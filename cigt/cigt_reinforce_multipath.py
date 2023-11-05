from collections import OrderedDict

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.custom_layers.basic_block_with_cbam import BasicBlockWithCbam


class CigtReinforceMultipath(CigtIgGatherScatterImplementation):
    def __init__(self, configs, run_id, model_definition, num_classes, model_mac_info, is_debug_mode):
        super().__init__(configs, run_id, model_definition, num_classes)
        self.iteration_id = 0
        # Parameters for the Policy Network Architecture
        self.policyNetworksCbamLayerCount = configs.policy_networks_cbam_layer_count
        self.policyNetworksCbamFeatureMapCount = configs.policy_networks_cbam_feature_map_count
        self.policyNetworksCbamLayerInputReductionRatio = configs.policy_networks_cbam_layer_input_reduction_ratio
        self.policyNetworksCbamReductionRatio = configs.policy_networks_cbam_reduction_ratio
        self.policyNetworksLstmDimension = configs.policy_networks_lstm_dimension
        self.policyNetworksMacLambda = configs.policy_networks_mac_lambda
        self.policyNetworksLogitTemperature = configs.policy_networks_logit_temperature
        self.policyNetworksNllConstant = 1e-10
        self.policyNetworksDiscountFactor = configs.policy_networks_discount_factor
        self.policyNetworksApplyRewardWhitening = configs.policy_networks_apply_reward_whitening

        # Inputs to the policy networks, per layer.

        # Parameters for the Policy Network Solver
        self.policyNetworkInitialLr = configs.policy_networks_initial_lr
        self.policyNetworkPolynomialSchedulerPower = configs.policy_networks_polynomial_scheduler_power
        self.policyNetworkWd = configs.policy_networks_wd

        # General Parameters for Training
        self.policyNetworkTotalNumOfEpochs = configs.policy_networks_total_num_of_epochs

        self.policyGradientsCrossEntropy = nn.CrossEntropyLoss(reduction="none")
        self.policyGradientsNegativeLogLikelihood = nn.NLLLoss(reduction="none")

        self.macCountsPerBlock = model_mac_info
        self.singlePathMacCost = sum([sum(d_.values()) for d_ in self.macCountsPerBlock])
        self.macCostPerLayer = torch.from_numpy(
            np.array([sum(d_.values()) for d_ in self.macCountsPerBlock])).to(self.device).to(torch.float32)
        self.isDebugMode = is_debug_mode

        self.policyNetworks = nn.ModuleList()
        self.valueNetworks = nn.ModuleList()

        self.create_policy_networks()
        self.policyGradientsModelOptimizer = None

        self.create_value_networks()
        self.valueModelOptimizer = None

        self.discountCoefficientArray = []
        for step_id in range(len(self.pathCounts) - 1):
            self.discountCoefficientArray.append(torch.pow(torch.tensor(self.policyNetworksDiscountFactor), step_id))
        self.discountCoefficientArray = torch.tensor(self.discountCoefficientArray)

        print("X")

    def test_convert_actions_to_routing_matrix(self,
                                               rl_hard_routing_matrix,
                                               p_n_given_x_soft,
                                               layer_sample_indices_unified,
                                               actions):
        rl_hard_routing_matrix_np = rl_hard_routing_matrix.detach().cpu().numpy()
        rl_hard_routing_matrix_np_test = np.zeros_like(rl_hard_routing_matrix_np)
        p_n_given_x_soft_np = p_n_given_x_soft.detach().cpu().numpy()
        layer_sample_indices_unified_np = layer_sample_indices_unified.detach().cpu().numpy()
        for idx in range(layer_sample_indices_unified_np.shape[0]):
            sample_id = layer_sample_indices_unified_np[idx]
            action_sample = actions.detach().cpu().numpy()[sample_id]
            routing_probs = p_n_given_x_soft_np[idx]
            paths_sorted = np.argsort(routing_probs)[::-1]
            for pid in range(action_sample + 1):
                rl_hard_routing_matrix_np_test[idx, paths_sorted[pid]] = 1.0
        assert np.array_equal(rl_hard_routing_matrix_np, rl_hard_routing_matrix_np_test)

    def test_format_input_for_policy_network_formatting_data(self,
                                                             pg_input,
                                                             layer_outputs_unified,
                                                             index_array,
                                                             pg_input_shape):
        pg_input_np = pg_input.detach().cpu().numpy()
        layer_outputs_unified_np = layer_outputs_unified.detach().cpu().numpy()
        index_array_np = index_array.detach().cpu().numpy()
        index_combinations = pg_input_shape[:-len(layer_outputs_unified.shape[1:])]
        index_combinations = [list(range(e_)) for e_ in index_combinations]
        index_combinations = Utilities.get_cartesian_product(list_of_lists=index_combinations)
        index_dict = {tuple(index_array_np[row_idx]): row_idx for row_idx in range(index_array_np.shape[0])}
        for idx_tpl in index_combinations:
            if idx_tpl not in index_dict:
                assert np.array_equal(pg_input_np[idx_tpl], np.zeros_like(pg_input_np[idx_tpl]))
            else:
                assert np.array_equal(pg_input_np[idx_tpl], layer_outputs_unified_np[index_dict[idx_tpl]])

    def test_format_input_for_policy_network_linearizing_data(self,
                                                              pg_input,
                                                              pg_input_linearized,
                                                              current_route_combinations,
                                                              channel_count):
        # Test that linearization has been correctly done.
        pg_input_np = pg_input.detach().cpu().numpy()
        pg_input_linearized_np = pg_input_linearized.detach().cpu().numpy()
        dimension_combinations = Utilities.get_cartesian_product(
            list_of_lists=[list(range(e_)) for e_ in current_route_combinations])
        for sample_id in range(pg_input_np.shape[0]):
            for tpl_id, tpl in enumerate(dimension_combinations):
                A_ = pg_input_np[(sample_id, *tpl)]
                B_ = pg_input_linearized_np[sample_id, tpl_id * channel_count:(tpl_id + 1) * channel_count]
                assert np.array_equal(A_, B_)

    def test_cross_entropy_loss_calculation(self,
                                            batch_size,
                                            labels_original,
                                            list_of_logits_unified,
                                            layer_sample_indices_unified,
                                            moe_probabilities_calculated,
                                            nll_calculated,
                                            accuracy_calculated):
        softmax_probs_np = torch.softmax(list_of_logits_unified / self.policyNetworksLogitTemperature,
                                         dim=1).detach().cpu().numpy()
        layer_sample_indices_unified_np = layer_sample_indices_unified.detach().cpu().numpy()
        assert softmax_probs_np.shape[0] == layer_sample_indices_unified_np.shape[0]
        probs_dict = {}
        for idx in range(layer_sample_indices_unified_np.shape[0]):
            sample_id = layer_sample_indices_unified_np[idx]
            if sample_id not in probs_dict:
                probs_dict[sample_id] = []
            probs_dict[sample_id].append(softmax_probs_np[idx])
        # Now each list in probs_dict contains all posterior probabilities for every sample. Calculate the mean
        # posteriors.
        moe_probabilities = []
        for idx in range(batch_size):
            assert idx in probs_dict
            expert_probs = probs_dict[idx]
            expert_probs = np.stack(expert_probs, axis=0)
            final_probs = np.mean(expert_probs, axis=0)
            moe_probabilities.append(final_probs)
        moe_probabilities = np.stack(moe_probabilities, axis=0)
        moe_probabilities_calculated_np = moe_probabilities_calculated.detach().cpu().numpy()
        # Assert that the calculated posteriors are the same for each sample.
        assert np.allclose(moe_probabilities, moe_probabilities_calculated_np)
        moe_probabilities_torch = torch.from_numpy(moe_probabilities).to(self.device)
        nll_loss = nn.NLLLoss()
        log_moe_probabilities_torch = torch.log(moe_probabilities_torch + self.policyNetworksNllConstant)
        loss_1 = nll_loss(log_moe_probabilities_torch, labels_original).detach().cpu().numpy()
        loss_2 = np.mean(nll_calculated.detach().cpu().numpy())
        assert np.allclose(loss_1, loss_2)
        # Accuracy control
        predicted_labels = np.argmax(moe_probabilities, axis=1)
        accuracy_1 = np.mean(predicted_labels == labels_original.detach().cpu().numpy())
        assert np.array_equal(accuracy_1, accuracy_calculated.detach().cpu().numpy())

    def test_mac_calculation(self,
                             batch_size,
                             list_of_sample_indices_per_layer,
                             mac_list_calculated):
        mac_list = []
        for layer_id in range(len(self.pathCounts)):
            macs_per_sample = []
            sample_id_list = list_of_sample_indices_per_layer[layer_id + 1].detach().cpu().numpy()
            for sample_id in range(batch_size):
                block_count = np.sum(sample_id_list == sample_id)
                macs_per_sample.append(block_count)
            macs_per_sample = np.array(macs_per_sample)
            mac_list.append(macs_per_sample)

        for layer_id in range(len(self.pathCounts)):
            assert np.array_equal(mac_list_calculated.detach().cpu().numpy()[:, layer_id],
                                  mac_list[layer_id])

    def create_policy_networks(self):
        for layer_id, path_count in enumerate(self.pathCounts[1:]):
            layers = OrderedDict()
            action_space_size = path_count
            if self.policyNetworksCbamLayerInputReductionRatio > 1:
                conv_block_reduction_layer = nn.MaxPool2d(
                    kernel_size=self.policyNetworksCbamLayerInputReductionRatio,
                    stride=self.policyNetworksCbamLayerInputReductionRatio)
                layers["policy_gradients_block_{0}_max_pool_dimension_reduction_layer".format(layer_id)] \
                    = conv_block_reduction_layer

            # Network entry
            # input_layer = nn.LazyConv2d(
            #     kernel_size=1,
            #     out_channels=self.policyNetworksCbamFeatureMapCount
            # )
            # Cifar10ResnetCigtConfigs.layer_config_list = [
            #     {"path_count": 1,
            #      "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
            #     {"path_count": 2,
            #      "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
            #                          {"layer_count": 18, "feature_map_count": 16}]},
            #     {"path_count": 4,
            #      "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]

            single_path_feature_count = self.layerConfigList[layer_id]["layer_structure"][-1]["feature_map_count"]
            current_route_combinations = self.pathCounts[:(layer_id + 1)]
            input_feature_count = np.prod(current_route_combinations) * single_path_feature_count
            input_layer = nn.Conv2d(
                kernel_size=1,
                in_channels=input_feature_count,
                out_channels=self.policyNetworksCbamFeatureMapCount
            )
            layers["policy_gradients_input_block_{0}".format(layer_id)] = input_layer

            for cid in range(self.policyNetworksCbamLayerCount):
                block = BasicBlockWithCbam(in_planes=self.policyNetworksCbamFeatureMapCount,
                                           planes=self.policyNetworksCbamFeatureMapCount,
                                           stride=1,
                                           cbam_reduction_ratio=self.policyNetworksCbamReductionRatio,
                                           norm_type=self.batchNormType)
                layers["policy_gradients_block_{0}_cbam_layer_{1}".format(layer_id, cid)] = block

            layers["policy_gradients_block_{0}_avg_pool".format(layer_id)] = nn.AvgPool2d(
                self.decisionAveragePoolingStrides[layer_id],
                stride=self.decisionAveragePoolingStrides[layer_id])
            layers["policy_gradients_block_{0}_flatten".format(layer_id)] = nn.Flatten()
            # layers["policy_gradients_block_{0}_relu".format(layer_id)] = nn.ReLU()
            layers["policy_gradients_block_{0}_feature_fc".format(layer_id)] = nn.LazyLinear(
                out_features=self.decisionDimensions[layer_id])
            layers["policy_gradients_block_{0}_action_space_fc".format(layer_id)] = nn.Linear(
                in_features=self.decisionDimensions[layer_id], out_features=action_space_size)
            layers["policy_gradients_block_{0}_softmax".format(layer_id)] = nn.Softmax(dim=1)

            policy_gradient_network_backbone = nn.Sequential(layers)
            self.policyNetworks.append(policy_gradient_network_backbone)

    def create_value_networks(self):
        for layer_id, path_count in enumerate(self.pathCounts[1:]):
            layers = OrderedDict()
            if self.policyNetworksCbamLayerInputReductionRatio > 1:
                conv_block_reduction_layer = nn.MaxPool2d(
                    kernel_size=self.policyNetworksCbamLayerInputReductionRatio,
                    stride=self.policyNetworksCbamLayerInputReductionRatio)
                layers["value_networks_block_{0}_max_pool_dimension_reduction_layer".format(layer_id)] \
                    = conv_block_reduction_layer

            # Network entry
            # input_layer = nn.LazyConv2d(
            #     kernel_size=1,
            #     out_channels=self.policyNetworksCbamFeatureMapCount
            # )
            # Cifar10ResnetCigtConfigs.layer_config_list = [
            #     {"path_count": 1,
            #      "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
            #     {"path_count": 2,
            #      "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
            #                          {"layer_count": 18, "feature_map_count": 16}]},
            #     {"path_count": 4,
            #      "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]

            single_path_feature_count = self.layerConfigList[layer_id]["layer_structure"][-1]["feature_map_count"]
            current_route_combinations = self.pathCounts[:(layer_id + 1)]
            input_feature_count = np.prod(current_route_combinations) * single_path_feature_count
            input_layer = nn.Conv2d(
                kernel_size=1,
                in_channels=input_feature_count,
                out_channels=self.policyNetworksCbamFeatureMapCount
            )
            layers["value_networks_input_block_{0}".format(layer_id)] = input_layer

            for cid in range(self.policyNetworksCbamLayerCount):
                block = BasicBlockWithCbam(in_planes=self.policyNetworksCbamFeatureMapCount,
                                           planes=self.policyNetworksCbamFeatureMapCount,
                                           stride=1,
                                           cbam_reduction_ratio=self.policyNetworksCbamReductionRatio,
                                           norm_type=self.batchNormType)
                layers["value_networks_block_{0}_cbam_layer_{1}".format(layer_id, cid)] = block

            layers["value_networks_block_{0}_avg_pool".format(layer_id)] = nn.AvgPool2d(
                self.decisionAveragePoolingStrides[layer_id],
                stride=self.decisionAveragePoolingStrides[layer_id])
            layers["value_networks_block_{0}_flatten".format(layer_id)] = nn.Flatten()
            # layers["policy_gradients_block_{0}_relu".format(layer_id)] = nn.ReLU()
            layers["value_networks_block_{0}_feature_fc".format(layer_id)] = nn.LazyLinear(
                out_features=self.decisionDimensions[layer_id])
            layers["value_networks_block_{0}_value_fc".format(layer_id)] = nn.Linear(
                in_features=self.decisionDimensions[layer_id], out_features=1)
            # layers["value_networks_block_{0}_softmax".format(layer_id)] = nn.Softmax(dim=1)

            value_network_backbone = nn.Sequential(layers)
            self.valueNetworks.append(value_network_backbone)

    def create_optimizer(self):
        paths = []
        for pc in self.pathCounts:
            paths.append([i_ for i_ in range(pc)])
        path_variaties = Utilities.get_cartesian_product(list_of_lists=paths)

        for idx in range(len(self.pathCounts)):
            cnt = len([tpl for tpl in path_variaties if tpl[idx] == 0])
            self.layerCoefficients.append(len(path_variaties) / cnt)

        # Create parameter groups per CIGT layer and shared parameters
        shared_parameters = []
        parameters_per_cigt_layers = []
        for idx in range(len(self.pathCounts)):
            parameters_per_cigt_layers.append([])
        # Policy Network parameters.
        policy_networks_parameters = []
        # Value Networks parameters.
        value_networks_parameters = []

        for name, param in self.named_parameters():
            assert not (("cigtLayers" in name and "policyNetworks" in name) or
                        ("cigtLayers" in name and "valueNetworks" in name) or
                        ("policyNetworks" in name and "valueNetworks" in name))
            if "cigtLayers" in name:
                assert "policyNetworks" not in name and "valueNetworks" not in name
                param_name_splitted = name.split(".")
                layer_id = int(param_name_splitted[1])
                assert 0 <= layer_id <= len(self.pathCounts) - 1
                parameters_per_cigt_layers[layer_id].append(param)
            elif "policyNetworks" in name:
                assert "cigtLayers" not in name and "valueNetworks" not in name
                policy_networks_parameters.append(param)
            elif "valueNetworks" in name:
                assert "cigtLayers" not in name and "policyNetworks" not in name
                value_networks_parameters.append(param)
            else:
                shared_parameters.append(param)

        num_shared_parameters = len(shared_parameters)
        num_policy_network_parameters = len(policy_networks_parameters)
        num_value_networks_parameters = len(value_networks_parameters)
        num_cigt_layer_parameters = sum([len(arr) for arr in parameters_per_cigt_layers])
        num_all_parameters = len([tpl for tpl in self.named_parameters()])
        assert num_shared_parameters + num_policy_network_parameters + \
               num_value_networks_parameters + num_cigt_layer_parameters == num_all_parameters

        parameter_groups = []
        # Add parameter groups with respect to their cigt layers
        for layer_id in range(len(self.pathCounts)):
            parameter_groups.append(
                {'params': parameters_per_cigt_layers[layer_id],
                 # 'lr': self.initialLr * self.layerCoefficients[layer_id],
                 'lr': self.initialLr,
                 'weight_decay': self.classificationWd})

        # Shared parameters, always the group
        parameter_groups.append(
            {'params': shared_parameters,
             'lr': self.initialLr,
             'weight_decay': self.classificationWd})

        if self.optimizerType == "SGD":
            model_optimizer = optim.SGD(parameter_groups, momentum=0.9)
        elif self.optimizerType == "Adam":
            model_optimizer = optim.Adam(parameter_groups)
        else:
            raise ValueError("{0} is not supported as optimizer.".format(self.optimizerType))

        # Create a separate optimizer that only optimizes the policy networks.
        policy_networks_optimizer = optim.AdamW(
            [{'params': policy_networks_parameters,
              'lr': self.policyNetworkInitialLr,
              'weight_decay': self.policyNetworkWd}])

        # Create a separate optimizer that only optimizers the value networks. Use the same parameters with the policy
        # network.
        value_networks_optimizer = optim.AdamW(
            [{'params': value_networks_parameters,
              'lr': self.policyNetworkInitialLr,
              'weight_decay': self.policyNetworkWd}])

        return model_optimizer, policy_networks_optimizer, value_networks_optimizer

    def adjust_learning_rate_polynomial(self, iteration, num_of_total_iterations):
        lr = self.policyNetworkInitialLr
        where = np.clip(iteration / num_of_total_iterations, a_min=0.0, a_max=1.0)
        modified_lr = lr * (1 - where) ** self.policyNetworkPolynomialSchedulerPower
        self.policyGradientsModelOptimizer.param_groups[0]['lr'] = modified_lr

    def merge_cigt_outputs_into_structured_array(self,
                                                 layer_id,
                                                 batch_size,
                                                 layer_outputs_unified,
                                                 layer_sample_indices_unified,
                                                 layer_route_indices_unified):
        # Step 1: Reshape all intermediate node outputs into
        # pg_input.shape = (batch_size, *route_dim, channels, feature_width, feature_height) format.
        current_route_combinations = self.pathCounts[:(layer_id + 1)]
        channel_count = layer_outputs_unified.shape[1]
        pg_input_shape = (batch_size, *current_route_combinations, *layer_outputs_unified.shape[1:])
        pg_input = torch.zeros(size=pg_input_shape, dtype=layer_outputs_unified.dtype, device=self.device)

        # Step 2: Put all samples in layer_outputs_unified into the intermediate input array.
        index_array = torch.concat([torch.unsqueeze(layer_sample_indices_unified, dim=-1),
                                    layer_route_indices_unified[:, 1:]], dim=1)
        index_array_list = [index_array[:, c] for c in range(index_array.shape[1])]
        pg_input[index_array_list] = layer_outputs_unified

        # Comment out the following lines, these are for just for testing if the copy operation is done correctly.
        # ************** TEST **************
        if self.isDebugMode:
            self.test_format_input_for_policy_network_formatting_data(
                pg_input=pg_input,
                layer_outputs_unified=layer_outputs_unified,
                index_array=index_array,
                pg_input_shape=pg_input_shape)
        # ************** TEST **************
        return pg_input

    def format_input_for_policy_network(self,
                                        layer_id,
                                        batch_size,
                                        layer_outputs_unified,
                                        layer_sample_indices_unified,
                                        layer_route_indices_unified):
        current_route_combinations = self.pathCounts[:(layer_id + 1)]
        channel_count = layer_outputs_unified.shape[1]
        pg_input = self.merge_cigt_outputs_into_structured_array(
            layer_id=layer_id,
            batch_size=batch_size,
            layer_outputs_unified=layer_outputs_unified,
            layer_sample_indices_unified=layer_sample_indices_unified,
            layer_route_indices_unified=layer_route_indices_unified)
        # Step 3: Reshape the input array, such that input arrays are concatenated along the channels.
        # pg_input.shape = (batch_size, *route_dim, channels, feature_width, feature_height) will be transformed
        # into (batch_size, np.prod(route_dim) * channels, feature_width, feature_height).
        pg_input_linearized = torch.reshape(pg_input, shape=(batch_size, -1, *layer_outputs_unified.shape[2:]))
        # ************** TEST **************
        if self.isDebugMode:
            self.test_format_input_for_policy_network_linearizing_data(
                pg_input=pg_input,
                pg_input_linearized=pg_input_linearized,
                current_route_combinations=current_route_combinations,
                channel_count=channel_count)
        # ************** TEST **************
        return pg_input_linearized

    def select_action(self, layer_id, pg_input, sample_action):
        # Step 1: Feed the input into the policy gradient.
        print("Layer{0} input shape:{1}".format(layer_id, pg_input.shape))
        action_probs = self.policyNetworks[layer_id](pg_input)
        # sample an action using the probability distribution
        dist = Categorical(action_probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def convert_actions_to_routing_matrix(self, actions, p_n_given_x_soft, layer_sample_indices_unified):
        rl_hard_routing_matrix = torch.zeros(size=p_n_given_x_soft.shape, dtype=p_n_given_x_soft.dtype)
        actions_per_sample_and_path = actions[layer_sample_indices_unified]
        path_selections_sorted = torch.argsort(p_n_given_x_soft, dim=1, descending=True)
        batch_indices = torch.arange(p_n_given_x_soft.shape[0])
        for col_id in range(p_n_given_x_soft.shape[1]):
            choice_vector = actions_per_sample_and_path >= col_id
            layer_sample_indices_selected = batch_indices[choice_vector]
            path_selections_sorted_selected = path_selections_sorted[choice_vector][:, col_id]
            rl_hard_routing_matrix[(layer_sample_indices_selected, path_selections_sorted_selected)] = 1.0
        # Comment out the following lines, these are for just for testing if the routing matrix is prepared correctly,
        # ************** TEST **************
        if self.isDebugMode:
            self.test_convert_actions_to_routing_matrix(rl_hard_routing_matrix=rl_hard_routing_matrix,
                                                        actions=actions,
                                                        layer_sample_indices_unified=layer_sample_indices_unified,
                                                        p_n_given_x_soft=p_n_given_x_soft)
        # ************** TEST **************
        return rl_hard_routing_matrix

    def calculate_final_reward(self,
                               batch_size,
                               labels_original,
                               list_of_logits_unified,
                               layer_labels_unified,
                               list_of_sample_indices_per_layer,
                               list_of_route_indices_per_layer):
        layer_sample_indices_unified = list_of_sample_indices_per_layer[-1]
        layer_route_indices_unified = list_of_route_indices_per_layer[-1]
        # PART 1: CALCULATE THE MULTIPATH ACCURACY
        list_of_logits_unified_tempered = list_of_logits_unified / self.policyNetworksLogitTemperature
        class_probabilities_unified = torch.softmax(list_of_logits_unified_tempered, dim=1)
        class_probabilities_structured = self.merge_cigt_outputs_into_structured_array(
            layer_id=len(self.pathCounts) - 1,
            batch_size=batch_size,
            layer_outputs_unified=class_probabilities_unified,
            layer_sample_indices_unified=layer_sample_indices_unified,
            layer_route_indices_unified=layer_route_indices_unified)

        layer_labels_unified = torch.unsqueeze(layer_labels_unified, dim=1)
        layer_labels_structured = self.merge_cigt_outputs_into_structured_array(
            layer_id=len(self.pathCounts) - 1,
            batch_size=batch_size,
            layer_outputs_unified=layer_labels_unified,
            layer_sample_indices_unified=layer_sample_indices_unified,
            layer_route_indices_unified=layer_route_indices_unified)

        # Calculate probability outputs
        # Reshape the structured probabilities array, such that input arrays are concatenated along the channels.
        # class_probabilities_structured.shape = (batch_size, *route_dim, number_of_classes) will be transformed
        # into (batch_size, np.prod(route_dim), number_of_classes).
        path_combination_dimensions = class_probabilities_structured.shape[1:-1]
        class_probabilities_linearized = torch.reshape(class_probabilities_structured,
                                                       shape=(batch_size,
                                                              np.prod(path_combination_dimensions),
                                                              class_probabilities_structured.shape[-1]))
        expert_counts = torch.sum(class_probabilities_linearized, dim=(1, 2))
        class_probabilities_summed = torch.sum(class_probabilities_linearized, dim=1)
        expert_coefficients = torch.reciprocal(expert_counts)
        expert_probabilities = class_probabilities_summed * torch.unsqueeze(expert_coefficients, dim=1)

        # Linearize the labels
        layer_labels_linearized = torch.reshape(layer_labels_structured,
                                                shape=(batch_size,
                                                       np.prod(path_combination_dimensions),
                                                       layer_labels_structured.shape[-1]))
        layer_labels_sum = torch.squeeze(torch.sum(layer_labels_linearized, dim=(1, 2)))
        layer_labels_final = torch.round(layer_labels_sum * expert_coefficients)
        layer_labels_final = layer_labels_final.to(torch.int64)

        # cross_entropy_loss_per_sample = self.policyGradientsCrossEntropy(expert_probabilities, layer_labels_final)
        log_expert_probabilities = torch.log(expert_probabilities + self.policyNetworksNllConstant)
        nll_per_sample = self.policyGradientsNegativeLogLikelihood(log_expert_probabilities, layer_labels_final)
        predicted_labels = torch.argmax(expert_probabilities, dim=1)
        accuracy_vector = predicted_labels == layer_labels_final
        accuracy = torch.mean(accuracy_vector.to(expert_probabilities.dtype))
        # PART 1: CALCULATE THE MULTIPATH ACCURACY

        # PART 2: CALCULATE THE MAC LOSS
        assert len(list_of_sample_indices_per_layer) == len(list_of_route_indices_per_layer)
        mac_list = []
        for layer_id in range(len(self.pathCounts)):
            sample_indices = list_of_sample_indices_per_layer[layer_id + 1]
            route_indices = list_of_route_indices_per_layer[layer_id + 1]
            current_route_combinations = self.pathCounts[:(layer_id + 1)]
            block_process_shape = (batch_size, *current_route_combinations)
            block_process_array = torch.zeros(size=block_process_shape, device=self.device)
            index_array = torch.concat([torch.unsqueeze(sample_indices, dim=-1), route_indices[:, 1:]], dim=1)
            index_array_list = [index_array[:, c] for c in range(index_array.shape[1])]
            block_process_array[index_array_list] = 1
            mac_list_layer = \
                torch.sum(block_process_array, dim=tuple([i_ + 1 for i_ in range(len(current_route_combinations))]))
            mac_list.append(mac_list_layer)
        mac_list = torch.stack(mac_list, dim=1)
        mac_list_op_count = mac_list * torch.unsqueeze(self.macCostPerLayer, dim=0)
        mac_per_sample = torch.sum(mac_list_op_count, dim=1)
        # PART 2: CALCULATE THE MAC LOSS

        cross_entropy_rewards = -1.0 * nll_per_sample
        mac_rewards = -1.0 * (mac_per_sample / self.singlePathMacCost - 1.0)
        final_rewards = (1.0 - self.policyNetworksMacLambda) * cross_entropy_rewards + \
                        self.policyNetworksMacLambda * mac_rewards

        if self.isDebugMode:
            assert np.array_equal(layer_labels_final.numpy(), labels_original.numpy())
            self.test_cross_entropy_loss_calculation(
                batch_size=batch_size,
                nll_calculated=nll_per_sample,
                labels_original=labels_original,
                layer_sample_indices_unified=layer_sample_indices_unified,
                list_of_logits_unified=list_of_logits_unified,
                moe_probabilities_calculated=expert_probabilities,
                accuracy_calculated=accuracy)

            self.test_mac_calculation(batch_size=batch_size,
                                      list_of_sample_indices_per_layer=list_of_sample_indices_per_layer,
                                      mac_list_calculated=mac_list)
        return expert_probabilities, final_rewards

    def forward(self, x, labels, temperature):
        sample_indices = torch.arange(0, labels.shape[0], device=self.device)
        balance_coefficient_list = self.informationGainBalanceCoeffList
        policy_gradient_network_states = []
        policy_gradient_network_actions = []
        policy_gradient_network_log_probs = []
        policy_gradient_network_rewards = []
        # Initial layer
        net = self.preprocess_input(x=x)
        layer_outputs = [{"net": net,
                          "labels": labels,
                          "sample_indices": sample_indices,
                          "routing_matrix_hard": torch.ones(size=(x.shape[0], 1),
                                                            dtype=torch.float32,
                                                            device=self.device),
                          "routing_matrices_soft": torch.ones(size=(x.shape[0], 1),
                                                              dtype=torch.float32,
                                                              device=self.device),
                          "routing_activations": torch.ones(size=(x.shape[0], 1),
                                                            dtype=torch.float32,
                                                            device=self.device),
                          "route_indices": torch.zeros(size=(x.shape[0], 1),
                                                       dtype=torch.int64,
                                                       device=self.device)}]

        for layer_id, cigt_layer_blocks in enumerate(self.cigtLayers):
            net_masked = self.divide_tensor_wrt_routing_matrix(
                tens=layer_outputs[-1]["net"],
                routing_matrix=layer_outputs[-1]["routing_matrix_hard"])
            labels_masked = self.divide_tensor_wrt_routing_matrix(
                tens=layer_outputs[-1]["labels"],
                routing_matrix=layer_outputs[-1]["routing_matrix_hard"])
            sample_indices_masked = self.divide_tensor_wrt_routing_matrix(
                tens=layer_outputs[-1]["sample_indices"],
                routing_matrix=layer_outputs[-1]["routing_matrix_hard"])
            route_indices_masked = self.divide_tensor_wrt_routing_matrix(
                tens=layer_outputs[-1]["route_indices"],
                routing_matrix=layer_outputs[-1]["routing_matrix_hard"])

            curr_layer_outputs = []
            curr_route_indices = []

            for block_id, block_obj in enumerate(cigt_layer_blocks):
                block_output = block_obj(net_masked[block_id])
                route_indices_block = route_indices_masked[block_id]
                curr_layer_outputs.append(block_output)
                block_indices_arr = block_id * torch.ones(size=(block_output.shape[0], 1),
                                                          dtype=torch.int64, device=self.device)
                route_indices_block = torch.concat([route_indices_block, block_indices_arr], dim=1)
                curr_route_indices.append(route_indices_block)

            layer_outputs_unified = torch.concat(curr_layer_outputs, dim=0)
            layer_labels_unified = torch.concat(labels_masked, dim=0)
            layer_sample_indices_unified = torch.concat(sample_indices_masked, dim=0)
            layer_route_indices_unified = torch.concat(curr_route_indices, dim=0)

            # Routing Layer
            if layer_id < len(self.cigtLayers) - 1:
                # Calculate routing weights for the next layer
                p_n_given_x_soft, routing_activations = self.blockEndLayers[layer_id](layer_outputs_unified,
                                                                                      layer_labels_unified,
                                                                                      temperature,
                                                                                      balance_coefficient_list[
                                                                                          layer_id])

                # Format layer outputs such that they are suitable for the policy gradients
                pg_input_linearized = self.format_input_for_policy_network(
                    layer_id=layer_id,
                    batch_size=x.shape[0],
                    layer_outputs_unified=layer_outputs_unified,
                    layer_sample_indices_unified=layer_sample_indices_unified,
                    layer_route_indices_unified=layer_route_indices_unified
                )
                policy_gradient_network_states.append(pg_input_linearized)

                # Sample action from the policy network
                actions, log_probs = \
                    self.select_action(layer_id=layer_id, pg_input=pg_input_linearized, sample_action=self.training)
                policy_gradient_network_actions.append(actions)
                policy_gradient_network_log_probs.append(log_probs)
                if layer_id < len(self.pathCounts) - 2:
                    policy_gradient_network_rewards.append(torch.zeros_like(log_probs))

                # Create appropriate hard routing matrices with respect to chosen actions
                rl_hard_routing_matrix = self.convert_actions_to_routing_matrix(
                    actions=actions,
                    p_n_given_x_soft=p_n_given_x_soft,
                    layer_sample_indices_unified=layer_sample_indices_unified)

                # Calculate the hard routing matrix
                p_n_given_x_hard = self.routingManager.get_hard_routing_matrix(
                    model=self,
                    layer_id=layer_id,
                    p_n_given_x_soft=p_n_given_x_soft,
                    p_n_given_x_hard=rl_hard_routing_matrix)
                layer_outputs.append({"net": layer_outputs_unified,
                                      "labels": layer_labels_unified,
                                      "sample_indices": layer_sample_indices_unified,
                                      "route_indices": layer_route_indices_unified,
                                      "routing_matrix_hard": p_n_given_x_hard,
                                      "routing_matrices_soft": p_n_given_x_soft,
                                      "rl_hard_routing_matrix": rl_hard_routing_matrix,
                                      "routing_activations": routing_activations})
            # Loss Layer
            else:
                if self.lossCalculationKind == "SingleLogitSingleLoss":
                    raise NotImplementedError()
                # Calculate logits with all block separately
                elif self.lossCalculationKind == "MultipleLogitsMultipleLosses" \
                        or self.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
                    list_of_logits = self.calculate_logits(p_n_given_x_hard=None,
                                                           loss_block_outputs=curr_layer_outputs)
                    logits_unified = torch.concat(list_of_logits, dim=0)
                    list_of_route_indices_per_layer = [d_["route_indices"] for d_ in layer_outputs]
                    list_of_route_indices_per_layer.append(layer_route_indices_unified)
                    list_of_sample_indices_per_layer = [d_["sample_indices"] for d_ in layer_outputs]
                    list_of_sample_indices_per_layer.append(layer_sample_indices_unified)

                    expert_probabilities, final_rewards = self.calculate_final_reward(
                        batch_size=x.shape[0],
                        list_of_logits_unified=logits_unified,
                        list_of_route_indices_per_layer=list_of_route_indices_per_layer,
                        list_of_sample_indices_per_layer=list_of_sample_indices_per_layer,
                        layer_labels_unified=layer_labels_unified,
                        labels_original=labels)
                    # No gradients should flow from the rewards.
                    final_rewards = final_rewards.detach()
                    policy_gradient_network_rewards.append(final_rewards)

                    layer_outputs.append({"net": layer_outputs_unified,
                                          "labels": layer_labels_unified,
                                          "sample_indices": layer_sample_indices_unified,
                                          "route_indices": layer_route_indices_unified,
                                          "labels_masked": labels_masked,
                                          "list_of_logits": list_of_logits,
                                          "logits_unified": logits_unified,
                                          "expert_probabilities": expert_probabilities,
                                          "final_rewards": final_rewards,
                                          "policy_gradient_network_states": policy_gradient_network_states,
                                          "policy_gradient_network_actions": policy_gradient_network_actions,
                                          "policy_gradient_network_log_probs": policy_gradient_network_log_probs,
                                          "policy_gradient_network_rewards": policy_gradient_network_rewards
                                          })

                else:
                    raise ValueError("Unknown logit calculation method: {0}".format(self.lossCalculationKind))

        return layer_outputs

    def process_rewards(self, batch_size, network_rewards):
        ''' Converts our rewards history into cumulative discounted rewards
        Args:
        - rewards (Array): array of rewards

        Returns:
        - G (Array): array of cumulative discounted rewards
        '''

        rewards_matrix = torch.stack(network_rewards, dim=1)
        cumulative_rewards = []

        for t_ in range(rewards_matrix.shape[1]):
            discounts_arr = self.discountCoefficientArray[0:rewards_matrix.shape[1] - t_]
            discounts_arr = torch.unsqueeze(discounts_arr, dim=0)
            rewards_partial = rewards_matrix[:, t_:]
            rewards_partial_discounted = rewards_partial * discounts_arr
            cumulative_reward = torch.sum(rewards_partial_discounted, dim=1)
            cumulative_rewards.append(cumulative_reward)

        if self.isDebugMode:
            sample_reward_arrays = []
            for sample_id in range(batch_size):
                rewards = []
                for layer_id, rarr in enumerate(network_rewards):
                    rewards.append(rarr[sample_id])

                # Calculate Gt (cumulative discounted rewards)
                G = []

                # track cumulative reward
                total_r = 0

                # iterate rewards from Gt to G0
                for r in reversed(rewards):
                    # Base case: G(T) = r(T)
                    # Recursive: G(t) = r(t) + G(t+1)^DISCOUNT
                    total_r = r + total_r * self.policyNetworksDiscountFactor

                    # add to front of G
                    G.insert(0, total_r)

                # whitening rewards
                G = torch.tensor(G).to(self.device)
                if self.policyNetworksApplyRewardWhitening:
                    G = (G - G.mean()) / G.std()
                sample_reward_arrays.append(G)

            sample_reward_arrays = torch.stack(sample_reward_arrays, dim=0)

            for t_ in range(rewards_matrix.shape[1]):
                assert np.allclose(cumulative_rewards[t_].cpu().numpy(), sample_reward_arrays[:, t_].cpu().numpy())

        return cumulative_rewards

    def train_value_network(self, cumulative_rewards, network_states):
        # Calculate state values and train state value network
        # total_loss = torch.zeros_like(cumulative_rewards[0])
        total_loss_list = []
        for layer_id in range(len(self.pathCounts) - 1):
            layer_states = network_states[layer_id]
            value_predictions = self.valueNetworks[layer_id](layer_states)
            value_predictions = torch.squeeze(value_predictions)
            val_loss = F.mse_loss(cumulative_rewards[layer_id], value_predictions, reduction="none")
            total_loss_list.append(val_loss)
        # Backpropagate
        total_loss_matrix = torch.stack(total_loss_list, dim=1)
        total_loss = torch.mean(total_loss_matrix, dim=(0, 1))
        total_loss.backward()
        self.valueModelOptimizer.step()
        print("X")

    def fit_policy_network(self, train_loader, test_loader):
        self.to(self.device)
        torch.manual_seed(1)
        best_performance = 0.0
        num_of_total_iterations = self.policyNetworkTotalNumOfEpochs * len(train_loader)

        # Run a forward pass first to initialize each LazyXXX layer.
        self.execute_forward_with_random_input()
        # test_accuracy = self.validate(loader=test_loader, epoch=-1, data_kind="test", temperature=0.1)

        # Create the model optimizer, we should have every parameter initialized right now.
        self.modelOptimizer, self.policyGradientsModelOptimizer, self.valueModelOptimizer = self.create_optimizer()

        # self.test_route_indices(test_loader=test_loader)

        temp_warm_up_state = self.isInWarmUp
        temp_random_routing_ratio = self.routingRandomizationRatio
        self.isInWarmUp = False
        self.routingRandomizationRatio = -1.0
        # Train the policy network for one epoch
        iteration_id = 0
        for epoch_id in range(0, self.policyNetworkTotalNumOfEpochs):
            self.train()
            for i, (input_, target) in enumerate(train_loader):
                print("*************Policy Network Training Epoch:{0} Iteration:{1}*************".format(
                    epoch_id, self.iteration_id))

                # Print learning rates
                print("Policy Network Lr:{0}".format(self.policyGradientsModelOptimizer.param_groups[0]["lr"]))
                self.policyGradientsModelOptimizer.zero_grad()
                self.valueModelOptimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    input_var = torch.autograd.Variable(input_).to(self.device)
                    target_var = torch.autograd.Variable(target).to(self.device)
                    batch_size = input_var.size(0)
                    outputs = self(x=input_var, labels=target_var, temperature=1.0)
                    network_states = outputs[-1]["policy_gradient_network_states"]
                    network_rewards = outputs[-1]["policy_gradient_network_rewards"]
                    network_actions = outputs[-1]["policy_gradient_network_actions"]
                    network_log_probs = outputs[-1]["policy_gradient_network_log_probs"]

                    # Prepare cumulative reward arrays
                    cumulative_rewards = self.process_rewards(batch_size=batch_size, network_rewards=network_rewards)
                    # Train value network
                    self.train_value_network(cumulative_rewards=cumulative_rewards, network_states=network_states)


                self.iteration_id += 1

        self.isInWarmUp = temp_warm_up_state
        self.routingRandomizationRatio = temp_random_routing_ratio
