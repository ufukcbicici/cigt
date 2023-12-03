from collections import OrderedDict
from collections import Counter

import torch
import time
import inspect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from auxillary.db_logger import DbLogger
from auxillary.average_meter import AverageMeter
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.custom_layers.basic_block_with_cbam import BasicBlockWithCbam
from tqdm import tqdm


class CigtReinforceV2(CigtIgGatherScatterImplementation):
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
        self.policyNetworksEvaluationPeriod = configs.policy_networks_evaluation_period
        self.policyNetworksEnforcedActions = []
        self.policyNetworksUseMovingAverageBaseline = configs.policy_networks_use_moving_average_baseline
        self.policyNetworksBaselineMomentum = configs.policy_networks_baseline_momentum
        self.policyNetworksEntropyLossCoeff = configs.policy_networks_policy_entropy_loss_coeff

        # Inputs to the policy networks, per layer.

        # Parameters for the Policy Network Solver
        self.policyNetworkInitialLr = configs.policy_networks_initial_lr
        self.policyNetworkPolynomialSchedulerPower = configs.policy_networks_polynomial_scheduler_power
        self.policyNetworkWd = configs.policy_networks_wd

        # General Parameters for Training
        self.policyNetworkTotalNumOfEpochs = configs.policy_networks_total_num_of_epochs

        self.policyGradientsCrossEntropy = nn.CrossEntropyLoss(reduction="none")
        self.policyGradientsNegativeLogLikelihood = nn.NLLLoss(reduction="none")
        self.policyGradientsMSELosses = [nn.MSELoss(reduction="none") for _ in range(len(self.pathCounts) - 1)]

        self.macCountsPerBlock = model_mac_info
        self.singlePathMacCost = sum([sum(d_.values()) for d_ in self.macCountsPerBlock])
        self.macCostPerLayer = torch.from_numpy(
            np.array([sum(d_.values()) for d_ in self.macCountsPerBlock])).to(self.device).to(torch.float32)
        self.isDebugMode = is_debug_mode

        self.policyNetworks = nn.ModuleList()
        self.valueNetworks = nn.ModuleList()

        self.create_policy_networks()
        self.policyGradientsModelOptimizer = None

        # self.create_value_networks()
        # self.valueModelOptimizer = None
        # self.trainingMode = False

        self.baselinesPerLayer = []
        for step_id in range(len(self.pathCounts) - 1):
            self.baselinesPerLayer.append(None)

        self.discountCoefficientArray = []
        for step_id in range(len(self.pathCounts) - 1):
            self.discountCoefficientArray.append(torch.pow(torch.tensor(self.policyNetworksDiscountFactor), step_id))
        self.discountCoefficientArray = torch.tensor(self.discountCoefficientArray).to(self.device)

    def execute_forward_with_random_input(self):
        max_batch_size = np.prod(self.pathCounts) * self.batchSize
        self.enforcedRoutingMatrices = []
        for path_count in self.pathCounts[1:]:
            self.enforcedRoutingMatrices.append(torch.ones(size=(max_batch_size, path_count),
                                                           dtype=torch.int64).to(self.device))
        fake_input = torch.from_numpy(
            np.random.uniform(size=(self.batchSize, *self.inputDims)).astype(dtype=np.float32)).to(self.device)
        fake_target = torch.ones(size=(self.batchSize,), dtype=torch.int64).to(self.device)
        print("fake_input.device:{0}".format(fake_input.device))
        print("fake_target.device:{0}".format(fake_target.device))
        for name, param in self.named_parameters():
            print("Parameter {0} Device:{1}".format(name, param.device))

        self.eval()
        self.forward_with_policies(x=fake_input, y=fake_target, greedy_actions=True)
        self.enforcedRoutingMatrices = []

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

    # def prepare_policy_network_input_s(self, batch_size, layer_id, current_paths_dict, cigt_outputs):
    #     route_combinations = Utilities.create_route_combinations(shape_=self.pathCounts[:(layer_id + 1)])
    #     for sample_id in range(batch_size):
    #         current_route_combinations = self.pathCounts[:(layer_id + 1)]
    #         channel_count = layer_outputs_unified.shape[1]
    #         pg_input_shape = (batch_size, *current_route_combinations, *layer_outputs_unified.shape[1:])
    #         pg_input = torch.zeros(size=pg_input_shape, dtype=layer_outputs_unified.dtype, device=self.device)

    def prepare_policy_network_input_f(self, batch_size, layer_id, current_paths_dict, cigt_outputs):
        paths_to_this_layer = self.pathCounts[:(layer_id + 1)]
        route_combinations = Utilities.create_route_combinations(shape_=paths_to_this_layer)
        route_combinations_dict = {idx: path for idx, path in enumerate(route_combinations)}
        route_combinations_dict_inverse = {path: idx for idx, path in enumerate(route_combinations)}
        output_arrays = {path: cigt_outputs["block_outputs_dict"][path] for path in route_combinations}
        output_shape = list(set([arr.shape for arr in output_arrays.values()]))
        assert len(output_shape) == 1
        output_shape = output_shape[0]
        output_types = [arr.dtype for arr in output_arrays.values()]
        assert len(set(output_types)) == 1

        # pg_input_shape = (batch_size, *paths_to_this_layer, *output_shape[1:])
        # pg_dense_input = torch.zeros(size=pg_input_shape, dtype=output_types[0], device=self.device)

        # Put each input from each path to the corresponding slice of the dense array.
        pg_dense_input = []
        for idx in range(len(route_combinations)):
            pg_dense_input.append(output_arrays[route_combinations_dict[idx]])
        pg_dense_input = torch.stack(pg_dense_input, dim=1)

        # Sparsify input regions
        pg_sparsifier_array_shape = pg_dense_input.shape[0:2]
        pg_sparsifier_array = torch.zeros(size=pg_sparsifier_array_shape, dtype=output_types[0], device="cpu")
        for k, v in current_paths_dict.items():
            for idx in range(len(route_combinations)):
                path = route_combinations_dict[idx]
                if path in v:
                    pg_sparsifier_array[(k, idx)] = 1.0
        pg_sparsifier_array = pg_sparsifier_array.to(self.device)
        for _ in range(len(pg_dense_input.shape) - len(pg_sparsifier_array.shape)):
            pg_sparsifier_array = torch.unsqueeze(pg_sparsifier_array, dim=-1)

        # Sparsify the input array. Inactive input regions for a sample will be zero. Otherwise it is equal to the
        # original input.
        pg_sparse_input = pg_dense_input * pg_sparsifier_array

        # Check correctness
        if self.isDebugMode:
            index_arrs = [list(range(batch_size))]
            index_arrs.extend([list(range(pp)) for pp in paths_to_this_layer])
            index_combinations = Utilities.get_cartesian_product(index_arrs)
            for index_combination in index_combinations:
                sample_id = index_combination[0]
                path = tuple(index_combination[1:])
                s_ = current_paths_dict[sample_id]
                a_ = pg_sparse_input[(sample_id, route_combinations_dict_inverse[path])].detach().cpu().numpy()
                if path in s_:
                    b_ = output_arrays[path][sample_id].detach().cpu().numpy()
                    assert np.array_equal(a_, b_)
                else:
                    b_ = np.zeros_like(a_)
                    assert np.array_equal(a_, b_)

        pg_sparse_input = torch.reshape(pg_sparse_input, shape=(pg_sparse_input.shape[0],
                                                                pg_sparse_input.shape[1] * pg_sparse_input.shape[2],
                                                                *pg_sparse_input.shape[3:]))
        return pg_sparse_input

    def run_policy_networks(self, layer_id, pn_input):
        batch_size = pn_input.shape[0]
        policy_network = self.policyNetworks[layer_id]
        action_probs = policy_network(pn_input)
        log_action_probs = torch.log(action_probs + self.policyNetworksNllConstant)
        p_lp = torch.sum(-1.0 * (action_probs * log_action_probs), dim=1)
        mean_policy_entropy = torch.mean(p_lp)

        if len(self.policyNetworksEnforcedActions) == 0:
            dist = Categorical(action_probs)
            actions = dist.sample()
            log_probs_selected = dist.log_prob(actions)
            probs_selected = torch.exp(log_probs_selected)
            return mean_policy_entropy, log_probs_selected, probs_selected, action_probs, actions.detach().cpu().numpy()
        else:
            actions_enforced = self.policyNetworksEnforcedActions[layer_id][0:pn_input.shape[0]]
            probs_selected = action_probs[torch.arange(batch_size), actions_enforced]
            log_probs_selected = torch.log(probs_selected)
            return mean_policy_entropy, log_probs_selected, probs_selected, \
                action_probs, actions_enforced.detach().cpu().numpy()

    def extend_sample_trajectories_wrt_actions(self, layer_id, actions, cigt_outputs, current_paths_dict):
        paths_to_this_layer = self.pathCounts[:(layer_id + 1)]
        route_combinations = Utilities.create_route_combinations(shape_=paths_to_this_layer)
        route_combinations_dict = {idx: path for idx, path in enumerate(route_combinations)}
        route_combinations_dict_inverse = {path: idx for idx, path in enumerate(route_combinations)}
        routing_matrices_soft_dict = {path: cigt_outputs["routing_matrices_soft_dict"][path]
                                      for path in route_combinations}
        routing_matrices_sorting_indices_dict = \
            {path: torch.argsort(routing_matrices_soft_dict[path], dim=1, descending=True).detach().cpu().numpy()
             for path in route_combinations}

        new_paths_dict = {}
        for sample_id, paths_set in current_paths_dict.items():
            sample_action = actions[sample_id]
            new_paths_dict[sample_id] = set()
            for path in paths_set:
                s_indices = routing_matrices_sorting_indices_dict[path][sample_id]
                for a_ in range(sample_action + 1):
                    new_path = tuple([*path, s_indices[a_]])
                    new_paths_dict[sample_id].add(new_path)
        return new_paths_dict

    def calculate_rewards(self, cigt_outputs, complete_path_history):
        paths_to_this_layer = self.pathCounts
        route_combinations = Utilities.create_route_combinations(shape_=paths_to_this_layer)
        # route_combinations_dict = {idx: path for idx, path in enumerate(route_combinations)}
        # route_combinations_dict_inverse = {path: idx for idx, path in enumerate(route_combinations)}

        # Step1: Calculate per sample accuracy. This will be the first component for the loss.
        labels = [arr for arr in cigt_outputs["labels_dict"].values()]
        labels = torch.stack(labels, dim=1)
        labels = torch.mean(labels.to(torch.float32), dim=1).to(torch.int64)
        assert all([np.array_equal(labels.detach().cpu().numpy(), arr.detach().cpu().numpy())
                    for arr in cigt_outputs["labels_dict"].values()])
        labels = labels.detach().cpu().numpy()
        # logits = [arr for arr in cigt_outputs["logits_dict"]]
        softmax_arrays = {path: torch.nn.functional.softmax(cigt_outputs["logits_dict"][path],
                                                            dim=1).detach().cpu().numpy()
                          for path in route_combinations}
        # Mixture of experts for different leaf nodes.
        correctness_vec = []
        for sample_id in range(labels.shape[0]):
            leaf_ids_for_sample = complete_path_history[-1][sample_id]
            probs_arr = []
            for path in leaf_ids_for_sample:
                probs_arr.append(softmax_arrays[path][sample_id])
            probs_arr = np.stack(probs_arr, axis=0)
            mixture_probs = np.mean(probs_arr, axis=0)
            gt_label = labels[sample_id]
            pr_label = np.argmax(mixture_probs)
            correctness_vec.append(float(gt_label == pr_label))
        correctness_vec = torch.from_numpy(np.array(correctness_vec)).to(self.device)

        # Step2: Calculate per sample MAC cost.
        mac_vec = []
        single_path_cost = torch.sum(self.macCostPerLayer)
        for sample_id in range(labels.shape[0]):
            mac_total = 0.0
            for layer_id in range(len(self.pathCounts)):
                layer_cost = len(complete_path_history[layer_id][sample_id]) * self.macCostPerLayer[layer_id]
                mac_total += layer_cost
            relative_mac_cost = (mac_total / single_path_cost) - 1.0
            mac_vec.append(relative_mac_cost)
        mac_vec = -torch.Tensor(mac_vec).to(self.device)

        reward_array = (1.0 - self.policyNetworksMacLambda) * correctness_vec + self.policyNetworksMacLambda * mac_vec
        return reward_array, correctness_vec, mac_vec

    def get_cigt_outputs(self, x, y):
        training_state = self.training
        self.eval()
        cigt_outputs = self.forward_v2(x=x, labels=y, temperature=1.0)
        batch_size = x.shape[0]
        if training_state == self.training:
            self.train()
        else:
            self.eval()
        return cigt_outputs, batch_size

    def forward_with_policies(self, x, y, greedy_actions):
        cigt_outputs, batch_size = self.get_cigt_outputs(x=x, y=y)
        policy_entropies = []
        log_probs_trajectory = []
        probs_trajectory = []
        actions_trajectory = []
        full_action_probs_trajectory = []
        correctness_vec = None
        mac_vec = None
        reward_array = None
        paths_history = [{idx: {(0,)} for idx in range(batch_size)}]

        for layer_id in range(len(self.pathCounts)):
            if layer_id < len(self.pathCounts) - 1:
                # Get sparse input arrays for the policy networks
                pg_sparse_input = self.prepare_policy_network_input_f(
                    batch_size=batch_size,
                    layer_id=layer_id,
                    current_paths_dict=paths_history[layer_id],
                    cigt_outputs=cigt_outputs
                )
                # Execute this layers policy network, get log action probs, actions and policy entropies.
                mean_policy_entropy, log_probs_selected, probs_selected, action_probs, actions = \
                    self.run_policy_networks(layer_id=layer_id, pn_input=pg_sparse_input)
                policy_entropies.append(mean_policy_entropy)
                log_probs_trajectory.append(log_probs_selected)
                probs_trajectory.append(probs_selected)
                actions_trajectory.append(actions)
                full_action_probs_trajectory.append(action_probs)

                # Extend the trajectories for each sample based on the actions selected.
                new_paths_dict = self.extend_sample_trajectories_wrt_actions(actions=actions,
                                                                             cigt_outputs=cigt_outputs,
                                                                             current_paths_dict=paths_history[layer_id],
                                                                             layer_id=layer_id)
                paths_history.append(new_paths_dict)
            else:
                reward_array, correctness_vec, mac_vec = \
                    self.calculate_rewards(cigt_outputs=cigt_outputs, complete_path_history=paths_history)

        return {
            "policy_entropies": policy_entropies,
            "log_probs_trajectory": log_probs_trajectory,
            "probs_trajectory": probs_trajectory,
            "actions_trajectory": actions_trajectory,
            "paths_history": paths_history,
            "reward_array": reward_array,
            "correctness_vec": correctness_vec,
            "mac_vec": mac_vec,
            "full_action_probs_trajectory": full_action_probs_trajectory}

    def validate(self, loader, epoch, data_kind, temperature=None, print_avg_measurements=False,
                 return_network_outputs=False,
                 verbose=False):
        self.eval()
        batch_time = AverageMeter()
        mean_reward_for_batch_avg = AverageMeter()
        macs_per_batch_avg = AverageMeter()
        accuracy_per_batch_avg = AverageMeter()
        mean_state_network_loss_avg = AverageMeter()
        mean_policy_value_avg = AverageMeter()
        mean_policy_value_no_baseline_avg = AverageMeter()

        if temperature is None:
            temperature = 1.0
        if verbose is False:
            verbose_loader = enumerate(loader)
        else:
            verbose_loader = tqdm(enumerate(loader))

        for i, (input_, target) in verbose_loader:
            time_begin = time.time()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input_).to(self.device)
                target_var = torch.autograd.Variable(target).to(self.device)
                batch_size = input_var.size(0)
                outputs = self.forward_with_policies(x=input_var, y=target_var, greedy_actions=False)

                # Mean reward from the network execution.
                mean_reward_for_batch = torch.mean(outputs["reward_array"])
                mean_reward_for_batch = mean_reward_for_batch.detach().cpu().numpy().item()
                mean_reward_for_batch_avg.update(mean_reward_for_batch, batch_size)
                # Mean accuracy for the batch.
                accuracy_per_batch = torch.mean(outputs["correctness_vec"]).detach().cpu().numpy().item()
                accuracy_per_batch_avg.update(accuracy_per_batch, batch_size)
                # Mean mac for the batch.
                macs_per_batch = torch.mean(outputs["mac_vec"]).detach().cpu().numpy().item()
                macs_per_batch_avg.update(macs_per_batch, batch_size)
        return {
            "mean_reward_for_batch_avg": mean_reward_for_batch_avg.avg,
            "accuracy_per_batch_avg": accuracy_per_batch_avg.avg,
            "macs_per_batch_avg": macs_per_batch_avg.avg}

    def toggle_allways_ig_routing(self, enable):
        if enable:
            self.policyNetworksEnforcedActions = []
            max_branch_count = np.prod(self.pathCounts)
            for path_count in self.pathCounts[1:]:
                self.policyNetworksEnforcedActions.append(
                    torch.zeros(size=(max_branch_count * self.batchSize,),
                                dtype=torch.int64).to(self.device))
        else:
            self.policyNetworksEnforcedActions = []

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

        # Create a separate optimizer that only optimizes the policy networks.
        policy_networks_optimizer = optim.AdamW(
            [{'params': policy_networks_parameters,
              'lr': self.policyNetworkInitialLr,
              'weight_decay': self.policyNetworkWd}])

        return policy_networks_optimizer

    def get_explanation_string(self):
        kv_rows = []
        explanation = super().get_explanation_string()
        for elem in inspect.getmembers(self):
            if elem[0].startswith("policyNetworks") and \
                    (isinstance(elem[1], bool) or
                     isinstance(elem[1], float) or
                     isinstance(elem[1], int) or
                     isinstance(elem[1], str)):
                explanation = self.add_explanation(name_of_param=elem[0],
                                                   value=elem[1],
                                                   explanation=explanation,
                                                   kv_rows=kv_rows)
        DbLogger.write_into_table(rows=kv_rows, table="run_parameters")
        return explanation

    def adjust_learning_rate_polynomial(self, iteration, num_of_total_iterations):
        lr = self.policyNetworkInitialLr
        where = np.clip(iteration / num_of_total_iterations, a_min=0.0, a_max=1.0)
        modified_lr = lr * (1 - where) ** self.policyNetworkPolynomialSchedulerPower
        self.policyGradientsModelOptimizer.param_groups[0]['lr'] = modified_lr

    def calculate_cumulative_rewards(self, rewards_array):
        network_rewards = [torch.zeros_like(rewards_array, device=self.device) for _ in range(len(self.pathCounts) - 2)]
        network_rewards.append(rewards_array)
        rewards_matrix = torch.stack(network_rewards, dim=1)
        cumulative_rewards = []

        for t_ in range(rewards_matrix.shape[1]):
            discounts_arr = self.discountCoefficientArray[0:rewards_matrix.shape[1] - t_]
            discounts_arr = torch.unsqueeze(discounts_arr, dim=0)
            rewards_partial = rewards_matrix[:, t_:]
            rewards_partial_discounted = rewards_partial * discounts_arr
            cumulative_reward = torch.sum(rewards_partial_discounted, dim=1)
            cumulative_rewards.append(cumulative_reward)
        return cumulative_rewards

    def update_baselines(self, cumulative_rewards):
        gamma = self.policyNetworksBaselineMomentum
        for t_ in range(len(cumulative_rewards)):
            b_ = self.baselinesPerLayer[t_]
            r_ = torch.mean(cumulative_rewards[t_])
            if b_ is None:
                self.baselinesPerLayer[t_] = r_
            else:
                self.baselinesPerLayer[t_] = gamma * b_ + (1.0 - gamma) * r_

    def calculate_policy_loss(self, cumulative_rewards, log_policy_probs):
        policy_values = []
        for layer_id in range(len(self.pathCounts) - 1):
            G_t = cumulative_rewards[layer_id]
            B_t = self.baselinesPerLayer[layer_id]
            if B_t is None:
                B_t = 0.0
            Lp_t = -1.0 * log_policy_probs[layer_id]
            delta = G_t - B_t
            policy_value = delta * Lp_t
            policy_values.append(policy_value)
        policy_values_each_time_step = torch.stack(policy_values, dim=1)
        total_policy_values_per_sample = torch.sum(policy_values_each_time_step, dim=1)
        expected_policy_loss = torch.mean(total_policy_values_per_sample)
        return expected_policy_loss

    def fit_policy_network(self, train_loader, test_loader):
        self.to(self.device)
        torch.manual_seed(1)
        best_performance = 0.0
        num_of_total_iterations = self.policyNetworkTotalNumOfEpochs * len(train_loader)

        # Run a forward pass first to initialize each LazyXXX layer.
        self.execute_forward_with_random_input()

        # Test with enforced actions set to 0. The accuracy should be the naive IG accuracy.
        self.toggle_allways_ig_routing(enable=True)
        validation_dict = self.validate(loader=test_loader, epoch=-1, data_kind="test", temperature=1.0)
        self.toggle_allways_ig_routing(enable=False)
        print("test_ig_accuracy_avg:{0} test_ig_mac_avg:{1}".format(validation_dict["accuracy_per_batch_avg"],
                                                                    validation_dict["macs_per_batch_avg"]))

        # Create the model optimizer, we should have every parameter initialized right now.
        self.policyGradientsModelOptimizer = self.create_optimizer()

        temp_warm_up_state = self.isInWarmUp
        temp_random_routing_ratio = self.routingRandomizationRatio
        self.isInWarmUp = False
        self.routingRandomizationRatio = -1.0
        # Train the policy network for one epoch
        iteration_id = 0
        for epoch_id in range(0, self.policyNetworkTotalNumOfEpochs):
            for i, (input_, target) in enumerate(train_loader):
                self.train()
                print("*************Policy Network Training Epoch:{0} Iteration:{1}*************".format(
                    epoch_id, self.iteration_id))

                # Adjust the learning rate
                self.adjust_learning_rate_polynomial(iteration=self.iteration_id,
                                                     num_of_total_iterations=num_of_total_iterations)
                # Print learning rates
                print("Policy Network Lr:{0}".format(self.policyGradientsModelOptimizer.param_groups[0]["lr"]))
                self.policyGradientsModelOptimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    input_var = torch.autograd.Variable(input_).to(self.device)
                    target_var = torch.autograd.Variable(target).to(self.device)
                    batch_size = input_var.size(0)
                    outputs = self.forward_with_policies(x=input_var, y=target_var, greedy_actions=False)
                    cumulative_rewards = self.calculate_cumulative_rewards(rewards_array=outputs["reward_array"])
                    self.update_baselines(cumulative_rewards=cumulative_rewards)
                    print("Baseline Values:{0}".format(self.baselinesPerLayer))
                    policy_loss = self.calculate_policy_loss(cumulative_rewards=cumulative_rewards,
                                                             log_policy_probs=outputs["log_probs_trajectory"])
                    entropy_loss = torch.Tensor(outputs["policy_entropies"])
                    entropy_loss = -torch.sum(entropy_loss)
                    total_loss = policy_loss + (self.policyNetworksEntropyLossCoeff * entropy_loss)

                    # Step
                    if self.isDebugMode:
                        grad_check = [param.grad is None or np.array_equal(param.grad.cpu().numpy(),
                                                                           np.zeros_like(param.grad.cpu().numpy()))
                                      for param in self.policyGradientsModelOptimizer.param_groups[0]["params"]]
                        # print(self.policyGradientsModelOptimizer.param_groups[0]["params"][0].grad)
                        # print(grad_check)
                        assert all(grad_check)
                    total_loss.backward()
                    if self.isDebugMode:
                        grad_check = [isinstance(param.grad, torch.Tensor) for param in
                                      self.policyGradientsModelOptimizer.param_groups[0]["params"]]
                        assert all(grad_check)
                    self.policyGradientsModelOptimizer.step()

                    print("Epoch:{0} Iteration:{1} Reward:{2} Policy Loss:{3} Entropy Loss:{4}".format(
                        epoch_id,
                        self.iteration_id,
                        torch.mean(outputs["reward_array"]).detach().cpu().numpy(),
                        policy_loss.detach().cpu().numpy(),
                        entropy_loss.detach().cpu().numpy()))
                self.iteration_id += 1

            # Validation
            if epoch_id % self.policyNetworksEvaluationPeriod == 0 or \
                    epoch_id >= (self.policyNetworkTotalNumOfEpochs - 10):
                print("***************Db:{0} RunId:{1} Epoch {2} End, Training Evaluation***************".format(
                    DbLogger.log_db_path, self.runId, epoch_id))
                train_dict = self.validate(loader=train_loader, epoch=epoch_id, data_kind="train", temperature=1.0)
                print("train_reward:{0} train_accuracy:{1} train_mac_avg:{2}".format(
                    train_dict["mean_reward_for_batch_avg"],
                    train_dict["accuracy_per_batch_avg"],
                    train_dict["macs_per_batch_avg"]))
                print("***************Db:{0} RunId:{1} Epoch {2} End, Test Evaluation***************".format(
                    DbLogger.log_db_path, self.runId, epoch_id))
                test_dict = self.validate(loader=test_loader, epoch=epoch_id, data_kind="test", temperature=1.0)
                print("test_reward:{0} test_accuracy:{1} test_mac_avg:{2}".format(
                    test_dict["mean_reward_for_batch_avg"],
                    test_dict["accuracy_per_batch_avg"],
                    test_dict["macs_per_batch_avg"]))
                self.toggle_allways_ig_routing(enable=True)
                validation_dict = self.validate(loader=test_loader, epoch=-1, data_kind="test", temperature=1.0)
                self.toggle_allways_ig_routing(enable=False)
                print("test_ig_accuracy:{0} test_mac_ig_avg:{1}".format(
                    validation_dict["accuracy_per_batch_avg"], validation_dict["macs_per_batch_avg"]))
                # self.save_cigt_model(epoch=epoch_id)

                DbLogger.write_into_table(
                    rows=[(self.runId,
                           self.iteration_id,
                           epoch_id,
                           train_dict["accuracy_per_batch_avg"],
                           train_dict["macs_per_batch_avg"],
                           test_dict["accuracy_per_batch_avg"],
                           test_dict["macs_per_batch_avg"],
                           0.0,
                           0.0,
                           "YYY")], table=DbLogger.logsTable)
