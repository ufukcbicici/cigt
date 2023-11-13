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
        self.run_with_policies(x=fake_input, y=fake_target, training=False, greedy_actions=True)
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
        policy_network = self.policyNetworks[layer_id]
        action_probs = policy_network(pn_input)
        log_action_probs = torch.log(action_probs + self.policyNetworksNllConstant)
        p_lp = torch.sum(-1.0 * (action_probs * log_action_probs), dim=1)
        mean_policy_entropy = torch.mean(p_lp)

        dist = Categorical(action_probs)
        actions = dist.sample()
        log_probs_selected = dist.log_prob(actions)
        probs_selected = torch.exp(log_probs_selected)
        return mean_policy_entropy, log_probs_selected, probs_selected, action_probs, actions.detach().cpu().numpy()

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
        route_combinations_dict = {idx: path for idx, path in enumerate(route_combinations)}
        route_combinations_dict_inverse = {path: idx for idx, path in enumerate(route_combinations)}

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
        mac_vec = torch.from_numpy(np.array(mac_vec)).to(self.device)

        reward_array = (1.0 - self.policyNetworksMacLambda) * correctness_vec + self.policyNetworksMacLambda * mac_vec
        return reward_array, correctness_vec, mac_vec

    def run_with_policies(self, x, y, training, greedy_actions):
        self.eval()
        cigt_outputs = self.forward_v2(x=x, labels=y, temperature=1.0)

        if training:
            self.train()

        policy_entropies = []
        log_probs_trajectory = []
        actions_trajectory = []
        correctness_vec = None
        mac_vec = None
        reward_array = None
        paths_history = [{idx: {(0,)} for idx in range(x.shape[0])}]

        for layer_id in range(len(self.pathCounts)):
            if layer_id < len(self.pathCounts) - 1:
                # Get sparse input arrays for the policy networks
                pg_sparse_input = self.prepare_policy_network_input_f(
                    batch_size=x.shape[0],
                    layer_id=layer_id,
                    current_paths_dict=paths_history[layer_id],
                    cigt_outputs=cigt_outputs
                )
                # Execute this layers policy network, get log action probs, actions and policy entropies.
                mean_policy_entropy, log_probs_selected, probs_selected, action_probs, actions = \
                    self.run_policy_networks(layer_id=layer_id, pn_input=pg_sparse_input)
                policy_entropies.append(mean_policy_entropy)
                log_probs_trajectory.append(log_probs_selected)
                actions_trajectory.append(actions)

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
            "actions_trajectory": actions_trajectory,
            "paths_history": paths_history,
            "reward_array": reward_array,
            "correctness_vec": correctness_vec,
            "mac_vec": mac_vec}
    def fit_policy_network(self, train_loader, test_loader):
        self.to(self.device)
        torch.manual_seed(1)
        best_performance = 0.0
        num_of_total_iterations = self.policyNetworkTotalNumOfEpochs * len(train_loader)

        # Run a forward pass first to initialize each LazyXXX layer.
        self.execute_forward_with_random_input()
