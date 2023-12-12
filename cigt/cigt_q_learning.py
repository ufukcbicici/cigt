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
from sklearn.metrics import mean_squared_error, r2_score

from auxillary.db_logger import DbLogger
from auxillary.average_meter import AverageMeter
from auxillary.time_profiler import TimeProfiler
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.custom_layers.basic_block_with_cbam import BasicBlockWithCbam
from tqdm import tqdm

from cigt.cigt_reinforce_v2 import CigtReinforceV2


# Algorithm Structure:
# Since the episodes are finite and we can enumerate all state-action pairs for a reasonably structured CIGT
# (like [2,4]), we can calculate the optimal Q tables for every minibatch and convert the Reinforcement Learning problem
# into a supervised regression problem.

# At every minibatch:
# Calculate the forward pass for every state combination from CIGT.
# Using the Bellman Equation, recursively calculate the optimal Q-Tables for every CIGT layer.
# For training, sample actions with respect to the OPTIMAL Q-Tables with epsilon exploration.
# Calculate the estimated Q-Tables from the policy networks. Use a conventional MSE or MAE based regression to learn
# the optimal tables.
# ???
# PROFIT!!!

class CigtQLearning(CigtReinforceV2):
    def __init__(self, configs, run_id, model_definition,
                 num_classes, model_mac_info, is_debug_mode, precalculated_datasets_dict):
        if precalculated_datasets_dict is not None:
            self.trainDataset = precalculated_datasets_dict["train_dataset"]
            self.testDataset = precalculated_datasets_dict["test_dataset"]
            self.usingPrecalculatedDatasets = True
        else:
            self.trainDataset = None
            self.testDataset = None
            self.usingPrecalculatedDatasets = False

        self.policyGradientsUseLstm = configs.policy_networks_use_lstm
        self.policyNetworksLstmNumLayers = configs.policy_networks_lstm_num_layers
        self.policyNetworksLstmBidirectional = configs.policy_networks_lstm_bidirectional
        self.policyNetworksEpsilonDecayCoeff = configs.policy_networks_epsilon_decay_coeff
        self.policyNetworksTrainOnlyActionHeads = configs.policy_networks_train_only_action_heads

        super().__init__(configs, run_id, model_definition, num_classes, model_mac_info, is_debug_mode)
        self.policyNetworkQNetRegressionLayers = nn.ModuleList()

        if self.policyGradientsUseLstm:
            assert len(set([dim for dim in self.decisionDimensions])) == 1
            self.lstmInputDimension = self.decisionDimensions[0]
            self.lstm = nn.LSTM(input_size=self.lstmInputDimension,
                                hidden_size=self.policyNetworksLstmDimension,
                                num_layers=self.policyNetworksLstmNumLayers,
                                batch_first=True,
                                bidirectional=self.policyNetworksLstmBidirectional)
            for layer_id, path_count in enumerate(self.pathCounts[1:]):
                action_space_size = path_count
                loss_layer = nn.Linear(in_features=self.decisionDimensions[layer_id],
                                       out_features=action_space_size)
                self.policyNetworkQNetRegressionLayers.append(loss_layer)
        else:
            self.lstm = None

        self.epsilonValue = 1.0
        self.actionSpaces = self.pathCounts[1:]

    def create_policy_networks(self):
        for layer_id, path_count in enumerate(self.pathCounts[1:]):
            layers = OrderedDict()
            action_space_size = path_count

            if not self.usingPrecalculatedDatasets and self.policyNetworksCbamLayerInputReductionRatio > 1:
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
            layers["policy_gradients_block_{0}_relu".format(layer_id)] = nn.ReLU()
            layers["policy_gradients_block_{0}_feature_fc".format(layer_id)] = nn.LazyLinear(
                out_features=self.decisionDimensions[layer_id])

            if not self.policyGradientsUseLstm:
                loss_layer = nn.Linear(in_features=self.decisionDimensions[layer_id], out_features=action_space_size)
                layers["policy_gradients_block_{0}_action_space_fc".format(layer_id)] = loss_layer

            policy_gradient_network_backbone = nn.Sequential(layers)
            self.policyNetworks.append(policy_gradient_network_backbone)

    def get_cigt_outputs(self, x, y):
        if not self.usingPrecalculatedDatasets:
            training_state = self.training
            self.eval()
            cigt_outputs = self.forward_v2(x=x, labels=y, temperature=1.0)
            batch_size = x.shape[0]
            if training_state == self.training:
                self.train()
            else:
                self.eval()
        else:
            assert isinstance(x, dict)
            assert y is None
            cigt_outputs = x
            batch_size = cigt_outputs["block_outputs_dict"][(0,)].shape[0]

        # Add matrices for sorting the routing indices.
        cigt_outputs["routing_matrices_sorting_indices_dict"] = {}
        for path, arr in cigt_outputs["routing_matrices_soft_dict"].items():
            sorted_indices_arr = torch.argsort(arr, dim=1, descending=True)
            cigt_outputs["routing_matrices_sorting_indices_dict"][path] = sorted_indices_arr

        # Add softmax probabilities
        cigt_outputs["softmax_dict"] = {}
        for path, arr in cigt_outputs["logits_dict"].items():
            softmax_arr = torch.nn.functional.softmax(arr, dim=1)
            cigt_outputs["softmax_dict"][path] = softmax_arr

        # for arr_name, d_ in cigt_outputs.items():
        #     if not isinstance(d_, dict):
        #         continue
        #     print("Array Name:{0}".format(arr_name))
        #     for k, v in d_.items():
        #         print("Key:{0} Device:{1}".format(k, v.device))

        return cigt_outputs, batch_size

    def create_optimizer(self):
        paths = []
        for pc in self.pathCounts:
            paths.append([i_ for i_ in range(pc)])
        path_variaties = Utilities.get_cartesian_product(list_of_lists=paths)

        # for idx in range(len(self.pathCounts)):
        #     cnt = len([tpl for tpl in path_variaties if tpl[idx] == 0])
        #     self.layerCoefficients.append(len(path_variaties) / cnt)
        #
        # # Create parameter groups per CIGT layer and shared parameters
        # shared_parameters = []
        # parameters_per_cigt_layers = []
        # for idx in range(len(self.pathCounts)):
        #     parameters_per_cigt_layers.append([])
        # # Policy Network parameters.
        # policy_networks_parameters = []
        # # Value Networks parameters.
        # value_networks_parameters = []
        #
        # for name, param in self.named_parameters():
        #     assert not (("cigtLayers" in name and "policyNetworks" in name) or
        #                 ("cigtLayers" in name and "valueNetworks" in name) or
        #                 ("policyNetworks" in name and "valueNetworks" in name))
        #     if "cigtLayers" in name:
        #         assert "policyNetworks" not in name and "valueNetworks" not in name
        #         param_name_splitted = name.split(".")
        #         layer_id = int(param_name_splitted[1])
        #         assert 0 <= layer_id <= len(self.pathCounts) - 1
        #         parameters_per_cigt_layers[layer_id].append(param)
        #     elif "policyNetworks" in name:
        #         assert "cigtLayers" not in name and "valueNetworks" not in name
        #         policy_networks_parameters.append(param)
        #     elif "valueNetworks" in name:
        #         assert "cigtLayers" not in name and "policyNetworks" not in name
        #         value_networks_parameters.append(param)
        #     else:
        #         shared_parameters.append(param)
        #
        # num_shared_parameters = len(shared_parameters)
        # num_policy_network_parameters = len(policy_networks_parameters)
        # num_value_networks_parameters = len(value_networks_parameters)
        # num_cigt_layer_parameters = sum([len(arr) for arr in parameters_per_cigt_layers])
        # num_all_parameters = len([tpl for tpl in self.named_parameters()])
        # assert num_shared_parameters + num_policy_network_parameters + \
        #        num_value_networks_parameters + num_cigt_layer_parameters == num_all_parameters
        #
        # # Create a separate optimizer that only optimizes the policy networks.
        # policy_networks_optimizer = optim.AdamW(
        #     [{'params': policy_networks_parameters,
        #       'lr': self.policyNetworkInitialLr,
        #       'weight_decay': self.policyNetworkWd}])
        #
        # return policy_networks_optimizer

    def execute_forward_with_random_input(self):
        # max_batch_size = np.prod(self.pathCounts) * self.batchSize
        # self.enforcedRoutingMatrices = []
        # for path_count in self.pathCounts[1:]:
        #     self.enforcedRoutingMatrices.append(torch.ones(size=(max_batch_size, path_count),
        #                                                    dtype=torch.int64).to(self.device))
        if not self.usingPrecalculatedDatasets:
            raise NotImplementedError()
        else:
            self.eval()
            cigt_outputs, batch_size = self.get_cigt_outputs(x=next(iter(self.testDataset)), y=None)
            cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
            action_trajectories = torch.Tensor((0, 0)).to(self.device).to(torch.int64)
            action_trajectories = torch.unsqueeze(action_trajectories, dim=0)
            action_trajectories = torch.tile(action_trajectories, dims=(batch_size, 1))
            res_dict = self.forward_with_actions(cigt_outputs=cigt_outputs, batch_size=batch_size,
                                                 action_trajectories=action_trajectories)

        self.enforcedRoutingMatrices = []

    def get_executed_nodes_wrt_trajectories(self, cigt_outputs, batch_size, action_trajectories):
        trajectory_length = action_trajectories.shape[1]
        # First nodes are always selected
        node_selection_arrays = [torch.ones(size=(batch_size, 1), dtype=torch.float32, device=self.device)]
        for t in range(trajectory_length):
            previous_layer_node_combinations = Utilities.create_route_combinations(shape_=self.pathCounts[:(t + 1)])
            next_level_selection_array_shape = (batch_size, *self.pathCounts[:(t + 2)])
            next_level_selection_array = torch.zeros(size=next_level_selection_array_shape, dtype=torch.float32,
                                                     device=self.device)
            curr_level_actions = action_trajectories[:, t]
            # Extend to next level nodes.
            for prev_node_combination in previous_layer_node_combinations:
                index_array = [torch.arange(batch_size).to(self.device).to(torch.int64)]
                for a_ in prev_node_combination:
                    i_arr = torch.Tensor([a_] * batch_size).to(self.device).to(torch.int64)
                    index_array.append(i_arr)
                parent_node_selections = node_selection_arrays[-1][index_array]
                for a_ in range(self.actionSpaces[t]):
                    current_node_selections = (a_ <= curr_level_actions).to(torch.float32)
                    final_node_selections = parent_node_selections * current_node_selections
                    next_level_index_array = []
                    next_level_index_array.extend(index_array)
                    # print("index_array.device:{0}".format(set([arr.device for arr in index_array])))
                    next_level_index_array.append(
                        cigt_outputs["routing_matrices_sorting_indices_dict"][prev_node_combination][:, a_])
                    # print("index_array_last.device:{0}".format(next_level_index_array[-1].device))
                    # print("All devices:{0}".format(set([arr.device for arr in next_level_index_array])))
                    next_level_selection_array[next_level_index_array] = final_node_selections
            node_selection_arrays.append(next_level_selection_array)
        return node_selection_arrays

    def get_executed_nodes_wrt_trajectories_baseline(self, cigt_outputs, batch_size, action_trajectories):
        trajectory_length = action_trajectories.shape[1]
        node_selection_arrays = [[{(0,)} for idx in range(batch_size)]]
        for t in range(trajectory_length):
            next_level_node_selections_array = []
            for sample_id in range(batch_size):
                next_level_node_selections = set()
                selected_nodes_for_level_t = node_selection_arrays[-1][sample_id]
                action = action_trajectories[sample_id, t].item()
                for selected_node_for_level_t in selected_nodes_for_level_t:
                    node_ordering_all_samples = \
                        cigt_outputs["routing_matrices_sorting_indices_dict"][selected_node_for_level_t]
                    node_ordering = node_ordering_all_samples[sample_id].detach().cpu().numpy()
                    for a_ in range(action + 1):
                        next_level_tpl = (*selected_node_for_level_t, node_ordering[a_].item())
                        next_level_node_selections.add(next_level_tpl)
                next_level_node_selections_array.append(next_level_node_selections)
            node_selection_arrays.append(next_level_node_selections_array)
        return node_selection_arrays

    def calculate_moe_for_final_layer(self, cigt_outputs, batch_size, executed_nodes_array):
        # ************** First calculate the MoE accuracies **************
        path_combinations_for_t = Utilities.create_route_combinations(shape_=self.pathCounts)
        mixture_of_experts_list = []
        for path_combination in path_combinations_for_t:
            softmax_probs = cigt_outputs["softmax_dict"][path_combination]
            path_trajectories = torch.Tensor(path_combination).to(self.device).to(torch.int64)
            path_trajectories = torch.unsqueeze(path_trajectories, dim=0)
            path_trajectories = torch.tile(path_trajectories, dims=(batch_size, 1))
            index_array = [torch.arange(batch_size, device=self.device)]
            for t in range(path_trajectories.shape[1]):
                index_array.append(path_trajectories[:, t])
            selection_array = executed_nodes_array[index_array]
            selection_array = torch.unsqueeze(selection_array, dim=1)
            # print("selection_array.device:{0}".format(selection_array.device))
            # print("softmax_probs.device:{0}".format(softmax_probs.device))

            weighted_softmax_probs = selection_array * softmax_probs
            mixture_of_experts_list.append(weighted_softmax_probs)
        # Generate ensemble probabilities
        mixture_of_experts_matrix = torch.stack(mixture_of_experts_list, dim=1)
        experts_sum = torch.sum(mixture_of_experts_matrix, dim=1)
        # Total number of involved experts. This is the sum of executed_nodes_array expect the batch dimension.
        expert_dims = tuple([idx for idx in range(1, len(executed_nodes_array.shape))])
        expert_counts = torch.sum(executed_nodes_array, dim=expert_dims)
        expert_coeffs = torch.reciprocal(expert_counts)

        expert_probs = torch.unsqueeze(expert_coeffs, dim=1) * experts_sum
        expert_probs_np = expert_probs.detach().cpu().numpy()
        expert_probs_sum_np = np.sum(expert_probs_np, axis=1)
        assert np.allclose(expert_probs_sum_np, np.ones_like(expert_probs_sum_np))
        predicted_labels = torch.argmax(expert_probs, dim=1)

        # Calculate prediction validities, per sample
        true_labels = [arr for arr in cigt_outputs["labels_dict"].values()]
        true_labels = torch.stack(true_labels, dim=1)
        true_labels = torch.mean(true_labels.to(torch.float32), dim=1).to(torch.int64)
        assert all([np.array_equal(true_labels.detach().cpu().numpy(), arr.detach().cpu().numpy())
                    for arr in cigt_outputs["labels_dict"].values()])

        correctness_vector = (true_labels == predicted_labels).to(torch.float32)
        return correctness_vector, expert_probs

    def calculate_moe_for_final_layer_baseline(self, cigt_outputs, batch_size, executed_nodes_array):
        path_combinations_for_t = Utilities.create_route_combinations(shape_=self.pathCounts)
        true_labels = [arr for arr in cigt_outputs["labels_dict"].values()]
        true_labels = torch.stack(true_labels, dim=1)
        true_labels = torch.mean(true_labels.to(torch.float32), dim=1).to(torch.int64)
        assert all([np.array_equal(true_labels.detach().cpu().numpy(), arr.detach().cpu().numpy())
                    for arr in cigt_outputs["labels_dict"].values()])

        all_expert_probs = []
        for sample_id in range(batch_size):
            selected_expert_nodes = executed_nodes_array[sample_id]
            expert_probs = []
            assert isinstance(selected_expert_nodes, set)
            for path_combination in selected_expert_nodes:
                softmax_probs = cigt_outputs["softmax_dict"][path_combination][sample_id]
                expert_probs.append(softmax_probs)
            expert_probs = torch.stack(expert_probs, dim=0)
            expert_probs = torch.mean(expert_probs, dim=0)
            all_expert_probs.append(expert_probs)

        all_expert_probs = torch.stack(all_expert_probs, dim=0)
        predicted_labels = torch.argmax(all_expert_probs, dim=1)

        correctness_vector = (true_labels == predicted_labels).to(torch.float32)
        return correctness_vector, all_expert_probs

    def calculate_mac_vector_for_layer(self, layer, executed_nodes_array):
        # ************** Secondly calculate the MAC costs for the final layer **************
        assert tuple(executed_nodes_array.shape[1:]) == tuple(self.pathCounts[:(layer + 1)])
        node_dims = tuple([idx for idx in range(1, len(executed_nodes_array.shape))])
        node_counts = torch.sum(executed_nodes_array, dim=node_dims)
        extra_node_counts = node_counts - 1
        mac_values_for_layer = extra_node_counts * self.macCostPerLayer[layer]
        relative_mac_values_for_layer = mac_values_for_layer / self.singlePathMacCost
        # ************** Secondly calculate the MAC costs for the final layer **************
        return relative_mac_values_for_layer

    def calculate_mac_vector_for_layer_baseline(self, layer, executed_nodes_array):
        relative_mac_values_for_layer = []
        base_cost = self.macCostPerLayer[layer].detach().cpu().numpy()
        for sample_id, selected_expert_nodes in enumerate(executed_nodes_array):
            node_count = len(selected_expert_nodes)
            total_layer_cost = node_count * base_cost
            extra_cost = total_layer_cost - base_cost
            relative_mac_values_for_layer.append(extra_cost / self.singlePathMacCost)
        relative_mac_values_for_layer = np.array(relative_mac_values_for_layer)
        return relative_mac_values_for_layer

    def compare_trajectory_evaluation_methods(self, dataset, repeat_count):
        time_profiler = TimeProfiler()
        for i, batch in enumerate(dataset):
            print("Iteration:{0}".format(i))
            if self.usingPrecalculatedDatasets:
                cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
                cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
            else:
                input_var = torch.autograd.Variable(batch[0]).to(self.device)
                target_var = torch.autograd.Variable(batch[1]).to(self.device)
                cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)

            torch_trajectory_computation_times = []
            numpy_trajectory_computation_times = []
            torch_moe_computation_times = []
            numpy_moe_computation_times = []
            torch_mac_computation_times = []
            numpy_mac_computation_times = []

            for repeat_id in tqdm(range(repeat_count)):
                actions = []
                for a_ in self.actionSpaces:
                    actions.append(np.random.randint(low=0, high=a_, size=(batch_size,)))
                actions = np.stack(actions, axis=1)
                actions_torch = torch.from_numpy(actions).to(self.device)

                time_profiler.start_measurement()
                node_selection_arrays_torch = self.get_executed_nodes_wrt_trajectories(
                    cigt_outputs=cigt_outputs,
                    batch_size=batch_size,
                    action_trajectories=actions_torch)
                time_profiler.end_measurement()
                torch_trajectory_computation_times.append(time_profiler.get_time())

                time_profiler.start_measurement()
                correctness_vector_torch, expert_probs_torch = self.calculate_moe_for_final_layer(
                    cigt_outputs=cigt_outputs,
                    batch_size=batch_size,
                    executed_nodes_array=
                    node_selection_arrays_torch[-1])
                time_profiler.end_measurement()
                torch_moe_computation_times.append(time_profiler.get_time())

                time_profiler.start_measurement()
                mac_vectors_torch = []
                for layer, trajectory_single_step in enumerate(node_selection_arrays_torch):
                    mac_vector_torch = \
                        self.calculate_mac_vector_for_layer(layer=layer,
                                                            executed_nodes_array=trajectory_single_step)
                    mac_vectors_torch.append(mac_vector_torch)
                total_mac_vectors_torch = torch.stack(mac_vectors_torch, dim=1)
                total_mac_vectors_torch = torch.sum(total_mac_vectors_torch, dim=1)
                time_profiler.end_measurement()
                torch_mac_computation_times.append(time_profiler.get_time())

                time_profiler.start_measurement()
                node_selection_arrays_numpy = self.get_executed_nodes_wrt_trajectories_baseline(
                    cigt_outputs=cigt_outputs,
                    batch_size=batch_size,
                    action_trajectories=actions)
                time_profiler.end_measurement()
                numpy_trajectory_computation_times.append(time_profiler.get_time())

                time_profiler.start_measurement()
                correctness_vector_numpy, expert_probs_numpy = self.calculate_moe_for_final_layer_baseline(
                    cigt_outputs=cigt_outputs,
                    batch_size=batch_size,
                    executed_nodes_array=
                    node_selection_arrays_numpy[-1])
                time_profiler.end_measurement()
                numpy_moe_computation_times.append(time_profiler.get_time())

                time_profiler.start_measurement()
                mac_vectors_numpy = []
                for layer, trajectory_single_step in enumerate(node_selection_arrays_numpy):
                    mac_vector_numpy = \
                        self.calculate_mac_vector_for_layer_baseline(layer=layer,
                                                                     executed_nodes_array=trajectory_single_step)
                    mac_vectors_numpy.append(mac_vector_numpy)
                total_mac_vectors_numpy = np.stack(mac_vectors_numpy, axis=1)
                total_mac_vectors_numpy = np.sum(total_mac_vectors_numpy, axis=1)
                time_profiler.end_measurement()
                numpy_mac_computation_times.append(time_profiler.get_time())

                # Compare executed node arrays
                assert len(node_selection_arrays_torch) == len(self.pathCounts)
                assert len(node_selection_arrays_numpy) == len(self.pathCounts)
                for t in range(len(self.pathCounts)):
                    numpy_selections_array = node_selection_arrays_numpy[t]
                    selections_torch = node_selection_arrays_torch[t].detach().cpu().numpy()
                    torch_selections_array = [set() for _ in range(batch_size)]
                    non_zero_indices = np.nonzero(selections_torch)
                    non_zero_indices = np.stack(non_zero_indices, axis=1)
                    for idx in range(non_zero_indices.shape[0]):
                        index_tpl = non_zero_indices[idx]
                        sample_id = index_tpl[0]
                        selection_tuple = tuple(index_tpl[1:])
                        torch_selections_array[sample_id].add(selection_tuple)

                    assert len(numpy_selections_array) == len(torch_selections_array)
                    for idx in range(len(numpy_selections_array)):
                        assert numpy_selections_array[idx] == torch_selections_array[idx]

                # Compare MoE results
                assert np.array_equal(correctness_vector_torch.detach().cpu().numpy(),
                                      correctness_vector_numpy.detach().cpu().numpy())
                assert np.allclose(expert_probs_torch.detach().cpu().numpy(),
                                   expert_probs_numpy.detach().cpu().numpy())

                # Compare MAC results
                assert np.allclose(total_mac_vectors_torch.detach().cpu().numpy(),
                                   total_mac_vectors_numpy)

            print("Torch Trajectory Computation time:{0}".format(np.mean(np.array(torch_trajectory_computation_times))))
            print("Numpy Trajectory Computation time:{0}".format(np.mean(np.array(numpy_trajectory_computation_times))))
            print("Torch MoE Computation time:{0}".format(np.mean(np.array(torch_moe_computation_times))))
            print("Numpy MoE Computation time:{0}".format(np.mean(np.array(numpy_moe_computation_times))))
            print("Torch MAC Computation time:{0}".format(np.mean(np.array(torch_mac_computation_times))))
            print("Numpy MAC Computation time:{0}".format(np.mean(np.array(numpy_mac_computation_times))))

    def calculate_optimal_q_tables(self, cigt_outputs, batch_size):
        # Always start with a fixed action, that is the execution of the root node.
        action_spaces = [1]
        action_spaces.extend(self.actionSpaces)
        q_tables = []
        for t in range(len(action_spaces)):
            q_table_shape = (batch_size, *action_spaces[:(t + 1)])
            q_table = torch.zeros(size=q_table_shape, dtype=torch.float32, device=self.device)
            q_tables.append(q_table)
        action_trajectory_to_executed_nodes_dict = {}

        for t in range(len(action_spaces) - 1, -1, -1):
            action_trajectories_for_t = Utilities.create_route_combinations(shape_=action_spaces[:(t + 1)])
            if t == len(action_spaces) - 1:
                # Last layer
                for action_trajectory in action_trajectories_for_t:
                    action_trajectories = torch.Tensor(action_trajectory).to(self.device).to(torch.int64)
                    action_trajectories = torch.unsqueeze(action_trajectories, dim=0)
                    action_trajectories = torch.tile(action_trajectories, dims=(batch_size, 1))
                    # action_trajectories = action_trajectories[:, 1:]
                    executed_nodes_array = self.get_executed_nodes_wrt_trajectories(
                        cigt_outputs=cigt_outputs,
                        batch_size=batch_size,
                        action_trajectories=action_trajectories[:, 1:])
                    # Add execution nodes
                    for tt in range(len(action_trajectory)):
                        sub_trajectory = action_trajectory[:(tt + 1)]
                        if sub_trajectory not in action_trajectory_to_executed_nodes_dict:
                            action_trajectory_to_executed_nodes_dict[sub_trajectory] = executed_nodes_array[tt]
                        else:
                            comparison_flag = torch.equal(executed_nodes_array[tt],
                                                          action_trajectory_to_executed_nodes_dict[sub_trajectory])
                            assert comparison_flag

                    correctness_vector, expert_probs = self.calculate_moe_for_final_layer(
                        cigt_outputs=cigt_outputs,
                        batch_size=batch_size,
                        executed_nodes_array=executed_nodes_array[-1])
                    mac_vector = \
                        self.calculate_mac_vector_for_layer(layer=t, executed_nodes_array=executed_nodes_array[-1])
                    lmb = self.policyNetworksMacLambda
                    reward_array = (1.0 - lmb) * correctness_vector - lmb * mac_vector
                    # Prepare the index array to write into the optimal q table
                    index_array = [torch.arange(batch_size, device=self.device)]
                    for idx in range(action_trajectories.shape[1]):
                        index_array.append(action_trajectories[:, idx])
                    q_tables[t][index_array] = reward_array
            else:
                # Intermediate layers, apply Bellman equation
                # Q(s_t,a_t) = E_{s_{t+1}}[r(s_t,a_t) + \gamma \max_{a_{t+1}} Q(s_{t+1},a_{t+1})]
                # In our case, p(s_{t+1}|s_t,a_t) is deterministic: s_{t+1}=f(s_t,a_t)
                # Q(s_t,a_t) = r(s_t,a_t) + \gamma \max_{a_{t+1}} Q(f(s_t,a_t),a_{t+1})]
                for action_trajectory in action_trajectories_for_t:
                    # print(action_trajectory)
                    action_trajectories = torch.Tensor(action_trajectory).to(self.device).to(torch.int64)
                    action_trajectories = torch.unsqueeze(action_trajectories, dim=0)
                    action_trajectories = torch.tile(action_trajectories, dims=(batch_size, 1))
                    index_array = [torch.arange(batch_size, device=self.device)]
                    for idx in range(action_trajectories.shape[1]):
                        index_array.append(action_trajectories[:, idx])
                    q_table_next_step = q_tables[t + 1][index_array]
                    assert len(q_table_next_step.shape) == 2
                    assert q_table_next_step.shape[1] == action_spaces[t + 1]
                    q_max_t_plus_1 = torch.max(q_table_next_step, dim=1)[0]
                    gamma_q_max_t_plus_1 = q_max_t_plus_1 * self.policyNetworksDiscountFactor
                    assert action_trajectory in action_trajectory_to_executed_nodes_dict
                    mac_vector = \
                        self.calculate_mac_vector_for_layer(
                            layer=t, executed_nodes_array=action_trajectory_to_executed_nodes_dict[action_trajectory])
                    layer_reward = -self.policyNetworksMacLambda * mac_vector
                    q_tables[t][index_array] = layer_reward + gamma_q_max_t_plus_1
        return q_tables

    def calculate_optimal_q_tables_baseline(self, cigt_outputs, batch_size):
        action_spaces = [1]
        action_spaces.extend(self.actionSpaces)
        q_tables = []
        for t in range(len(action_spaces)):
            q_table_shape = (batch_size, *action_spaces[:(t + 1)])
            q_table = np.zeros(shape=q_table_shape, dtype=np.float32)
            q_tables.append(q_table)
        action_trajectory_to_executed_nodes_dict = {}

        for t in range(len(action_spaces) - 1, -1, -1):
            action_trajectories_for_t = Utilities.create_route_combinations(shape_=action_spaces[:(t + 1)])
            if t == len(action_spaces) - 1:
                for action_trajectory in action_trajectories_for_t:
                    action_trajectories = np.expand_dims(action_trajectory, axis=0)
                    action_trajectories = np.tile(action_trajectories, reps=(batch_size, 1))
                    node_selection_arrays = self.get_executed_nodes_wrt_trajectories_baseline(
                        cigt_outputs=cigt_outputs,
                        batch_size=batch_size,
                        action_trajectories=action_trajectories[:, 1:])
                    # Add execution nodes
                    for tt in range(len(action_trajectory)):
                        sub_trajectory = action_trajectory[:(tt + 1)]
                        if sub_trajectory not in action_trajectory_to_executed_nodes_dict:
                            action_trajectory_to_executed_nodes_dict[sub_trajectory] = node_selection_arrays[tt]
                        else:
                            comparison_flag = np.array_equal(node_selection_arrays[tt],
                                                             action_trajectory_to_executed_nodes_dict[sub_trajectory])
                            assert comparison_flag

                    correctness_vector, expert_probs = \
                        self.calculate_moe_for_final_layer_baseline(cigt_outputs=cigt_outputs,
                                                                    batch_size=batch_size,
                                                                    executed_nodes_array=node_selection_arrays[-1])
                    mac_vector_numpy = \
                        self.calculate_mac_vector_for_layer_baseline(layer=t,
                                                                     executed_nodes_array=node_selection_arrays[-1])
                    for sample_id in range(batch_size):
                        reward_accuracy = (1.0 - self.policyNetworksMacLambda) * \
                                          correctness_vector[sample_id].detach().cpu().numpy()
                        mac_penalty = self.policyNetworksMacLambda * mac_vector_numpy[sample_id]
                        final_reward = reward_accuracy - mac_penalty
                        q_tables[-1][(sample_id, *action_trajectory)] = final_reward
            else:
                for action_trajectory in action_trajectories_for_t:
                    mac_vector_numpy = \
                        self.calculate_mac_vector_for_layer_baseline(
                            layer=t,
                            executed_nodes_array=action_trajectory_to_executed_nodes_dict[action_trajectory])
                    for sample_id in range(batch_size):
                        q_plus_one = q_tables[t + 1][(sample_id, *action_trajectory)]
                        max_q = np.max(q_plus_one)
                        mac_sample = mac_vector_numpy[sample_id]
                        q_tables[t][(sample_id, *action_trajectory)] = \
                            (self.policyNetworksDiscountFactor * max_q) - self.policyNetworksMacLambda * mac_sample
        return q_tables

    def compare_q_table_calculation_types(self, dataset):
        time_profiler = TimeProfiler()
        times_torch = []
        times_baseline = []
        for i, batch in enumerate(dataset):
            print("Iteration:{0}".format(i))
            if self.usingPrecalculatedDatasets:
                cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
                cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
            else:
                input_var = torch.autograd.Variable(batch[0]).to(self.device)
                target_var = torch.autograd.Variable(batch[1]).to(self.device)
                cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)

            time_profiler.start_measurement()
            q_tables_torch = self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs, batch_size=batch_size)
            time_profiler.end_measurement()
            times_torch.append(time_profiler.get_time())

            time_profiler.start_measurement()
            q_tables_baseline = \
                self.calculate_optimal_q_tables_baseline(cigt_outputs=cigt_outputs, batch_size=batch_size)
            time_profiler.end_measurement()
            times_baseline.append(time_profiler.get_time())

            assert len(q_tables_torch) == len(q_tables_baseline)
            for idx in range(len(q_tables_torch)):
                q_t = q_tables_torch[idx].detach().cpu().numpy()
                q_t_b = q_tables_baseline[idx]
                assert np.allclose(q_t, q_t_b)

        print("times_torch:{0}".format(np.mean(np.array(times_torch))))
        print("times_baseline:{0}".format(np.mean(np.array(times_baseline))))
        print("Test has been successfully completed!!!")

    def compare_q_net_input_calculation_types(self, dataset):
        time_profiler = TimeProfiler()
        times_torch = []
        times_baseline = []
        for i, batch in enumerate(dataset):
            print("Iteration:{0}".format(i))
            if self.usingPrecalculatedDatasets:
                cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
                cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
            else:
                input_var = torch.autograd.Variable(batch[0]).to(self.device)
                target_var = torch.autograd.Variable(batch[1]).to(self.device)
                cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)

            q_tables_torch = self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs, batch_size=batch_size)

            # time_profiler.start_measurement()
            # q_tables_baseline = \
            #     self.calculate_optimal_q_tables_baseline(cigt_outputs=cigt_outputs, batch_size=batch_size)
            # time_profiler.end_measurement()
            # times_baseline.append(time_profiler.get_time())

            action_trajectories = self.sample_action_trajectories(optimal_q_tables=q_tables_torch,
                                                                  batch_size=batch_size)
            # Calculate the arrays of executed nodes.
            executed_nodes_array_torch = self.get_executed_nodes_wrt_trajectories(
                cigt_outputs=cigt_outputs,
                batch_size=batch_size,
                action_trajectories=action_trajectories[:, 1:])

            executed_nodes_array_numpy = self.get_executed_nodes_wrt_trajectories_baseline(
                cigt_outputs=cigt_outputs,
                batch_size=batch_size,
                action_trajectories=action_trajectories.detach().cpu().numpy()[:, 1:])

            time_profiler.start_measurement()
            sparse_inputs_array_torch = self.prepare_q_net_inputs(cigt_outputs=cigt_outputs,
                                                                  batch_size=batch_size,
                                                                  executed_nodes_array=executed_nodes_array_torch)
            time_profiler.end_measurement()
            times_torch.append(time_profiler.get_time())

            time_profiler.start_measurement()
            sparse_inputs_array_numpy = self.prepare_q_net_inputs_baseline(
                cigt_outputs=cigt_outputs,
                batch_size=batch_size,
                executed_nodes_array=executed_nodes_array_numpy)
            time_profiler.end_measurement()
            times_baseline.append(time_profiler.get_time())

            # Compare sparse input arrays
            assert len(sparse_inputs_array_torch) == len(self.pathCounts) - 1
            assert len(sparse_inputs_array_numpy) == len(self.pathCounts) - 1

            for tt in range(len(self.pathCounts) - 1):
                arr_torch = sparse_inputs_array_torch[tt].detach().cpu().numpy()
                arr_numpy = sparse_inputs_array_numpy[tt]
                assert np.allclose(arr_torch, arr_numpy)
                assert np.array_equal(arr_torch, arr_numpy)

            print("times_torch:{0}".format(np.mean(np.array(times_torch))))
            print("times_baseline:{0}".format(np.mean(np.array(times_baseline))))
            print("Test has been successfully completed!!!")

    def sample_action_trajectories(self, q_tables, batch_size):
        action_trajectories = []
        sample_indices = torch.arange(batch_size).to(self.device)
        for t, main_q_table in enumerate(q_tables):
            index_array = [sample_indices]
            for a_arr in action_trajectories:
                index_array.append(a_arr)
            q_table = main_q_table[index_array]
            # Greedy, optimal policy choices
            greedy_a = torch.argmax(q_table, dim=1)
            # Random actions
            random_q_table = torch.randint_like(input=q_table, low=0, high=1000, device=self.device)
            random_a = torch.argmax(random_q_table, dim=1)
            # Random vector in U[0,1]
            random_decision_vector = torch.rand(size=(batch_size,), device=self.device)
            select_random_action = self.epsilonValue >= random_decision_vector
            final_a = torch.where(select_random_action, random_a, greedy_a)
            action_trajectories.append(final_a)
        action_trajectories = torch.stack(action_trajectories, dim=1)
        return action_trajectories

    def create_index_array_for_q_table(self, batch_size, path_combination):
        path_trajectories = torch.Tensor(path_combination).to(self.device).to(torch.int64)
        path_trajectories = torch.unsqueeze(path_trajectories, dim=0)
        path_trajectories = torch.tile(path_trajectories, dims=(batch_size, 1))
        index_array = [torch.arange(batch_size, device=self.device)]
        for t in range(path_trajectories.shape[1]):
            index_array.append(path_trajectories[:, t])
        return index_array

    def create_q_net_input_array(self, layer_id, cigt_outputs, batch_size):
        t = layer_id
        action_trajectories_for_t = Utilities.create_route_combinations(shape_=self.pathCounts[:(t + 1)])
        input_array_shapes = set()
        for path_combination in action_trajectories_for_t:
            input_array_shapes.add(cigt_outputs["block_outputs_dict"][path_combination].shape)
        assert len(input_array_shapes) == 1
        input_array_shape = list(input_array_shapes)[0]
        assert input_array_shape[0] == batch_size
        destination_array_shape = (batch_size, *self.pathCounts[:(t + 1)], *input_array_shape[1:])
        sparse_q_net_input = torch.zeros(size=destination_array_shape, dtype=torch.float32, device=self.device)
        return sparse_q_net_input

    def prepare_q_net_inputs(self, cigt_outputs, batch_size, executed_nodes_array):
        sparse_inputs_array = []
        for t in range(len(self.pathCounts) - 1):
            action_trajectories_for_t = Utilities.create_route_combinations(shape_=self.pathCounts[:(t + 1)])
            sparse_q_net_input = self.create_q_net_input_array(layer_id=t,
                                                               cigt_outputs=cigt_outputs,
                                                               batch_size=batch_size)
            for path_combination in action_trajectories_for_t:
                index_array = self.create_index_array_for_q_table(batch_size=batch_size,
                                                                  path_combination=path_combination)
                sparsity_array_for_path_combination = executed_nodes_array[t][index_array]
                assert sparsity_array_for_path_combination.shape == (batch_size,)
                output_array_for_path_combination = cigt_outputs["block_outputs_dict"][path_combination]
                for _ in range(len(output_array_for_path_combination.shape) - 1):
                    sparsity_array_for_path_combination = torch.unsqueeze(sparsity_array_for_path_combination, dim=-1)
                partial_sparse_q_net_input = sparsity_array_for_path_combination * output_array_for_path_combination
                sparse_q_net_input[index_array] = partial_sparse_q_net_input
            sparse_inputs_array.append(sparse_q_net_input)
        return sparse_inputs_array

    def prepare_q_net_inputs_baseline(self, cigt_outputs, batch_size, executed_nodes_array):
        sparse_inputs_array = []
        for t in range(len(self.pathCounts) - 1):
            sparse_q_net_input = self.create_q_net_input_array(layer_id=t,
                                                               cigt_outputs=cigt_outputs,
                                                               batch_size=batch_size).detach().cpu().numpy()
            for sample_id in range(batch_size):
                executed_blocks_set = executed_nodes_array[t][sample_id]
                for block_id_tpl in executed_blocks_set:
                    sample_block = cigt_outputs["block_outputs_dict"][block_id_tpl][sample_id].detach().cpu().numpy()
                    # Place it into the appropriate location in sparse_q_net_input
                    block_index = (sample_id, *block_id_tpl)
                    sparse_q_net_input[block_index] = sample_block
            sparse_inputs_array.append(sparse_q_net_input)
        return sparse_inputs_array

    def execute_q_networks(self, sparse_inputs_array):
        assert len(sparse_inputs_array) == len(self.policyNetworks)
        assert self.policyGradientsUseLstm
        q_network_outputs = []
        for layer_id, input_arr in enumerate(sparse_inputs_array):
            # Reshape sparse inputs into (B,C,W,H) shaped arrays.
            sparse_input = torch.reshape(input_arr, shape=(input_arr.shape[0],
                                                           np.prod(input_arr.shape[1:layer_id + 3]),
                                                           input_arr.shape[-2], input_arr.shape[-1]))
            output_arr = self.policyNetworks[layer_id](sparse_input)
            q_network_outputs.append(output_arr)
        q_network_outputs = torch.stack(q_network_outputs, dim=1)
        lstm_outputs = self.lstm(q_network_outputs)[0]
        lstm_outputs = torch.transpose_copy(lstm_outputs, dim0=0, dim1=1)
        q_net_outputs = []
        for layer_id in range(len(self.pathCounts) - 1):
            q_features = lstm_outputs[layer_id]
            q_net_output = self.policyNetworkQNetRegressionLayers[layer_id](q_features)
            q_net_outputs.append(q_net_output)
        return q_net_outputs

    def calculate_regression_loss(self, q_net_outputs, optimal_q_tables, action_trajectories):
        pass

    def forward_with_actions(self, cigt_outputs, batch_size, action_trajectories):
        # cigt_outputs, batch_size = self.get_cigt_outputs(x=x, y=y)
        # Calculate the arrays of executed nodes.
        executed_nodes_array = self.get_executed_nodes_wrt_trajectories(
            cigt_outputs=cigt_outputs,
            batch_size=batch_size,
            action_trajectories=action_trajectories)
        # Prepare the (possibly) sparse inputs for the q networks, for every layer.
        sparse_inputs_array = self.prepare_q_net_inputs(cigt_outputs=cigt_outputs,
                                                        batch_size=batch_size,
                                                        executed_nodes_array=executed_nodes_array)
        # Execute the Q-Nets. Obtain the regression outputs for every Q-Net layer.
        q_net_outputs = self.execute_q_networks(sparse_inputs_array=sparse_inputs_array)

        # Calculate the correctness, MoE probabilities
        correctness_vector, expert_probs = self.calculate_moe_for_final_layer(
            cigt_outputs=cigt_outputs,
            batch_size=batch_size,
            executed_nodes_array=executed_nodes_array[-1])

        # Calculate the extra MAC incurred due to multiple path executions
        mac_vectors = []
        for tt, e_nodes in enumerate(executed_nodes_array):
            mac_vector = self.calculate_mac_vector_for_layer(layer=tt, executed_nodes_array=e_nodes)
            mac_vectors.append(mac_vector)
        mac_vectors = torch.stack(mac_vectors, dim=1)
        total_mac_vector = torch.sum(mac_vectors, dim=1)

        results_dict = {
            "q_net_outputs": q_net_outputs,
            "correctness_vector": correctness_vector,
            "expert_probs": expert_probs,
            "mac_vectors": mac_vectors,
            "total_mac_vector": total_mac_vector
        }
        return results_dict

    def validate_with_expectation(self, loader, temperature=None):
        self.eval()
        time_profiler = TimeProfiler()
        # If temperature is None, then it is treated as greedy.
        if temperature is None:
            temperature = 1e-10
        action_space = [1]
        action_space.extend(self.actionSpaces)
        all_trajectories = Utilities.create_route_combinations(shape_=action_space)

        optimal_q_tables_dataset = []
        predicted_q_tables_dataset = []
        for _ in range(len(action_space)):
            optimal_q_tables_dataset.append([])
            predicted_q_tables_dataset.append([])
        trajectory_probabilities_dict = {}
        correctness_vectors_dict = {}
        mac_vectors_dict = {}
        greedy_correctness_vectors = []
        greedy_mac_vectors = []
        time_spent = []
        policy_distributions_dict = {}
        mse_dict = {}
        r2_dict = {}

        print("Device:{0}".format(self.device))
        total_sample_count = 0
        for i__, batch in tqdm(enumerate(loader)):
            time_profiler.start_measurement()
            with torch.no_grad():
                if self.usingPrecalculatedDatasets:
                    cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
                    cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
                else:
                    input_var = torch.autograd.Variable(batch[0]).to(self.device)
                    target_var = torch.autograd.Variable(batch[1]).to(self.device)
                    cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)

                # Calculate the optimal q tables. Add to the dataset-wide lists at every layer.
                optimal_q_tables = self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs, batch_size=batch_size)
                for tt, optimal_q_table in enumerate(optimal_q_tables):
                    optimal_q_tables_dataset[tt].append(optimal_q_table)

                # Holds predicted q tables per action trajectory.
                batch_predicted_q_tables_dict = {}

                # Calculate results for every possible trajectory.
                for action_trajectory in all_trajectories:
                    action_trajectory_torch = torch.Tensor(action_trajectory).to(self.device).to(torch.int64)
                    action_trajectory_torch = torch.unsqueeze(action_trajectory_torch, dim=0)
                    action_trajectory_torch = torch.tile(action_trajectory_torch, dims=(batch_size, 1))
                    result_dict = self.forward_with_actions(cigt_outputs=cigt_outputs, batch_size=batch_size,
                                                            action_trajectories=action_trajectory_torch[:, 1:])
                    # Record predicted q_tables in a structured way.
                    trajectory_probabilities = []
                    for idx, predicted_q_table in enumerate(result_dict["q_net_outputs"]):
                        q_table_idx = action_trajectory[:(idx + 1)]
                        if q_table_idx not in batch_predicted_q_tables_dict:
                            batch_predicted_q_tables_dict[q_table_idx] = predicted_q_table
                        else:
                            q_tables_close = torch.allclose(batch_predicted_q_tables_dict[q_table_idx],
                                                            predicted_q_table)
                            assert q_tables_close
                        # Calculate the selection probabilities for the given trajectory, for every sample,
                        # from the predicted q_tables
                        predicted_q_table_tempered = predicted_q_table / temperature
                        step_probabilities = torch.nn.functional.softmax(predicted_q_table_tempered, dim=1)
                        # Get probabilities for this step.
                        step_action = action_trajectory[idx + 1]
                        step_action_probabilities = step_probabilities[:, step_action]
                        trajectory_probabilities.append(step_action_probabilities)
                    # Save the joint probability of every action trajectory.
                    trajectory_probabilities = torch.stack(trajectory_probabilities, dim=1)
                    trajectory_probabilities = torch.prod(trajectory_probabilities, dim=1)
                    if action_trajectory not in trajectory_probabilities_dict:
                        trajectory_probabilities_dict[action_trajectory] = []
                    trajectory_probabilities_dict[action_trajectory].append(trajectory_probabilities)

                    # Record predicted correctness vectors and mac vectors
                    if action_trajectory not in correctness_vectors_dict:
                        correctness_vectors_dict[action_trajectory] = []
                    correctness_vectors_dict[action_trajectory].append(result_dict["correctness_vector"])
                    if action_trajectory not in mac_vectors_dict:
                        mac_vectors_dict[action_trajectory] = []
                    mac_vectors_dict[action_trajectory].append(result_dict["total_mac_vector"])

                # Add predicted q tables into the proper locations
                for tt, optimal_q_table in enumerate(optimal_q_tables):
                    # We dont need to predict the first q table, it is trivially always single selection.
                    if tt == 0:
                        continue
                    predicted_q_table_full = torch.zeros_like(optimal_q_table)
                    previous_trajectories = Utilities.create_route_combinations(shape_=action_space[:tt])
                    for previous_trajectory in previous_trajectories:
                        assert previous_trajectory in batch_predicted_q_tables_dict
                        index_array = \
                            self.create_index_array_for_q_table(batch_size=batch_size,
                                                                path_combination=previous_trajectory)
                        predicted_q_table_full[index_array] = batch_predicted_q_tables_dict[previous_trajectory]
                    predicted_q_tables_dataset[tt].append(predicted_q_table_full)

                # Prepare the greedy results:
                # For debug purposes: Always use the greedy policy for each step: a_t = argmax_x Q_t(s_t,x).
                # Compare it with the expectation method where temperature is very small.
                # Since tempereture -> 0 means the policy distributions derived from the predicted q tables will
                # approach to one-hot vectors, where one entry is the argmax,
                # we must obtain the same results. (ONLY FOR VERY SMALL TEMPERATURES!!!)
                index_array = [torch.arange(batch_size, device=self.device),
                               torch.zeros(size=(batch_size,), dtype=torch.int64, device=self.device)]
                for t in range(len(action_space)):
                    if t == 0:
                        continue
                    q_table_t = predicted_q_tables_dataset[t][-1][index_array]
                    greedy_actions = torch.argmax(q_table_t, dim=1)
                    index_array.append(greedy_actions)
                index_array = torch.stack(index_array, dim=1)
                greedy_action_trajectory = index_array[:, 2:]
                greedy_results_dict = self.forward_with_actions(cigt_outputs=cigt_outputs,
                                                                action_trajectories=greedy_action_trajectory,
                                                                batch_size=batch_size)
                greedy_correctness_vectors.append(greedy_results_dict["correctness_vector"])
                greedy_mac_vectors.append(greedy_results_dict["total_mac_vector"])

            time_profiler.end_measurement()
            time_spent.append(time_profiler.get_time())
            total_sample_count += batch_size

        # Concatenate predicted q_tables and optimal q_tables. Measure MSE and R2 scores between each compatible table.
        for tt in range(len(action_space)):
            if tt == 0:
                continue
            q_true = torch.concat(optimal_q_tables_dataset[tt], dim=0)
            q_pred = torch.concat(predicted_q_tables_dataset[tt], dim=0)
            assert q_true.shape == q_pred.shape
            previous_trajectories = Utilities.create_route_combinations(shape_=action_space[:tt])
            for previous_trajectory in previous_trajectories:
                index_array = \
                    self.create_index_array_for_q_table(batch_size=q_true.shape[0],
                                                        path_combination=previous_trajectory)
                q_partial_true = q_true[index_array].cpu().numpy()
                q_partial_pred = q_pred[index_array].cpu().numpy()
                mse_ = mean_squared_error(y_true=q_partial_true, y_pred=q_partial_pred)
                r2_ = r2_score(y_true=q_partial_true, y_pred=q_partial_pred)
                # Measure the policy distribution as well.
                policy_distribution = torch.nn.functional.softmax(q_pred[index_array], dim=1)
                mean_policy_distribution = torch.mean(policy_distribution, dim=0).cpu().numpy()
                policy_distributions_dict[previous_trajectory] = mean_policy_distribution
                mse_dict[previous_trajectory] = mse_
                r2_dict[previous_trajectory] = r2_
                print("Trajectory:{0} MSE:{1} R2:{2} Mean Policy Distribution:{3}".format(
                    previous_trajectory, mse_, r2_, mean_policy_distribution))

            optimal_q_tables_dataset[tt] = q_true
            predicted_q_tables_dataset[tt] = q_pred

        # Concatenate correctness vector and mac vectors from every trajectory. Measure the expected accuracy and mac.
        all_trajectories = Utilities.create_route_combinations(shape_=action_space)
        action_probabilities_matrix = []
        correctness_vectors_matrix = []
        mac_vectors_matrix = []
        for actions in all_trajectories:
            probs_full = torch.concat(trajectory_probabilities_dict[actions], dim=0)
            correctness_full = torch.concat(correctness_vectors_dict[actions], dim=0)
            mac_full = torch.concat(mac_vectors_dict[actions], dim=0)
            action_probabilities_matrix.append(probs_full)
            correctness_vectors_matrix.append(correctness_full)
            mac_vectors_matrix.append(mac_full)
        action_probabilities_matrix = torch.stack(action_probabilities_matrix, dim=1)
        correctness_vectors_matrix = torch.stack(correctness_vectors_matrix, dim=1)
        mac_vectors_matrix = torch.stack(mac_vectors_matrix, dim=1)
        sum_prob = torch.sum(action_probabilities_matrix, dim=1)
        assert torch.allclose(sum_prob, torch.ones_like(sum_prob))
        expected_accuracy = torch.mean(torch.sum(action_probabilities_matrix * correctness_vectors_matrix, dim=1))
        expected_mac = torch.mean(torch.sum(action_probabilities_matrix * mac_vectors_matrix, dim=1))
        expected_accuracy = expected_accuracy.cpu().numpy()
        expected_mac = expected_mac.cpu().numpy()
        expected_time = np.mean(np.array(time_spent))

        # if get_greedy_prediction:
        greedy_correctness_vector_full = torch.concat(greedy_correctness_vectors, dim=0)
        greedy_mac_vector_full = torch.concat(greedy_mac_vectors, dim=0)
        greedy_accuracy = torch.mean(greedy_correctness_vector_full).cpu().numpy()
        greedy_mac = torch.mean(greedy_mac_vector_full).cpu().numpy()
        # assert np.allclose(greedy_accuracy, expected_accuracy)
        # assert np.allclose(greedy_mac, expected_mac)

        return {"expected_accuracy": expected_accuracy,
                "expected_mac": expected_mac,
                "expected_time": expected_time,
                "greedy_accuracy": greedy_accuracy,
                "greedy_mac": greedy_mac,
                "policy_distributions_dict": policy_distributions_dict,
                "mse_dict": mse_dict,
                "r2_dict": r2_dict}

    def validate_with_single_action_trajectory(self, loader, action_trajectory):
        self.eval()
        assert isinstance(action_trajectory, tuple)
        action_trajectory_torch = torch.Tensor(action_trajectory).to(self.device).to(torch.int64)
        action_trajectory_torch = torch.unsqueeze(action_trajectory_torch, dim=0)

        all_q_net_outputs = []
        for _ in range(len(self.actionSpaces)):
            all_q_net_outputs.append([])
        all_correctness_vectors = []
        all_expert_probs = []
        all_total_mac_vectors = []
        time_spent_arr = []
        time_profiler = TimeProfiler()

        print("Device:{0}".format(self.device))
        for i__, batch in tqdm(enumerate(loader)):
            time_profiler.start_measurement()
            if self.usingPrecalculatedDatasets:
                cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
                cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
            else:
                input_var = torch.autograd.Variable(batch[0]).to(self.device)
                target_var = torch.autograd.Variable(batch[1]).to(self.device)
                cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)

            action_trajectories = torch.tile(action_trajectory_torch, dims=(batch_size, 1))
            result_dict = self.forward_with_actions(cigt_outputs=cigt_outputs, batch_size=batch_size,
                                                    action_trajectories=action_trajectories)
            # for idx, q_arr in enumerate(result_dict["q_net_outputs"]):
            #     all_q_net_outputs[idx].append(q_arr)
            all_correctness_vectors.append(result_dict["correctness_vector"])
            # all_expert_probs.append(result_dict["expert_probs"])
            all_total_mac_vectors.append(result_dict["total_mac_vector"])
            time_profiler.end_measurement()
            time_spent_arr.append(time_profiler.get_time())

        # for idx in range(len(all_q_net_outputs)):
        #     all_q_net_outputs[idx] = torch.concat(all_q_net_outputs[idx], dim=0)
        all_correctness_vectors = torch.concat(all_correctness_vectors, dim=0)
        # all_expert_probs = torch.concat(all_expert_probs, dim=0)
        all_total_mac_vectors = torch.concat(all_total_mac_vectors, dim=0)

        accuracy = torch.mean(all_correctness_vectors).detach().cpu().numpy()
        mac_avg = torch.mean(all_total_mac_vectors).detach().cpu().numpy()
        time_avg = np.mean(np.array(time_spent_arr))
        return accuracy, mac_avg, time_avg

    def evaluate_datasets(self, train_loader, test_loader, epoch):
        print("************** Epoch:{0} **************".format(epoch))
        kv_rows = []
        results_summary = {"Train": {}, "Test": {}}
        for data_type, data_loader in [("Test", test_loader), ("Train", train_loader)]:
            results_dict = self.validate_with_expectation(loader=data_loader)
            print("Expected {0} Accuracy:{1}".format(data_type, results_dict["expected_accuracy"]))
            print("Expected {0} Mac:{1}".format(data_type, results_dict["expected_mac"]))
            print("Expected {0} Mean Time:{1}".format(data_type, results_dict["expected_time"]))
            print("Greedy {0} Accuracy:{1}".format(data_type, results_dict["greedy_accuracy"]))
            print("Greedy {0} Mac:{1}".format(data_type, results_dict["greedy_mac"]))
            policy_distributions_dict = results_dict["policy_distributions_dict"]
            results_summary[data_type]["Accuracy"] = results_dict["expected_accuracy"]
            results_summary[data_type]["Mac"] = results_dict["expected_mac"]
            mse_dict = results_dict["mse_dict"]
            r2_dict = results_dict["r2_dict"]
            trajectories = set(policy_distributions_dict.keys())
            assert trajectories == set(mse_dict.keys()) and trajectories == set(r2_dict.keys())
            for trajectory in trajectories:
                policy_distribution = policy_distributions_dict[trajectory]
                mse_ = mse_dict[trajectory]
                r2_ = r2_dict[trajectory]
                print("{0} Policy Distribution {1}:{2}".format(data_type, trajectory, policy_distribution))
                print("{0} Q-Table MSE {1}:{2}".format(data_type, trajectory, mse_))
                print("{0} Q-Table R2 {1}:{2}".format(data_type, trajectory, r2_))

                kv_rows.append((self.runId,
                                epoch,
                                "{0} Policy Distribution {1}".format(data_type, trajectory),
                                "{0}".format(policy_distribution)))
                kv_rows.append((self.runId,
                                epoch,
                                "{0} Q-Table MSE {1}".format(data_type, trajectory),
                                "{0}".format(mse_)))
                kv_rows.append((self.runId,
                                epoch,
                                "{0} Q-Table R2 {1}".format(data_type, trajectory),
                                "{0}".format(r2_)))

        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)

        DbLogger.write_into_table(
            rows=[(self.runId,
                   self.iteration_id,
                   epoch,
                   results_summary["Train"]["Accuracy"].item(),
                   results_summary["Train"]["Mac"].item(),
                   results_summary["Test"]["Accuracy"].item(),
                   results_summary["Test"]["Mac"].item(),
                   0.0)], table=DbLogger.logsTableQCigt)

        print("************** Epoch:{0} **************".format(epoch))

    def fit_policy_network(self, train_loader, test_loader):
        self.to(self.device)
        print("Device:{0}".format(self.device))
        torch.manual_seed(1)
        best_performance = 0.0
        num_of_total_iterations = self.policyNetworkTotalNumOfEpochs * len(train_loader)

        # Run a forward pass first to initialize each LazyXXX layer.
        self.execute_forward_with_random_input()

        test_ig_accuracy, test_ig_mac, test_ig_time = self.validate_with_single_action_trajectory(
            loader=test_loader, action_trajectory=(0, 0))
        print("Test Ig Accuracy:{0} Test Ig Mac:{1} Test Ig Mean Validation Time:{2}".format(
            test_ig_accuracy, test_ig_mac, test_ig_time))

        train_ig_accuracy, train_ig_mac, train_ig_time = self.validate_with_single_action_trajectory(
            loader=train_loader, action_trajectory=(0, 0))
        print("Train Ig Accuracy:{0} Train Ig Mac:{1} Train Ig Mean Validation Time:{2}".format(
            train_ig_accuracy, train_ig_mac, train_ig_time))

        print("Device:{0}".format(self.device))
        # for epoch_id in range(0, self.policyNetworkTotalNumOfEpochs):
        #     for i, cigt_outputs in enumerate(train_loader):
        #         self.train()
        #         print("*************CIGT Q-Net Training Epoch:{0} Iteration:{1}*************".format(
        #             epoch_id, self.iteration_id))
        #         cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
        #

        # for i__, batch in tqdm(enumerate(loader)):

        # policy_entropies = []
        # log_probs_trajectory = []
        # probs_trajectory = []
        # actions_trajectory = []
        # full_action_probs_trajectory = []
        # correctness_vec = None
        # mac_vec = None
        # reward_array = None
        # paths_history = [{idx: {(0,)} for idx in range(batch_size)}]
        #
        # for layer_id in range(len(self.pathCounts)):
        #     if layer_id < len(self.pathCounts) - 1:
        #         # Get sparse input arrays for the policy networks
        #         pg_sparse_input = self.prepare_policy_network_input_f(
        #             batch_size=batch_size,
        #             layer_id=layer_id,
        #             current_paths_dict=paths_history[layer_id],
        #             cigt_outputs=cigt_outputs
        #         )
        #         # Execute this layers policy network, get log action probs, actions and policy entropies.
        #         mean_policy_entropy, log_probs_selected, probs_selected, action_probs, actions = \
        #             self.run_policy_networks(layer_id=layer_id, pn_input=pg_sparse_input)
        #         policy_entropies.append(mean_policy_entropy)
        #         log_probs_trajectory.append(log_probs_selected)
        #         probs_trajectory.append(probs_selected)
        #         actions_trajectory.append(actions)
        #         full_action_probs_trajectory.append(action_probs)
        #
        #         # Extend the trajectories for each sample based on the actions selected.
        #         new_paths_dict = self.extend_sample_trajectories_wrt_actions(actions=actions,
        #                                                                      cigt_outputs=cigt_outputs,
        #                                                                      current_paths_dict=paths_history[layer_id],
        #                                                                      layer_id=layer_id)
        #         paths_history.append(new_paths_dict)
        #     else:
        #         reward_array, correctness_vec, mac_vec = \
        #             self.calculate_rewards(cigt_outputs=cigt_outputs, complete_path_history=paths_history)
        #
        # return {
        #     "policy_entropies": policy_entropies,
        #     "log_probs_trajectory": log_probs_trajectory,
        #     "probs_trajectory": probs_trajectory,
        #     "actions_trajectory": actions_trajectory,
        #     "paths_history": paths_history,
        #     "reward_array": reward_array,
        #     "correctness_vec": correctness_vec,
        #     "mac_vec": mac_vec,
        #     "full_action_probs_trajectory": full_action_probs_trajectory}
