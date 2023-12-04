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

        super().__init__(configs, run_id, model_definition, num_classes, model_mac_info, is_debug_mode)

        if self.policyGradientsUseLstm:
            assert len(set([dim for dim in self.decisionDimensions])) == 1
            self.lstmInputDimension = self.decisionDimensions[0]
            self.lstm = nn.LSTM(input_size=self.lstmInputDimension,
                                hidden_size=self.policyNetworksLstmDimension,
                                num_layers=self.policyNetworksLstmNumLayers,
                                batch_first=True,
                                bidirectional=self.policyNetworksLstmBidirectional)
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
            # layers["policy_gradients_block_{0}_relu".format(layer_id)] = nn.ReLU()
            layers["policy_gradients_block_{0}_feature_fc".format(layer_id)] = nn.LazyLinear(
                out_features=self.decisionDimensions[layer_id])

            if not self.policyGradientsUseLstm:
                layers["policy_gradients_block_{0}_action_space_fc".format(layer_id)] = nn.Linear(
                    in_features=self.decisionDimensions[layer_id], out_features=action_space_size)

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
        return cigt_outputs, batch_size

    def execute_forward_with_random_input(self):
        max_batch_size = np.prod(self.pathCounts) * self.batchSize
        self.enforcedRoutingMatrices = []
        for path_count in self.pathCounts[1:]:
            self.enforcedRoutingMatrices.append(torch.ones(size=(max_batch_size, path_count),
                                                           dtype=torch.int64).to(self.device))
        if not self.usingPrecalculatedDatasets:
            fake_input = torch.from_numpy(
                np.random.uniform(size=(self.batchSize, *self.inputDims)).astype(dtype=np.float32)).to(self.device)
            fake_target = torch.ones(size=(self.batchSize,), dtype=torch.int64).to(self.device)
            print("fake_input.device:{0}".format(fake_input.device))
            print("fake_target.device:{0}".format(fake_target.device))
            for name, param in self.named_parameters():
                print("Parameter {0} Device:{1}".format(name, param.device))

            self.eval()
            self.forward_with_policies(x=fake_input, y=fake_target, greedy_actions=True)
        else:
            cigt_outputs = next(iter(self.testDataset))
            cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)

            self.eval()
            self.forward_with_policies(x=cigt_outputs, y=None, greedy_actions=True)
            self.enforcedRoutingMatrices = []

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
                    next_level_index_array.append(
                        cigt_outputs["routing_matrices_sorting_indices_dict"][prev_node_combination][:, a_])
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

    def compare_trajectory_evaluation_methods(self, dataset, repeat_count):
        for i, batch in enumerate(dataset):
            print("Iteration:{0}".format(i))
            if self.usingPrecalculatedDatasets:
                cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
            else:
                input_var = torch.autograd.Variable(batch[0]).to(self.device)
                target_var = torch.autograd.Variable(batch[1]).to(self.device)
                cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)

            torch_trajectory_computation_times = []
            numpy_trajectory_computation_times = []
            torch_moe_computation_times = []
            numpy_moe_computation_times = []

            for repeat_id in tqdm(range(repeat_count)):
                actions = []
                for a_ in self.actionSpaces:
                    actions.append(np.random.randint(low=0, high=a_, size=(batch_size,)))
                actions = np.stack(actions, axis=1)
                actions_torch = torch.from_numpy(actions).to(self.device)

                t0 = time.time()
                node_selection_arrays_torch = self.get_executed_nodes_wrt_trajectories(
                    cigt_outputs=cigt_outputs,
                    batch_size=batch_size,
                    action_trajectories=actions_torch)
                t1 = time.time()
                torch_trajectory_computation_times.append(t1 - t0)

                t2 = time.time()
                correctness_vector_torch, expert_probs_torch = self.calculate_moe_for_final_layer(
                    cigt_outputs=cigt_outputs,
                    batch_size=batch_size,
                    executed_nodes_array=
                    node_selection_arrays_torch[-1])
                t3 = time.time()
                torch_moe_computation_times.append(t3 - t2)

                t4 = time.time()
                node_selection_arrays_numpy = self.get_executed_nodes_wrt_trajectories_baseline(
                    cigt_outputs=cigt_outputs,
                    batch_size=batch_size,
                    action_trajectories=actions)
                t5 = time.time()
                numpy_trajectory_computation_times.append(t5 - t4)

                t6 = time.time()
                correctness_vector_numpy, expert_probs_numpy = self.calculate_moe_for_final_layer_baseline(
                    cigt_outputs=cigt_outputs,
                    batch_size=batch_size,
                    executed_nodes_array=
                    node_selection_arrays_numpy[-1])
                t7 = time.time()
                numpy_moe_computation_times.append(t7 - t6)

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

            print("Torch Trajectory Computation time:{0}".format(np.mean(np.array(torch_trajectory_computation_times))))
            print("Numpy Trajectory Computation time:{0}".format(np.mean(np.array(numpy_trajectory_computation_times))))
            print("Torch MoE Computation time:{0}".format(np.mean(np.array(torch_moe_computation_times))))
            print("Numpy MoE Computation time:{0}".format(np.mean(np.array(numpy_moe_computation_times))))

    def calculate_mac_vector_for_layer(self, layer, batch_size, executed_nodes_array):
        # ************** Secondly calculate the MAC costs for the final layer **************
        assert executed_nodes_array.shape[1:] == self.pathCounts[:(layer + 2)]
        node_dims = tuple([idx for idx in range(1, len(executed_nodes_array.shape))])
        node_counts = torch.sum(executed_nodes_array, dim=node_dims)
        extra_node_counts = node_counts - 1
        mac_values_for_layer = extra_node_counts * self.macCostPerLayer[layer + 1]
        relative_mac_values_for_layer = mac_values_for_layer / self.singlePathMacCost
        # ************** Secondly calculate the MAC costs for the final layer **************
        return relative_mac_values_for_layer

    def calculate_optimal_q_tables(self, cigt_outputs, batch_size):
        # Always start with a fixed action, that is the execution of the root node.
        action_spaces = [1]
        action_spaces.extend(self.actionSpaces)
        for t in range(len(action_spaces) - 2, -1, -1):
            action_trajectories_for_t = Utilities.create_route_combinations(shape_=action_spaces[:(t + 2)])
            if t == len(action_spaces) - 2:
                for action_trajectory in action_trajectories_for_t:
                    action_trajectories = torch.Tensor(action_trajectory).to(self.device).to(torch.int64)
                    action_trajectories = torch.unsqueeze(action_trajectories, dim=0)
                    action_trajectories = torch.tile(action_trajectories, dims=(batch_size, 1))
                    action_trajectories = action_trajectories[:, 1:]
                    executed_nodes_array = self.get_executed_nodes_wrt_trajectories(
                        cigt_outputs=cigt_outputs,
                        batch_size=batch_size,
                        action_trajectories=action_trajectories)
                    self.calculate_accuracy_vector_for_final_layer(cigt_outputs=cigt_outputs,
                                                                   batch_size=batch_size,
                                                                   executed_nodes_array=executed_nodes_array[
                                                                       -1])

            print("X")

            # path_combinations_for_t = Utilities.create_route_combinations(shape_=self.pathCounts[:(t + 2)])
            # q_table_shape = (batch_size, *self.pathCounts[:(t + 2)])
            # # Last layer (the loss layer). Calculate sample accuracies and MAC costs for the final layer.
            # if t == len(self.pathCounts) - 2:
            #     for path_combination in path_combinations_for_t:
            #         action_trajectory = path_combination
            #         action_trajectories = torch.Tensor(action_trajectory).to(self.device).to(torch.int64)
            #         action_trajectories = torch.unsqueeze(action_trajectories, dim=0)
            #         action_trajectories = torch.tile(action_trajectories, dims=(batch_size, 1))
            #         action_trajectories = action_trajectories[:, 1:]
            #         executed_nodes_array = self.get_executed_nodes_wrt_trajectories(
            #             cigt_outputs=cigt_outputs,
            #             batch_size=batch_size,
            #             action_trajectories=action_trajectories)
            #         self.calculate_final_rewards(cigt_outputs=cigt_outputs,
            #                                      batch_size=batch_size,
            #                                      executed_nodes_array=executed_nodes_array[-1])

    def forward_with_policies(self, x, y, greedy_actions=None):
        cigt_outputs, batch_size = self.get_cigt_outputs(x=x, y=y)
        # Calculate optimal Q-Tables
        self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs, batch_size=batch_size)

    def fit_policy_network(self, train_loader, test_loader):

        for epoch_id in range(0, self.policyNetworkTotalNumOfEpochs):
            for i, (input_, target) in enumerate(train_loader):
                self.forward_with_policies(x=input_, y=target)

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
