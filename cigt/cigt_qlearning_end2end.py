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
from cigt.cigt_q_learning import CigtQLearning
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

class CigtQlearningEnd2End(CigtQLearning):
    def __init__(self, configs, run_id, model_definition,
                 num_classes, model_mac_info, is_debug_mode, precalculated_datasets_dict):
        super().__init__(configs, run_id, model_definition, num_classes, model_mac_info, is_debug_mode,
                         precalculated_datasets_dict)

    def get_cigt_outputs(self, x, y, **kwargs):
        assert "temperature" in kwargs
        temperature = kwargs["temperature"]
        cigt_outputs = self.forward_v2(x=x, labels=y, temperature=temperature)
        batch_size = x.shape[0]
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
        fake_input = torch.from_numpy(
            np.random.uniform(size=(self.batchSize, *self.inputDims)).astype(dtype=np.float32)).to(self.device)
        fake_target = torch.ones(size=(self.batchSize,), dtype=torch.int64).to(self.device)
        print("fake_input.device:{0}".format(fake_input.device))
        print("fake_target.device:{0}".format(fake_target.device))
        for name, param in self.named_parameters():
            print("Parameter {0} Device:{1}".format(name, param.device))

        self.eval()
        cigt_outputs = self.forward_v2(x=fake_input, labels=fake_target, temperature=1.0)
        cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
        action_trajectories = torch.Tensor((0, 0)).to(self.device).to(torch.int64)
        action_trajectories = torch.unsqueeze(action_trajectories, dim=0)
        action_trajectories = torch.tile(action_trajectories, dims=(self.batchSize, 1))
        self.forward_with_actions(cigt_outputs=cigt_outputs, batch_size=self.batchSize,
                                  action_trajectories=action_trajectories)

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
        lstm_outputs = self.policyNetworksLstm(q_network_outputs)[0]
        lstm_outputs = torch.transpose_copy(lstm_outputs, dim0=0, dim1=1)
        q_net_outputs = []
        for layer_id in range(len(self.pathCounts) - 1):
            q_features = lstm_outputs[layer_id]
            q_net_output = self.policyNetworksQNetRegressionLayers[layer_id](q_features)
            q_net_outputs.append(q_net_output)
        return q_net_outputs
    #
    # def calculate_regression_loss(self, batch_size, q_net_outputs, optimal_q_tables, action_trajectories):
    #     assert action_trajectories.shape[1] == len(self.actionSpaces)
    #     sample_indices = torch.arange(batch_size, device=self.device)
    #     index_array = [torch.arange(batch_size, device=self.device),
    #                    torch.zeros(size=(batch_size,), dtype=torch.int64, device=self.device)]
    #
    #     if self.policyNetworksTrainOnlyActionHeads:
    #         losses_arr = []
    #         for t in range(len(self.actionSpaces)):
    #             a_t = action_trajectories[:, t]
    #             index_array.append(a_t)
    #             q_pred = q_net_outputs[t][sample_indices, a_t]
    #             q_truth = optimal_q_tables[t + 1][index_array]
    #             mse_t = torch.nn.functional.mse_loss(input=q_pred, target=q_truth)
    #             losses_arr.append(mse_t)
    #         total_loss = sum(losses_arr)
    #         return total_loss
    #     else:
    #         losses_arr = []
    #         for t in range(len(self.actionSpaces)):
    #             q_pred = q_net_outputs[t]
    #             q_truth = optimal_q_tables[t + 1][index_array]
    #             mse_t = torch.nn.functional.mse_loss(input=q_pred, target=q_truth)
    #             a_t = action_trajectories[:, t]
    #             index_array.append(a_t)
    #             losses_arr.append(mse_t)
    #         total_loss = sum(losses_arr)
    #         return total_loss
    #
    # def forward_with_actions(self, cigt_outputs, batch_size, action_trajectories):
    #     # cigt_outputs, batch_size = self.get_cigt_outputs(x=x, y=y)
    #     # Calculate the arrays of executed nodes.
    #     executed_nodes_array = self.get_executed_nodes_wrt_trajectories(
    #         cigt_outputs=cigt_outputs,
    #         batch_size=batch_size,
    #         action_trajectories=action_trajectories)
    #     # Prepare the (possibly) sparse inputs for the q networks, for every layer.
    #     sparse_inputs_array = self.prepare_q_net_inputs(cigt_outputs=cigt_outputs,
    #                                                     batch_size=batch_size,
    #                                                     executed_nodes_array=executed_nodes_array)
    #     # Execute the Q-Nets. Obtain the regression outputs for every Q-Net layer.
    #     q_net_outputs = self.execute_q_networks(sparse_inputs_array=sparse_inputs_array)
    #
    #     # Calculate the correctness, MoE probabilities
    #     correctness_vector, expert_probs = self.calculate_moe_for_final_layer(
    #         cigt_outputs=cigt_outputs,
    #         batch_size=batch_size,
    #         executed_nodes_array=executed_nodes_array[-1])
    #
    #     # Calculate the extra MAC incurred due to multiple path executions
    #     mac_vectors = []
    #     for tt, e_nodes in enumerate(executed_nodes_array):
    #         mac_vector = self.calculate_mac_vector_for_layer(layer=tt, executed_nodes_array=e_nodes)
    #         mac_vectors.append(mac_vector)
    #     mac_vectors = torch.stack(mac_vectors, dim=1)
    #     total_mac_vector = torch.sum(mac_vectors, dim=1)
    #
    #     results_dict = {
    #         "q_net_outputs": q_net_outputs,
    #         "correctness_vector": correctness_vector,
    #         "expert_probs": expert_probs,
    #         "mac_vectors": mac_vectors,
    #         "total_mac_vector": total_mac_vector
    #     }
    #     return results_dict
    #
    # def validate_with_expectation(self, loader, temperature=None):
    #     self.eval()
    #     time_profiler = TimeProfiler()
    #     # If temperature is None, then it is treated as greedy.
    #     if temperature is None:
    #         temperature = 1e-10
    #     action_space = [1]
    #     action_space.extend(self.actionSpaces)
    #     all_trajectories = Utilities.create_route_combinations(shape_=action_space)
    #
    #     optimal_q_tables_dataset = []
    #     predicted_q_tables_dataset = []
    #     for _ in range(len(action_space)):
    #         optimal_q_tables_dataset.append([])
    #         predicted_q_tables_dataset.append([])
    #     trajectory_probabilities_dict = {}
    #     correctness_vectors_dict = {}
    #     mac_vectors_dict = {}
    #     greedy_correctness_vectors = []
    #     greedy_mac_vectors = []
    #     time_spent = []
    #     policy_distributions_dict = {}
    #     mse_dict = {}
    #     r2_dict = {}
    #
    #     print("Device:{0}".format(self.device))
    #     total_sample_count = 0
    #     for i__, batch in tqdm(enumerate(loader)):
    #         time_profiler.start_measurement()
    #         with torch.no_grad():
    #             if self.usingPrecalculatedDatasets:
    #                 cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
    #                 cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
    #             else:
    #                 input_var = torch.autograd.Variable(batch[0]).to(self.device)
    #                 target_var = torch.autograd.Variable(batch[1]).to(self.device)
    #                 cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)
    #
    #             # Calculate the optimal q tables. Add to the dataset-wide lists at every layer.
    #             optimal_q_tables = self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs, batch_size=batch_size)
    #             for tt, optimal_q_table in enumerate(optimal_q_tables):
    #                 optimal_q_tables_dataset[tt].append(optimal_q_table)
    #
    #             # Holds predicted q tables per action trajectory.
    #             batch_predicted_q_tables_dict = {}
    #
    #             # Calculate results for every possible trajectory.
    #             for action_trajectory in all_trajectories:
    #                 action_trajectory_torch = torch.Tensor(action_trajectory).to(self.device).to(torch.int64)
    #                 action_trajectory_torch = torch.unsqueeze(action_trajectory_torch, dim=0)
    #                 action_trajectory_torch = torch.tile(action_trajectory_torch, dims=(batch_size, 1))
    #                 result_dict = self.forward_with_actions(cigt_outputs=cigt_outputs, batch_size=batch_size,
    #                                                         action_trajectories=action_trajectory_torch[:, 1:])
    #                 # Record predicted q_tables in a structured way.
    #                 trajectory_probabilities = []
    #                 for idx, predicted_q_table in enumerate(result_dict["q_net_outputs"]):
    #                     q_table_idx = action_trajectory[:(idx + 1)]
    #                     if q_table_idx not in batch_predicted_q_tables_dict:
    #                         batch_predicted_q_tables_dict[q_table_idx] = predicted_q_table
    #                     else:
    #                         q_tables_close = torch.allclose(batch_predicted_q_tables_dict[q_table_idx],
    #                                                         predicted_q_table)
    #                         assert q_tables_close
    #                     # Calculate the selection probabilities for the given trajectory, for every sample,
    #                     # from the predicted q_tables
    #                     predicted_q_table_tempered = predicted_q_table / temperature
    #                     step_probabilities = torch.nn.functional.softmax(predicted_q_table_tempered, dim=1)
    #                     # Get probabilities for this step.
    #                     step_action = action_trajectory[idx + 1]
    #                     step_action_probabilities = step_probabilities[:, step_action]
    #                     trajectory_probabilities.append(step_action_probabilities)
    #                 # Save the joint probability of every action trajectory.
    #                 trajectory_probabilities = torch.stack(trajectory_probabilities, dim=1)
    #                 trajectory_probabilities = torch.prod(trajectory_probabilities, dim=1)
    #                 if action_trajectory not in trajectory_probabilities_dict:
    #                     trajectory_probabilities_dict[action_trajectory] = []
    #                 trajectory_probabilities_dict[action_trajectory].append(trajectory_probabilities)
    #
    #                 # Record predicted correctness vectors and mac vectors
    #                 if action_trajectory not in correctness_vectors_dict:
    #                     correctness_vectors_dict[action_trajectory] = []
    #                 correctness_vectors_dict[action_trajectory].append(result_dict["correctness_vector"])
    #                 if action_trajectory not in mac_vectors_dict:
    #                     mac_vectors_dict[action_trajectory] = []
    #                 mac_vectors_dict[action_trajectory].append(result_dict["total_mac_vector"])
    #
    #             # Add predicted q tables into the proper locations
    #             for tt, optimal_q_table in enumerate(optimal_q_tables):
    #                 # We dont need to predict the first q table, it is trivially always single selection.
    #                 if tt == 0:
    #                     continue
    #                 predicted_q_table_full = torch.zeros_like(optimal_q_table)
    #                 previous_trajectories = Utilities.create_route_combinations(shape_=action_space[:tt])
    #                 for previous_trajectory in previous_trajectories:
    #                     assert previous_trajectory in batch_predicted_q_tables_dict
    #                     index_array = \
    #                         self.create_index_array_for_q_table(batch_size=batch_size,
    #                                                             path_combination=previous_trajectory)
    #                     predicted_q_table_full[index_array] = batch_predicted_q_tables_dict[previous_trajectory]
    #                 predicted_q_tables_dataset[tt].append(predicted_q_table_full)
    #
    #             # Prepare the greedy results:
    #             # For debug purposes: Always use the greedy policy for each step: a_t = argmax_x Q_t(s_t,x).
    #             # Compare it with the expectation method where temperature is very small.
    #             # Since tempereture -> 0 means the policy distributions derived from the predicted q tables will
    #             # approach to one-hot vectors, where one entry is the argmax,
    #             # we must obtain the same results. (ONLY FOR VERY SMALL TEMPERATURES!!!)
    #             index_array = [torch.arange(batch_size, device=self.device),
    #                            torch.zeros(size=(batch_size,), dtype=torch.int64, device=self.device)]
    #             for t in range(len(action_space)):
    #                 if t == 0:
    #                     continue
    #                 q_table_t = predicted_q_tables_dataset[t][-1][index_array]
    #                 greedy_actions = torch.argmax(q_table_t, dim=1)
    #                 index_array.append(greedy_actions)
    #             index_array = torch.stack(index_array, dim=1)
    #             greedy_action_trajectory = index_array[:, 2:]
    #             greedy_results_dict = self.forward_with_actions(cigt_outputs=cigt_outputs,
    #                                                             action_trajectories=greedy_action_trajectory,
    #                                                             batch_size=batch_size)
    #             greedy_correctness_vectors.append(greedy_results_dict["correctness_vector"])
    #             greedy_mac_vectors.append(greedy_results_dict["total_mac_vector"])
    #
    #         time_profiler.end_measurement()
    #         time_spent.append(time_profiler.get_time())
    #         total_sample_count += batch_size
    #
    #     # Concatenate predicted q_tables and optimal q_tables. Measure MSE and R2 scores between each compatible table.
    #     for tt in range(len(action_space)):
    #         if tt == 0:
    #             continue
    #         q_true = torch.concat(optimal_q_tables_dataset[tt], dim=0)
    #         q_pred = torch.concat(predicted_q_tables_dataset[tt], dim=0)
    #         assert q_true.shape == q_pred.shape
    #         previous_trajectories = Utilities.create_route_combinations(shape_=action_space[:tt])
    #         for previous_trajectory in previous_trajectories:
    #             index_array = \
    #                 self.create_index_array_for_q_table(batch_size=q_true.shape[0],
    #                                                     path_combination=previous_trajectory)
    #             q_partial_true = q_true[index_array].cpu().numpy()
    #             q_partial_pred = q_pred[index_array].cpu().numpy()
    #             mse_ = mean_squared_error(y_true=q_partial_true, y_pred=q_partial_pred)
    #             r2_ = r2_score(y_true=q_partial_true, y_pred=q_partial_pred)
    #             # Measure the policy distribution as well.
    #             policy_distribution = torch.nn.functional.softmax(q_pred[index_array], dim=1)
    #             mean_policy_distribution = torch.mean(policy_distribution, dim=0).cpu().numpy()
    #             policy_distributions_dict[previous_trajectory] = mean_policy_distribution
    #             mse_dict[previous_trajectory] = mse_
    #             r2_dict[previous_trajectory] = r2_
    #             print("Trajectory:{0} MSE:{1} R2:{2} Mean Policy Distribution:{3}".format(
    #                 previous_trajectory, mse_, r2_, mean_policy_distribution))
    #
    #         optimal_q_tables_dataset[tt] = q_true
    #         predicted_q_tables_dataset[tt] = q_pred
    #
    #     # Concatenate correctness vector and mac vectors from every trajectory. Measure the expected accuracy and mac.
    #     all_trajectories = Utilities.create_route_combinations(shape_=action_space)
    #     action_probabilities_matrix = []
    #     correctness_vectors_matrix = []
    #     mac_vectors_matrix = []
    #     for actions in all_trajectories:
    #         probs_full = torch.concat(trajectory_probabilities_dict[actions], dim=0)
    #         correctness_full = torch.concat(correctness_vectors_dict[actions], dim=0)
    #         mac_full = torch.concat(mac_vectors_dict[actions], dim=0)
    #         action_probabilities_matrix.append(probs_full)
    #         correctness_vectors_matrix.append(correctness_full)
    #         mac_vectors_matrix.append(mac_full)
    #     action_probabilities_matrix = torch.stack(action_probabilities_matrix, dim=1)
    #     correctness_vectors_matrix = torch.stack(correctness_vectors_matrix, dim=1)
    #     mac_vectors_matrix = torch.stack(mac_vectors_matrix, dim=1)
    #     sum_prob = torch.sum(action_probabilities_matrix, dim=1)
    #     assert torch.allclose(sum_prob, torch.ones_like(sum_prob))
    #     expected_accuracy = torch.mean(torch.sum(action_probabilities_matrix * correctness_vectors_matrix, dim=1))
    #     expected_mac = torch.mean(torch.sum(action_probabilities_matrix * mac_vectors_matrix, dim=1))
    #     expected_accuracy = expected_accuracy.cpu().numpy()
    #     expected_mac = expected_mac.cpu().numpy()
    #     expected_time = np.mean(np.array(time_spent))
    #
    #     # if get_greedy_prediction:
    #     greedy_correctness_vector_full = torch.concat(greedy_correctness_vectors, dim=0)
    #     greedy_mac_vector_full = torch.concat(greedy_mac_vectors, dim=0)
    #     greedy_accuracy = torch.mean(greedy_correctness_vector_full).cpu().numpy()
    #     greedy_mac = torch.mean(greedy_mac_vector_full).cpu().numpy()
    #     # assert np.allclose(greedy_accuracy, expected_accuracy)
    #     # assert np.allclose(greedy_mac, expected_mac)
    #
    #     return {"expected_accuracy": expected_accuracy,
    #             "expected_mac": expected_mac,
    #             "expected_time": expected_time,
    #             "greedy_accuracy": greedy_accuracy,
    #             "greedy_mac": greedy_mac,
    #             "policy_distributions_dict": policy_distributions_dict,
    #             "mse_dict": mse_dict,
    #             "r2_dict": r2_dict}
    #
    # def validate_with_single_action_trajectory(self, loader, action_trajectory):
    #     self.eval()
    #     assert isinstance(action_trajectory, tuple)
    #     action_trajectory_torch = torch.Tensor(action_trajectory).to(self.device).to(torch.int64)
    #     action_trajectory_torch = torch.unsqueeze(action_trajectory_torch, dim=0)
    #
    #     all_q_net_outputs = []
    #     for _ in range(len(self.actionSpaces)):
    #         all_q_net_outputs.append([])
    #     all_correctness_vectors = []
    #     all_expert_probs = []
    #     all_total_mac_vectors = []
    #     time_spent_arr = []
    #     time_profiler = TimeProfiler()
    #
    #     print("Device:{0}".format(self.device))
    #     for i__, batch in tqdm(enumerate(loader)):
    #         time_profiler.start_measurement()
    #         if self.usingPrecalculatedDatasets:
    #             cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
    #             cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
    #         else:
    #             input_var = torch.autograd.Variable(batch[0]).to(self.device)
    #             target_var = torch.autograd.Variable(batch[1]).to(self.device)
    #             cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)
    #
    #         action_trajectories = torch.tile(action_trajectory_torch, dims=(batch_size, 1))
    #         result_dict = self.forward_with_actions(cigt_outputs=cigt_outputs, batch_size=batch_size,
    #                                                 action_trajectories=action_trajectories)
    #         # for idx, q_arr in enumerate(result_dict["q_net_outputs"]):
    #         #     all_q_net_outputs[idx].append(q_arr)
    #         all_correctness_vectors.append(result_dict["correctness_vector"])
    #         # all_expert_probs.append(result_dict["expert_probs"])
    #         all_total_mac_vectors.append(result_dict["total_mac_vector"])
    #         time_profiler.end_measurement()
    #         time_spent_arr.append(time_profiler.get_time())
    #
    #     # for idx in range(len(all_q_net_outputs)):
    #     #     all_q_net_outputs[idx] = torch.concat(all_q_net_outputs[idx], dim=0)
    #     all_correctness_vectors = torch.concat(all_correctness_vectors, dim=0)
    #     # all_expert_probs = torch.concat(all_expert_probs, dim=0)
    #     all_total_mac_vectors = torch.concat(all_total_mac_vectors, dim=0)
    #
    #     accuracy = torch.mean(all_correctness_vectors).detach().cpu().numpy()
    #     mac_avg = torch.mean(all_total_mac_vectors).detach().cpu().numpy()
    #     time_avg = np.mean(np.array(time_spent_arr))
    #     return accuracy, mac_avg, time_avg
    #
    # def evaluate_datasets(self, train_loader, test_loader, epoch):
    #     print("************** Epoch:{0} **************".format(epoch))
    #     kv_rows = []
    #     results_summary = {"Train": {}, "Test": {}}
    #     for data_type, data_loader in [("Test", test_loader), ("Train", train_loader)]:
    #         results_dict = self.validate_with_expectation(loader=data_loader, temperature=None)
    #         print("Expected {0} Accuracy:{1}".format(data_type, results_dict["expected_accuracy"]))
    #         print("Expected {0} Mac:{1}".format(data_type, results_dict["expected_mac"]))
    #         print("Expected {0} Mean Time:{1}".format(data_type, results_dict["expected_time"]))
    #         print("Greedy {0} Accuracy:{1}".format(data_type, results_dict["greedy_accuracy"]))
    #         print("Greedy {0} Mac:{1}".format(data_type, results_dict["greedy_mac"]))
    #         policy_distributions_dict = results_dict["policy_distributions_dict"]
    #         results_summary[data_type]["Accuracy"] = results_dict["expected_accuracy"]
    #         results_summary[data_type]["Mac"] = results_dict["expected_mac"]
    #         mse_dict = results_dict["mse_dict"]
    #         r2_dict = results_dict["r2_dict"]
    #         trajectories = set(policy_distributions_dict.keys())
    #         assert trajectories == set(mse_dict.keys()) and trajectories == set(r2_dict.keys())
    #         for trajectory in trajectories:
    #             policy_distribution = policy_distributions_dict[trajectory]
    #             mse_ = mse_dict[trajectory]
    #             r2_ = r2_dict[trajectory]
    #             print("{0} Policy Distribution {1}:{2}".format(data_type, trajectory, policy_distribution))
    #             print("{0} Q-Table MSE {1}:{2}".format(data_type, trajectory, mse_))
    #             print("{0} Q-Table R2 {1}:{2}".format(data_type, trajectory, r2_))
    #
    #             kv_rows.append((self.runId,
    #                             epoch,
    #                             "{0} Policy Distribution {1}".format(data_type, trajectory),
    #                             "{0}".format(policy_distribution)))
    #             kv_rows.append((self.runId,
    #                             epoch,
    #                             "{0} Q-Table MSE {1}".format(data_type, trajectory),
    #                             "{0}".format(mse_)))
    #             kv_rows.append((self.runId,
    #                             epoch,
    #                             "{0} Q-Table R2 {1}".format(data_type, trajectory),
    #                             "{0}".format(r2_)))
    #
    #     DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
    #
    #     DbLogger.write_into_table(
    #         rows=[(self.runId,
    #                self.iteration_id,
    #                epoch,
    #                results_summary["Train"]["Accuracy"].item(),
    #                results_summary["Train"]["Mac"].item(),
    #                results_summary["Test"]["Accuracy"].item(),
    #                results_summary["Test"]["Mac"].item(),
    #                0.0)], table=DbLogger.logsTableQCigt)
    #
    #     print("************** Epoch:{0} **************".format(epoch))
    #     results = {
    #         "train_accuracy": results_summary["Train"]["Accuracy"].item(),
    #         "train_mac": results_summary["Train"]["Mac"].item(),
    #         "test_accuracy": results_summary["Test"]["Accuracy"].item(),
    #         "test_mac": results_summary["Test"]["Mac"].item(),
    #         "greedy_accuracy": results_dict["greedy_accuracy"],
    #         "greedy_mac": results_dict["greedy_mac"]
    #     }
    #     return results
    #
    # def adjust_learning_rate_polynomial(self, iteration, num_of_total_iterations):
    #     lr = self.policyNetworkInitialLr
    #     where = np.clip(iteration / num_of_total_iterations, a_min=0.0, a_max=1.0)
    #     modified_lr = lr * (1 - where) ** self.policyNetworkPolynomialSchedulerPower
    #     self.qNetOptimizer.param_groups[0]['lr'] = modified_lr
    #
    # def fit_policy_network(self, train_loader, test_loader):
    #     self.to(self.device)
    #     print("Device:{0}".format(self.device))
    #     torch.manual_seed(1)
    #     best_performance = 0.0
    #     num_of_total_iterations = self.policyNetworkTotalNumOfEpochs * len(train_loader)
    #
    #     # Run a forward pass first to initialize each LazyXXX layer.
    #     self.execute_forward_with_random_input()
    #
    #     test_ig_accuracy, test_ig_mac, test_ig_time = self.validate_with_single_action_trajectory(
    #         loader=test_loader, action_trajectory=(0, 0))
    #     print("Test Ig Accuracy:{0} Test Ig Mac:{1} Test Ig Mean Validation Time:{2}".format(
    #         test_ig_accuracy, test_ig_mac, test_ig_time))
    #
    #     train_ig_accuracy, train_ig_mac, train_ig_time = self.validate_with_single_action_trajectory(
    #         loader=train_loader, action_trajectory=(0, 0))
    #     print("Train Ig Accuracy:{0} Train Ig Mac:{1} Train Ig Mean Validation Time:{2}".format(
    #         train_ig_accuracy, train_ig_mac, train_ig_time))
    #
    #     self.evaluate_datasets(train_loader=train_loader, test_loader=test_loader, epoch=-1)
    #
    #     # Create the model optimizer, we should have every parameter initialized right now.
    #     self.qNetOptimizer = self.create_optimizer()
    #
    #     self.isInWarmUp = False
    #     self.routingRandomizationRatio = -1.0
    #
    #     loss_buffer = []
    #     best_accuracy = 0.0
    #     best_mac = 0.0
    #     epochs_without_improvement = 0
    #
    #     print("Device:{0}".format(self.device))
    #     for epoch_id in range(0, self.policyNetworkTotalNumOfEpochs):
    #         for i__, batch in enumerate(train_loader):
    #             self.train()
    #             print("*************CIGT Q-Net Training Epoch:{0} Iteration:{1}*************".format(
    #                 epoch_id, self.iteration_id))
    #             if self.usingPrecalculatedDatasets:
    #                 cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
    #                 cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
    #             else:
    #                 input_var = torch.autograd.Variable(batch[0]).to(self.device)
    #                 target_var = torch.autograd.Variable(batch[1]).to(self.device)
    #                 cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)
    #
    #             # Adjust the learning rate
    #             self.adjust_learning_rate_polynomial(iteration=self.iteration_id,
    #                                                  num_of_total_iterations=num_of_total_iterations)
    #             # Print learning rates
    #             self.qNetOptimizer.zero_grad()
    #             with torch.set_grad_enabled(True):
    #                 optimal_q_tables = self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs, batch_size=batch_size)
    #                 action_trajectories = self.sample_action_trajectories(q_tables=optimal_q_tables,
    #                                                                       batch_size=batch_size)
    #                 result_dict = self.forward_with_actions(cigt_outputs=cigt_outputs, batch_size=batch_size,
    #                                                         action_trajectories=action_trajectories[:, 1:])
    #                 regression_loss = self.calculate_regression_loss(batch_size=batch_size,
    #                                                                  optimal_q_tables=optimal_q_tables,
    #                                                                  q_net_outputs=result_dict["q_net_outputs"],
    #                                                                  action_trajectories=action_trajectories[:, 1:])
    #                 regression_loss.backward()
    #                 self.qNetOptimizer.step()
    #                 self.epsilonValue = self.epsilonValue * self.policyNetworksEpsilonDecayCoeff
    #                 loss_buffer.append(regression_loss.detach().cpu().numpy())
    #                 if len(loss_buffer) >= 10:
    #                     print("Policy Network Lr:{0}".format(self.qNetOptimizer.param_groups[0]["lr"]))
    #                     print("Epoch:{0} Iteration:{1} MSE:{2}".format(
    #                         epoch_id,
    #                         self.iteration_id,
    #                         np.mean(np.array(loss_buffer))))
    #                     loss_buffer = []
    #             self.iteration_id += 1
    #         # Validation
    #         if epoch_id % self.policyNetworksEvaluationPeriod == 0 or \
    #                 epoch_id >= (self.policyNetworkTotalNumOfEpochs - self.policyNetworksLastEvalStart):
    #             results_dict = \
    #                 self.evaluate_datasets(train_loader=train_loader, test_loader=test_loader, epoch=epoch_id)
    #             if results_dict["test_accuracy"] > best_accuracy:
    #                 best_accuracy = results_dict["test_accuracy"]
    #                 best_mac = results_dict["test_mac"]
    #                 epochs_without_improvement = 0
    #                 print("BEST ACCURACY SO FAR: {0} MAC: {1}".format(best_accuracy, best_mac))
    #             else:
    #                 epochs_without_improvement += 1
    #             if epochs_without_improvement >= self.policyNetworksNoImprovementStopCount:
    #                 print("NO IMPROVEMENTS FOR {0} EPOCHS, STOPPING.".format(epochs_without_improvement))
    #                 break
    #
    #
