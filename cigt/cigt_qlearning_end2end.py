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
        self.policyNetworkInitialLr = configs.policy_networks_initial_lr
        self.policyNetworkBackboneLrCoefficient = configs.policy_networks_backbone_lr_coefficient

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
        cigt_outputs, batch_size = self.get_cigt_outputs(x=fake_input, y=fake_target, temperature=1.0)
        assert self.batchSize == batch_size
        # cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
        action_trajectories = torch.Tensor((0, 0)).to(self.device).to(torch.int64)
        action_trajectories = torch.unsqueeze(action_trajectories, dim=0)
        action_trajectories = torch.tile(action_trajectories, dims=(self.batchSize, 1))
        self.forward_with_actions(cigt_outputs=cigt_outputs, batch_size=self.batchSize,
                                  action_trajectories=action_trajectories)

    # Modify validate_with_expectation
    # Modify validate_with_single_action_trajectory
    # Modify evaluate_datasets

    def evaluate_greedy(self, data_loader):
        self.eval()
        action_space = [1]
        action_space.extend(self.actionSpaces)
        all_trajectories = Utilities.create_route_combinations(shape_=action_space)
        all_q_net_outputs = []
        for _ in range(len(self.actionSpaces)):
            all_q_net_outputs.append([])
        all_correctness_vectors = []
        all_expert_probs = []
        all_total_mac_vectors = []
        optimal_q_tables_dataset = []
        predicted_q_tables_dataset = []
        for _ in range(len(action_space)):
            optimal_q_tables_dataset.append([])
            predicted_q_tables_dataset.append([])
        greedy_actions = []
        time_spent_arr = []
        time_profiler = TimeProfiler()

        print("Device:{0}".format(self.device))
        for i__, batch in tqdm(enumerate(data_loader)):
            time_profiler.start_measurement()
            input_var = torch.autograd.Variable(batch[0]).to(self.device)
            target_var = torch.autograd.Variable(batch[1]).to(self.device)
            cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var, temperature=1.0)
            actions_array = torch.zeros(size=(batch_size, len(self.pathCounts) - 1), dtype=torch.int64).to(self.device)
            optimal_q_tables = self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs, batch_size=batch_size)
            for tt, optimal_q_table in enumerate(optimal_q_tables):
                optimal_q_tables_dataset[tt].append(optimal_q_table.detach().cpu().numpy())
            # Call forward len(self.pathCounts) - 1 times. At each call, update the q-table.
            for t in range(len(self.pathCounts) - 1):
                results_dict = self.forward_with_actions(cigt_outputs=cigt_outputs,
                                                         action_trajectories=actions_array, batch_size=batch_size)
                q_nets = results_dict["q_net_outputs"]
                actions_selected = torch.argmax(q_nets[t], dim=1)
                actions_array[:, t] = actions_selected
            # Here we have the final results.
            results_dict = self.forward_with_actions(cigt_outputs=cigt_outputs,
                                                     action_trajectories=actions_array, batch_size=batch_size)
            greedy_actions.append(actions_array.detach().cpu().numpy())
            for idx, q_net in enumerate(results_dict["q_net_outputs"]):
                predicted_q_tables_dataset[idx + 1].append(q_net.detach().cpu().numpy())

            all_correctness_vectors.append(results_dict["correctness_vector"].detach().cpu().numpy())
            all_total_mac_vectors.append(results_dict["total_mac_vector"].detach().cpu().numpy())
            time_profiler.end_measurement()
            time_spent_arr.append(time_profiler.get_time())

        # for idx in range(len(all_q_net_outputs)):
        #     all_q_net_outputs[idx] = torch.concat(all_q_net_outputs[idx], dim=0)
        all_correctness_vectors = np.concatenate(all_correctness_vectors, axis=0)
        # all_expert_probs = torch.concat(all_expert_probs, dim=0)
        all_total_mac_vectors = np.concatenate(all_total_mac_vectors, axis=0)
        for idx in range(len(predicted_q_tables_dataset)):
            if idx == 0:
                continue
            predicted_q_tables_dataset[idx] = np.concatenate(predicted_q_tables_dataset[idx], axis=0)
        greedy_actions = np.concatenate(greedy_actions, axis=0)

        accuracy = np.mean(all_correctness_vectors)
        mac_avg = np.mean(all_total_mac_vectors)
        time_avg = np.mean(np.array(time_spent_arr))
        return {"accuracy": accuracy,
                "mac_avg": mac_avg,
                "time_avg": time_avg,
                "predicted_q_tables_dataset": predicted_q_tables_dataset,
                "greedy_actions": greedy_actions}

    # TODO: Test and complete this
    def evaluate_datasets(self, train_loader, test_loader, epoch):
        print("************** Epoch:{0} **************".format(epoch))
        kv_rows = []
        results_summary = {"Train": {}, "Test": {}}
        for data_type, data_loader in [("Test", test_loader), ("Train", train_loader)]:
            results_dict_greedy = self.evaluate_greedy(data_loader=data_loader)
            results_dict = self.validate_with_expectation(loader=data_loader, temperature=None)
            greedy_actions = results_dict_greedy["greedy_actions"]
            predicted_q_tables_greedy = results_dict_greedy["predicted_q_tables_dataset"]
            predicted_q_tables_expectation = results_dict["predicted_q_tables_dataset"]
            action_indices = [np.arange(greedy_actions.shape[0]),
                              np.zeros_like(greedy_actions[:, 0])]
            for idx in range(greedy_actions.shape[1]):
                a_greedy = predicted_q_tables_greedy[idx + 1]
                a_expected = predicted_q_tables_expectation[idx + 1][action_indices].detach().cpu().numpy()
                assert np.allclose(a_greedy, a_expected)
                action_indices.append(greedy_actions[:, idx])
            print("Test with {0} is complete! No errors found.".format(data_type))

            # assert results_dict["expected_accuracy"] == results_dict_greedy["accuracy"]
            # assert results_dict["mac_avg"] == results_dict_greedy["mac_avg"]
            # print("X")

        #     print("Expected {0} Accuracy:{1}".format(data_type, results_dict["expected_accuracy"]))
        #     print("Expected {0} Mac:{1}".format(data_type, results_dict["expected_mac"]))
        #     print("Expected {0} Mean Time:{1}".format(data_type, results_dict["expected_time"]))
        #     print("Greedy {0} Accuracy:{1}".format(data_type, results_dict["greedy_accuracy"]))
        #     print("Greedy {0} Mac:{1}".format(data_type, results_dict["greedy_mac"]))
        #     policy_distributions_dict = results_dict["policy_distributions_dict"]
        #     results_summary[data_type]["Accuracy"] = results_dict["expected_accuracy"]
        #     results_summary[data_type]["Mac"] = results_dict["expected_mac"]
        #     mse_dict = results_dict["mse_dict"]
        #     r2_dict = results_dict["r2_dict"]
        #     trajectories = set(policy_distributions_dict.keys())
        #     assert trajectories == set(mse_dict.keys()) and trajectories == set(r2_dict.keys())
        #     for trajectory in trajectories:
        #         policy_distribution = policy_distributions_dict[trajectory]
        #         mse_ = mse_dict[trajectory]
        #         r2_ = r2_dict[trajectory]
        #         print("{0} Policy Distribution {1}:{2}".format(data_type, trajectory, policy_distribution))
        #         print("{0} Q-Table MSE {1}:{2}".format(data_type, trajectory, mse_))
        #         print("{0} Q-Table R2 {1}:{2}".format(data_type, trajectory, r2_))
        #
        #         kv_rows.append((self.runId,
        #                         epoch,
        #                         "{0} Policy Distribution {1}".format(data_type, trajectory),
        #                         "{0}".format(policy_distribution)))
        #         kv_rows.append((self.runId,
        #                         epoch,
        #                         "{0} Q-Table MSE {1}".format(data_type, trajectory),
        #                         "{0}".format(mse_)))
        #         kv_rows.append((self.runId,
        #                         epoch,
        #                         "{0} Q-Table R2 {1}".format(data_type, trajectory),
        #                         "{0}".format(r2_)))
        #
        # DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
        #
        # DbLogger.write_into_table(
        #     rows=[(self.runId,
        #            self.iteration_id,
        #            epoch,
        #            results_summary["Train"]["Accuracy"].item(),
        #            results_summary["Train"]["Mac"].item(),
        #            results_summary["Test"]["Accuracy"].item(),
        #            results_summary["Test"]["Mac"].item(),
        #            0.0)], table=DbLogger.logsTableQCigt)
        #
        # print("************** Epoch:{0} **************".format(epoch))
        # results = {
        #     "train_accuracy": results_summary["Train"]["Accuracy"].item(),
        #     "train_mac": results_summary["Train"]["Mac"].item(),
        #     "test_accuracy": results_summary["Test"]["Accuracy"].item(),
        #     "test_mac": results_summary["Test"]["Mac"].item(),
        #     "greedy_accuracy": results_dict["greedy_accuracy"],
        #     "greedy_mac": results_dict["greedy_mac"]
        # }
        # return results

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
        policy_network_parameter_names = []
        shared_parameter_names = []

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
                policy_network_parameter_names.append(name)
            elif "valueNetworks" in name:
                assert "cigtLayers" not in name and "policyNetworks" not in name
                value_networks_parameters.append(param)
            else:
                shared_parameters.append(param)
                shared_parameter_names.append(name)

        num_shared_parameters = len(shared_parameters)
        num_policy_network_parameters = len(policy_networks_parameters)
        num_value_networks_parameters = len(value_networks_parameters)
        num_cigt_layer_parameters = sum([len(arr) for arr in parameters_per_cigt_layers])
        num_all_parameters = len([tpl for tpl in self.named_parameters()])
        assert num_shared_parameters + num_policy_network_parameters + \
               num_value_networks_parameters + num_cigt_layer_parameters == num_all_parameters

        # Include all backbone parameters, except the routing networks, into the same set.
        back_bone_parameters = []
        for idx in range(len(parameters_per_cigt_layers)):
            back_bone_parameters.extend(parameters_per_cigt_layers[idx])
        for name, param in self.named_parameters():
            if name in set(shared_parameter_names) and "blockEndLayers" not in name:
                back_bone_parameters.append(param)
        print("X")

        # # Create a separate optimizer that only optimizes the policy networks.
        # policy_networks_optimizer = optim.AdamW(
        #     [{'params': policy_networks_parameters,
        #       'lr': self.policyNetworkInitialLr,
        #       'weight_decay': self.policyNetworksWd}])
        #
        # parameter_groups = []
        # # Add parameter groups with respect to their cigt layers
        # for layer_id in range(len(self.pathCounts)):
        #     parameter_groups.append(
        #         {'params': parameters_per_cigt_layers[layer_id],
        #          # 'lr': self.initialLr * self.layerCoefficients[layer_id],
        #          'lr': self.initialLr,
        #          'weight_decay': self.classificationWd})
        #
        # # Shared parameters, always the group
        # parameter_groups.append(
        #     {'params': shared_parameters,
        #      'lr': self.initialLr,
        #      'weight_decay': self.classificationWd})
        #
        # if self.optimizerType == "SGD":
        #     model_optimizer = optim.SGD(parameter_groups, momentum=0.9)

        return policy_networks_optimizer

    def adjust_learning_rate_stepwise(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 after 150 and 250 epochs"""
        # Calculate base learning rate
        lower_bounds = [0]
        lower_bounds.extend([tpl[0] for tpl in self.learningRateSchedule])
        upper_bounds = [tpl[0] for tpl in self.learningRateSchedule]
        upper_bounds.append(np.inf)
        bounds = np.stack([lower_bounds, upper_bounds], axis=1)
        lr_coeffs = [1.0]
        lr_coeffs.extend([tpl[1] for tpl in self.learningRateSchedule])
        lower_comparison = bounds[:, 0] <= epoch
        upper_comparison = epoch < bounds[:, 1]
        bounds_binary = np.stack([lower_comparison, upper_comparison], axis=1)
        res = np.all(bounds_binary, axis=1)
        idx = np.argmax(res)
        lr_coeff = lr_coeffs[idx]

        # Update learning rates for the backbone and the policy networks accordingly

        # base_lr = lr * lr_coeff
        #
        # assert len(self.modelOptimizer.param_groups) == len(self.pathCounts) + 1
        #
        # # Cigt layers with boosted lrs.
        # for layer_id in range(len(self.pathCounts)):
        #     if self.boostLearningRatesLayerWise:
        #         self.modelOptimizer.param_groups[layer_id]['lr'] = self.layerCoefficients[layer_id] * base_lr
        #     else:
        #         self.modelOptimizer.param_groups[layer_id]['lr'] = base_lr
        # assert len(self.pathCounts) == len(self.modelOptimizer.param_groups) - 1
        # # Shared parameters
        # self.modelOptimizer.param_groups[-1]['lr'] = base_lr
        #
        # if not self.boostLearningRatesLayerWise:
        #     for p_group in self.modelOptimizer.param_groups:
        #         assert p_group["lr"] == base_lr

    def fit_policy_network(self, train_loader, test_loader):
        self.to(self.device)
        print("Device:{0}".format(self.device))
        torch.manual_seed(1)
        best_performance = 0.0
        num_of_total_iterations = self.policyNetworkTotalNumOfEpochs * len(train_loader)

        # Run a forward pass first to initialize each LazyXXX layer.
        self.execute_forward_with_random_input()

        print("X")

        test_ig_accuracy, test_ig_mac, test_ig_time = self.validate_with_single_action_trajectory(
            loader=test_loader, action_trajectory=(0, 0))
        print("Test Ig Accuracy:{0} Test Ig Mac:{1} Test Ig Mean Validation Time:{2}".format(
            test_ig_accuracy, test_ig_mac, test_ig_time))

        train_ig_accuracy, train_ig_mac, train_ig_time = self.validate_with_single_action_trajectory(
            loader=train_loader, action_trajectory=(0, 0))
        print("Train Ig Accuracy:{0} Train Ig Mac:{1} Train Ig Mean Validation Time:{2}".format(
            train_ig_accuracy, train_ig_mac, train_ig_time))

        self.evaluate_datasets(train_loader=train_loader, test_loader=test_loader, epoch=-1)

        # Create the model optimizer, we should have every parameter initialized right now.
        # self.qNetOptimizer = self.create_optimizer()
        #
        # self.isInWarmUp = False
        # self.routingRandomizationRatio = -1.0
        #
        # loss_buffer = []
        # best_accuracy = 0.0
        # best_mac = 0.0
        # epochs_without_improvement = 0
        #
        # print("Device:{0}".format(self.device))
        # for epoch_id in range(0, self.policyNetworkTotalNumOfEpochs):
        #     for i__, batch in enumerate(train_loader):
        #         self.train()
        #         print("*************CIGT Q-Net Training Epoch:{0} Iteration:{1}*************".format(
        #             epoch_id, self.iteration_id))
        #         if self.usingPrecalculatedDatasets:
        #             cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
        #             cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
        #         else:
        #             input_var = torch.autograd.Variable(batch[0]).to(self.device)
        #             target_var = torch.autograd.Variable(batch[1]).to(self.device)
        #             cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)
        #
        #         # Adjust the learning rate
        #         self.adjust_learning_rate_polynomial(iteration=self.iteration_id,
        #                                              num_of_total_iterations=num_of_total_iterations)
        #         # Print learning rates
        #         self.qNetOptimizer.zero_grad()
        #         with torch.set_grad_enabled(True):
        #             optimal_q_tables = self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs, batch_size=batch_size)
        #             action_trajectories = self.sample_action_trajectories(q_tables=optimal_q_tables,
        #                                                                   batch_size=batch_size)
        #             result_dict = self.forward_with_actions(cigt_outputs=cigt_outputs, batch_size=batch_size,
        #                                                     action_trajectories=action_trajectories[:, 1:])
        #             regression_loss = self.calculate_regression_loss(batch_size=batch_size,
        #                                                              optimal_q_tables=optimal_q_tables,
        #                                                              q_net_outputs=result_dict["q_net_outputs"],
        #                                                              action_trajectories=action_trajectories[:, 1:])
        #             regression_loss.backward()
        #             self.qNetOptimizer.step()
        #             self.epsilonValue = self.epsilonValue * self.policyNetworksEpsilonDecayCoeff
        #             loss_buffer.append(regression_loss.detach().cpu().numpy())
        #             if len(loss_buffer) >= 10:
        #                 print("Policy Network Lr:{0}".format(self.qNetOptimizer.param_groups[0]["lr"]))
        #                 print("Epoch:{0} Iteration:{1} MSE:{2}".format(
        #                     epoch_id,
        #                     self.iteration_id,
        #                     np.mean(np.array(loss_buffer))))
        #                 loss_buffer = []
        #         self.iteration_id += 1
        #     # Validation
        #     if epoch_id % self.policyNetworksEvaluationPeriod == 0 or \
        #             epoch_id >= (self.policyNetworkTotalNumOfEpochs - self.policyNetworksLastEvalStart):
        #         results_dict = \
        #             self.evaluate_datasets(train_loader=train_loader, test_loader=test_loader, epoch=epoch_id)
        #         if results_dict["test_accuracy"] > best_accuracy:
        #             best_accuracy = results_dict["test_accuracy"]
        #             best_mac = results_dict["test_mac"]
        #             epochs_without_improvement = 0
        #             print("BEST ACCURACY SO FAR: {0} MAC: {1}".format(best_accuracy, best_mac))
        #         else:
        #             epochs_without_improvement += 1
        #         if epochs_without_improvement >= self.policyNetworksNoImprovementStopCount:
        #             print("NO IMPROVEMENTS FOR {0} EPOCHS, STOPPING.".format(epochs_without_improvement))
        #             break
        #
