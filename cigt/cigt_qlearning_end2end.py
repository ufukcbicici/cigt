from collections import OrderedDict
from collections import Counter

import random
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
        self.policyNetworkBackboneFreezeBnLayers = configs.policy_network_backbone_freeze_bn_layers
        self.igClassificationLoss = nn.CrossEntropyLoss()

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

    def freeze_bn_layers(self):
        for name, module in self.named_modules():
            if hasattr(module, 'track_running_stats'):
                assert isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d)
                if "policyNetworks" not in name:
                    module.eval()

    def evaluate_dataset(self, data_loader):
        self.eval()
        action_space = [1]
        action_space.extend(self.actionSpaces)
        all_trajectories = Utilities.create_route_combinations(shape_=action_space)
        all_q_table_trajectories = []
        for t_ in range(len(action_space) - 1):
            all_q_table_trajectories.extend(list(set([tpl[:(t_ + 1)] for tpl in all_trajectories])))
        all_q_net_outputs = []
        for _ in range(len(self.actionSpaces)):
            all_q_net_outputs.append([])
        all_correctness_vectors = []
        all_expert_probs = []
        all_total_mac_vectors = []
        optimal_q_tables_dataset = []
        predicted_q_tables_dataset = []
        predicted_q_tables_dict = {}
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

            # ************ Calculate optimal Q-Tables ************
            optimal_q_tables = self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs, batch_size=batch_size)
            for tt, optimal_q_table in enumerate(optimal_q_tables):
                optimal_q_tables_dataset[tt].append(optimal_q_table.detach().cpu().numpy())
            # ************ Calculate optimal Q-Tables ************

            # ************ Calculate greedy action results ************
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
            # ************ Calculate greedy action results ************

            # ************ Calculate all possible predicted Q-Tables ************
            for t_, tpl in enumerate(all_q_table_trajectories):
                actions_array = torch.zeros(size=(batch_size, len(self.pathCounts) - 1), dtype=torch.int64).to(
                    self.device)
                for tt_ in range(1, len(tpl)):
                    actions_array[:, tt_ - 1] = tpl[tt_]
                results_dict_2 = self.forward_with_actions(cigt_outputs=cigt_outputs,
                                                           action_trajectories=actions_array,
                                                           batch_size=batch_size)
                q_nets = results_dict_2["q_net_outputs"]
                if tpl not in predicted_q_tables_dict:
                    predicted_q_tables_dict[tpl] = []
                predicted_q_tables_dict[tpl].append(q_nets[len(tpl) - 1].detach().cpu().numpy())
            # ************ Calculate all possible predicted Q-Tables ************

        # for idx in range(len(all_q_net_outputs)):
        #     all_q_net_outputs[idx] = torch.concat(all_q_net_outputs[idx], dim=0)
        all_correctness_vectors = np.concatenate(all_correctness_vectors, axis=0)
        # all_expert_probs = torch.concat(all_expert_probs, dim=0)
        all_total_mac_vectors = np.concatenate(all_total_mac_vectors, axis=0)
        for idx in range(len(predicted_q_tables_dataset)):
            optimal_q_tables_dataset[idx] = np.concatenate(optimal_q_tables_dataset[idx], axis=0)
            if idx > 0:
                predicted_q_tables_dataset[idx] = np.concatenate(predicted_q_tables_dataset[idx], axis=0)
        greedy_actions = np.concatenate(greedy_actions, axis=0)

        # MSE and R2 score calculations
        r2_dict = {}
        mse_dict = {}
        for k in predicted_q_tables_dict.keys():
            predicted_q_tables_dict[k] = np.concatenate(predicted_q_tables_dict[k], axis=0)
            index_array = \
                self.create_index_array_for_q_table(batch_size=predicted_q_tables_dict[k].shape[0], path_combination=k)
            index_array = tuple([arr.detach().cpu().numpy() for arr in index_array])
            q_pred = predicted_q_tables_dict[k]
            q_truth = optimal_q_tables_dataset[len(k)][index_array]
            mse_ = mean_squared_error(y_true=q_truth, y_pred=q_pred)
            r2_ = r2_score(y_true=q_truth, y_pred=q_pred)
            mse_dict[k] = mse_
            r2_dict[k] = r2_

        accuracy = np.mean(all_correctness_vectors)
        mac_avg = np.mean(all_total_mac_vectors)
        time_avg = np.mean(np.array(time_spent_arr))

        return {"accuracy": accuracy,
                "mac_avg": mac_avg,
                "time_avg": time_avg,
                "predicted_q_tables_dataset": predicted_q_tables_dataset,
                "optimal_q_tables_dataset": optimal_q_tables_dataset,
                "greedy_actions": greedy_actions,
                "predicted_q_tables_dict": predicted_q_tables_dict,
                "r2_dict": r2_dict,
                "mse_dict": mse_dict,
                "all_q_table_trajectories": all_q_table_trajectories}

    # TODO: Test and complete this
    def evaluate_datasets(self, train_loader, test_loader, epoch):
        print("************** Epoch:{0} **************".format(epoch))
        kv_rows = []
        results_summary = {"Train": {}, "Test": {}}
        for data_type, data_loader in [("Test", test_loader), ("Train", train_loader)]:
            results_dict_greedy = self.evaluate_dataset(data_loader=data_loader)
            greedy_actions = results_dict_greedy["greedy_actions"]
            optimal_q_tables_dataset = results_dict_greedy["optimal_q_tables_dataset"]
            predicted_q_tables_greedy = results_dict_greedy["predicted_q_tables_dataset"]
            greedy_accuracy = results_dict_greedy["accuracy"]
            greedy_mac = results_dict_greedy["mac_avg"]
            all_q_table_trajectories = results_dict_greedy["all_q_table_trajectories"]
            greedy_r2_dict = results_dict_greedy["r2_dict"]
            greedy_mse_dict = results_dict_greedy["mse_dict"]
            print("{0} greedy_accuracy:{1}".format(data_type, greedy_accuracy))
            print("{0} greedy_mac:{1}".format(data_type, greedy_mac))
            results_summary[data_type]["Accuracy"] = greedy_accuracy
            results_summary[data_type]["Mac"] = greedy_mac

            for trajectory in all_q_table_trajectories:
                # policy_distribution = policy_distributions_dict[trajectory]
                mse_ = greedy_mse_dict[trajectory]
                r2_ = greedy_r2_dict[trajectory]
                # print("{0} Policy Distribution {1}:{2}".format(data_type, trajectory, policy_distribution))
                print("{0} Q-Table MSE {1}:{2}".format(data_type, trajectory, mse_))
                print("{0} Q-Table R2 {1}:{2}".format(data_type, trajectory, r2_))

                # kv_rows.append((self.runId,
                #                 epoch,
                #                 "{0} Policy Distribution {1}".format(data_type, trajectory),
                #                 "{0}".format(policy_distribution)))
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
        print("Num of back bone parameters before:{0}".format(len(back_bone_parameters)))
        for name, param in self.named_parameters():
            if name in set(shared_parameter_names) and "blockEndLayers" not in name:
                print("Backbone parameter:{0}".format(name))
                back_bone_parameters.append(param)
        print("Num of back bone parameters later:{0}".format(len(back_bone_parameters)))
        print("X")

        # Create separate parameter groups for the backbone and the policy network parameters.
        parameter_groups = [{'params': back_bone_parameters,
                             'lr': self.initialLr * self.policyNetworkBackboneLrCoefficient,
                             'weight_decay': self.classificationWd},
                            {'params': policy_networks_parameters,
                             'lr': self.initialLr,
                             'weight_decay': self.classificationWd}]
        if self.optimizerType == "SGD":
            model_optimizer = optim.SGD(parameter_groups, momentum=0.9)
        elif self.optimizerType == "AdamW":
            model_optimizer = optim.AdamW(parameter_groups)
        else:
            raise NotImplementedError()
        return model_optimizer

    def get_ig_based_losses(self, cigt_outputs, batch_size, executed_nodes):
        merged_arrays_dict = {}
        routing_matrices_soft = []
        logits_array = None
        labels = []
        for lid in range(len(self.pathCounts)):
            if lid < len(self.pathCounts) - 1:
                output_name = "routing_matrices_soft_dict"
            else:
                output_name = "logits_dict"
            trajectories = Utilities.create_route_combinations(shape_=self.pathCounts[:(lid + 1)])
            arrays_merged = torch.stack([cigt_outputs[output_name][tpl] for tpl in trajectories], dim=1)
            arrays_merged = torch.reshape(arrays_merged, shape=(batch_size,
                                                                *self.pathCounts[:(lid + 1)],
                                                                *arrays_merged.shape[2:]))
            arrays_merged_sparse = arrays_merged * torch.unsqueeze(executed_nodes[lid], dim=-1)
            reduce_dimensions = tuple([idx + 1 for idx in range(len(self.pathCounts[:(lid + 1)]))])
            final_array = torch.sum(arrays_merged_sparse, dim=reduce_dimensions)
            merged_arrays_dict[(output_name, lid)] = final_array
            if output_name == "routing_matrices_soft_dict":
                routing_matrices_soft.append(final_array)
                labels.append(cigt_outputs["labels_dict"][()])
            else:
                logits_array = final_array

        # Test code
        # merged_arrays_dict_v2 = {}
        # for sample_id in range(batch_size):
        #     curr_node_id = [0, ]
        #     while True:
        #         assert tuple(curr_node_id) in cigt_outputs["routing_matrices_soft_dict"] \
        #                or tuple(curr_node_id) in cigt_outputs["logits_dict"]
        #         if tuple(curr_node_id) in cigt_outputs["routing_matrices_soft_dict"]:
        #             if ("routing_matrices_soft_dict", len(curr_node_id) - 1) not in merged_arrays_dict_v2:
        #                 merged_arrays_dict_v2[("routing_matrices_soft_dict", len(curr_node_id) - 1)] = []
        #             next_probs = cigt_outputs["routing_matrices_soft_dict"][tuple(curr_node_id)][sample_id]
        #             merged_arrays_dict_v2[("routing_matrices_soft_dict", len(curr_node_id) - 1)].append(next_probs)
        #             next_level_id = torch.argmax(next_probs).detach().cpu().numpy().item()
        #             curr_node_id.append(next_level_id)
        #         else:
        #             logits = cigt_outputs["logits_dict"][tuple(curr_node_id)][sample_id]
        #             if ("logits_dict", len(curr_node_id) - 1) not in merged_arrays_dict_v2:
        #                 merged_arrays_dict_v2[("logits_dict", len(curr_node_id) - 1)] = []
        #             merged_arrays_dict_v2[("logits_dict", len(curr_node_id) - 1)].append(logits)
        #             break
        # for tpl in merged_arrays_dict_v2.keys():
        #     merged_arrays_dict_v2[tpl] = torch.stack(merged_arrays_dict_v2[tpl], dim=0)
        #
        # assert set(merged_arrays_dict.keys()) == set(merged_arrays_dict_v2.keys())
        # for k in merged_arrays_dict.keys():
        #     assert np.array_equal(merged_arrays_dict[k].detach().cpu().numpy(),
        #                           merged_arrays_dict_v2[k].detach().cpu().numpy())
        # print("Correct!!!!!")

        # Calculate information gain
        information_gain_losses = self.calculate_information_gain_losses(
            routing_matrices=routing_matrices_soft, labels=labels,
            balance_coefficient_list=self.informationGainBalanceCoeffList)
        # Calculate accuracy
        reduce_dimensions = tuple([idx + 1 for idx in range(len(self.pathCounts) - 1)])
        last_layer_routing_matrix = torch.sum(executed_nodes[-1], dim=reduce_dimensions)
        logits_masked = self.divide_tensor_wrt_routing_matrix(
            tens=logits_array,
            routing_matrix=last_layer_routing_matrix)
        labels_masked = self.divide_tensor_wrt_routing_matrix(
            tens=cigt_outputs["labels_dict"][()],
            routing_matrix=last_layer_routing_matrix)
        classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
            list_of_logits=logits_masked,
            routing_matrices=None,
            target_var=labels_masked)
        losses_dict = {
            "information_gain_losses": information_gain_losses,
            "classification_loss": classification_loss,
            "batch_accuracy": batch_accuracy
        }
        return losses_dict

    def calculate_losses_standard_way(self, input_var, target_var, temperature):
        layer_outputs = self(input_var, target_var, temperature)
        if self.lossCalculationKind == "SingleLogitSingleLoss":
            classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                list_of_logits=layer_outputs[-1]["list_of_logits"],
                routing_matrices=None,
                target_var=[layer_outputs[-1]["labels"]])
        # Calculate logits with all block separately
        elif self.lossCalculationKind == "MultipleLogitsMultipleLosses" \
                or self.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
            classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                list_of_logits=layer_outputs[-1]["list_of_logits"],
                routing_matrices=None,
                target_var=layer_outputs[-1]["labels_masked"])
        else:
            raise ValueError("Unknown logit calculation method: {0}".format(self.lossCalculationKind))
        # Calculate the information gain losses, with respect to each routing layer
        routing_matrices_soft = [od["routing_matrices_soft"] for od in layer_outputs[1:-1]]
        labels_per_routing_layer = [od["labels"] for od in layer_outputs[1:-1]]
        information_gain_losses = self.calculate_information_gain_losses(
            routing_matrices=routing_matrices_soft, labels=labels_per_routing_layer,
            balance_coefficient_list=self.informationGainBalanceCoeffList)
        losses_dict = {
            "information_gain_losses": information_gain_losses,
            "classification_loss": classification_loss,
            "batch_accuracy": batch_accuracy
        }
        return losses_dict

    def adjust_learning_rate_stepwise(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 after 150 and 250 epochs"""
        lr = self.initialLr
        # if epoch >= 150:
        #     lr = 0.1 * lr
        # if epoch >= 250:
        #     lr = 0.1 * lr
        # learning_schedule = [(150, 0.1), (250, 0.01)]

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
        base_lr = lr * lr_coeff

        # Backbone
        self.modelOptimizer.param_groups[0]['lr'] = base_lr * self.policyNetworkBackboneLrCoefficient
        # Policy network
        self.modelOptimizer.param_groups[1]['lr'] = base_lr

    def fit_policy_network(self, train_loader, test_loader):
        self.to(self.device)
        print("Device:{0}".format(self.device))
        torch.manual_seed(1)
        best_performance = 0.0
        num_of_total_iterations = self.policyNetworkTotalNumOfEpochs * len(train_loader)

        # Run a forward pass first to initialize each LazyXXX layer.
        self.execute_forward_with_random_input()
        # test_ig_accuracy, test_ig_mac, test_ig_time = self.validate_with_single_action_trajectory(
        #     loader=test_loader, action_trajectory=(0, 0))
        # print("Test Ig Accuracy:{0} Test Ig Mac:{1} Test Ig Mean Validation Time:{2}".format(
        #     test_ig_accuracy, test_ig_mac, test_ig_time))
        #
        # train_ig_accuracy, train_ig_mac, train_ig_time = self.validate_with_single_action_trajectory(
        #     loader=train_loader, action_trajectory=(0, 0))
        # print("Train Ig Accuracy:{0} Train Ig Mac:{1} Train Ig Mean Validation Time:{2}".format(
        #     train_ig_accuracy, train_ig_mac, train_ig_time))
        #
        # self.evaluate_datasets(train_loader=train_loader, test_loader=test_loader, epoch=-1)

        # Create the model optimizer, we should have every parameter initialized right now.
        self.modelOptimizer = self.create_optimizer()

        self.isInWarmUp = False
        self.routingRandomizationRatio = -1.0

        loss_buffer = []
        best_accuracy = 0.0
        best_mac = 0.0
        epochs_without_improvement = 0

        # self.eval_ig_based(data_loader=test_loader)

        print("Device:{0}".format(self.device))
        for epoch_id in range(0, self.policyNetworkTotalNumOfEpochs):
            batch_time = AverageMeter()
            losses = AverageMeter()
            losses_c = AverageMeter()
            losses_t = AverageMeter()
            losses_r = AverageMeter()
            losses_t_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
            grad_magnitude = AverageMeter()
            accuracy_avg = AverageMeter()

            print("*************Epoch:{0} Starts*************".format(epoch_id))
            self.train()
            if self.policyNetworkBackboneFreezeBnLayers:
                self.freeze_bn_layers()
            self.adjust_learning_rate_stepwise(epoch_id)
            # Print learning rates
            print("Back bone learning rate:{0}".format(self.modelOptimizer.param_groups[0]['lr']))
            print("Policy learning rate:{0}".format(self.modelOptimizer.param_groups[1]['lr']))

            for i__, batch in enumerate(train_loader):
                print("*************CIGT Q-Net Training Epoch:{0} Iteration:{1}*************".format(
                    epoch_id, self.iteration_id))

                self.modelOptimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    input_var = torch.autograd.Variable(batch[0]).to(self.device)
                    target_var = torch.autograd.Variable(batch[1]).to(self.device)

                    # Open for test purposes
                    # losses_dict_v1 = self.calculate_losses_standard_way(input_var=input_var,
                    #                                                     target_var=target_var,
                    #                                                     temperature=1.0)

                    decision_loss_coeff = self.routingManager.adjust_decision_loss_coeff(model=self)
                    temperature = 0.1
                    cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var, temperature=temperature)
                    zero_actions = torch.zeros(size=(batch_size, len(self.pathCounts) - 1), dtype=torch.int64,
                                               device=self.device)
                    # with torch.set_grad_enabled(False):
                    executed_nodes = self.get_executed_nodes_wrt_trajectories(cigt_outputs=cigt_outputs,
                                                                              action_trajectories=zero_actions,
                                                                              batch_size=batch_size)
                    # Calculate ig based loss function
                    losses_dict = self.get_ig_based_losses(cigt_outputs=cigt_outputs,
                                                           batch_size=batch_size,
                                                           executed_nodes=executed_nodes)
                    information_gain_losses = losses_dict["information_gain_losses"]
                    classification_loss = losses_dict["classification_loss"]
                    batch_accuracy = losses_dict["batch_accuracy"]
                    # Calculate the total backbone loss
                    total_routing_loss = 0.0
                    for t_loss in information_gain_losses:
                        total_routing_loss += t_loss
                    total_routing_loss = -1.0 * decision_loss_coeff * total_routing_loss
                    total_backbone_loss = classification_loss + total_routing_loss

                    # Calculate Q-Learning loss
                    # with torch.set_grad_enabled(False):
                    optimal_q_tables = self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs,
                                                                       batch_size=batch_size)
                    sampled_action_trajectories = self.sample_action_trajectories(q_tables=optimal_q_tables,
                                                                                  batch_size=batch_size)
                    sampled_action_trajectories = sampled_action_trajectories[:, 1:]
                    sampled_executed_nodes_array = self.get_executed_nodes_wrt_trajectories(
                        cigt_outputs=cigt_outputs,
                        batch_size=batch_size,
                        action_trajectories=sampled_action_trajectories)
                    # Prepare the (possibly) sparse inputs for the q networks, for every layer.
                    sparse_inputs_array = self.prepare_q_net_inputs(cigt_outputs=cigt_outputs,
                                                                    batch_size=batch_size,
                                                                    executed_nodes_array=sampled_executed_nodes_array)
                    # Execute the Q-Nets. Obtain the regression outputs for every Q-Net layer.
                    q_net_outputs = self.execute_q_networks(sparse_inputs_array=sparse_inputs_array)

                    regression_loss = self.calculate_regression_loss(
                        batch_size=batch_size,
                        optimal_q_tables=optimal_q_tables,
                        q_net_outputs=q_net_outputs,
                        action_trajectories=sampled_action_trajectories)

                    total_model_loss = total_backbone_loss + regression_loss
                    total_model_loss.backward()
                    self.modelOptimizer.step()

                losses.update(total_backbone_loss.detach().cpu().numpy().item(), 1)
                losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
                losses_r.update(regression_loss.detach().cpu().numpy().item(), 1)
                accuracy_avg.update(batch_accuracy, batch_size)
                losses_t.update(total_routing_loss.detach().cpu().numpy().item(), 1)
                for lid in range(len(self.pathCounts) - 1):
                    losses_t_layer_wise[lid].update(information_gain_losses[lid].detach().cpu().numpy().item(), 1)

                print("batch_accuracy:{0}".format(batch_accuracy))
                print("decision_loss_coeff:{0}".format(decision_loss_coeff))
                print("total_loss:{0}".format(losses.avg))
                print("classification_loss:{0}".format(losses_c.avg))
                print("routing_loss:{0}".format(losses_t.avg))
                print("regression loss:{0}".format(losses_r.avg))
                for lid in range(len(self.pathCounts) - 1):
                    print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))
                print("accuracy_avg:{0}".format(accuracy_avg.avg))
                print("batch_time:{0}".format(batch_time.avg))
                print("grad_magnitude:{0}".format(grad_magnitude.avg))

                self.iteration_id += 1

                #     if self.usingPrecalculatedDatasets:
                #         cigt_outputs, batch_size = self.get_cigt_outputs(x=batch, y=None)
                #         cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
                #     else:
                #         input_var = torch.autograd.Variable(batch[0]).to(self.device)
                #         target_var = torch.autograd.Variable(batch[1]).to(self.device)
                #         cigt_outputs, batch_size = self.get_cigt_outputs(x=input_var, y=target_var)
                #
                #     # Adjust the learning rate
                #     self.adjust_learning_rate_polynomial(iteration=self.iteration_id,
                #                                          num_of_total_iterations=num_of_total_iterations)
                #     # Print learning rates
                #     self.qNetOptimizer.zero_grad()
                #     with torch.set_grad_enabled(True):
                #         optimal_q_tables = self.calculate_optimal_q_tables(cigt_outputs=cigt_outputs, batch_size=batch_size)
                #         action_trajectories = self.sample_action_trajectories(q_tables=optimal_q_tables,
                #                                                               batch_size=batch_size)
                #         result_dict = self.forward_with_actions(cigt_outputs=cigt_outputs, batch_size=batch_size,
                #                                                 action_trajectories=action_trajectories[:, 1:])
                #         regression_loss = self.calculate_regression_loss(batch_size=batch_size,
                #                                                          optimal_q_tables=optimal_q_tables,
                #                                                          q_net_outputs=result_dict["q_net_outputs"],
                #                                                          action_trajectories=action_trajectories[:, 1:])
                #         regression_loss.backward()
                #         self.qNetOptimizer.step()
                #         self.epsilonValue = self.epsilonValue * self.policyNetworksEpsilonDecayCoeff
                #         loss_buffer.append(regression_loss.detach().cpu().numpy())
                #         if len(loss_buffer) >= 10:
                #             print("Policy Network Lr:{0}".format(self.qNetOptimizer.param_groups[0]["lr"]))
                #             print("Epoch:{0} Iteration:{1} MSE:{2}".format(
                #                 epoch_id,
                #                 self.iteration_id,
                #                 np.mean(np.array(loss_buffer))))
                #             loss_buffer = []
                #     self.iteration_id += 1
                # # Validation
                # if epoch_id % self.policyNetworksEvaluationPeriod == 0 or \
                #         epoch_id >= (self.policyNetworkTotalNumOfEpochs - self.policyNetworksLastEvalStart):
                #     results_dict = \
                #         self.evaluate_datasets(train_loader=train_loader, test_loader=test_loader, epoch=epoch_id)
                #     if results_dict["test_accuracy"] > best_accuracy:
                #         best_accuracy = results_dict["test_accuracy"]
                #         best_mac = results_dict["test_mac"]
                #         epochs_without_improvement = 0
                #         print("BEST ACCURACY SO FAR: {0} MAC: {1}".format(best_accuracy, best_mac))
                #     else:
                #         epochs_without_improvement += 1
                #     if epochs_without_improvement >= self.policyNetworksNoImprovementStopCount:
                #         print("NO IMPROVEMENTS FOR {0} EPOCHS, STOPPING.".format(epochs_without_improvement))
                #         break
                #

            print("*************Epoch:{0} Ending Measurements*************".format(epoch_id))
            print("decision_loss_coeff:{0}".format(decision_loss_coeff))
            print("total_loss:{0}".format(losses.avg))
            print("classification_loss:{0}".format(losses_c.avg))
            print("routing_loss:{0}".format(losses_t.avg))
            for lid in range(len(self.pathCounts) - 1):
                print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))
            print("accuracy_avg:{0}".format(accuracy_avg.avg))
            print("batch_time:{0}".format(batch_time.avg))
            print("grad_magnitude:{0}".format(grad_magnitude.avg))
            print("*************Epoch:{0} Ending Measurements*************".format(epoch_id))

            if epoch_id % self.policyNetworksEvaluationPeriod == 0 or \
                    epoch_id >= (self.policyNetworkTotalNumOfEpochs - 10):
                test_ig_accuracy, test_ig_mac, test_ig_time = self.validate_with_single_action_trajectory(
                    loader=test_loader, action_trajectory=(0, 0))
                print("Test Ig Accuracy:{0} Test Ig Mac:{1} Test Ig Mean Validation Time:{2}".format(
                    test_ig_accuracy, test_ig_mac, test_ig_time))

                train_ig_accuracy, train_ig_mac, train_ig_time = self.validate_with_single_action_trajectory(
                    loader=train_loader, action_trajectory=(0, 0))
                print("Train Ig Accuracy:{0} Train Ig Mac:{1} Train Ig Mean Validation Time:{2}".format(
                    train_ig_accuracy, train_ig_mac, train_ig_time))

                self.evaluate_datasets(train_loader=train_loader, test_loader=test_loader, epoch=epoch_id)
