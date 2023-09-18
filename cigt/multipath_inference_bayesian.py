import os.path
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
from auxillary.bayesian_optimizer import BayesianOptimizer
from auxillary.softmax_temperature_optimizer import SoftmaxTemperatureOptimizer
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
from tqdm import tqdm


class NetworkOutput(object):
    def __init__(self):
        self.routingActivationMatrices = []
        self.logits = []
        self.labels = []
        self.optimalTemperatures = []
        self.igAccuracy = None


class MultiplePathBayesianOptimizer(BayesianOptimizer):
    def __init__(self, data_root_path, model,
                 train_dataset, test_dataset, xi, init_points, n_iter, mac_counts_per_block,
                 train_dataset_repeat_count, mac_lambda, max_probabilities, evaluate_network_first):
        super().__init__(init_points, n_iter)
        self.dataRootPath = data_root_path
        self.maxProbabilities = max_probabilities
        self.macLambda = mac_lambda
        self.repeatCount = train_dataset_repeat_count
        self.macCountsPerBlock = mac_counts_per_block
        self.trainDataset = train_dataset
        self.testDataset = test_dataset
        self.maxEntropies = []
        self.optimization_bounds_continuous = {}
        self.model = model
        max_branch_count = np.prod(self.model.pathCounts)
        self.calculate_max_entropies()
        if evaluate_network_first:
            test_acc = self.model.validate(data_kind="test", epoch=0, loader=self.testDataset, temperature=0.1,
                                           verbose=True)
            print("Standard test accuracy:{0}".format(test_acc))

        for lid in range(len(self.model.pathCounts) - 1):
            for pid in range(self.model.pathCounts[lid]):
                self.optimization_bounds_continuous["threshold_{0},{1}".format(lid, pid)] = (0.0,
                                                                                             self.maxProbabilities[lid])

        # self.assert_gather_scatter_model_output_correctness(dataloader=self.testDataset, repeat_count=1)
        self.trainOutputs = self.create_outputs(dataloader=self.trainDataset, repeat_count=self.repeatCount)
        self.testOutputs = self.create_outputs(dataloader=self.testDataset, repeat_count=1,
                                               optimal_temperatures=self.trainOutputs.optimalTemperatures)

    def calculate_max_entropies(self):
        for layer_id, block_count in enumerate(self.model.pathCounts):
            if layer_id == len(self.model.pathCounts) - 1:
                break
            max_entropy = (-np.log(1.0 / self.model.pathCounts[layer_id + 1])).item()
            self.maxEntropies.append(max_entropy)

    def get_start_offset_for_gather_scatter_model(self, route_, batch_id_, curr_batch_size):
        path_list = list(zip(list(range(len(route_))), route_))
        offset = batch_id_ * np.prod(self.model.pathCounts[:len(route_)], dtype=np.int64) * self.model.batchSize
        for tpl in reversed(path_list):
            block_id = tpl[0]
            block_path_index = tpl[1]
            offset_step = block_path_index * np.prod(self.model.pathCounts[:block_id], dtype=np.int64) * curr_batch_size
            offset += offset_step
        return offset

    def fill_output_array(self, outputs_dict, output_array, block_id, results_array_shape, arr_type):
        data_size = outputs_dict["list_of_labels"][0].shape[0]
        interpreted_results_array = np.zeros(shape=results_array_shape, dtype=arr_type)
        interpreted_results_array[:] = np.nan
        route_combinations = Utilities.create_route_combinations(shape_=self.model.pathCounts[:(block_id + 1)])
        batch_sizes = [len(outputs_dict["list_of_labels"][0][idx:idx + self.model.batchSize])
                       for idx in range(0, data_size, self.model.batchSize)]
        for route_combination in route_combinations:
            for i_, curr_batch_size in tqdm(enumerate(batch_sizes)):
                route_offset = self.get_start_offset_for_gather_scatter_model(route_=route_combination,
                                                                              batch_id_=i_,
                                                                              curr_batch_size=curr_batch_size)
                if len(output_array.shape) > 1:
                    route_activations_array = output_array[route_offset:route_offset + curr_batch_size, :]
                else:
                    route_activations_array = output_array[route_offset:route_offset + curr_batch_size]
                interpreted_results_array[route_combination
                ][i_ * self.model.batchSize:i_ * self.model.batchSize + curr_batch_size] = route_activations_array
        assert np.sum(np.isnan(interpreted_results_array)) == 0
        return interpreted_results_array

    def interpret_gather_scatter_model_outputs(self, outputs_dict, ig_accuracy):
        network_output = NetworkOutput()
        data_size = outputs_dict["list_of_labels"][0].shape[0]
        assert outputs_dict["list_of_original_labels"].shape[0] == data_size
        for block_id in range(len(self.model.pathCounts)):
            # Routing activations
            if block_id < len(self.model.pathCounts) - 1:
                output_array = outputs_dict["list_of_routing_activations"][block_id]
                results_array_shape = (*self.model.pathCounts[:(block_id + 1)],
                                       data_size, self.model.pathCounts[block_id + 1])
                interpreted_results_array = self.fill_output_array(
                    outputs_dict=outputs_dict,
                    arr_type=np.float32,
                    block_id=block_id,
                    output_array=output_array,
                    results_array_shape=results_array_shape)
                network_output.routingActivationMatrices.append(interpreted_results_array)
            else:
                # Logits
                output_array = outputs_dict["list_of_logits_unified"]
                results_array_shape = (*self.model.pathCounts[:(block_id + 1)], data_size, self.model.numClasses)
                interpreted_results_array = self.fill_output_array(
                    outputs_dict=outputs_dict,
                    arr_type=np.float32,
                    block_id=block_id,
                    output_array=output_array,
                    results_array_shape=results_array_shape)
                network_output.logits.append(interpreted_results_array)
                # Labels
                output_array = outputs_dict["list_of_final_block_labels"]
                results_array_shape = (*self.model.pathCounts[:(block_id + 1)], data_size,)
                interpreted_results_array = self.fill_output_array(
                    outputs_dict=outputs_dict,
                    arr_type=np.float32,
                    block_id=block_id,
                    output_array=output_array,
                    results_array_shape=results_array_shape)
                interpreted_results_array = interpreted_results_array.astype(np.int64)
                network_output.labels.append(interpreted_results_array)

        network_output.igAccuracy = ig_accuracy
        return network_output

    def assert_gather_scatter_model_output_correctness(self, network_output, results_dict2):
        array_comparison_information = [
            {"arr1": network_output.routingActivationMatrices[block_id],
             "arr2": results_dict2["routing_activations_complete"],
             "route_combinations": Utilities.create_route_combinations(shape_=self.model.pathCounts[:(block_id + 1)]),
             "output_type": "routing_activations_complete"}
            for block_id in range(len(self.model.pathCounts) - 1)
        ]
        array_comparison_information.append(
            {"arr1": network_output.logits[0],
             "arr2": results_dict2["logits_complete"],
             "route_combinations": Utilities.create_route_combinations(
                 shape_=self.model.pathCounts[:(len(self.model.pathCounts))]),
             "output_type": "logits_complete"}
        )
        array_comparison_information.append(
            {"arr1": network_output.labels[0],
             "arr2": results_dict2["labels_complete"],
             "route_combinations": Utilities.create_route_combinations(
                 shape_=self.model.pathCounts[:(len(self.model.pathCounts))]),
             "output_type": "labels_complete"}
        )

        for d_ in array_comparison_information:
            for route in d_["route_combinations"]:
                print("Difference of results in {0}[{1}]".format(d_["output_type"], route))
                sub_arr1 = d_["arr1"][route]
                sub_arr2 = d_["arr2"][route]
                assert sub_arr1.dtype == sub_arr2.dtype
                num_distant_entries = np.sum(np.isclose(sub_arr1, sub_arr1, rtol=1e-3) == False)
                ratio_of_distant_entries = num_distant_entries / np.prod(sub_arr1.shape)
                if ratio_of_distant_entries > 1e-3:
                    print("WARNING: TWO RESULTS DIFFER SIGNIFICANTLY. ratio_of_distant_entries:{0}".format(
                        ratio_of_distant_entries))
                else:
                    print("Two results are close. ratio_of_distant_entries:{0}".format(
                        ratio_of_distant_entries))

    def merge_multiple_outputs(self, network_outputs):
        complete_output = NetworkOutput()
        for block_id in range(len(self.model.pathCounts) - 1):
            batch_sizes = [n_o.routingActivationMatrices[block_id].shape[block_id + 1] for n_o in network_outputs]
            assert len(set(batch_sizes)) == 1
            total_arr = np.concatenate([n_o.routingActivationMatrices[block_id] for n_o in network_outputs],
                                       axis=block_id + 1)
            complete_output.routingActivationMatrices.append(total_arr)
        batch_sizes = [n_o.logits[0].shape[len(self.model.pathCounts)] for n_o in network_outputs]
        assert len(set(batch_sizes)) == 1
        total_logits = np.concatenate([n_o.logits[0] for n_o in network_outputs], axis=len(self.model.pathCounts))
        complete_output.logits.append(total_logits)
        total_labels = np.concatenate([n_o.labels[0] for n_o in network_outputs], axis=len(self.model.pathCounts))
        complete_output.labels.append(total_labels)

        total_accuracy = 0.0
        for n_idx in range(len(network_outputs)):
            batch_size = network_outputs[n_idx].logits[0].shape[len(self.model.pathCounts)]
            weight = batch_size / sum(batch_sizes)
            total_accuracy += weight * network_outputs[n_idx].igAccuracy

        complete_output.igAccuracy = total_accuracy
        return complete_output

    def create_outputs(self, dataloader, repeat_count, optimal_temperatures=None):
        network_outputs = []
        data_kind = "test" if not dataloader.dataset.train else "train"
        for epoch_id in range(repeat_count):
            print("Processing Data:{0} Epoch:{1}".format(data_kind, epoch_id))
            raw_outputs_file_path = "{0}_{1}_raw_outputs_dict.sav".format(data_kind, epoch_id)
            if not os.path.isfile(raw_outputs_file_path):
                # Get outputs from multiple path execution (Type 1)
                self.model.toggle_all_paths_routing(enable=True)
                raw_outputs_type1_dict = self.model.validate(loader=dataloader, epoch=0, temperature=0.1,
                                                             return_network_outputs=True, data_kind=data_kind,
                                                             verbose=True)
                x_tensor = raw_outputs_type1_dict["list_of_original_inputs"]
                y_tensor = raw_outputs_type1_dict["list_of_original_labels"]
                assert np.array_equal(y_tensor, raw_outputs_type1_dict["list_of_labels"][0])
                validation_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.from_numpy(x_tensor), torch.from_numpy(y_tensor)),
                    shuffle=False, batch_size=self.model.batchSize)

                # Get outputs from IG execution
                self.model.toggle_all_paths_routing(enable=False)
                ig_outputs_dict = self.model.validate(loader=dataloader, epoch=0, temperature=0.1,
                                                      return_network_outputs=True, data_kind=data_kind,
                                                      verbose=True)
                ig_accuracy = ig_outputs_dict["accuracy"]

                # Get outputs from graph based execution (Type 2)
                self.model.toggle_all_paths_routing(enable=True)
                raw_outputs_type2_dict = self.model.validate_v2(validation_loader,
                                                                temperature=0.1,
                                                                verbose=True)

                # Save the outputs
                Utilities.pickle_save_to_file(path=raw_outputs_file_path,
                                              file_content={"raw_outputs_type1_dict": raw_outputs_type1_dict,
                                                            "raw_outputs_type2_dict": raw_outputs_type2_dict,
                                                            "ig_accuracy": ig_accuracy})
            else:
                outputs_loaded = Utilities.pickle_load_from_file(raw_outputs_file_path)
                raw_outputs_type1_dict = outputs_loaded["raw_outputs_type1_dict"]
                raw_outputs_type2_dict = outputs_loaded["raw_outputs_type2_dict"]
                ig_accuracy = outputs_loaded["ig_accuracy"]

            interpreted_network_outputs = self.interpret_gather_scatter_model_outputs(
                outputs_dict=raw_outputs_type1_dict, ig_accuracy=ig_accuracy)
            self.assert_gather_scatter_model_output_correctness(
                network_output=interpreted_network_outputs,
                results_dict2=raw_outputs_type2_dict)
            network_outputs.append(interpreted_network_outputs)

        # Merge all outputs into a single object
        if repeat_count > 1:
            complete_output = self.merge_multiple_outputs(network_outputs=network_outputs)
        else:
            assert len(network_outputs) == 1
            complete_output = network_outputs[0]

        # Calculate optimal temperatures for routing probabilities
        if optimal_temperatures is None:
            self.optimize_routing_temperatures(complete_output=complete_output)
        else:
            complete_output.optimalTemperatures = optimal_temperatures

        # accuracy_graph, mac_cost_graph = \
        #     self.evaluate_thresholds_graph_based(
        #         thresholds=[[0.2, 0.25], [0.1, 0.12, 0.15, .2]],
        #         outputs=complete_output)
        # accuracy_vector, mac_cost_vector = self.evaluate_thresholds_array_based(
        #     thresholds=[[0.2, 0.25], [0.1, 0.12, 0.15, .2]],
        #     outputs=complete_output)
        # print("X")
        #
        # accuracy_graph_ig, mac_cost_graph_ig = \
        #     self.evaluate_thresholds_graph_based(
        #         thresholds=[[10.0, 10.0], [10.0, 10.0, 10.0, 10.0]],
        #         outputs=complete_output)
        # accuracy_vector_ig, mac_cost_vector_ig = self.evaluate_thresholds_array_based(
        #     thresholds=[[10.0, 10.0], [10.0, 10.0, 10.0, 10.0]],
        #     outputs=complete_output)
        return complete_output

    def optimize_routing_temperatures(self, complete_output):
        for layer_id in range(len(self.model.pathCounts) - 1):
            route_combinations = Utilities.create_route_combinations(shape_=self.model.pathCounts[:(layer_id + 1)])
            results_array_shape = (*self.model.pathCounts[:(layer_id + 1)],)
            temperature_array = np.zeros(shape=results_array_shape)
            temperature_array[:] = np.nan
            for route_combination in route_combinations:
                routing_activations = complete_output.routingActivationMatrices[layer_id][route_combination]
                temperature_optimizer = SoftmaxTemperatureOptimizer()
                entropies_before_low \
                    = Utilities.calculate_entropy_from_activations(activations=routing_activations,
                                                                   temperature=1.0)
                temperature = temperature_optimizer.run(routing_activations=routing_activations)
                entropies_after \
                    = Utilities.calculate_entropy_from_activations(activations=routing_activations,
                                                                   temperature=temperature)
                entropies_before_high \
                    = Utilities.calculate_entropy_from_activations(activations=routing_activations,
                                                                   temperature=temperature * 3.0)
                fig, ax = plt.subplots(3, 1)
                ax[0].set_title("Entropies [{0},{1}] with temperature 1.0".format(layer_id, route_combination))
                ax[0].hist(entropies_before_low, density=False, histtype='stepfilled',
                           alpha=1.0, bins=100, range=(0, self.maxEntropies[layer_id]))
                ax[0].legend(loc='best', frameon=False)

                ax[1].set_title("Entropies [{0},{1}] with temperature {2}".format(layer_id, route_combination,
                                                                                  temperature))
                ax[1].hist(entropies_after, density=False, histtype='stepfilled',
                           alpha=1.0, bins=100, range=(0, self.maxEntropies[layer_id]))
                ax[1].legend(loc='best', frameon=False)

                ax[2].set_title("Entropies [{0},{1}] with temperature {2}".format(layer_id, route_combination,
                                                                                  temperature * 3.0))
                ax[2].hist(entropies_before_high, density=False, histtype='stepfilled',
                           alpha=1.0, bins=100, range=(0, self.maxEntropies[layer_id]))
                ax[2].legend(loc='best', frameon=False)

                plt.tight_layout()
                plt.show()
                plt.close()

                temperature_array[route_combination] = temperature

            assert np.sum(np.isnan(temperature_array)) == 0

            complete_output.optimalTemperatures.append(temperature_array)

    def evaluate_thresholds_graph_based(self, thresholds, outputs):
        data_size = outputs.logits[0].shape[len(self.model.pathCounts)]
        validness_vector = []
        mac_cost_vector = []
        # Single path mac cost
        single_path_mac_cost = sum([sum(d_.values()) for d_ in self.macCountsPerBlock])
        for sample_idx in tqdm(range(data_size)):
            sample_history = []
            ig_path = []
            # Always execute the ig path.
            ig_block = (0,)
            blocks_to_execute_in_this_layer = {(0,)}
            for layer_id in range(len(self.model.pathCounts)):
                # We will use sample history to count the number of executions through the network
                sample_history.append(deepcopy(blocks_to_execute_in_this_layer))
                ig_path.append(ig_block)
                block_to_execute_next_layer = set()
                # Routing layers
                if layer_id < len(self.model.pathCounts) - 1:
                    for block_id in blocks_to_execute_in_this_layer:
                        tpl = (*block_id, sample_idx)
                        routing_activations = outputs.routingActivationMatrices[layer_id][tpl]
                        temperature = outputs.optimalTemperatures[layer_id][block_id]
                        routing_activations_tempered = routing_activations / temperature
                        routing_probabilities = torch.softmax(
                            torch.from_numpy(np.expand_dims(routing_activations_tempered, axis=0)), dim=1).numpy()
                        routing_probabilities = np.squeeze(routing_probabilities)
                        layer_thresholds = thresholds[layer_id]
                        assert len(routing_probabilities) == len(layer_thresholds)
                        for i__ in range(len(routing_probabilities)):
                            prob = routing_probabilities[i__]
                            thresh = layer_thresholds[i__]
                            if prob >= thresh:
                                block_to_execute_next_layer.add((*block_id, i__))
                        # Add block of maximum probability if this block is on the ig path.
                        if block_id == ig_block:
                            ig_index = np.argmax(routing_probabilities)
                            # Change the ig block
                            ig_block = (*block_id, ig_index.item())
                            block_to_execute_next_layer.add(ig_block)
                    blocks_to_execute_in_this_layer = block_to_execute_next_layer
                    assert len(ig_block) == layer_id + 2
                # Loss layers
                else:
                    logits_arr = []
                    labels_arr = []
                    for block_id in blocks_to_execute_in_this_layer:
                        tpl = (*block_id, sample_idx)
                        logits = outputs.logits[0][tpl]
                        label = outputs.labels[0][tpl]
                        logits_arr.append(logits)
                        labels_arr.append(label)
                    assert len(set(labels_arr)) == 1
                    gt_label = list(set(labels_arr))[0]
                    # Combine the logits, calculate the accuracy
                    logits_arr = np.stack(logits_arr, axis=0)
                    posteriors_arr = torch.softmax(torch.from_numpy(logits_arr), dim=1).numpy()
                    posteriors_ensemble = np.mean(posteriors_arr, axis=0)
                    predicted_label = np.argmax(posteriors_ensemble)
                    validness_vector.append(predicted_label.item() == gt_label.item())
                    # Count the executed blocks and their mac costs
                    mac_count_for_thresholds = [
                        len(blocks_executed_per_layer) * sum(self.macCountsPerBlock[bi].values())
                        for bi, blocks_executed_per_layer in enumerate(sample_history)]
                    total_mac = sum(mac_count_for_thresholds)
                    mac_ratio = total_mac / single_path_mac_cost
                    mac_cost_vector.append(mac_ratio)

        accuracy = np.mean(validness_vector)
        mac_cost = np.mean(mac_cost_vector)
        return accuracy, mac_cost

    def evaluate_thresholds_array_based(self, thresholds, outputs):
        data_size = outputs.logits[0].shape[len(self.model.pathCounts)]
        routing_matrices_dict = {(): np.ones(shape=(data_size, 1), dtype=np.bool)}
        path_history_dict = {}
        # Single path mac cost
        single_path_mac_cost = sum([sum(d_.values()) for d_ in self.macCountsPerBlock])
        for layer_id in range(len(self.model.pathCounts)):
            route_combinations = Utilities.create_route_combinations(shape_=
                                                                     self.model.pathCounts[:(layer_id + 1)])
            # Routing layers
            if layer_id < len(self.model.pathCounts) - 1:
                layer_thresholds = thresholds[layer_id]
                for route_combination in route_combinations:
                    routing_activations = outputs.routingActivationMatrices[layer_id][route_combination]
                    temperature = outputs.optimalTemperatures[layer_id][route_combination]
                    routing_activations_tempered = routing_activations / temperature
                    routing_probabilities = torch.softmax(torch.from_numpy(routing_activations_tempered), dim=1).numpy()
                    # Calculate the IG routing matrix
                    ig_indices = np.argmax(routing_probabilities, axis=1)
                    ig_routing_matrix = np.zeros_like(routing_activations)
                    ig_routing_matrix[np.arange(ig_routing_matrix.shape[0]), ig_indices] = 1.0
                    # Calculate the routing matrix based on probability thresholds
                    threshold_routing_matrix = routing_probabilities >= np.expand_dims(np.array(layer_thresholds),
                                                                                       axis=0)
                    # Apply logical or to both ig_routing_matrix and threshold_routing_matrix,
                    # so that routing is always done through the ig path plus the routes where threshold values are
                    # below the routing probabilities.
                    final_routing_matrix = np.logical_or(ig_routing_matrix.astype(np.bool),
                                                         threshold_routing_matrix.astype(np.bool))
                    final_routing_matrix = final_routing_matrix.astype(np.float)
                    # Get the parent routing result
                    parent_route = route_combination[:-1]
                    parent_routing_matrix = routing_matrices_dict[parent_route]
                    child_index_in_parent = route_combination[-1]
                    parent_routing_vector = parent_routing_matrix[:, child_index_in_parent]
                    path_history_dict[route_combination] = np.copy(parent_routing_vector)
                    final_routing_matrix = final_routing_matrix * np.expand_dims(
                        parent_routing_vector.astype(np.float), axis=-1)
                    routing_matrices_dict[route_combination] = final_routing_matrix
            # Loss layers
            else:
                selections_arr = []
                weighted_posteriors_arr = []
                labels_arr = []
                for route_combination in route_combinations:
                    logits = outputs.logits[0][route_combination]
                    labels = outputs.labels[0][route_combination]
                    # Get the parent routing result
                    parent_route = route_combination[:-1]
                    parent_routing_matrix = routing_matrices_dict[parent_route]
                    child_index_in_parent = route_combination[-1]
                    parent_routing_vector = parent_routing_matrix[:, child_index_in_parent]
                    path_history_dict[route_combination] = np.copy(parent_routing_vector)
                    posteriors = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
                    weighted_posteriors = posteriors * np.expand_dims(parent_routing_vector.astype(np.float), axis=-1)
                    selections_arr.append(parent_routing_vector)
                    weighted_posteriors_arr.append(weighted_posteriors)
                    labels_arr.append(labels)

                selections_arr = np.stack(selections_arr, axis=-1)
                weighted_posteriors_arr = np.stack(weighted_posteriors_arr, axis=-1)
                labels_arr = np.stack(labels_arr, axis=-1)

                number_of_experts = np.sum(selections_arr, axis=-1)
                assert np.sum(number_of_experts < 1) == 0
                ensemble_posteriors = np.sum(weighted_posteriors_arr, axis=-1)
                ensemble_posteriors = ensemble_posteriors * np.expand_dims(np.reciprocal(number_of_experts), axis=-1)
                assert np.allclose(np.sum(ensemble_posteriors, axis=-1),
                                   np.ones_like(np.sum(ensemble_posteriors, axis=-1)))
                for idx in range(labels_arr.shape[-1] - 1):
                    A_ = labels_arr[:, idx]
                    B_ = labels_arr[:, idx + 1]
                    assert np.array_equal(A_, B_)

                predicted_labels = np.argmax(ensemble_posteriors, axis=-1)
                gt_labels = labels_arr[:, 0]
                accuracy = np.mean(predicted_labels == gt_labels)
                # Calculate the MAC by tracing the execution history
                mac_costs_list = []
                for lid in range(len(self.model.pathCounts)):
                    layer_history = [v for k, v in path_history_dict.items() if len(k) == lid + 1]
                    layer_history = np.stack(layer_history, axis=-1)
                    layer_execution_times = np.sum(layer_history, axis=-1)
                    unit_mac_cost_layer = sum(self.macCountsPerBlock[lid].values())
                    mac_costs_list.append(unit_mac_cost_layer * layer_execution_times)
                mac_costs_list = np.stack(mac_costs_list, axis=-1)
                mac_costs_list = np.sum(mac_costs_list, axis=-1)
                mac_costs_list = mac_costs_list / single_path_mac_cost
                mac_cost_final = np.mean(mac_costs_list)
                return accuracy, mac_cost_final

    def cost_function(self, **kwargs):
        # lr_initial_rate,
        # hyperbolic_exponent):
        thresholds = []
        for lid in range(len(self.model.pathCounts) - 1):
            thresholds.append([])
            for pid in range(self.model.pathCounts[lid]):
                thresholds[lid].append(self.optimization_bounds_continuous["threshold_{0},{1}".format(lid, pid)])

        accuracy_train, mac_cost_train = self.evaluate_thresholds_array_based(
            thresholds=thresholds, outputs=self.trainOutputs)
        accuracy_test, mac_cost_test = self.evaluate_thresholds_array_based(
            thresholds=thresholds, outputs=self.testOutputs)
        print("********************")
        print("Accuracy Train:{0}".format(accuracy_train))
        print("Mac Train:{0}".format(mac_cost_train))
        print("Accuracy Test:{0}".format(accuracy_test))
        print("Mac Test:{0}".format(mac_cost_test))
        print("********************")

        score = self.macLambda * accuracy_train - (1.0 - self.macLambda) * (mac_cost_train - 1.0)
        return score
