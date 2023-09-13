import os.path

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


class MultiplePathBayesianOptimizer(BayesianOptimizer):
    def __init__(self, data_root_path, model,
                 train_dataset, test_dataset, xi, init_points, n_iter, mac_counts_per_block,
                 train_dataset_repeat_count, evaluate_network_first):
        super().__init__(init_points, n_iter)
        self.dataRootPath = data_root_path
        self.repeatCount = train_dataset_repeat_count
        self.macCountsPerBlock = mac_counts_per_block
        self.trainDataset = train_dataset
        self.testDataset = test_dataset
        self.maxEntropies = []
        self.optimization_bounds_continuous = {}
        self.model = model
        max_branch_count = np.prod(self.model.pathCounts)
        self.create_entropy_bounds()
        if evaluate_network_first:
            test_acc = self.model.validate(data_kind="test", epoch=0, loader=self.testDataset, temperature=0.1,
                                           verbose=True)
            print("Standard test accuracy:{0}".format(test_acc))
        # self.assert_gather_scatter_model_output_correctness(dataloader=self.testDataset, repeat_count=1)
        self.create_outputs(dataloader=self.trainDataset, repeat_count=self.repeatCount)
        self.create_outputs(dataloader=self.testDataset, repeat_count=1)

    def create_entropy_bounds(self):
        self.optimization_bounds_continuous = {}
        for layer_id, block_count in enumerate(self.model.pathCounts):
            if layer_id == len(self.model.pathCounts) - 1:
                break
            max_entropy = (-np.log(1.0 / self.model.pathCounts[layer_id + 1])).item()
            self.maxEntropies.append(max_entropy)
            # Route combinations for that layer
            routes_for_this_layer = set([tpl[:layer_id] for tpl in self.routeCombinations])
            for route in routes_for_this_layer:
                self.optimization_bounds_continuous[str(route)[1:-1]] = (0.0, self.maxEntropies[layer_id])

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

    def interpret_gather_scatter_model_outputs(self, outputs_dict):
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
        total_arr = np.concatenate([n_o.logits[0] for n_o in network_outputs], axis=len(self.model.pathCounts))
        complete_output.logits.append(total_arr)
        return complete_output

    def create_outputs(self, dataloader, repeat_count):
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

            interpreted_network_outputs = self.interpret_gather_scatter_model_outputs(
                outputs_dict=raw_outputs_type1_dict)
            self.assert_gather_scatter_model_output_correctness(
                network_output=interpreted_network_outputs,
                results_dict2=raw_outputs_type2_dict)
            network_outputs.append(interpreted_network_outputs)
        # if repeat_count > 1:
        #     complete_output = self.merge_multiple_outputs(network_outputs=network_outputs)
        # else:
        #     assert len(network_outputs) == 1
        #     complete_output = network_outputs[0]

    def optimize_routing_temperatures(self, complete_output):
        for layer_id in range(len(self.model.pathCounts) - 1):
            route_combinations = Utilities.create_route_combinations(shape_=self.model.pathCounts[:(layer_id + 1)])
            for route_combination in route_combinations:
                routing_activations = complete_output.routingActivationMatrices[layer_id][route_combination]
                temperature_optimizer = SoftmaxTemperatureOptimizer()
                entropies_before \
                    = Utilities.calculate_entropy_from_activations(activations=routing_activations,
                                                                   temperature=1.0)
                temperature = temperature_optimizer.run(routing_activations=routing_activations)
                entropies_after \
                    = Utilities.calculate_entropy_from_activations(activations=routing_activations,
                                                                   temperature=temperature)
                # fig, ax = plt.subplots(2, 1)
                # ax[0].set_title("Entropies [{0},{1}] with temperature 1.0".format(layer_id, route_combination))
                # ax[0].hist(entropies_before, density=False, histtype='stepfilled',
                #            alpha=1.0, bins=100, range=(0, self.maxEntropies[p_id]))
                # ax[0].legend(loc='best', frameon=False)
                # ax[1].set_title("Entropies [{0},{1}] with temperature {1}".format(layer_id, route_combination))
                # ax[1].hist(entropies_after, density=False, histtype='stepfilled',
                #            alpha=1.0, bins=100, range=(0, self.maxEntropies[p_id]))
                # ax[1].legend(loc='best', frameon=False)
                #
                # plt.tight_layout()
                # plt.show()
                # plt.close()

    def evaluate_thresholds_graph_based(self, thresholds, network_outputs):
        pass
