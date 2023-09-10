import os.path

import numpy as np
import torch
from auxillary.bayesian_optimizer import BayesianOptimizer
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
from tqdm import tqdm


class NetworkOutput(object):
    def __init__(self):
        self.routingActivationMatrices = []
        self.logits = []


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
        if evaluate_network_first:
            test_acc = self.model.validate(data_kind="test", epoch=0, loader=self.testDataset, temperature=0.1,
                                           verbose=True)
            print("Standard test accuracy:{0}".format(test_acc))

        for path_count in self.model.pathCounts[1:]:
            self.model.enforcedRoutingMatrices.append(
                torch.ones(size=(max_branch_count * self.model.batchSize, path_count), dtype=torch.int64))

        # self.assert_gather_scatter_model_output_correctness(dataloader=self.testDataset, repeat_count=1)
        self.create_outputs(dataloader=self.trainDataset, repeat_count=self.repeatCount)
        self.create_outputs(dataloader=self.testDataset, repeat_count=1)

    def get_start_offset_for_gather_scatter_model(self, route_, batch_id_, curr_batch_size):
        path_list = list(zip(list(range(len(route_))), route_))
        offset = batch_id_ * np.prod(self.model.pathCounts[:len(route_)], dtype=np.int64) * self.model.batchSize
        for tpl in reversed(path_list):
            block_id = tpl[0]
            block_path_index = tpl[1]
            offset_step = block_path_index * np.prod(self.model.pathCounts[:block_id], dtype=np.int64) * curr_batch_size
            offset += offset_step
        return offset

    def interpret_gather_scatter_model_outputs(self, outputs_dict, dataloader):
        network_output = NetworkOutput()
        data_size = outputs_dict["list_of_labels"][0].shape[0]
        assert outputs_dict["list_of_original_labels"].shape[0] == data_size
        for block_id in range(len(self.model.pathCounts)):
            # Routing blocks
            if block_id < len(self.model.pathCounts) - 1:
                output_array = outputs_dict["list_of_routing_activations"][block_id]
                result_container = network_output.routingActivationMatrices
                results_array_shape = (*self.model.pathCounts[:(block_id + 1)],
                                       data_size, self.model.pathCounts[block_id + 1])
            # Loss calculation blocks
            else:
                output_array = outputs_dict["list_of_logits_unified"]
                result_container = network_output.logits
                results_array_shape = (*self.model.pathCounts[:(block_id + 1)], data_size, self.model.numClasses)

            interpreted_results_array = np.zeros(shape=results_array_shape, dtype=np.float32)
            interpreted_results_array[:] = np.nan
            route_combinations = Utilities.create_route_combinations(shape_=self.model.pathCounts[:(block_id + 1)])
            batch_sizes = [len(outputs_dict["list_of_labels"][0][idx:idx + self.model.batchSize])
                           for idx in range(0, data_size, self.model.batchSize)]

            for route_combination in route_combinations:
                for i_, curr_batch_size in tqdm(enumerate(batch_sizes)):
                    route_offset = self.get_start_offset_for_gather_scatter_model(route_=route_combination,
                                                                                  batch_id_=i_,
                                                                                  curr_batch_size=curr_batch_size)
                    route_activations_array = output_array[route_offset:route_offset + curr_batch_size, :]
                    interpreted_results_array[route_combination
                    ][i_ * self.model.batchSize:i_ * self.model.batchSize + curr_batch_size] = route_activations_array
            assert np.sum(np.isnan(interpreted_results_array)) == 0
            result_container.append(interpreted_results_array)
        return network_output

    def assert_gather_scatter_model_output_correctness(self, network_output, results_dict2):
        # class NetworkOutput(object):
        #     def __init__(self):
        #         self.routingActivationMatrices = []
        #         self.logits = []

        for block_id in range(len(self.model.pathCounts)):
            if block_id < len(self.model.pathCounts) - 1:
                arr_1 = network_output.routingActivationMatrices[block_id]
                arr_2 = results_dict2["routing_activations_complete"]
            else:
                arr_1 = network_output.logits[0]
                arr_2 = results_dict2["logits_complete"]
            route_combinations = Utilities.create_route_combinations(shape_=self.model.pathCounts[:(block_id + 1)])
            for route in route_combinations:
                sub_arr1 = arr_1[route]
                sub_arr2 = arr_2[route]
                assert sub_arr1.dtype == sub_arr2.dtype
                num_distant_entries = np.sum(np.isclose(sub_arr1, sub_arr1, rtol=1e-3) == False)
                ratio_of_distant_entries = num_distant_entries / np.prod(sub_arr1.shape)
                print("ratio_of_distant_entries={0}".format(ratio_of_distant_entries))
                assert np.allclose(sub_arr1, sub_arr2, rtol=1e-2)

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
                raw_outputs_type1_dict = self.model.validate(loader=dataloader, epoch=0, temperature=0.1,
                                                             enforced_hard_routing_kind="EnforcedRouting",
                                                             return_network_outputs=True, data_kind=data_kind,
                                                             verbose=True)
                x_tensor = raw_outputs_type1_dict["list_of_original_inputs"]
                y_tensor = raw_outputs_type1_dict["list_of_original_labels"]
                assert np.array_equal(y_tensor, raw_outputs_type1_dict["list_of_labels"][0])
                validation_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.from_numpy(x_tensor), torch.from_numpy(y_tensor)),
                    shuffle=False, batch_size=self.model.batchSize)
                raw_outputs_type2_dict = self.model.validate_v2(validation_loader,
                                                                temperature=0.1,
                                                                enforced_hard_routing_kind="EnforcedRouting",
                                                                verbose=True)
                Utilities.pickle_save_to_file(path=raw_outputs_file_path,
                                              file_content={"raw_outputs_type1_dict": raw_outputs_type1_dict,
                                                            "raw_outputs_type2_dict": raw_outputs_type2_dict})
            else:
                outputs_loaded = Utilities.pickle_load_from_file(raw_outputs_file_path)
                raw_outputs_type1_dict = outputs_loaded["raw_outputs_type1_dict"]
                raw_outputs_type2_dict = outputs_loaded["raw_outputs_type2_dict"]

            interpreted_network_outputs = self.interpret_gather_scatter_model_outputs(
                outputs_dict=raw_outputs_type1_dict, dataloader=dataloader)
            self.assert_gather_scatter_model_output_correctness(
                network_output=interpreted_network_outputs,
                results_dict2=raw_outputs_type2_dict)
            network_outputs.append(interpreted_network_outputs)
        if repeat_count > 1:
            complete_output = self.merge_multiple_outputs(network_outputs=network_outputs)
        else:
            assert len(network_outputs) == 1
            complete_output = network_outputs[0]
