import os.path

import numpy as np
import torch
from auxillary.bayesian_optimizer import BayesianOptimizer
from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.resnet_cigt_constants import ResnetCigtConstants
from tqdm import tqdm


class NetworkOutput(object):
    def __init__(self):
        self.routingActivationMatrices = []
        self.logits = []


class MultiplePathBayesianOptimizer(BayesianOptimizer):
    def __init__(self, data_root_path, model,
                 train_dataset, test_dataset, xi, init_points, n_iter, train_dataset_repeat_count,
                 evaluate_network_first):
        super().__init__(xi, init_points, n_iter)
        self.dataRootPath = data_root_path
        self.repeatCount = train_dataset_repeat_count
        self.trainDataset = train_dataset
        self.testDataset = test_dataset
        self.maxEntropies = []
        self.optimization_bounds_continuous = {}
        # Load the trained model
        ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
        self.model = model
        max_branch_count = np.prod(self.model.pathCounts)
        for path_count in self.model.pathCounts[1:]:
            self.model.enforcedRoutingMatrices.append(
                torch.ones(size=(max_branch_count * self.model.batchSize, path_count), dtype=torch.int64))

        if evaluate_network_first:
            test_acc = self.model.validate(data_kind="test", epoch=0, loader=self.testDataset, temperature=0.1)
            print("Standard test accuracy:{0}".format(test_acc))

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

            interpreted_results_array = np.zeros(shape=results_array_shape, dtype=np.float)
            interpreted_results_array[:] = np.nan
            route_combinations = Utilities.create_route_combinations(shape_=self.model.pathCounts[:(block_id + 1)])
            for route_combination in route_combinations:
                for i_, (x, y) in tqdm(enumerate(dataloader)):
                    route_offset = self.get_start_offset_for_gather_scatter_model(route_=route_combination,
                                                                                  batch_id_=i_,
                                                                                  curr_batch_size=x.shape[0])
                    route_activations_array = output_array[route_offset:route_offset + x.shape[0], :]
                    interpreted_results_array[route_combination
                    ][i_ * self.model.batchSize:i_ * self.model.batchSize + x.shape[0]] = route_activations_array
            assert np.sum(np.isnan(interpreted_results_array)) == 0
            result_container.append(interpreted_results_array)
        return network_output

    def assert_gather_scatter_model_output_correctness(self, dataloader, repeat_count):
        for epoch_id in range(repeat_count):
            outputs_dict = self.model.validate(loader=dataloader, epoch=0, temperature=0.1,
                                               enforced_hard_routing_kind="EnforcedRouting",
                                               return_network_outputs=True, data_kind="test")
            network_output = self.interpret_gather_scatter_model_outputs(outputs_dict=outputs_dict,
                                                                         dataloader=dataloader)
            block_outputs_complete, routing_matrices_soft_complete, \
                routing_matrices_hard_complete, \
                routing_activations_complete, logits_complete = \
                self.model.validate_v2(dataloader,
                                       temperature=0.1,
                                       enforced_hard_routing_kind="EnforcedRouting")
            for k, v in routing_activations_complete.items():
                if len(k) == 0:
                    continue
                arr_to_compare = network_output.routingActivationMatrices[len(k) - 1][k]
                assert np.allclose(v, arr_to_compare, rtol=1e-3)
            for k, v in logits_complete.items():
                if len(k) == 0:
                    continue
                arr_to_compare = network_output.logits[0][k]
                assert np.allclose(v, arr_to_compare, rtol=1e-3)

    def create_outputs(self, dataloader, repeat_count):
        network_outputs = []
        data_kind = "test" if not dataloader.dataset.train else "train"
        for epoch_id in range(repeat_count):
            print("Processing Data:{0} Epoch:{1}".format(data_kind, epoch_id))
            output_file_path = "{0}_{1}_outputs_dict.sav".format(data_kind, epoch_id)
            if os.path.isfile(output_file_path):
                epoch_results = Utilities.pickle_load_from_file(path=output_file_path)
                raw_outputs_dict = epoch_results["raw_outputs_dict"]
                interpreted_output = epoch_results["interpreted_output"]
            else:
                raw_outputs_dict = self.model.validate(loader=dataloader, epoch=0, temperature=0.1,
                                                       enforced_hard_routing_kind="EnforcedRouting",
                                                       return_network_outputs=True, data_kind=data_kind)

                if isinstance(self.model, CigtIgGatherScatterImplementation):
                    interpreted_output = self.interpret_gather_scatter_model_outputs(outputs_dict=raw_outputs_dict,
                                                                                     dataloader=dataloader)
                else:
                    raise NotImplementedError()

                Utilities.pickle_save_to_file(path=output_file_path, file_content={
                    "raw_outputs_dict": raw_outputs_dict, "interpreted_output": interpreted_output})
            network_outputs.append(interpreted_output)

        complete_output = NetworkOutput()
        for block_id in range(len(self.model.pathCounts) - 1):
            assert len(set([n_o.routingActivationMatrices[block_id].shape[block_id + 1]
                            for n_o in network_outputs])) == 1
            total_arr = np.concatenate([n_o.routingActivationMatrices[block_id] for n_o in network_outputs],
                                       axis=block_id + 1)
            complete_output.routingActivationMatrices.append(total_arr)
        assert len(set([n_o.logits[0].shape[len(self.model.pathCounts)] for n_o in network_outputs])) == 1
        total_arr = np.concatenate([n_o.logits[0] for n_o in network_outputs], axis=len(self.model.pathCounts))
        complete_output.logits.append(total_arr)

        #     outputs_dict_v2 = self.model.validate_v2(loader=dataloader, epoch=0, temperature=0.1,
        #                                              enforced_hard_routing_kind="EnforcedRouting",
        #                                              return_network_outputs=True, data_kind="test")
        #
        #     if isinstance(self.model, CigtIgGatherScatterImplementation):
        #         network_output = self.interpret_gather_scatter_model_outputs(outputs_dict=outputs_dict)
        #         network_outputs.append(network_output)
        #     else:
        #         raise NotImplementedError()
        # complete_output = NetworkOutput()
        # for block_id in range(len(self.model.pathCounts) - 1):
        #     total_arr = np.concatenate([n_o.routingActivationMatrices[block_id] for n_o in network_outputs],
        #                                axis=block_id + 1)
        #     complete_output.routingActivationMatrices.append(total_arr)
        # total_arr = np.concatenate([n_o.logits[0] for n_o in network_outputs], axis=len(self.model.pathCounts))
        # complete_output.logits.append(total_arr)

    # def create_outputs(self, dataloader, repeat_count):
    #     max_branch_count = np.prod(self.model.pathCounts)
    # for path_count in self.model.pathCounts[1:]:
    #     self.model.enforcedRoutingMatrices.append(
    #         torch.ones(size=(max_branch_count * self.model.batchSize, path_count), dtype=torch.int64))
    #
    # # def validate(self, loader, epoch, data_kind, temperature=None,
    # #              enforced_hard_routing_kind=None, print_avg_measurements=False, return_network_outputs=False):
    # #
    # for epoch_id in range(repeat_count):
    #     outputs_dict = self.model.validate(loader=dataloader, epoch=0, temperature=0.1,
    #                                        enforced_hard_routing_kind="EnforcedRouting",
    #                                        return_network_outputs=True, data_kind="test")
    #     print("X")

    # def create_entropy_bounds(self):
    #     self.optimization_bounds_continuous = {}
    #     for layer_id, block_count in enumerate(self.model.pathCounts):
    #         if layer_id == len(self.model.pathCounts) - 1:
    #             break
    #         max_entropy = (-np.log(1.0 / self.model.pathCounts[layer_id + 1])).item()
    #         self.maxEntropies.append(max_entropy)
    #         # Route combinations for that layer
    #         routes_for_this_layer = set([tpl[:layer_id] for tpl in self.routeCombinations])
    #         for route in routes_for_this_layer:
    #             self.optimization_bounds_continuous[str(route)[1:-1]] = (0.0, self.maxEntropies[layer_id])

    # DbLogger.write_into_table(rows=[(self.runId, explanation)], table=DbLogger.runMetaData)
    # if not os.path.isdir(self.dataRootPath):
    #     os.mkdir(self.dataRootPath)
    # checkpoint = torch.load(self.checkpointPath, map_location="cpu")
    # self.model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    # self.routeCombinations = None
    # self.igResultsDict = None
    # self.dataset0 = None
    # self.dataset1 = None
    # self.routingResultsDict = None
    # self.allLabels = None
    # self.routingActivationsListUnified = []
    # self.logitsListUnified = None
    # self.igRouteSelections = None
    # self.labelCounters = []
    # self.idealLabelRouteAssignments = {}
    # self.correctlyRoutedSampleIndices = None
    # self.incorrectlyRoutedSampleIndices = None
    # self.maxEntropies = []
    # self.entropyBoundsDict = {}
    # self.optimalTemperatures = {}
    # self.create_route_combinations()
    # self.create_entropy_bounds()
