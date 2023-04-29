import os
from collections import Counter

import torch
import numpy as np
from auxillary.db_logger import DbLogger
from torchvision import transforms
import torchvision.datasets as datasets

from auxillary.rump_dataset import RumpDataset
from auxillary.similar_dataset_division_algorithm import SimilarDatasetDivisionAlgorithm
from auxillary.utilities import Utilities
from cigt.cigt_ig_refactored import CigtIgHardRoutingX


class MultiplePathOptimizer(object):
    def __init__(self, checkpoint_path, data_root_path, dataset):
        self.checkpointPath = checkpoint_path
        self.dataRootPath = data_root_path
        self.dataset = dataset
        # Load the trained model
        self.runId = DbLogger.get_run_id()
        self.model = CigtIgHardRoutingX(
            run_id=self.runId,
            model_definition="Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Multiple Path Inference",
            num_classes=10)
        explanation = self.model.get_explanation_string()
        DbLogger.write_into_table(rows=[(self.runId, explanation)], table=DbLogger.runMetaData)
        if not os.path.isdir(self.dataRootPath):
            os.mkdir(self.dataRootPath)
        checkpoint = torch.load(self.checkpointPath, map_location="cpu")
        self.model.load_state_dict(state_dict=checkpoint["model_state_dict"])
        self.routeCombinations = None
        self.igResultsDict = None
        self.dataset0 = None
        self.dataset1 = None
        self.routingResultsDict = None
        self.allLabels = None
        self.routingActivationsListUnified = []
        self.logitsListUnified = None
        self.igRouteSelections = None
        self.labelCounters = []
        self.create_route_combinations()

    def create_pure_ig_results(self):
        information_gain_routing_outputs_path = os.path.join(self.dataRootPath, "ig_routing_results.sav")
        if not os.path.isfile(information_gain_routing_outputs_path):
            self.igResultsDict = self.model.validate(loader=self.dataset, temperature=0.1, epoch=0, data_kind="test",
                                                     enforced_hard_routing_kind="InformationGainRouting",
                                                     return_network_outputs=True)
            Utilities.pickle_save_to_file(file_content=self.igResultsDict, path=information_gain_routing_outputs_path)
        else:
            self.igResultsDict = Utilities.pickle_load_from_file(path=information_gain_routing_outputs_path)

        # Calculate ig routes
        self.igRouteSelections = []
        for block_id, routing_activations in enumerate(self.igResultsDict["list_of_routing_activations"]):
            path_count = routing_activations.shape[1]
            selected_paths = np.argmax(routing_activations, axis=1)
            self.igRouteSelections.append(selected_paths)
            self.labelCounters.append([])
            for path_id in range(path_count):
                path_labels = self.igResultsDict["list_of_labels"][selected_paths == path_id]
                label_counter = Counter(path_labels)
                self.labelCounters[-1].append(label_counter)
        self.igRouteSelections = np.stack(self.igRouteSelections, axis=1)

    def create_evenly_divided_datasets(self):
        dataset_path = os.path.join(self.dataRootPath, "multiple_inference_data.sav")
        if not os.path.isfile(dataset_path):
            dataset_0, dataset_1 = SimilarDatasetDivisionAlgorithm.run(model=self.model, parent_loader=self.dataset,
                                                                       save_path=dataset_path,
                                                                       division_ratio=0.5, max_accuracy_dif=0.002)
        else:
            d_ = Utilities.pickle_load_from_file(path=dataset_path)
            dataset_0 = d_["dataset_0"]
            dataset_1 = d_["dataset_1"]

        self.dataset0 = torch.utils.data.DataLoader(dataset_0, batch_size=1024, shuffle=False, **kwargs)
        self.dataset1 = torch.utils.data.DataLoader(dataset_1, batch_size=1024, shuffle=False, **kwargs)

    def create_route_combinations(self):
        list_of_path_choices = []
        for path_count in self.model.pathCounts[1:]:
            list_of_path_choices.append([i_ for i_ in range(path_count)])
        self.routeCombinations = Utilities.get_cartesian_product(list_of_lists=list_of_path_choices)

    def create_routing_results(self):
        # Load routing results for every possible path
        self.routingResultsDict = {}
        for selected_routes in self.routeCombinations:
            print("Executing route selection:{0}".format(selected_routes))
            data_file_path = os.path.join(self.dataRootPath, "{0}_{1}_data.sav".format("whole", selected_routes))
            if os.path.isfile(data_file_path):
                route_results = Utilities.pickle_load_from_file(path=data_file_path)
                self.routingResultsDict[selected_routes] = route_results
                continue
            routing_matrices = []
            for layer_id, route_id in enumerate(selected_routes):
                routing_matrix = torch.zeros(size=(self.dataset.batch_size, self.model.pathCounts[layer_id + 1]),
                                             dtype=torch.float32)
                routing_matrix[:, route_id] = 1.0
                routing_matrices.append(routing_matrix)

            self.model.enforcedRoutingMatrices = routing_matrices
            enforced_routing_res_dict = self.model.validate(loader=test_loader,
                                                            temperature=0.1, epoch=0, data_kind="test",
                                                            enforced_hard_routing_kind="EnforcedRouting",
                                                            return_network_outputs=True)
            Utilities.pickle_save_to_file(file_content=enforced_routing_res_dict, path=data_file_path)
            self.routingResultsDict[selected_routes] = enforced_routing_res_dict
            print("Saved {0}!!!".format(data_file_path))
        self.model.enforcedRoutingMatrices = []

    # Organize result arrays for fast access.
    def create_efficient_routing_arrays(self):
        # Routing activations
        self.routingActivationsListUnified = []
        for rid, p_count in enumerate(self.model.pathCounts[1:]):
            arr_shapes = list(set([dict_["list_of_routing_activations"][rid].shape
                                   for dict_ in self.routingResultsDict.values()]))
            arr_types = list(set([dict_["list_of_routing_activations"][rid].dtype
                                  for dict_ in self.routingResultsDict.values()]))
            assert len(arr_shapes) == 1
            assert len(arr_types) == 1
            activations_array_shape = (*self.model.pathCounts[1:], *arr_shapes[0])
            activations_arr = np.zeros(shape=activations_array_shape, dtype=arr_types[0])
            activations_arr[:] = np.nan
            for route in self.routeCombinations:
                activations_arr[route][:] = self.routingResultsDict[route]["list_of_routing_activations"][rid]
            assert np.sum(np.isnan(activations_arr)) == 0
            self.routingActivationsListUnified.append(activations_arr)

        # Logits
        arr_shapes = [arr.shape
                      for dict_ in self.routingResultsDict.values() for arr in dict_["list_of_logits_complete"]]
        arr_shapes = set(arr_shapes)
        assert len(arr_shapes) == 1
        arr_shape = list(arr_shapes)[0]
        arr_types = [arr.dtype
                     for dict_ in self.routingResultsDict.values() for arr in dict_["list_of_logits_complete"]]
        arr_types = set(arr_types)
        assert len(arr_types) == 1
        arr_type = list(arr_types)[0]
        for dict_ in self.routingResultsDict.values():
            assert len(dict_["list_of_logits_complete"]) == self.model.pathCounts[-1]

        logits_unified_shape = (*self.model.pathCounts[1:], *arr_shape)
        self.logitsListUnified = np.zeros(shape=logits_unified_shape, dtype=arr_type)
        self.logitsListUnified[:] = np.nan

        for routes_except_last in set([tpl[:-1] for tpl in self.routeCombinations]):
            logits_temp = []
            for k_ in self.routingResultsDict.keys():
                if k_[:-1] == routes_except_last:
                    logits_temp.append(np.stack(self.routingResultsDict[k_]["list_of_logits_complete"], axis=0))
            logits_mean = np.mean(np.stack(logits_temp, axis=0), axis=0)
            for all_last_layer_logits in logits_temp:
                assert np.allclose(logits_mean, all_last_layer_logits)
            self.logitsListUnified[routes_except_last][:] = logits_mean
        assert np.sum(np.isnan(self.logitsListUnified)) == 0

        # Labels
        self.allLabels = self.igResultsDict["list_of_labels"]
        for route in self.routeCombinations:
            assert np.array_equal(self.allLabels, self.routingResultsDict[route]["list_of_labels"])

    def measure_accuracy_with_routing_decisions(self, index_array, routes):
        selected_labels = self.allLabels[index_array]
        route_indices = [routes[:, idx] for idx in range(routes.shape[1])]
        route_indices.append(index_array)
        logits = self.logitsListUnified[route_indices]
        predicted_labels = np.argmax(logits, axis=1)
        correctness_vector = (selected_labels == predicted_labels)
        accuracy = np.mean(correctness_vector)
        return accuracy


if __name__ == "__main__":
    DbLogger.log_db_path = DbLogger.home_asus
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    # Cifar 10 Dataset
    kwargs = {'num_workers': 2, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
        batch_size=1024, shuffle=False, **kwargs)
    train_loader_test_time_augmentation = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform_test),
        batch_size=1024, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=1024, shuffle=False, **kwargs)

    # Paths for results
    chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                             "randig_cigtlogger2_23_epoch1390.pth")
    data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                             "randig_cigtlogger2_23_epoch1390_data")
    multiple_path_optimizer = MultiplePathOptimizer(checkpoint_path=chck_path, data_root_path=data_path,
                                                    dataset=test_loader)
    # Calculate information gain routing results
    multiple_path_optimizer.create_pure_ig_results()
    # Load evenly divided datasets
    multiple_path_optimizer.create_evenly_divided_datasets()
    # Create routing results for every possible routing combination
    multiple_path_optimizer.create_routing_results()
    # Create fast-accessible arrays for routing results
    multiple_path_optimizer.create_efficient_routing_arrays()

    multiple_path_optimizer.measure_accuracy_with_routing_decisions(index_array=np.arange(10000),
                                                                    routes=multiple_path_optimizer.igRouteSelections)

    # trained_model.validate(loader=dataset_0_loader, temperature=0.1, epoch=0, data_kind="test",
    # #                        enforced_hard_routing_kind="InformationGainRouting")
    # # trained_model.validate(loader=dataset_1_loader, temperature=0.1, epoch=0, data_kind="test",
    # #                        enforced_hard_routing_kind="InformationGainRouting")
    #

    #
    # # Create fast-to-access arrays out of routing results
    # create_routing_arrays(model=trained_model, routing_results_dict=route_results_dict)
    #
    # # Detect ideal routing behaviors
    # # res_dict = {
    # #     "accuracy": accuracy_avg.avg,
    # #     "list_of_labels": list_of_labels,
    # #     "list_of_routing_probability_matrices": list_of_routing_probability_matrices,
    # #     "list_of_routing_activations": list_of_routing_activations,
    # #     "list_of_logits_complete": list_of_logits_complete
    # # }
    #
    # label_counters = []
    # route_selections = []
    # for block_id, routing_activations in enumerate(ig_res_dict["list_of_routing_activations"]):
    #     path_count = routing_activations.shape[1]
    #     selected_paths = np.argmax(routing_activations, axis=1)
    #     route_selections.append(selected_paths)
    #     label_counters.append([])
    #     for path_id in range(path_count):
    #         path_labels = ig_res_dict["list_of_labels"][selected_paths == path_id]
    #         label_counter = Counter(path_labels)
    #         label_counters[-1].append(label_counter)
    # route_selections = np.stack(route_selections, axis=1)
    #
    # # Label to route assignments
    # label_route_assignments = {}
    # labels_set = set(ig_res_dict["list_of_labels"])
    #
    # for label in labels_set:
    #     label_route_assignments[label] = []
    #     for block_id, counters_for_block in enumerate(label_counters):
    #         route_frequencies_for_label = [(route_id, cntr[label]) if label in cntr else (route_id, 0)
    #                                        for route_id, cntr in enumerate(counters_for_block)]
    #         route_frequencies_for_label_sorted = sorted(route_frequencies_for_label, key=lambda tpl: tpl[1],
    #                                                     reverse=True)
    #         label_route_assignments[label].append(route_frequencies_for_label_sorted[0][0])
    #
    # # Measure the accuracy of correctly routed samples.
    # # for block_id, routing_activations in enumerate(ig_res_dict["list_of_routing_activations"]):
    # routing_correctness_vector = []
    # for idx in range(ig_res_dict["list_of_labels"].shape[0]):
    #     selected_route = route_selections[idx, :]
    #     ideal_route = np.array(label_route_assignments[ig_res_dict["list_of_labels"][idx]])
    #     routing_correctness_vector.append(np.array_equal(selected_route, ideal_route))
    # print("X")
    #
    # # print("X")
