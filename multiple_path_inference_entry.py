import os
from collections import Counter

import torch
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
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
        self.idealLabelRouteAssignments = {}
        self.correctlyRoutedSampleIndices = None
        self.incorrectlyRoutedSampleIndices = None
        self.maxEntropies = []
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
            max_entropy = np.asscalar(-np.log(1.0 / routing_activations.shape[1]))
            self.maxEntropies.append(max_entropy)
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

            all_past_routes = set([route[:rid] for route in self.routeCombinations])
            mean_activations_array_shape = [max([tpl[i_] for tpl in self.routeCombinations]) + 1 for i_ in range(rid)]
            mean_activations_array_shape = (*mean_activations_array_shape, *arr_shapes[0])
            total_routing_array = np.zeros(shape=mean_activations_array_shape, dtype=arr_types[0])
            total_routing_array[:] = np.nan
            for past_route in all_past_routes:
                matching_arr = []
                for route in self.routeCombinations:
                    if past_route == route[:rid]:
                        matching_arr.append(activations_arr[route])
                mean_arr = np.mean(np.stack(matching_arr, axis=0), axis=0)
                for a_ in matching_arr:
                    assert np.allclose(mean_arr, a_)
                total_routing_array[past_route][:] = mean_arr
            assert np.sum(np.isnan(activations_arr)) == 0
            assert np.sum(np.isnan(total_routing_array)) == 0
            self.routingActivationsListUnified.append(total_routing_array)

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
        acc = np.mean(correctness_vector)
        return acc

    def determine_ideal_routes_for_labels(self):
        self.idealLabelRouteAssignments = {}
        labels_set = set(self.igResultsDict["list_of_labels"])

        for label in labels_set:
            self.idealLabelRouteAssignments[label] = []
            for block_id, counters_for_block in enumerate(self.labelCounters):
                route_frequencies_for_label = [(route_id, cntr[label]) if label in cntr else (route_id, 0)
                                               for route_id, cntr in enumerate(counters_for_block)]
                route_frequencies_for_label_sorted = sorted(route_frequencies_for_label, key=lambda tpl: tpl[1],
                                                            reverse=True)
                self.idealLabelRouteAssignments[label].append(route_frequencies_for_label_sorted[0][0])

        # Determine the indices of correctly and incorrectly routed samples
        routing_correctness_vector = []
        for idx in range(self.allLabels.shape[0]):
            selected_route = self.igRouteSelections[idx, :]
            ideal_route = np.array(self.idealLabelRouteAssignments[self.allLabels[idx]])
            routing_correctness_vector.append(np.array_equal(selected_route, ideal_route))

        print("{0} of samples have been correctly routed.".format(np.mean(routing_correctness_vector)))
        self.correctlyRoutedSampleIndices = np.nonzero(routing_correctness_vector)[0]
        correct_routes = self.igRouteSelections[self.correctlyRoutedSampleIndices]
        acc_correct_routes = self.measure_accuracy_with_routing_decisions(index_array=self.correctlyRoutedSampleIndices,
                                                                          routes=correct_routes)
        print("Accuracy of Correct Routes:{0}".format(acc_correct_routes))

        self.incorrectlyRoutedSampleIndices = np.nonzero(np.logical_not(routing_correctness_vector))[0]
        incorrect_routes = self.igRouteSelections[self.incorrectlyRoutedSampleIndices]
        acc_incorrect_routes = self.measure_accuracy_with_routing_decisions(
            index_array=self.incorrectlyRoutedSampleIndices,
            routes=incorrect_routes)
        print("Accuracy of Incorrect Routes:{0}".format(acc_incorrect_routes))

    def measure_correctness_of_ideally_and_wrongly_routed_samples(self, random_correction_ratio=0.5):
        # Measure the accuracy of correctly routed samples.
        # for block_id, routing_activations in enumerate(ig_res_dict["list_of_routing_activations"]):
        for trial_id in range(100):
            # Correct "random_correction_ratio" of wrongly routed samples such that they follow the correct paths
            # and measure the accuracy
            improved_routes = np.copy(self.igRouteSelections)
            selected_incorrect_routed_sample_indices = np.random.choice(
                self.incorrectlyRoutedSampleIndices,
                size=int(random_correction_ratio * self.incorrectlyRoutedSampleIndices.shape[0]),
                replace=False)
            # Change the selected wrongly routed samples with their ideal routes.
            for correction_idx in selected_incorrect_routed_sample_indices:
                ideal_route = np.array(self.idealLabelRouteAssignments[self.allLabels[correction_idx]])
                improved_routes[correction_idx, :] = ideal_route
            acc_improved_routes = self.measure_accuracy_with_routing_decisions(
                index_array=np.arange(self.allLabels.shape[0], ),
                routes=improved_routes)
            print("Trial {0} New Accuracy:{1}".format(trial_id, acc_improved_routes))

        print("X")

    def measure_routing_entropies_of_samples(self, sample_indices, do_plot=False, plot_name=""):
        curr_selected_routes = [sample_indices]
        entropies_per_layer = []
        eps = 1e-30
        for p_id in range(len(self.model.pathCounts) - 1):
            activations = self.routingActivationsListUnified[p_id][curr_selected_routes]
            routing_probs = torch.softmax(torch.from_numpy(activations), dim=1).numpy()
            log_routing_probs = np.log(routing_probs + eps)
            prob_log_prob = routing_probs * log_routing_probs
            entropies = -1.0 * np.sum(prob_log_prob, axis=1)
            entropies_per_layer.append(entropies)
            selected_routes = np.argmax(activations, axis=1)
            curr_selected_routes = [*curr_selected_routes[:-1], selected_routes, sample_indices]

        for p_id in range(len(self.model.pathCounts) - 1):
            mean_entropy = np.mean(np.array(entropies_per_layer[p_id]))
            std_entropy = np.std(np.array(entropies_per_layer[p_id]))

            if do_plot:
                fig, ax = plt.subplots(1, 1)
                ax.set_title("{0} Entropy distribution of layer {1}".format(plot_name, p_id))
                ax.hist(np.array(entropies_per_layer[p_id]), density=False, histtype='stepfilled',
                        alpha=1.0, bins=100, range=(0, self.maxEntropies[p_id]))
                ax.legend(loc='best', frameon=False)
                plt.tight_layout()
                plt.show()
                plt.close()

            print("Layer {0} mean entropy:{1} std entropy:{2}".format(p_id, mean_entropy, std_entropy))
        return entropies_per_layer

    def per_entropy_accuracy_analysis(self, sample_indices, entropy_percentile):
        entropies_per_layer = self.measure_routing_entropies_of_samples(sample_indices=sample_indices)
        percentile_index = int(len(sample_indices) * (1.0 - entropy_percentile))
        high_entropy_indices_all = set()
        for p_id in range(len(self.model.pathCounts) - 1):
            sorted_indices = np.argsort(entropies_per_layer[p_id])
            # low_entropy_indices = sorted_indices[:percentile_index]
            high_entropy_indices = sorted_indices[percentile_index:]
            high_entropy_indices_all = high_entropy_indices_all.union(set(high_entropy_indices))

        low_entropy_indices_all = set(sample_indices).difference(high_entropy_indices_all)
        low_entropy_indices_all = np.array(list(low_entropy_indices_all))
        high_entropy_indices_all = np.array(list(high_entropy_indices_all, ))
        low_entropy_routing_decisions = self.igRouteSelections[low_entropy_indices_all]
        high_entropy_routing_decisions = self.igRouteSelections[high_entropy_indices_all]
        low_entropy_accuracy = \
            self.measure_accuracy_with_routing_decisions(index_array=low_entropy_indices_all,
                                                         routes=low_entropy_routing_decisions)
        high_entropy_accuracy = \
            self.measure_accuracy_with_routing_decisions(index_array=high_entropy_indices_all,
                                                         routes=high_entropy_routing_decisions)
        print("Low Entropy Accuracy:{0} Sample Count:{1}".format(low_entropy_accuracy,
                                                                 low_entropy_indices_all.shape[0]))
        print("High Entropy Accuracy:{0} Sample Count:{1}".format(high_entropy_accuracy,
                                                                  high_entropy_indices_all.shape[0]))

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
    # Measure the accuracy of the original IG based routing
    accuracy = multiple_path_optimizer.measure_accuracy_with_routing_decisions(
        index_array=np.arange(10000), routes=multiple_path_optimizer.igRouteSelections)
    print("Accuracy of the original IG routing:{0}".format(accuracy))
    # Determine ideal route assignments for every label (a basic heuristic)
    multiple_path_optimizer.determine_ideal_routes_for_labels()
    # Determine the correctness of correcty and wrongly routed samples' accuracies
    multiple_path_optimizer.measure_correctness_of_ideally_and_wrongly_routed_samples()

    multiple_path_optimizer.measure_routing_entropies_of_samples(
        sample_indices=multiple_path_optimizer.correctlyRoutedSampleIndices, do_plot=True,
        plot_name="Correctly Routed Samples")
    multiple_path_optimizer.measure_routing_entropies_of_samples(
        sample_indices=multiple_path_optimizer.incorrectlyRoutedSampleIndices, do_plot=True,
        plot_name="Incorrectly Routed Samples")

    multiple_path_optimizer.per_entropy_accuracy_analysis(sample_indices=np.arange(10000), entropy_percentile=0.15)
