import os.path
from multiprocessing import Process

import numpy as np
import inspect
import pandas as pd

import torch.cuda
from sqlalchemy import create_engine
from tqdm import tqdm
from auxillary.db_logger import DbLogger
from auxillary.time_profiler import TimeProfiler
from auxillary.utilities import Utilities
from cigt.multipath_evaluator import MultipathEvaluator
from cigt.multivariate_sigmoid_mixture import MultivariateSigmoidMixture


class MultipathInferenceCrossEntropyV2(object):
    def __init__(self,
                 run_id,
                 path_counts,
                 mac_lambda,
                 max_probabilities,
                 multipath_evaluator,
                 n_iter,
                 quantile_interval,
                 num_of_components,
                 covariance_type,
                 single_threshold_for_each_layer,
                 num_samples_each_iteration,
                 maximum_iterations_without_improvement,
                 num_jobs
                 ):
        super().__init__()
        self.parameterRunId = run_id
        self.parameterPathCounts = path_counts
        self.parameterMaxProbabilities = max_probabilities
        self.parameterMacLambda = mac_lambda
        self.parameterNumOfIterations = n_iter
        self.parameterQuantileInterval = quantile_interval
        self.parameterNumOfComponents = num_of_components
        self.parameterCovarianceType = covariance_type
        self.parameterSingleThresholdForEachLayer = single_threshold_for_each_layer
        self.parameterNumOfSamples = num_samples_each_iteration
        self.parameterMaximumIterationsWithoutImprovement = maximum_iterations_without_improvement

        self.numJobs = num_jobs
        self.multipathEvaluator = multipath_evaluator
        self.thresholdMappingDict, self.intervalBoundsDict, self.intervalsArray \
            = self.determine_threshold_mapping()
        self.distribution = MultivariateSigmoidMixture(
            num_of_components=self.parameterNumOfComponents,
            covariance_type=self.parameterCovarianceType,
            dimension_intervals=self.intervalsArray)

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def determine_threshold_mapping(self):
        threshold_mapping_dict = {}
        interval_bounds_dict = {}
        if self.parameterSingleThresholdForEachLayer:
            curr_layer_start_dimension_id = 0
            for layer_id in range(len(self.parameterPathCounts) - 1):
                route_combinations = Utilities.create_route_combinations(
                    shape_=self.parameterPathCounts[:(layer_id + 1)])
                next_layer_node_count = self.parameterPathCounts[layer_id + 1]
                for route_combination in route_combinations:
                    dimension_id = 0
                    for next_node_id in range(next_layer_node_count):
                        next_combination_tuple = (*route_combination, next_node_id)
                        threshold_mapping_dict[next_combination_tuple] = curr_layer_start_dimension_id + dimension_id
                        dimension_id += 1
                curr_layer_start_dimension_id += next_layer_node_count
        else:
            dimension_id = 0
            for layer_id in range(len(self.parameterPathCounts) - 1):
                route_combinations = Utilities.create_route_combinations(
                    shape_=self.parameterPathCounts[:(layer_id + 1)])
                next_layer_node_count = self.parameterPathCounts[layer_id + 1]
                for route_combination in route_combinations:
                    for next_node_id in range(next_layer_node_count):
                        next_combination_tuple = (*route_combination, next_node_id)
                        threshold_mapping_dict[next_combination_tuple] = dimension_id
                        dimension_id += 1

        min_tpl_length = min([len(tpl) for tpl in threshold_mapping_dict.keys()])
        for tpl in threshold_mapping_dict.keys():
            interval_bounds_dict[tpl] = (0.0, self.parameterMaxProbabilities[len(tpl) - min_tpl_length])

        assert set([idx for idx in range(max(threshold_mapping_dict.values()) + 1)]) \
               == set(threshold_mapping_dict.values())
        intervals_array = np.zeros(shape=(max(threshold_mapping_dict.values()) + 1, 2))

        for tpl, dim_index in threshold_mapping_dict.items():
            interval = interval_bounds_dict[tpl]
            intervals_array[dim_index, 0] = interval[0]
            intervals_array[dim_index, 1] = interval[1]

        return threshold_mapping_dict, interval_bounds_dict, intervals_array

    def get_explanation_string(self):
        kv_rows = []
        explanation = "Multivariate Sigmoid Gaussian Based Cross Entropy Search \n"
        for elem in inspect.getmembers(self):
            if elem[0].startswith("parameter"):
                name_of_param = elem[0]
                value = elem[1]
                explanation += "{0}:{1}\n".format(elem[0], elem[1])
                kv_rows.append((self.parameterRunId, name_of_param, "{0}".format(value)))
        DbLogger.write_into_table(rows=kv_rows, table="run_parameters")
        return explanation

    def fit(self):
        explanation = self.get_explanation_string()
        DbLogger.write_into_table(rows=[(self.parameterRunId, explanation)], table=DbLogger.runMetaData)
        db_engine = create_engine(url="sqlite:///" + DbLogger.log_db_path, echo=False)
        time_profiler = TimeProfiler()
        train_outputs_cuda = self.multipathEvaluator.trainOutputs.move_to_torch(device=self.device)
        test_outputs_cuda = self.multipathEvaluator.testOutputs.move_to_torch(device=self.device)
        best_train_accuracies = {}
        best_test_accuracies = {}
        best_objective_scores = {}
        iterations_without_improvement = 0
        best_acc_overall = 0.0

        for iteration_id in range(self.parameterNumOfIterations):
            # Generate samples with the current distributions
            thresholds = self.distribution.sample(num_of_samples=self.parameterNumOfSamples)
            raw_results_dict = self.analyze_thresholds(
                thresholds=thresholds,
                best_objective_scores=best_objective_scores,
                best_test_accuracies=best_test_accuracies,
                best_train_accuracies=best_train_accuracies,
                iteration_id=iteration_id,
                test_outputs=test_outputs_cuda,
                train_outputs=train_outputs_cuda,
                verbose=True)

            # Store all results into a Pandas Dataframe
            raw_results_df = pd.DataFrame(raw_results_dict)
            sorted_raw_results_df = raw_results_df.sort_values(by=["score"], inplace=False, ascending=False)
            ideal_scores_start = int(sorted_raw_results_df.shape[0] * min(self.parameterQuantileInterval))
            ideal_scores_end = int(sorted_raw_results_df.shape[0] * max(self.parameterQuantileInterval))
            best_scores_df = sorted_raw_results_df.iloc[ideal_scores_start:ideal_scores_end]

            best_scores_dict = {"accuracy_train": [],
                                "mac_cost_train": [],
                                "accuracy_test": [],
                                "mac_cost_test": [],
                                "score": [],
                                "head_count": []}
            for perc in [0.05, 0.1, 0.15, 0.25, 0.5, 1.0]:
                head_count = int(best_scores_df.shape[0] * perc)
                head_df = best_scores_df.head(head_count)
                for metric_name in best_scores_dict.keys():
                    if metric_name == "head_count":
                        best_scores_dict[metric_name].append(head_count)
                    else:
                        avg_metric = head_df[metric_name].mean()
                        if head_count == best_scores_df.shape[0]:
                            print("Iteration:{0} Head Count:{1} Average {2}:{3}".format(
                                iteration_id, head_count, metric_name, avg_metric))
                        best_scores_dict[metric_name].append(avg_metric)

            best_scores_summary_df = pd.DataFrame(best_scores_dict)
            best_scores_summary_df["run_id"] = self.parameterRunId
            best_scores_summary_df["iteration_id"] = iteration_id
            # Dump best scores summary results to DB.
            with db_engine.connect() as connection:
                best_scores_summary_df.to_sql("logs_table_cross_entropy_summary",
                                              con=connection, if_exists='append', index=False)
            best_acc_this_iteration = best_scores_summary_df["accuracy_test"].max()

            sorted_raw_result_metrics_df = sorted_raw_results_df[["accuracy_train",
                                                                  "mac_cost_train",
                                                                  "accuracy_test",
                                                                  "mac_cost_test",
                                                                  "score"]]
            sorted_raw_result_metrics_df["run_id"] = self.parameterRunId
            sorted_raw_result_metrics_df["iteration_id"] = iteration_id
            # Dump raw results to DB.
            with db_engine.connect() as connection:
                sorted_raw_result_metrics_df.to_sql("logs_table_cross_entropy",
                                                    con=connection, if_exists='append', index=False)

            if best_acc_this_iteration > best_acc_overall:
                best_acc_overall = best_acc_this_iteration
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= self.parameterMaximumIterationsWithoutImprovement:
                print("NO IMPROVEMENT. ENDING.")
                return

            # Update the Sigmoid Gaussian Mixture according the best thresholds.
            thresholds_selected = best_scores_df["threshold_vector"]
            thresholds_arr = [thresholds_selected.iloc[idx] for idx in range(thresholds_selected.shape[0])]
            thresholds_arr = np.stack(thresholds_arr, axis=0)
            self.distribution.fit(data=thresholds_arr)

    def fit_test(self):
        explanation = self.get_explanation_string()
        DbLogger.write_into_table(rows=[(self.parameterRunId, explanation)], table=DbLogger.runMetaData)

        train_outputs_cuda = self.multipathEvaluator.trainOutputs.move_to_torch(device=self.device)
        test_outputs_cuda = self.multipathEvaluator.testOutputs.move_to_torch(device=self.device)

        for iteration_id in range(self.parameterNumOfIterations):
            # Generate samples with the current distributions
            thresholds = self.distribution.sample(num_of_samples=self.parameterNumOfSamples)
            # Distribute sampled thresholds to processes
            sample_indices = np.arange(self.parameterNumOfSamples)
            sample_index_chunks = Utilities.divide_array_into_chunks(arr=sample_indices, count=self.numJobs)
            thresholds_per_process = {}
            for process_id in range(self.numJobs):
                index_chunk = sample_index_chunks[process_id]
                thresholds_per_process[process_id] = thresholds[index_chunk]
            assert np.array_equal(thresholds,
                                  np.concatenate([thresholds_per_process[pid]
                                                  for pid in range(self.numJobs)], axis=0))
            method_1_time = 0.0
            method_2_time = 0.0
            t_counter = tqdm(range(thresholds.shape[0]))
            # Call each process with their own set of thresholds
            for tid in t_counter:
                threshs = thresholds[tid]
                # self.multipathEvaluator.trainOutputs,
                # self.multipathEvaluator.testOutputs,

                time_profiler = TimeProfiler()
                time_profiler.start_measurement()
                train_res1 = MultipathEvaluator.evaluate_thresholds_static(
                    mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
                    thresholds=[threshs[0:2], threshs[2:6]],
                    outputs=self.multipathEvaluator.trainOutputs,
                    path_counts=self.parameterPathCounts)
                test_res1 = MultipathEvaluator.evaluate_thresholds_static(
                    mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
                    thresholds=[threshs[0:2], threshs[2:6]],
                    outputs=self.multipathEvaluator.testOutputs,
                    path_counts=self.parameterPathCounts)
                time_profiler.end_measurement()
                method_1_time += time_profiler.get_time()

                time_profiler.start_measurement()
                threshs_dict = {tpl: threshs[dim_id] for tpl, dim_id in self.thresholdMappingDict.items()}
                train_res2 = MultipathEvaluator.evaluate_thresholds_static_v2(
                    mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
                    thresholds=threshs_dict,
                    outputs=train_outputs_cuda,
                    path_counts=self.parameterPathCounts,
                    device=self.device)
                test_res2 = MultipathEvaluator.evaluate_thresholds_static_v2(
                    mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
                    thresholds=threshs_dict,
                    outputs=test_outputs_cuda,
                    path_counts=self.parameterPathCounts,
                    device=self.device)
                time_profiler.end_measurement()
                method_2_time += time_profiler.get_time()

                if not np.allclose(np.array(train_res1), torch.Tensor(train_res2).cpu().numpy()):
                    print("NOT CLOSE:{0}-{1}".format(np.array(train_res1),
                                                     torch.Tensor(train_res2).cpu().numpy()))
                    break
                if not np.allclose(np.array(test_res1), torch.Tensor(test_res2).cpu().numpy()):
                    print("NOT CLOSE:{0}-{1}".format(np.array(test_res1),
                                                     torch.Tensor(test_res2).cpu().numpy()))
                    break
                desc = "method_1_time:{0} method_2_time:{1}".format(method_1_time, method_2_time)
                t_counter.set_description(desc)
            break

    def calculate_score(self, accuracy, mac_cost):
        score = (1.0 - self.parameterMacLambda) * accuracy - self.parameterMacLambda * (mac_cost - 1.0)
        return score

    def analyze_thresholds(self, thresholds, train_outputs, test_outputs,
                           best_train_accuracies, best_test_accuracies, best_objective_scores,
                           verbose, iteration_id):
        time_profiler = TimeProfiler()
        iteration_times = []
        t_counter = tqdm(range(thresholds.shape[0]))

        raw_results_dict = {
            "accuracy_train": [],
            "mac_cost_train": [],
            "accuracy_test": [],
            "mac_cost_test": [],
            "score": [],
            "threshold_vector": []
        }
        # Call each process with their own set of thresholds
        for tid in t_counter:
            threshold_vector = thresholds[tid]
            threshs_dict = {tpl: threshold_vector[dim_id] for tpl, dim_id in self.thresholdMappingDict.items()}

            time_profiler.start_measurement()
            train_res = MultipathEvaluator.evaluate_thresholds_static_v2(
                mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
                thresholds=threshs_dict,
                outputs=train_outputs,
                path_counts=self.parameterPathCounts,
                device=self.device)
            test_res = MultipathEvaluator.evaluate_thresholds_static_v2(
                mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
                thresholds=threshs_dict,
                outputs=test_outputs,
                path_counts=self.parameterPathCounts,
                device=self.device)
            time_profiler.end_measurement()
            iteration_times.append(time_profiler.get_time())

            accuracy_train = train_res[0].cpu().numpy().item()
            mac_cost_train = train_res[1].cpu().numpy().item()
            accuracy_test = test_res[0].cpu().numpy().item()
            mac_cost_test = test_res[1].cpu().numpy().item()
            score = self.calculate_score(accuracy=accuracy_train, mac_cost=mac_cost_train)

            raw_results_dict["accuracy_train"].append(accuracy_train)
            raw_results_dict["mac_cost_train"].append(mac_cost_train)
            raw_results_dict["accuracy_test"].append(accuracy_test)
            raw_results_dict["mac_cost_test"].append(mac_cost_test)
            raw_results_dict["score"].append(score)
            raw_results_dict["threshold_vector"].append(threshold_vector)

            if (best_train_accuracies is not None) \
                    and (len(best_train_accuracies) == 0 or
                         best_train_accuracies["accuracy_train"] < accuracy_train):
                best_train_accuracies["accuracy_train"] = accuracy_train
                best_train_accuracies["mac_cost_train"] = mac_cost_train
                best_train_accuracies["accuracy_test"] = accuracy_test
                best_train_accuracies["mac_cost_test"] = mac_cost_test
                best_train_accuracies["score"] = score

            if (best_test_accuracies is not None) \
                    and (len(best_test_accuracies) == 0 or
                         best_test_accuracies["accuracy_test"] < accuracy_test):
                best_test_accuracies["accuracy_train"] = accuracy_train
                best_test_accuracies["mac_cost_train"] = mac_cost_train
                best_test_accuracies["accuracy_test"] = accuracy_test
                best_test_accuracies["mac_cost_test"] = mac_cost_test
                best_test_accuracies["score"] = score

            if (best_objective_scores is not None) \
                    and (len(best_objective_scores) == 0 or
                         best_objective_scores["score"] < score):
                best_objective_scores["accuracy_train"] = accuracy_train
                best_objective_scores["mac_cost_train"] = mac_cost_train
                best_objective_scores["accuracy_test"] = accuracy_test
                best_objective_scores["mac_cost_test"] = mac_cost_test
                best_objective_scores["score"] = score

            if verbose and (tid + 1) % 1000 == 0:
                print("************ Iteration {0} has completed {1} samples************".format(
                    iteration_id, (tid + 1)))
                print("Best Train Stats:")
                print(best_train_accuracies)

                print("Best Test Stats:")
                print(best_test_accuracies)

                print("Best Objective Score Stats:")
                print(best_objective_scores)

                print("average_iteration_time:{0}".format(np.mean(np.array(iteration_times))))
                iteration_times = []

        return raw_results_dict

    def histogram_analysis(self, path_to_saved_output, repeat_count, bin_size):
        if not os.path.isfile(path_to_saved_output):
            train_outputs_cuda = self.multipathEvaluator.trainOutputs.move_to_torch(device=self.device)
            test_outputs_cuda = self.multipathEvaluator.testOutputs.move_to_torch(device=self.device)
            data_frames = []
            for idx in range(repeat_count):
                print("Iteration {0}".format(idx))
                self.distribution.isFitted = False
                thresholds = self.distribution.sample(num_of_samples=self.parameterNumOfSamples)
                raw_results_dict = self.analyze_thresholds(thresholds=thresholds,
                                                           train_outputs=train_outputs_cuda,
                                                           test_outputs=test_outputs_cuda,
                                                           best_objective_scores=None,
                                                           best_test_accuracies=None,
                                                           best_train_accuracies=None,
                                                           iteration_id=None,
                                                           verbose=False)
                raw_results_df = pd.DataFrame(raw_results_dict)
                raw_results_df = raw_results_df[[col for col in raw_results_df.columns if col != "threshold_vector"]]
                data_frames.append(raw_results_df)
            all_results_df = pd.concat(data_frames, axis=0)
            sorted_all_results_df = all_results_df.sort_values(by=["score"], inplace=False, ascending=False)
            Utilities.pickle_save_to_file(path=path_to_saved_output, file_content=sorted_all_results_df)

        all_results_df = Utilities.pickle_load_from_file(path=path_to_saved_output)
        # Reorganize with respect to new mac measure.
        train_acc = all_results_df["accuracy_train"]
        train_mac = all_results_df["mac_cost_train"]
        score_vector = (1.0 - self.parameterMacLambda) * train_acc - self.parameterMacLambda * (train_mac - 1.0)
        all_results_df["score"] = score_vector

        sorted_all_results_df = all_results_df.sort_values(by=["score"], inplace=False, ascending=False)
        # db_engine = create_engine(url="sqlite:///" + DbLogger.log_db_path, echo=False)
        # with db_engine.connect() as connection:
        #     sorted_all_results_df.to_sql("random_thresholds", con=connection, if_exists='append', index=False)

        largest_test_score = 0.0
        for bin_id in range(0, sorted_all_results_df.shape[0], bin_size):
            bin_start = bin_id
            bin_end = bin_start + bin_size
            bin_df = sorted_all_results_df.iloc[bin_start: bin_end]
            mean_score = bin_df["score"].mean()
            max_score = bin_df["score"].max()
            min_score = bin_df["score"].min()
            mean_test_accuracy = bin_df["accuracy_test"].mean()
            mean_test_mac = bin_df["mac_cost_test"].mean()
            mean_test_mac2 = bin_df.sort_values(by=["accuracy_test"], inplace=False, ascending=False).head(
                int(bin_size / 2))["accuracy_test"].mean()
            if mean_test_accuracy > largest_test_score:
                largest_test_score = mean_test_accuracy
            print("Bin({0},{1}) Mean Score:{2} Mean Test Accuracy:{3} Mean Test Accuracy2:{4} Mean Test Mac:{5} "
                  "Max Score:{6} Min Score:{7}".format(bin_start, bin_end,
                                                       mean_score, mean_test_accuracy, mean_test_mac2,
                                                       mean_test_mac, max_score, min_score))

        print(largest_test_score)


