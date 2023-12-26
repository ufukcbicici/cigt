import os.path
from multiprocessing import Process

import numpy as np
import inspect
import pandas as pd
from scipy.stats import norm
import torch.cuda
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from tqdm import tqdm
from auxillary.db_logger import DbLogger
from auxillary.time_profiler import TimeProfiler
from auxillary.utilities import Utilities
from cigt.multipath_evaluator import MultipathEvaluator
from cigt.multipath_inference_cross_entropy import MultipathInferenceCrossEntropy
from cigt.multipath_inference_cross_entropy_v2 import MultipathInferenceCrossEntropyV2
from cigt.multivariate_sigmoid_mixture import MultivariateSigmoidMixture


class MultipathInferenceCrossEntropyValidationScoring(MultipathInferenceCrossEntropyV2):
    def __init__(self, run_id, path_counts, validation_ratio, mac_lambda, max_probabilities, multipath_evaluator,
                 n_iter, quantile_interval, num_of_components, covariance_type, single_threshold_for_each_layer,
                 num_samples_each_iteration, maximum_iterations_without_improvement, num_jobs):
        super().__init__(run_id, path_counts, mac_lambda, max_probabilities, multipath_evaluator, n_iter,
                         quantile_interval, num_of_components, covariance_type, single_threshold_for_each_layer,
                         num_samples_each_iteration, maximum_iterations_without_improvement, num_jobs)
        self.parameterValidationRatio = validation_ratio
        self.trainOutputs, self.testOutputs = \
            multipath_evaluator.testOutputs.separate_into_validation_set(
                ratio=self.parameterValidationRatio)
        self.multipathEvaluator.optimize_routing_temperatures(complete_output=self.trainOutputs)
        self.testOutputs.optimalTemperatures = self.trainOutputs.optimalTemperatures
        self.trainOutputs = self.trainOutputs.move_to_torch(self.device)
        self.testOutputs = self.testOutputs.move_to_torch(self.device)

        threshold_vector = np.array([1.0] * 10)
        threshs_dict = {tpl: threshold_vector[dim_id] for tpl, dim_id in self.thresholdMappingDict.items()}

        train_res = MultipathEvaluator.evaluate_thresholds_static_v2(
            mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
            thresholds=threshs_dict,
            outputs=self.trainOutputs,
            path_counts=self.parameterPathCounts,
            device=self.device)
        test_res = MultipathEvaluator.evaluate_thresholds_static_v2(
            mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
            thresholds=threshs_dict,
            outputs=self.testOutputs,
            path_counts=self.parameterPathCounts,
            device=self.device)
        print("train_res:{0}".format(train_res))
        print("test_res:{0}".format(test_res))

        # self.trainOutputs, self.validationOutputs = \
        #     train_outputs.separate_into_validation_set(ratio=self.parameterValidationRatio)
        # self.multipathEvaluator.optimize_routing_temperatures(complete_output=self.trainOutputs)
        # self.testOutputs.optimalTemperatures = self.trainOutputs.optimalTemperatures
        # self.validationOutputs.optimalTemperatures = self.trainOutputs.optimalTemperatures
        #
        # self.trainOutputs = self.trainOutputs.move_to_torch(self.device)
        # self.testOutputs = self.testOutputs.move_to_torch(self.device)
        # self.validationOutputs = self.validationOutputs.move_to_torch(self.device)
    #
    # def analyze_thresholds(self, thresholds,
    #                        train_outputs, test_outputs,
    #                        best_train_accuracies, best_test_accuracies, best_objective_scores,
    #                        verbose, iteration_id, validation_outputs=None):
    #     time_profiler = TimeProfiler()
    #     iteration_times = []
    #     t_counter = tqdm(range(thresholds.shape[0]))
    #
    #     raw_results_dict = {
    #         "accuracy_train": [],
    #         "mac_cost_train": [],
    #         "accuracy_test": [],
    #         "mac_cost_test": [],
    #         "accuracy_validation": [],
    #         "mac_cost_validation": [],
    #         "score": [],
    #         "threshold_vector": []
    #     }
    #     # Call each process with their own set of thresholds
    #     for tid in t_counter:
    #         threshold_vector = thresholds[tid]
    #         threshs_dict = {tpl: threshold_vector[dim_id] for tpl, dim_id in self.thresholdMappingDict.items()}
    #
    #         time_profiler.start_measurement()
    #         train_res = MultipathEvaluator.evaluate_thresholds_static_v2(
    #             mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
    #             thresholds=threshs_dict,
    #             outputs=train_outputs,
    #             path_counts=self.parameterPathCounts,
    #             device=self.device)
    #         test_res = MultipathEvaluator.evaluate_thresholds_static_v2(
    #             mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
    #             thresholds=threshs_dict,
    #             outputs=test_outputs,
    #             path_counts=self.parameterPathCounts,
    #             device=self.device)
    #         val_res = MultipathEvaluator.evaluate_thresholds_static_v2(
    #             mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
    #             thresholds=threshs_dict,
    #             outputs=validation_outputs,
    #             path_counts=self.parameterPathCounts,
    #             device=self.device)
    #         time_profiler.end_measurement()
    #         iteration_times.append(time_profiler.get_time())
    #
    #         accuracy_train = train_res[0].cpu().numpy().item()
    #         mac_cost_train = train_res[1].cpu().numpy().item()
    #         accuracy_test = test_res[0].cpu().numpy().item()
    #         mac_cost_test = test_res[1].cpu().numpy().item()
    #         accuracy_validation = val_res[0].cpu().numpy().item()
    #         mac_cost_validation = val_res[1].cpu().numpy().item()
    #
    #         score = self.calculate_score(accuracy=accuracy_train, mac_cost=mac_cost_train)
    #
    #         raw_results_dict["accuracy_train"].append(accuracy_train)
    #         raw_results_dict["mac_cost_train"].append(mac_cost_train)
    #         raw_results_dict["accuracy_test"].append(accuracy_test)
    #         raw_results_dict["mac_cost_test"].append(mac_cost_test)
    #         raw_results_dict["accuracy_validation"].append(accuracy_validation)
    #         raw_results_dict["mac_cost_validation"].append(mac_cost_validation)
    #         raw_results_dict["score"].append(score)
    #         raw_results_dict["threshold_vector"].append(threshold_vector)
    #
    #         if (best_train_accuracies is not None) \
    #                 and (len(best_train_accuracies) == 0 or
    #                      best_train_accuracies["accuracy_train"] < accuracy_train):
    #             best_train_accuracies["accuracy_train"] = accuracy_train
    #             best_train_accuracies["mac_cost_train"] = mac_cost_train
    #             best_train_accuracies["accuracy_test"] = accuracy_test
    #             best_train_accuracies["mac_cost_test"] = mac_cost_test
    #             best_train_accuracies["score"] = score
    #
    #         if (best_test_accuracies is not None) \
    #                 and (len(best_test_accuracies) == 0 or
    #                      best_test_accuracies["accuracy_test"] < accuracy_test):
    #             best_test_accuracies["accuracy_train"] = accuracy_train
    #             best_test_accuracies["mac_cost_train"] = mac_cost_train
    #             best_test_accuracies["accuracy_test"] = accuracy_test
    #             best_test_accuracies["mac_cost_test"] = mac_cost_test
    #             best_test_accuracies["score"] = score
    #
    #         if (best_objective_scores is not None) \
    #                 and (len(best_objective_scores) == 0 or
    #                      best_objective_scores["score"] < score):
    #             best_objective_scores["accuracy_train"] = accuracy_train
    #             best_objective_scores["mac_cost_train"] = mac_cost_train
    #             best_objective_scores["accuracy_test"] = accuracy_test
    #             best_objective_scores["mac_cost_test"] = mac_cost_test
    #             best_objective_scores["score"] = score
    #
    #         if verbose and (tid + 1) % 1000 == 0:
    #             print("************ Iteration {0} has completed {1} samples************".format(
    #                 iteration_id, (tid + 1)))
    #             print("Best Train Stats:")
    #             print(best_train_accuracies)
    #
    #             print("Best Test Stats:")
    #             print(best_test_accuracies)
    #
    #             print("Best Objective Score Stats:")
    #             print(best_objective_scores)
    #
    #             print("average_iteration_time:{0}".format(np.mean(np.array(iteration_times))))
    #             iteration_times = []
    #
    #     return raw_results_dict

    # def histogram_analysis(self, path_to_saved_output, repeat_count, bin_size):
    #     if not os.path.isfile(path_to_saved_output):
    #         data_frames = []
    #         for idx in range(repeat_count):
    #             print("Iteration {0}".format(idx))
    #             self.distribution.isFitted = False
    #             thresholds = self.distribution.sample(num_of_samples=self.parameterNumOfSamples)
    #             raw_results_dict = self.analyze_thresholds(thresholds=thresholds,
    #                                                        train_outputs=self.trainOutputs,
    #                                                        test_outputs=self.testOutputs,
    #                                                        best_objective_scores=None,
    #                                                        best_test_accuracies=None,
    #                                                        best_train_accuracies=None,
    #                                                        iteration_id=None,
    #                                                        verbose=False)
    #             raw_results_df = pd.DataFrame(raw_results_dict)
    #             raw_results_df = raw_results_df[[col for col in raw_results_df.columns if col != "threshold_vector"]]
    #             data_frames.append(raw_results_df)
    #         all_results_df = pd.concat(data_frames, axis=0)
    #         sorted_all_results_df = all_results_df.sort_values(by=["score"], inplace=False, ascending=False)
    #         Utilities.pickle_save_to_file(path=path_to_saved_output, file_content=sorted_all_results_df)
    #
    #     all_results_df = Utilities.pickle_load_from_file(path=path_to_saved_output)
    #     # Reorganize with respect to new mac measure.
    #     train_acc = all_results_df["accuracy_train"]
    #     train_mac = all_results_df["mac_cost_train"]
    #     score_vector = (1.0 - self.parameterMacLambda) * train_acc - self.parameterMacLambda * (train_mac - 1.0)
    #     all_results_df["score"] = score_vector
    #
    #     sorted_all_results_df = all_results_df.sort_values(by=["score"], inplace=False, ascending=False)
    #     # db_engine = create_engine(url="sqlite:///" + DbLogger.log_db_path, echo=False)
    #     # with db_engine.connect() as connection:
    #     #     sorted_all_results_df.to_sql("random_thresholds", con=connection, if_exists='append', index=False)
    #
    #     largest_test_score = 0.0
    #     for bin_id in range(0, sorted_all_results_df.shape[0], bin_size):
    #         bin_start = bin_id
    #         bin_end = bin_start + bin_size
    #         bin_df = sorted_all_results_df.iloc[bin_start: bin_end]
    #         mean_score = bin_df["score"].mean()
    #         max_score = bin_df["score"].max()
    #         min_score = bin_df["score"].min()
    #         mean_test_accuracy = bin_df["accuracy_test"].mean()
    #         mean_test_mac = bin_df["mac_cost_test"].mean()
    #         mean_train_accuracy = bin_df["accuracy_train"].mean()
    #         mean_train_mac = bin_df["mac_cost_train"].mean()
    #         mean_validation_accuracy = bin_df["accuracy_validation"].mean()
    #         mean_mac_accuracy = bin_df["mac_cost_validation"].mean()
    #
    #         mean_test_accuracy2 = bin_df.sort_values(by=["accuracy_test"],
    #                                                  inplace=False, ascending=False).head(
    #             int(bin_size / 2))["accuracy_test"].mean()
    #         text = "Bin({0},{1})".format(bin_start, bin_end)
    #         text += " Mean Score:{0}".format(mean_score)
    #         text += " Mean Test Accuracy:{0}".format(mean_test_accuracy)
    #         text += " Mean Test Accuracy2:{0}".format(mean_test_accuracy2)
    #         text += " Mean Test Mac:{0}".format(mean_test_mac)
    #
    #         text += " Mean Train Accuracy:{0}".format(mean_train_accuracy)
    #         text += " Mean Train Mac:{0}".format(mean_train_mac)
    #
    #         text += " Mean Validation Accuracy:{0}".format(mean_validation_accuracy)
    #         text += " Mean Validation Mac:{0}".format(mean_mac_accuracy)
    #
    #         if mean_test_accuracy > largest_test_score:
    #             largest_test_score = mean_test_accuracy
    #         print(text)
    #
    #     print(largest_test_score)

    def histogram_analysis(self, path_to_saved_output, repeat_count, bin_size):
        if not os.path.isfile(path_to_saved_output):
            data_frames = []
            for idx in range(repeat_count):
                print("Iteration {0}".format(idx))
                self.distribution.isFitted = False
                thresholds = self.distribution.sample(num_of_samples=self.parameterNumOfSamples)
                raw_results_dict = self.analyze_thresholds(thresholds=thresholds,
                                                           train_outputs=self.trainOutputs,
                                                           test_outputs=self.testOutputs,
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


