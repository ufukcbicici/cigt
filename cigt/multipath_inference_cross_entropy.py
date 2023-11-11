import os.path
from collections import deque
from copy import deepcopy
from multiprocessing import Process

import numpy as np
import pandas as pd
from sqlalchemy import insert, create_engine, update, distinct, func, asc
from sqlalchemy import select, Table, and_, MetaData
import torch
import matplotlib.pyplot as plt
from auxillary.bayesian_optimizer import BayesianOptimizer
from auxillary.db_logger import DbLogger
from auxillary.softmax_temperature_optimizer import SoftmaxTemperatureOptimizer
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.multipath_evaluator import MultipathEvaluator
from cigt.sigmoid_gaussian_mixture import SigmoidGaussianMixture
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
from tqdm import tqdm


class MultipathInferenceCrossEntropy(object):
    def __init__(self, data_root_path,
                 model, multipath_evaluator, n_iter, mac_lambda, max_probabilities, distribution_type,
                 quantile, num_of_components, num_samples_each_iteration, num_jobs):
        super().__init__()
        self.model = model
        self.multipathEvaluator = multipath_evaluator
        self.dataRootPath = data_root_path
        self.maxProbabilities = max_probabilities
        self.macLambda = mac_lambda
        self.bestTrainScores = {}
        self.bestTestScores = {}
        self.bestObjectiveScores = {}
        self.maxNumOfIterations = n_iter
        self.iterationCount = 0
        self.quantile = quantile
        self.distributions = {}
        self.distributionType = distribution_type
        self.numOfComponents = num_of_components
        self.numOfSamplesEachIteration = num_samples_each_iteration
        self.numJobs = num_jobs
        for lid in range(len(self.model.pathCounts) - 1):
            for pid in range(self.model.pathCounts[lid + 1]):
                distribution = SigmoidGaussianMixture(
                    num_of_components=self.numOfComponents,
                    low_end=0.0,
                    high_end=self.maxProbabilities[lid])
                samples = distribution.sample(num_of_samples=10000)
                distribution.fit(data=samples)
                self.distributions[(lid, pid)] = distribution

        assert self.distributionType in {"Gaussian"}

    def generate_sample(self, num_samples):
        thresholds = {}
        for lid in range(len(self.model.pathCounts) - 1):
            for pid in range(self.model.pathCounts[lid + 1]):
                sample_threshold = self.distributions[(lid, pid)].sample(num_of_samples=num_samples)
                thresholds[(lid, pid)] = sample_threshold
        return thresholds

    @staticmethod
    def evaluate_sample_thresholds(process_id,
                                   iteration_id,
                                   path_counts,
                                   sampled_thresholds,
                                   model_outputs_train,
                                   model_outputs_test,
                                   mac_values,
                                   mac_lambda):
        results_dict = {
            "accuracy_train": [],
            "mac_cost_train": [],
            "accuracy_test": [],
            "mac_cost_test": [],
            "score": []
        }
        for lid in range(len(path_counts) - 1):
            for pid in range(path_counts[lid + 1]):
                results_dict["threshold_({0},{1})".format(lid, pid)] = []

        best_train_scores = {}
        best_test_scores = {}
        best_objective_scores = {}

        print("********* Process {0} is starting. *********".format(process_id))
        sample_counts = set([threshold_array.shape[0] for threshold_array in sampled_thresholds.values()])
        assert len(sample_counts) == 1
        sample_count = list(sample_counts)[0]
        for sample_id in range(sample_count):
            threshold_sample = []
            for lid in range(len(path_counts) - 1):
                threshold_sample.append([])
                for pid in range(path_counts[lid + 1]):
                    threshold_sample[-1].append(sampled_thresholds[(lid, pid)][sample_id])
            accuracy_train, mac_cost_train = MultipathEvaluator.evaluate_thresholds_static(
                thresholds=threshold_sample,
                outputs=model_outputs_train,
                mac_counts_per_block=mac_values,
                path_counts=path_counts)
            accuracy_test, mac_cost_test = MultipathEvaluator.evaluate_thresholds_static(
                thresholds=threshold_sample,
                outputs=model_outputs_test,
                mac_counts_per_block=mac_values,
                path_counts=path_counts)
            score = mac_lambda * accuracy_train - (1.0 - mac_lambda) * (mac_cost_train - 1.0)
            results_dict["accuracy_train"].append(accuracy_train)
            results_dict["mac_cost_train"].append(mac_cost_train)
            results_dict["accuracy_test"].append(accuracy_test)
            results_dict["mac_cost_test"].append(mac_cost_test)
            results_dict["score"].append(score)
            for lid in range(len(path_counts) - 1):
                for pid in range(path_counts[lid + 1]):
                    results_dict["threshold_({0},{1})".format(lid, pid)].append(threshold_sample[lid][pid])

            if len(best_train_scores) == 0 or best_train_scores["accuracy_train"] < accuracy_train:
                best_train_scores["accuracy_train"] = accuracy_train
                best_train_scores["mac_cost_train"] = mac_cost_train
                best_train_scores["accuracy_test"] = accuracy_test
                best_train_scores["mac_cost_test"] = mac_cost_test
                best_train_scores["score"] = score

            if len(best_test_scores) == 0 or best_test_scores["accuracy_test"] < accuracy_test:
                best_test_scores["accuracy_train"] = accuracy_train
                best_test_scores["mac_cost_train"] = mac_cost_train
                best_test_scores["accuracy_test"] = accuracy_test
                best_test_scores["mac_cost_test"] = mac_cost_test
                best_test_scores["score"] = score

            if len(best_objective_scores) == 0 or best_objective_scores["score"] < score:
                best_objective_scores["accuracy_train"] = accuracy_train
                best_objective_scores["mac_cost_train"] = mac_cost_train
                best_objective_scores["accuracy_test"] = accuracy_test
                best_objective_scores["mac_cost_test"] = mac_cost_test
                best_objective_scores["score"] = score

            if (sample_id + 1) % 100 == 0:
                print("************ Process {0} has completed {1} samples************".format(
                    process_id, (sample_id + 1)
                ))
                print("Best Train Stats:")
                print(best_train_scores)

                print("Best Test Stats:")
                print(best_test_scores)

                print("Best Objective Score Stats:")
                print(best_objective_scores)

        df = pd.DataFrame(results_dict)
        csv_file_path = "results_process_{0}_iteration_{1}.csv".format(process_id, iteration_id)
        if os.path.isfile(csv_file_path):
            os.remove(csv_file_path)
        df.to_csv(path_or_buf=csv_file_path, index=False)

    def fit(self):
        run_id = DbLogger.get_run_id()
        config = "Component Count:{0}\n".format(self.numOfComponents) \
                 + "Max Probabilities:{0}\n".format(self.maxProbabilities) \
                 + "Quantile:{0}\n".format(self.quantile) \
                 + "Sample Count:{0}\n".format(self.numOfSamplesEachIteration) \
                 + "Mac Lambda:{0}\n".format(self.macLambda)
        DbLogger.write_into_table(rows=[(run_id, config)], table=DbLogger.runMetaData)
        db_engine = create_engine(url="sqlite:///" + DbLogger.log_db_path, echo=False)
        # with db_engine.connect() as connection:
        #     meta_df = pd.read_sql(sql="SELECT * FROM run_meta_data", con=connection)

        for iteration_id in range(self.maxNumOfIterations):
            # Generate samples with the current distributions
            all_sampled_thresholds_dict = self.generate_sample(num_samples=self.numOfSamplesEachIteration)
            # Distribute sampled thresholds to processes
            sample_indices = np.arange(self.numOfSamplesEachIteration)
            sample_index_chunks = Utilities.divide_array_into_chunks(arr=sample_indices, count=self.numJobs)
            thresholds_per_process = {}
            for process_id in range(self.numJobs):
                index_chunk = sample_index_chunks[process_id]
                thresholds_per_process[process_id] = {}
                for k, v in all_sampled_thresholds_dict.items():
                    thresholds_per_process[process_id][k] = v[index_chunk]
            # Call each process with their own set of thresholds
            list_of_processes = []
            for process_id in range(self.numJobs):
                # def evaluate_sample_thresholds(process_id,
                #                                iteration_id,
                #                                path_counts,
                #                                sampled_thresholds,
                #                                model_outputs_train,
                #                                model_outputs_test,
                #                                mac_values,
                #                                mac_lambda):
                process = Process(target=MultipathInferenceCrossEntropy.evaluate_sample_thresholds,
                                  args=(process_id,
                                        iteration_id,
                                        self.model.pathCounts,
                                        thresholds_per_process[process_id],
                                        self.multipathEvaluator.trainOutputs,
                                        self.multipathEvaluator.testOutputs,
                                        self.multipathEvaluator.macCountsPerBlock,
                                        self.macLambda))
                list_of_processes.append(process)
                process.start()

            for process in list_of_processes:
                process.join()

            # Read the results, sort by the scores.
            df_list = []
            for process_id in range(self.numJobs):
                df = pd.read_csv("results_process_{0}_iteration_{1}.csv".format(process_id, iteration_id))
                df_list.append(df)
            all_df = pd.concat(df_list, axis=0)
            all_df = all_df.sort_values(by=["score"], inplace=False, ascending=False)
            best_samples_count = int(all_df.shape[0] * self.quantile)
            best_scores_df = all_df.iloc[0:best_samples_count]
            average_df = best_scores_df.mean()
            print("Iteration {0} best scores:{1}".format(iteration_id, average_df))

            all_df["run_id"] = run_id
            all_df["iteration_id"] = iteration_id
            # Dump results to DB.
            with db_engine.connect() as connection:
                all_df.to_sql("cross_entropy_results", con=connection, if_exists='append')

            # Update the distributions according the best thresholds.
            for lid in range(len(self.model.pathCounts) - 1):
                for pid in range(self.model.pathCounts[lid + 1]):
                    thresholds = best_scores_df["threshold_({0},{1})".format(lid, pid)].to_numpy()
                    self.distributions[(lid, pid)].fit(data=thresholds)


