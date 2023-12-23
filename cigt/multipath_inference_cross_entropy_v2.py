from multiprocessing import Process

import numpy as np
import inspect

import torch.cuda
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
                 quantile,
                 num_of_components,
                 covariance_type,
                 single_threshold_for_each_layer,
                 num_samples_each_iteration,
                 num_jobs
                 ):
        super().__init__()
        self.parameterRunId = run_id
        self.parameterPathCounts = path_counts
        self.parameterMaxProbabilities = max_probabilities
        self.parameterMacLambda = mac_lambda
        self.parameterNumOfIterations = n_iter
        self.parameterQuantileSamples = quantile
        self.parameterNumOfComponents = num_of_components
        self.parameterCovarianceType = covariance_type
        self.parameterSingleThresholdForEachLayer = single_threshold_for_each_layer
        self.parameterNumOfSamples = num_samples_each_iteration

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
        time_profiler = TimeProfiler()
        train_outputs_cuda = self.multipathEvaluator.trainOutputs.move_to_torch(device=self.device)
        test_outputs_cuda = self.multipathEvaluator.testOutputs.move_to_torch(device=self.device)

        for iteration_id in range(self.parameterNumOfIterations):
            # Generate samples with the current distributions
            thresholds = self.distribution.sample(num_of_samples=self.parameterNumOfSamples)
            iteration_times = []
            t_counter = tqdm(range(thresholds.shape[0]))
            # Call each process with their own set of thresholds
            for tid in t_counter:
                threshold_vector = thresholds[tid]
                threshs_dict = {tpl: threshold_vector[dim_id] for tpl, dim_id in self.thresholdMappingDict.items()}

                time_profiler.start_measurement()
                train_res = MultipathEvaluator.evaluate_thresholds_static_v2(
                    mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
                    thresholds=threshs_dict,
                    outputs=train_outputs_cuda,
                    path_counts=self.parameterPathCounts,
                    device=self.device)
                test_res = MultipathEvaluator.evaluate_thresholds_static_v2(
                    mac_counts_per_block=self.multipathEvaluator.macCountsPerBlock,
                    thresholds=threshs_dict,
                    outputs=test_outputs_cuda,
                    path_counts=self.parameterPathCounts,
                    device=self.device)
                time_profiler.end_measurement()
                iteration_times.append(time_profiler.get_time())

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
                                                     np.array(torch.Tensor(train_res2).cpu().numpy())))
                    break
                if not np.allclose(np.array(test_res1), torch.Tensor(test_res2).cpu().numpy()):
                    print("NOT CLOSE:{0}-{1}".format(np.array(test_res1),
                                                     np.array(torch.Tensor(test_res2).cpu().numpy())))
                    break
                desc = "method_1_time:{0} method_2_time:{1}".format(method_1_time, method_2_time)
                t_counter.set_description(desc)
            break

            # list_of_processes = []
            # for process_id in range(self.numJobs):
            #     process = Process(target=MultipathInferenceCrossEntropy.evaluate_sample_thresholds,
            #                       args=(process_id,
            #                             iteration_id,
            #                             self.model.pathCounts,
            #                             thresholds_per_process[process_id],
            #                             self.multipathEvaluator.trainOutputs,
            #                             self.multipathEvaluator.testOutputs,
            #                             self.multipathEvaluator.macCountsPerBlock,
            #                             self.macLambda))
            #     list_of_processes.append(process)
            #     process.start()
            #
            # for process in list_of_processes:
            #     process.join()

            print("X")
