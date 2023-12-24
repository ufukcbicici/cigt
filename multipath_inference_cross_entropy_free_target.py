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
from cigt.multipath_inference_cross_entropy_v2 import MultipathInferenceCrossEntropyV2
from cigt.multivariate_sigmoid_mixture import MultivariateSigmoidMixture


class MultipathInferenceCrossEntropyFreeTarget(MultipathInferenceCrossEntropyV2):
    def __init__(self, run_id, path_counts, mac_lambda, max_probabilities, multipath_evaluator, n_iter,
                 quantile_interval, num_of_components, covariance_type, single_threshold_for_each_layer,
                 num_samples_each_iteration, maximum_iterations_without_improvement, num_jobs,
                 accuracy_target_normalized, path_to_saved_output):
        super().__init__(run_id, path_counts, mac_lambda, max_probabilities, multipath_evaluator, n_iter,
                         quantile_interval, num_of_components, covariance_type, single_threshold_for_each_layer,
                         num_samples_each_iteration, maximum_iterations_without_improvement, num_jobs)
        assert os.path.isfile(path_to_saved_output)
        self.parameterAccuracyTargetNormalized = accuracy_target_normalized
        # Analyze the simulation statistics
        all_results_df = Utilities.pickle_load_from_file(path=path_to_saved_output)
        scaler = StandardScaler()
        scaler.fit(X=np.expand_dims(all_results_df["accuracy_train"].to_numpy(), axis=-1))
        self.accuracyTargetActual = norm.ppf(self.parameterAccuracyTargetNormalized,
                                             loc=scaler.mean_[0], scale=scaler.scale_[0])
        self.accuracyTargetMin = all_results_df["accuracy_train"].min()
        self.accuracyTargetMax = all_results_df["accuracy_train"].max()
        self.accuracyInterval = self.accuracyTargetMax - self.accuracyTargetMin

    def calculate_score(self, accuracy, mac_cost):
        accuracy_score = 1.0 - np.power(
            np.abs(self.accuracyTargetActual - accuracy) / self.accuracyInterval, 0.5)
        # accuracy_score = (self.accuracyInterval - accuracy_distance) / self.accuracyInterval
        score = (1.0 - self.parameterMacLambda) * accuracy_score - self.parameterMacLambda * (mac_cost - 1.0)
        return score
