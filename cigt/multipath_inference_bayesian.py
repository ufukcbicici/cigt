import os.path
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
from auxillary.bayesian_optimizer import BayesianOptimizer
from auxillary.softmax_temperature_optimizer import SoftmaxTemperatureOptimizer
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
from tqdm import tqdm


class MultiplePathBayesianOptimizer(BayesianOptimizer):
    def __init__(self, data_root_path,
                 model, multipath_evaluator, init_points, n_iter, mac_lambda, max_probabilities):
        super().__init__(init_points, n_iter)
        self.model = model
        self.multipathEvaluator = multipath_evaluator
        self.dataRootPath = data_root_path
        self.maxProbabilities = max_probabilities
        self.macLambda = mac_lambda
        self.bestTrainScores = {}
        self.bestTestScores = {}
        self.bestObjectiveScores = {}
        self.iterationCount = 0
        self.optimization_bounds_continuous = {}
        for lid in range(len(self.model.pathCounts) - 1):
            for pid in range(self.model.pathCounts[lid + 1]):
                self.optimization_bounds_continuous["threshold_{0},{1}".format(lid, pid)] = (0.0,
                                                                                             self.maxProbabilities[lid])

    def cost_function(self, **kwargs):
        # lr_initial_rate,
        # hyperbolic_exponent):
        thresholds = []
        for lid in range(len(self.model.pathCounts) - 1):
            thresholds.append([])
            for pid in range(self.model.pathCounts[lid + 1]):
                thresholds[lid].append(kwargs["threshold_{0},{1}".format(lid, pid)])

        accuracy_train, mac_cost_train = self.multipathEvaluator.evaluate_thresholds_array_based(
            thresholds=thresholds, outputs=self.multipathEvaluator.trainOutputs)
        accuracy_test, mac_cost_test = self.multipathEvaluator.evaluate_thresholds_array_based(
            thresholds=thresholds, outputs=self.multipathEvaluator.testOutputs)
        score = self.macLambda * accuracy_train - (1.0 - self.macLambda) * (mac_cost_train - 1.0)
        print("**********Iteration {0}**********".format(self.iterationCount))
        print("Accuracy Train:{0}".format(accuracy_train))
        print("Mac Train:{0}".format(mac_cost_train))
        print("Accuracy Test:{0}".format(accuracy_test))
        print("Mac Test:{0}".format(mac_cost_test))
        print("Score:{0}".format(score))

        if len(self.bestTrainScores) == 0 or self.bestTrainScores["accuracy_train"] < accuracy_train:
            self.bestTrainScores["accuracy_train"] = accuracy_train
            self.bestTrainScores["mac_cost_train"] = mac_cost_train
            self.bestTrainScores["accuracy_test"] = accuracy_test
            self.bestTrainScores["mac_cost_test"] = mac_cost_test
            self.bestTrainScores["score"] = score

        if len(self.bestTestScores) == 0 or self.bestTestScores["accuracy_test"] < accuracy_test:
            self.bestTestScores["accuracy_train"] = accuracy_train
            self.bestTestScores["mac_cost_train"] = mac_cost_train
            self.bestTestScores["accuracy_test"] = accuracy_test
            self.bestTestScores["mac_cost_test"] = mac_cost_test
            self.bestTestScores["score"] = score

        if len(self.bestObjectiveScores) == 0 or self.bestObjectiveScores["score"] < score:
            self.bestObjectiveScores["accuracy_train"] = accuracy_train
            self.bestObjectiveScores["mac_cost_train"] = mac_cost_train
            self.bestObjectiveScores["accuracy_test"] = accuracy_test
            self.bestObjectiveScores["mac_cost_test"] = mac_cost_test
            self.bestObjectiveScores["score"] = score

        print("Best Train Stats:")
        print(self.bestTrainScores)

        print("Best Test Stats:")
        print(self.bestTestScores)

        print("Best Objective Score Stats:")
        print(self.bestObjectiveScores)
        print("**********Iteration {0}**********".format(self.iterationCount))

        self.iterationCount += 1

        return score
