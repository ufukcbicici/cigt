import numpy as np
import torch
from auxillary.bayesian_optimizer import BayesianOptimizer
from auxillary.db_logger import DbLogger
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.resnet_cigt_constants import ResnetCigtConstants


class MultiplePathBayesianOptimizer(BayesianOptimizer):
    def __init__(self, checkpoint_path, data_root_path, dataset, xi, init_points, n_iter):
        super().__init__(xi, init_points, n_iter)
        self.checkpointPath = checkpoint_path
        self.dataRootPath = data_root_path
        self.dataset = dataset
        # Load the trained model
        self.runId = DbLogger.get_run_id()
        ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
        self.model = CigtIgGatherScatterImplementation(
            run_id=self.runId,
            model_definition="Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
            num_classes=10)

        explanation = self.model.get_explanation_string()
        checkpoint = torch.load(self.checkpointPath, map_location="cpu")
        self.model.load_state_dict(state_dict=checkpoint["model_state_dict"])
        print("X")

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