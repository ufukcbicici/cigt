from cigt.cigt_soft_routing import CigtSoftRouting
from cigt.routing_layers.constant_routing_layer import ConstantRoutingLayer


class CigtConstantRoutingWeights(CigtSoftRouting):
    def __init__(self, run_id, model_definition):
        super().__init__(run_id, model_definition)

    def get_routing_layer(self, cigt_layer_id, input_feature_map_size, input_feature_map_count):
        routing_layer = ConstantRoutingLayer(
            feature_dim=self.decisionDimensions[cigt_layer_id],
            avg_pool_stride=self.decisionAveragePoolingStrides[cigt_layer_id],
            path_count=self.pathCounts[cigt_layer_id + 1],
            class_count=self.numClasses,
            input_feature_map_size=input_feature_map_size,
            input_feature_map_count=input_feature_map_count,
            device=self.device)
        return routing_layer
