from cigt.cigt_soft_routing import CigtSoftRouting
from cigt.routing_layers.ideal_routing_layer import IdealRoutingLayer


class CigtIdealRouting(CigtSoftRouting):
    def __init__(self, run_id, model_definition):
        self.routes = [
            ({0, 1, 8, 9}, {2, 3, 4, 5, 6, 7}),
            ({0, 8}, {1, 9}, {2, 3, 6}, {4, 5, 7})]
        super().__init__(run_id, model_definition)

    def get_routing_layer(self, cigt_layer_id, input_feature_map_size, input_feature_map_count):
        routing_layer = IdealRoutingLayer(
            ideal_routes=self.routes[cigt_layer_id],
            path_count=self.pathCounts[cigt_layer_id + 1],
            class_count=self.numClasses,
            device=self.device)
        return routing_layer
