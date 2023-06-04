from cigt.cigt_ig_refactored import CigtIgHardRoutingX
import numpy as np
import torch
from torch import nn


class CigtMaskedRouting(CigtIgHardRoutingX):
    def __init__(self, run_id, model_definition, num_classes):
        super().__init__(run_id, model_definition, num_classes)

    def create_cigt_blocks(self):
        curr_input_shape = (self.batchSize, *self.inputDims)
        feature_edge_size = curr_input_shape[-1]
        # for cigt_layer_id, cigt_layer_info in self.blockParametersList:
        #     path_count_in_layer = self.pathCounts[cigt_layer_id]
        #     cigt_layer_blocks = nn.ModuleList()
        #     for path_id in range(path_count_in_layer):
        #         layers = []
        #         for inner_block_info in cigt_layer_info:
        #             block = BasicBlock(in_planes=inner_block_info["in_dimension"],
        #                                planes=inner_block_info["out_dimension"],
        #                                stride=inner_block_info["stride"])
        #             layers.append(block)
        #         block_obj = Sequential_ext(*layers)
        #         if self.useDataParallelism:
        #             block_obj = nn.DataParallel(block_obj)
        #         # block_obj.name = "block_{0}_{1}".format(cigt_layer_id, path_id)
        #         cigt_layer_blocks.append(block_obj)
        #     self.cigtLayers.append(cigt_layer_blocks)
        #     # Block end layers: Routing layers for inner layers, loss layer for the last one.
        #     if cigt_layer_id < len(self.blockParametersList) - 1:
        #         for inner_block_info in cigt_layer_info:
        #             feature_edge_size = int(feature_edge_size / inner_block_info["stride"])
        #         routing_layer = self.get_routing_layer(cigt_layer_id=cigt_layer_id,
        #                                                input_feature_map_size=feature_edge_size,
        #                                                input_feature_map_count=cigt_layer_info[-1]["out_dimension"])
        #         if self.useDataParallelism:
        #             routing_layer = nn.DataParallel(routing_layer)
        #         self.blockEndLayers.append(routing_layer)
        # # if cigt_layer_id == len(self.blockParametersList) - 1:
        # self.get_loss_layer()
