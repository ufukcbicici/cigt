from cigt.cigt_ig_refactored import CigtIgHardRoutingX
import numpy as np
import torch
from torch import nn

from convnet_aig import BasicBlock


class CigtMaskedRouting(CigtIgHardRoutingX):
    def __init__(self, run_id, model_definition, num_classes):
        super().__init__(run_id, model_definition, num_classes)

    def create_cigt_blocks(self):
        curr_input_shape = (self.batchSize, *self.inputDims)
        feature_edge_size = curr_input_shape[-1]
        for cigt_layer_id, cigt_layer_info in self.blockParametersList:
            path_count_in_layer = self.pathCounts[cigt_layer_id]
            cigt_layer_blocks = nn.ModuleList()
            layers = []
            for block_id, inner_block_info in enumerate(cigt_layer_info):
                prev_route_count = self.pathCounts[cigt_layer_id - 1] if cigt_layer_id > 0 else 1
                curr_route_count = self.pathCounts[cigt_layer_id]
                if block_id == 0:
                    in_channel_count = inner_block_info["in_dimension"] * prev_route_count
                else:
                    in_channel_count = inner_block_info["in_dimension"] * curr_route_count
                out_channel_count = inner_block_info["out_dimension"] * curr_route_count
                block = BasicBlock(in_planes=in_channel_count, planes=out_channel_count,
                                   stride=inner_block_info["stride"])
                print("X")

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
