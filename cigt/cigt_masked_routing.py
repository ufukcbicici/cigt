from collections import OrderedDict

from cigt.cigt_ig_refactored import CigtIgHardRoutingX
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer


class MaskedSequential(nn.Module):
    """A Sequential container extended to also propagate the gating information
    that is needed in the target rate loss.
    """

    def __init__(self, *args):
        super(MaskedSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input_, routing_matrix):
        x = input_
        for i, module in enumerate(self._modules.values()):
            x = module(x, routing_matrix)
            # print(x.max())
        return x


class MaskedConvBn(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, planes,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.maskConv = CigtMaskingLayer()

        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, routing_matrix):
        x = self.conv(x)
        x = self.maskConv(x, routing_matrix)
        x = self.bn(x)
        return x


class MaskedConvMaskedBn(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, planes,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.maskConv = CigtMaskingLayer()

        self.bn = nn.BatchNorm2d(planes)
        self.maskBn = CigtMaskingLayer()

    def forward(self, x, routing_matrix):
        x = self.conv(x)
        x = self.maskConv(x, routing_matrix)
        x = self.bn(x)
        x = self.maskBn(x, routing_matrix)
        return x


class MaskedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, mask_batch_norm_layers, in_planes, planes, stride=1):
        super(MaskedBasicBlock, self).__init__()
        if mask_batch_norm_layers:
            self.convBn1 = MaskedConvMaskedBn(
                in_planes=in_planes, planes=planes, kernel_size=3, stride=stride, padding=1
            )
            self.convBn2 = MaskedConvMaskedBn(
                in_planes=planes, planes=planes, kernel_size=3, stride=1, padding=1
            )
        else:
            self.convBn1 = MaskedConvBn(
                in_planes=in_planes, planes=planes, kernel_size=3, stride=stride, padding=1
            )
            self.convBn2 = MaskedConvBn(
                in_planes=planes, planes=planes, kernel_size=3, stride=1, padding=1
            )

        self.shortcut = MaskedSequential()

        if stride != 1 or in_planes != self.expansion * planes:
            if mask_batch_norm_layers:
                self.shortcut = MaskedConvMaskedBn(
                    in_planes=in_planes, planes=self.expansion * planes, kernel_size=1, stride=stride)
            else:
                self.shortcut = MaskedConvBn(
                    in_planes=in_planes, planes=self.expansion * planes, kernel_size=1, stride=stride)

    def forward(self, x, routing_matrix):
        out = F.relu(self.convBn1(x, routing_matrix))
        out = self.convBn2(out, routing_matrix)
        out = self.shortcut(x, routing_matrix) + out
        out = F.relu(out)
        return out


class CigtMaskedRouting(CigtIgHardRoutingX):
    def __init__(self, run_id, model_definition, num_classes):
        super().__init__(run_id, model_definition, num_classes)

    def create_cigt_blocks(self):
        curr_input_shape = (self.batchSize, *self.inputDims)
        feature_edge_size = curr_input_shape[-1]
        for cigt_layer_id, cigt_layer_info in self.blockParametersList:
            path_count_in_layer = self.pathCounts[cigt_layer_id]
            layers = []
            for block_id, inner_block_info in enumerate(cigt_layer_info):
                prev_route_count = self.pathCounts[cigt_layer_id - 1] if cigt_layer_id > 0 else 1
                curr_route_count = self.pathCounts[cigt_layer_id]
                if block_id == 0:
                    in_channel_count = inner_block_info["in_dimension"] * prev_route_count
                else:
                    in_channel_count = inner_block_info["in_dimension"] * curr_route_count
                out_channel_count = inner_block_info["out_dimension"] * curr_route_count
                block = MaskedBasicBlock(in_planes=in_channel_count, planes=out_channel_count,
                                         stride=inner_block_info["stride"],
                                         mask_batch_norm_layers=self.applyMaskToBatchNorm)
                layers.append(block)
            block_obj = MaskedSequential(*layers)
            if self.useDataParallelism:
                block_obj = nn.DataParallel(block_obj)
            self.cigtLayers.append(block_obj)

            # Block end layers: Routing layers for inner layers, loss layer for the last one.
            if cigt_layer_id < len(self.blockParametersList) - 1:
                for inner_block_info in cigt_layer_info:
                    feature_edge_size = int(feature_edge_size / inner_block_info["stride"])
                routing_layer = self.get_routing_layer(
                    cigt_layer_id=cigt_layer_id,
                    input_feature_map_size=feature_edge_size,
                    input_feature_map_count=self.pathCounts[cigt_layer_id] * cigt_layer_info[-1]["out_dimension"])
                if self.useDataParallelism:
                    routing_layer = nn.DataParallel(routing_layer)
                self.blockEndLayers.append(routing_layer)

        self.get_loss_layer()
