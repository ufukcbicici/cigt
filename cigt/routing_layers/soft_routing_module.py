import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cigt.cigt_model import BasicBlock, Sequential_ext


class SoftRoutingModule(nn.Module):
    def __init__(self,
                 blocks,
                 feature_dim,
                 avg_pool_stride,
                 path_count,
                 class_count,
                 input_feature_map_count,
                 input_feature_map_size,
                 device,
                 from_logits=True):
        super().__init__()
        layers = []
        for inner_block_info in blocks:
            block = BasicBlock(in_planes=inner_block_info["in_dimension"],
                               planes=inner_block_info["out_dimension"],
                               stride=1)
            layers.append(block)
        self.hModule = Sequential_ext(*layers)

        self.featureDim = feature_dim
        self.avgPoolStride = avg_pool_stride
        self.pathCount = path_count
        self.classCount = class_count
        self.device = device
        self.fromLogits = True
        self.inputFeatureMapCount = input_feature_map_count
        self.inputFeatureMapSize = input_feature_map_size
        self.linearLayerInputDim = \
            int(
                self.inputFeatureMapCount * \
                (self.inputFeatureMapSize / self.avgPoolStride) *
                (self.inputFeatureMapSize / self.avgPoolStride))
        self.identityLayer = nn.Identity()
        #  Change the GAP Layer with average pooling with size
        self.avgPool = nn.AvgPool2d(self.avgPoolStride, stride=self.avgPoolStride)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.linearLayerInputDim, self.featureDim)
        self.fc2 = nn.Linear(self.featureDim, self.pathCount)
        self.igBatchNorm = nn.BatchNorm1d(self.featureDim)

    def forward(self, layer_input, labels, temperature, balance_coefficient):
        # Feed it into information gain calculation
        h_out = self.hModule(layer_input)
        h_out = self.avgPool(h_out)
        h_out = self.flatten(h_out)
        h_out = self.fc1(h_out)
        h_out = self.igBatchNorm(h_out)
        activations = self.fc2(h_out)

        if self.fromLogits:
            activations_with_temperature = activations / temperature
            p_n_given_x = torch.softmax(activations_with_temperature, dim=1)
        else:
            p_n_given_x = self.identityLayer(activations)

        return p_n_given_x
