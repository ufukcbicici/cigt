import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SoftRoutingLayer(nn.Module):
    def __init__(self,
                 feature_dim,
                 avg_pool_stride,
                 path_count,
                 class_count,
                 input_feature_map_count,
                 input_feature_map_size,
                 device,
                 apply_relu_dropout,
                 dropout_probability,
                 from_logits=True):
        super().__init__()
        self.featureDim = feature_dim
        self.avgPoolStride = avg_pool_stride
        self.pathCount = path_count
        self.classCount = class_count
        self.device = device
        self.applyReluDropout = apply_relu_dropout
        self.dropoutProbability = dropout_probability
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
        self.reluNonlinearity = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropoutProbability)
        self.igBatchNorm = nn.BatchNorm1d(self.featureDim)
        self.fc2 = nn.Linear(self.featureDim, self.pathCount)

    def forward(self, layer_input, labels, temperature, balance_coefficient):
        # Feed it into information gain calculation
        h_out = self.avgPool(layer_input)
        h_out = self.flatten(h_out)
        h_out = self.fc1(h_out)
        if self.applyReluDropout:
            h_out = self.reluNonlinearity(h_out)
            h_out = self.dropout(h_out)
        h_out = self.igBatchNorm(h_out)
        activations = self.fc2(h_out)

        if self.fromLogits:
            activations_with_temperature = activations / temperature
            p_n_given_x = torch.softmax(activations_with_temperature, dim=1)
        else:
            p_n_given_x = self.identityLayer(activations)

        return p_n_given_x, activations
