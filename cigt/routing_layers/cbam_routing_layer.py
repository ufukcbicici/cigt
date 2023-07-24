from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from cigt.cigt_model import conv3x3, BasicBlock, Sequential_ext
from cigt.custom_layers.basic_block_with_cbam import BasicBlockWithCbam
from cigt.routing_layers.soft_routing_layer import SoftRoutingLayer


class CbamRoutingLayer(nn.Module):
    def __init__(self,
                 block_id,
                 norm_type,
                 conv_block_count,
                 conv_block_reduction,
                 cbam_reduction_ratio,
                 feature_dim, avg_pool_stride, path_count, class_count, input_feature_map_count,
                 device, apply_relu_dropout, dropout_probability, from_logits=True):
        super().__init__()
        layers = OrderedDict()
        self.convBlockReduction = conv_block_reduction
        if self.convBlockReduction > 1:
            conv_block_reduction_layer = nn.MaxPool2d(kernel_size=self.convBlockReduction,
                                                      stride=self.convBlockReduction)
            layers["max_pool_dimension_reduction_layer"] = conv_block_reduction_layer
        else:
            self.convBlockReduction = 1

        for cid in range(conv_block_count):
            block = BasicBlockWithCbam(in_planes=input_feature_map_count,
                                       planes=input_feature_map_count,
                                       stride=1,
                                       cbam_reduction_ratio=cbam_reduction_ratio,
                                       norm_type=norm_type)
            layers["block_{0}_conv_layer_{1}".format(block_id, cid)] = block

        self.cbamBlock = nn.Sequential(layers)

        # Linear layers for information gain calculation
        self.featureDim = feature_dim
        self.avgPoolStride = avg_pool_stride
        self.pathCount = path_count
        self.classCount = class_count
        self.device = device
        self.applyReluDropout = apply_relu_dropout
        self.dropoutProbability = dropout_probability
        self.fromLogits = True
        self.inputFeatureMapCount = input_feature_map_count
        self.identityLayer = nn.Identity()
        #  Change the GAP Layer with average pooling with size
        self.avgPool = nn.AvgPool2d(self.avgPoolStride, stride=self.avgPoolStride)
        self.flatten = nn.Flatten()
        # if input_dimension_predetermined is None:
        #     self.fc1 = nn.Linear(self.linearLayerInputDim, self.featureDim)
        # else:
        #     self.fc1 = nn.Linear(input_dimension_predetermined, self.featureDim)
        self.fc1 = nn.LazyLinear(self.featureDim)
        self.reluNonlinearity = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropoutProbability)
        self.igBatchNorm = nn.BatchNorm1d(self.featureDim)
        self.fc2 = nn.Linear(self.featureDim, self.pathCount)

    def forward(self, layer_input, labels, temperature, balance_coefficient):
        # First, pass the input through the CBAM layers.
        cbam_out = self.cbamBlock(layer_input)

        # Feed it into information gain calculation
        h_out = self.avgPool(cbam_out)
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
