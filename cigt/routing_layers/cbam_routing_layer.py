from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from cigt.cigt_model import conv3x3, BasicBlock, Sequential_ext
from cigt.routing_layers.soft_routing_layer import SoftRoutingLayer


class CbamRoutingLayer(SoftRoutingLayer):
    def __init__(self,
                 block_id,
                 conv_block_count,
                 in_planes,
                 feature_dim, avg_pool_stride, path_count, class_count, input_feature_map_count,
                 input_feature_map_size, device, apply_relu_dropout, dropout_probability, from_logits=True):

        super().__init__(feature_dim, avg_pool_stride, path_count, class_count, input_feature_map_count,
                         input_feature_map_size, device, apply_relu_dropout, dropout_probability, from_logits)

        layers = OrderedDict()
        for cid in range(conv_block_count):
            block = BasicBlock(in_planes=in_planes,
                               planes=in_planes,
                               stride=1)
            layers.append(block)


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
