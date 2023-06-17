from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from cigt.cigt_model import conv3x3, BasicBlock, Sequential_ext
from cigt.custom_layers.basic_block_with_cbam import BasicBlockWithCbam
from cigt.routing_layers.soft_routing_layer import SoftRoutingLayer


class CbamRoutingLayer(SoftRoutingLayer):
    def __init__(self,
                 block_id,
                 conv_block_count,
                 cbam_reduction_ratio,
                 feature_dim, avg_pool_stride, path_count, class_count, input_feature_map_count,
                 input_feature_map_size, device, apply_relu_dropout, dropout_probability, from_logits=True):

        super().__init__(feature_dim, avg_pool_stride, path_count, class_count, input_feature_map_count,
                         input_feature_map_size, device, apply_relu_dropout, dropout_probability, from_logits)

        layers = OrderedDict()
        for cid in range(conv_block_count):
            block = BasicBlockWithCbam(in_planes=input_feature_map_count,
                                       planes=input_feature_map_count,
                                       stride=1,
                                       cbam_reduction_ratio=cbam_reduction_ratio)
            layers["block_{0}_conv_layer_{1}".format(block_id, cid)] = block
        self.cbamBlock = nn.Sequential(layers)

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
