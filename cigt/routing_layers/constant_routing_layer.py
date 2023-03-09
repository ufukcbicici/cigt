import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstantRoutingLayer(nn.Module):
    def __init__(self,
                 feature_dim,
                 avg_pool_stride,
                 path_count,
                 class_count,
                 input_feature_map_count,
                 input_feature_map_size,
                 device,
                 from_logits=True):
        super().__init__()
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

    def forward(self, layer_input, labels, temperature, balance_coefficient):
        batch_size = layer_input.size(0)
        routing_weight = 1.0 / self.pathCount
        p_n_given_x = routing_weight * torch.ones(size=(batch_size, self.pathCount),
                                                  dtype=torch.float32,
                                                  device=layer_input.device)
        return p_n_given_x
