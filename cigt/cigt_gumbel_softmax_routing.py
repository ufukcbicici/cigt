from collections import OrderedDict, Counter
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets
from torch import optim

from auxillary.db_logger import DbLogger
from auxillary.average_meter import AverageMeter
from cigt.cigt_gumbel_softmax import GumbelSoftmax
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cigt_model import conv3x3, BasicBlock, Sequential_ext
from cigt.resnet_cigt_constants import ResnetCigtConstants


class CigtGumbelSoftmaxRouting(CigtIgHardRoutingX):
    def __init__(self, run_id, model_definition, num_classes):
        super().__init__(run_id, model_definition, num_classes)
        self.zSampleCount = ResnetCigtConstants.z_sample_count
        self.gumbelSoftmaxOperations = [GumbelSoftmax() for _ in range(len(self.pathCounts) - 1)]

    def apply_gumbel_softmax_routing(self, raw_activations, layer_id, temperature):
        # Convert raw activations to non-negative weights
        if self.decisionNonLinearity == "Softplus":
            logits = F.softplus(raw_activations)
        elif self.decisionNonLinearity == "Softmax":
            logits = torch.softmax(input=raw_activations, dim=1)
        else:
            raise NotImplementedError()

        # Apply Gumbel Softmax sampling
        p_n_given_x = self.gumbelSoftmaxOperations[layer_id](logits, temperature, hard=self.useStraightThrough)
        print("X")

    def forward(self, x, labels, temperature):
        balance_coefficient_list = self.informationGainBalanceCoeffList
        # Routing Matrices
        routing_matrices_hard = []
        routing_matrices_soft = []
        # Initial layer
        out = F.relu(self.bn1(self.conv1(x)))
        routing_matrices_hard.append(torch.ones(size=(x.shape[0], 1), dtype=torch.float32, device=self.device))
        routing_matrices_soft.append(torch.ones(size=(x.shape[0], 1), dtype=torch.float32, device=self.device))
        block_outputs = []
        routing_activations_list = []
        list_of_logits = None

        for layer_id, cigt_layer_blocks in enumerate(self.cigtLayers):
            block_outputs.append([])
            for block_id, block_obj in enumerate(cigt_layer_blocks):
                block_output = block_obj(out)
                block_outputs[-1].append(block_output)

            # Routing Layer
            if layer_id < len(self.cigtLayers) - 1:
                # Weighted sum of block outputs
                out = self.weighted_sum_of_tensors(routing_matrix=routing_matrices_hard[-1],
                                                   tensors=block_outputs[-1])
                # Calculate routing weights for the next layer
                p_n_given_x_soft, routing_activations = self.blockEndLayers[layer_id](out,
                                                                                      labels,
                                                                                      temperature,
                                                                                      balance_coefficient_list[
                                                                                          layer_id])
                routing_matrices_soft.append(p_n_given_x_soft)
                routing_activations_list.append(routing_activations)
                # Calculate the hard routing matrix
                p_n_given_x_hard = self.get_hard_routing_matrix(layer_id=layer_id,
                                                                p_n_given_x_soft=p_n_given_x_soft)
                routing_matrices_hard.append(p_n_given_x_hard)
            # Logits layer
            else:
                list_of_logits = self.calculate_logits(p_n_given_x_hard=routing_matrices_hard[-1],
                                                       loss_block_outputs=block_outputs[-1])

        return routing_matrices_hard, routing_matrices_soft, block_outputs, list_of_logits, routing_activations_list


        # eps = 1e-20
        # samples_shape = (logits.shape[0], logits.shape[1], self.zSampleCount)
        # U_ = torch.rand(size=samples_shape, dtype=torch.float32)
        # G_ = -torch.math.log(-torch.math.log(U_ + eps) + eps)
        # log_logits = torch.math.log(logits + eps)
        # log_logits = torch.unsqueeze(log_logits, dim=-1)
        # gumbel_logits = log_logits + G_
        # gumbel_logits_tempered = gumbel_logits / temperature
        # z_samples = torch.softmax(gumbel_logits_tempered, dim=1)
        #
        # # Convert Gumbel Softmax samples into soft routing probabilities, apply Straight Through trick.
        # p_n_given_x = torch.mean(z_samples, dim=-1)

