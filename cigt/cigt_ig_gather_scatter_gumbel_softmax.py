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


class CigtIgGatherScatterGumbelSoftmax(CigtIgGatherScatterImplementation):
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


    def forward(self, x, labels, temperature):
        sample_indices = torch.arange(0, labels.shape[0], device=self.device)
        balance_coefficient_list = self.informationGainBalanceCoeffList
        # Initial layer
        net = F.relu(self.bn1(self.conv1(x)))
        layer_outputs = [{"net": net,
                          "labels": labels,
                          "sample_indices": sample_indices,
                          "routing_matrix_hard": torch.ones(size=(x.shape[0], 1),
                                                            dtype=torch.float32,
                                                            device=self.device),
                          "routing_matrices_soft": torch.ones(size=(x.shape[0], 1),
                                                              dtype=torch.float32,
                                                              device=self.device),
                          "routing_activations": torch.ones(size=(x.shape[0], 1),
                                                            dtype=torch.float32,
                                                            device=self.device),
                          "block_indices": torch.zeros(size=(x.shape[0],),
                                                       dtype=torch.int64,
                                                       device=self.device)}]

        for layer_id, cigt_layer_blocks in enumerate(self.cigtLayers):
            net_masked = self.divide_tensor_wrt_routing_matrix(
                tens=layer_outputs[-1]["net"],
                routing_matrix=layer_outputs[-1]["routing_matrix_hard"])
            labels_masked = self.divide_tensor_wrt_routing_matrix(
                tens=layer_outputs[-1]["labels"],
                routing_matrix=layer_outputs[-1]["routing_matrix_hard"])
            sample_indices_masked = self.divide_tensor_wrt_routing_matrix(
                tens=layer_outputs[-1]["sample_indices"],
                routing_matrix=layer_outputs[-1]["routing_matrix_hard"])
            curr_layer_outputs = []
            curr_block_indices = []

            for block_id, block_obj in enumerate(cigt_layer_blocks):
                block_output = block_obj(net_masked[block_id])
                curr_layer_outputs.append(block_output)
                block_indices_arr = block_id * torch.ones(size=(block_output.shape[0],),
                                                          dtype=torch.int64, device=self.device)
                curr_block_indices.append(block_indices_arr)

            layer_outputs_unified = torch.concat(curr_layer_outputs, dim=0)
            layer_labels_unified = torch.concat(labels_masked, dim=0)
            layer_sample_indices_unified = torch.concat(sample_indices_masked, dim=0)
            layer_block_indices_unified = torch.concat(curr_block_indices, dim=0)

            # Routing Layer
            if layer_id < len(self.cigtLayers) - 1:
                # For Gumbel Softmax training, we only need the raw activations from the routing layers, we ignore
                # the soft probabilities.
                _, routing_activations = self.blockEndLayers[layer_id](layer_outputs_unified,
                                                                       layer_labels_unified,
                                                                       temperature,
                                                                       balance_coefficient_list[
                                                                           layer_id])
                self.apply_gumbel_softmax_routing(raw_activations=routing_activations, temperature=temperature,
                                                  layer_id=layer_id)

        #         # Calculate the hard routing matrix
        #         p_n_given_x_hard = self.get_hard_routing_matrix(layer_id=layer_id, p_n_given_x_soft=p_n_given_x_soft)
        #         layer_outputs.append({"net": layer_outputs_unified,
        #                               "labels": layer_labels_unified,
        #                               "sample_indices": layer_sample_indices_unified,
        #                               "block_indices": layer_block_indices_unified,
        #                               "routing_matrix_hard": p_n_given_x_hard,
        #                               "routing_matrices_soft": p_n_given_x_soft,
        #                               "routing_activations": routing_activations})
        #     # Loss Layer
        #     else:
        #         if self.lossCalculationKind == "SingleLogitSingleLoss":
        #             list_of_logits = self.calculate_logits(p_n_given_x_hard=None,
        #                                                    loss_block_outputs=[layer_outputs_unified])
        #         # Calculate logits with all block separately
        #         elif self.lossCalculationKind == "MultipleLogitsMultipleLosses" \
        #                 or self.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
        #             list_of_logits = self.calculate_logits(p_n_given_x_hard=None,
        #                                                    loss_block_outputs=curr_layer_outputs)
        #         else:
        #             raise ValueError("Unknown logit calculation method: {0}".format(self.lossCalculationKind))
        #
        #         layer_outputs.append({"net": layer_outputs_unified,
        #                               "labels": layer_labels_unified,
        #                               "sample_indices": layer_sample_indices_unified,
        #                               "block_indices": layer_block_indices_unified,
        #                               "labels_masked": labels_masked,
        #                               "list_of_logits": list_of_logits})
        #
        # return layer_outputs
        #
