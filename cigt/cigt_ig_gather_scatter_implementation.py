from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cigt.cigt_working_well_tetam_ubicici_01052023 import CigtIgHardRoutingX
from cigt.cigt_model import conv3x3, BasicBlock, Sequential_ext


class BlockOutput(object):
    def __init__(self, routing_activations, p_n_given_x_soft, p_n_given_x_hard,
                 masked_outputs, masked_labels, masked_indices):
        self.routingActivations = routing_activations
        self.softRoutingMatrix = p_n_given_x_soft
        self.hardRoutingMatrix = p_n_given_x_hard
        self.outputsToNextLayer = masked_outputs
        self.labelsToNextLayer = masked_labels
        self.indicesToNextLayer = masked_indices


class NetworkOutputs(object):
    def __init__(self):
        pass


class CigtIgGatherScatterImplementation(CigtIgHardRoutingX):
    def __init__(self, run_id, model_definition, num_classes):
        super().__init__(run_id, model_definition, num_classes)

    def divide_tensor_wrt_routing_matrix(self, tens, routing_matrix):
        tens_num_of_non_batch_dims = np.prod(tens.shape[1:]).item()
        masked_tensors = []
        p_count = routing_matrix.shape[1]
        for p_id in range(p_count):
            s_tensor = routing_matrix[:, p_id].to(torch.bool)
            for _ in range(len(tens.shape) - 1):
                s_tensor = s_tensor.unsqueeze(-1)
            tens_part = torch.masked_select(tens, s_tensor)
            tens_part = torch.reshape(input=tens_part,
                                      shape=(int(tens_part.shape[0] // tens_num_of_non_batch_dims), *tens.shape[1:]))
            masked_tensors.append(tens_part)
        return masked_tensors

    def forward(self, x, labels, temperature):
        sample_indices = torch.arange(0, labels.shape[0])
        balance_coefficient_list = self.informationGainBalanceCoeffList
        # Routing Matrices
        routing_matrices_hard = []
        routing_matrices_soft = []
        routing_activations_list = []
        # Initial layer
        first_output = F.relu(self.bn1(self.conv1(x)))
        list_of_logits = None

        block_outputs = [[(first_output, labels, sample_indices)]]
        for layer_id, cigt_layer_blocks in enumerate(self.cigtLayers):
            block_outputs.append([])
            routing_matrices_hard.append([])
            routing_matrices_soft.append([])
            routing_activations_list.append([])

            for block_id, block_obj in enumerate(cigt_layer_blocks):
                # Traverse the previous layer's outputs, get each previous layer block's particular output, that is
                # destinated into this block. Concatenate them and build a single input into the block.
                block_input_features = [b_output[block_id][0] for b_output in block_outputs[-2]]
                block_input_features = torch.concat(block_input_features, dim=0)
                block_input_labels = [b_output[block_id][1] for b_output in block_outputs[-2]]
                block_input_labels = torch.concat(block_input_labels, dim=0)
                block_input_sample_indices = [b_output[block_id][2] for b_output in block_outputs[-2]]
                block_input_sample_indices = torch.concat(block_input_sample_indices, dim=0)

                block_output = block_obj(block_input_features)

                # Routing Block
                if layer_id < len(self.cigtLayers) - 1:
                    p_n_given_x_soft, routing_activations = \
                        self.blockEndLayers[layer_id][block_id](block_output,
                                                                labels,
                                                                temperature,
                                                                balance_coefficient_list[layer_id])
                    p_n_given_x_hard = self.get_hard_routing_matrix(layer_id=layer_id,
                                                                    p_n_given_x_soft=p_n_given_x_soft)
                    routing_matrices_soft[-1].append(p_n_given_x_soft)
                    routing_matrices_hard[-1].append(p_n_given_x_hard)
                    routing_activations_list[-1].append(routing_activations)
                    # Masking and distributing the block outputs to the next layer.
                    masked_outputs = self.divide_tensor_wrt_routing_matrix(tens=block_output,
                                                                           routing_matrix=p_n_given_x_hard)
                    masked_labels = self.divide_tensor_wrt_routing_matrix(tens=block_input_labels,
                                                                          routing_matrix=p_n_given_x_hard)
                    masked_sample_indices = self.divide_tensor_wrt_routing_matrix(tens=block_input_sample_indices,
                                                                                  routing_matrix=p_n_given_x_hard)
                    next_layer_block_count = p_n_given_x_hard.shape[1]
                    next_layer_block_inputs = [(masked_outputs[nb_id],
                                                masked_labels[nb_id],
                                                masked_sample_indices[nb_id])
                                               for nb_id in range(next_layer_block_count)]
                    block_outputs[-1].append(next_layer_block_inputs)
