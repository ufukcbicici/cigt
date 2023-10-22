from collections import OrderedDict

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.custom_layers.basic_block_with_cbam import BasicBlockWithCbam


class CigtReinforceMultipath(CigtIgGatherScatterImplementation):
    def __init__(self, configs, run_id, model_definition, num_classes):
        super().__init__(configs, run_id, model_definition, num_classes)
        self.policyNetworksCbamLayerCount = configs.policy_networks_cbam_layer_count
        self.policyNetworksCbamFeatureMapCount = configs.policy_networks_cbam_feature_map_count
        self.policyNetworksCbamReductionRatio = configs.policy_networks_cbam_reduction_ratio
        self.policyNetworksCbamLayerInputReductionRatio = configs.policy_networks_cbam_layer_input_reduction_ratio
        self.policyNetworksLstmDimension = configs.policy_networks_lstm_dimension
        self.policyNetworksAvgPoolStride = configs.policy_networks_cbam_end_avg_pool_strode
        self.policyNetworks = nn.ModuleList()
        self.create_policy_networks()
        print("X")

    def create_policy_networks(self):
        for layer_id, path_count in enumerate(self.pathCounts[1:]):
            layers = OrderedDict()
            if self.policyNetworksCbamLayerInputReductionRatio > 1:
                conv_block_reduction_layer = nn.MaxPool2d(
                    kernel_size=self.policyNetworksCbamLayerInputReductionRatio,
                    stride=self.policyNetworksCbamLayerInputReductionRatio)
                layers["policy_gradients_max_pool_dimension_reduction_layer"] = conv_block_reduction_layer

            # Network entry
            # input_layer = nn.LazyConv2d(
            #     kernel_size=1,
            #     out_channels=self.policyNetworksCbamFeatureMapCount
            # )
            # Cifar10ResnetCigtConfigs.layer_config_list = [
            #     {"path_count": 1,
            #      "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
            #     {"path_count": 2,
            #      "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
            #                          {"layer_count": 18, "feature_map_count": 16}]},
            #     {"path_count": 4,
            #      "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]

            input_feature_count = self.layerConfigList[layer_id]["layer_structure"][-1]["feature_map_count"]
            input_layer = nn.Conv2d(
                kernel_size=1,
                in_channels=input_feature_count,
                out_channels=self.policyNetworksCbamFeatureMapCount
            )
            layers["policy_gradients_input_layer_{0}".format(layer_id)] = input_layer

            for cid in range(self.policyNetworksCbamLayerCount):
                block = BasicBlockWithCbam(in_planes=self.policyNetworksCbamFeatureMapCount,
                                           planes=self.policyNetworksCbamFeatureMapCount,
                                           stride=1,
                                           cbam_reduction_ratio=self.policyNetworksCbamReductionRatio,
                                           norm_type=self.batchNormType)
                layers["policy_gradients_block_{0}_cbam_layer_{1}".format(layer_id, cid)] = block

            # layers["policy_gradients_block_{0}_avg_pool".format(layer_id)] = nn.AvgPool2d(
            #     self.decisionAveragePoolingStrides[layer_id],
            #     stride=self.decisionAveragePoolingStrides[layer_id])
            # layers["policy_gradients_block_{0}_flatten".format(layer_id)] = nn.Flatten()
            # layers["policy_gradients_block_{0}_fc".format(layer_id)] = nn.LazyLinear(
            #     out_features=self.decisionDimensions[layer_id])
            # layers["policy_gradients_block_{0}_relu".format(layer_id)] = nn.ReLU()

            policy_gradient_network_backbone = nn.Sequential(layers)
            self.policyNetworks.append(policy_gradient_network_backbone)

    def forward(self, x, labels, temperature):
        sample_indices = torch.arange(0, labels.shape[0], device=self.device)
        balance_coefficient_list = self.informationGainBalanceCoeffList
        # Initial layer
        net = self.preprocess_input(x=x)
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
                          "route_indices": torch.zeros(size=(x.shape[0], 1),
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
            route_indices_masked = self.divide_tensor_wrt_routing_matrix(
                tens=layer_outputs[-1]["route_indices"],
                routing_matrix=layer_outputs[-1]["routing_matrix_hard"])

            curr_layer_outputs = []
            curr_route_indices = []

            for block_id, block_obj in enumerate(cigt_layer_blocks):
                block_output = block_obj(net_masked[block_id])
                route_indices_block = route_indices_masked[block_id]
                curr_layer_outputs.append(block_output)
                block_indices_arr = block_id * torch.ones(size=(block_output.shape[0], 1),
                                                          dtype=torch.int64, device=self.device)
                route_indices_block = torch.concat([route_indices_block, block_indices_arr], dim=1)
                curr_route_indices.append(route_indices_block)

            layer_outputs_unified = torch.concat(curr_layer_outputs, dim=0)
            layer_labels_unified = torch.concat(labels_masked, dim=0)
            layer_sample_indices_unified = torch.concat(sample_indices_masked, dim=0)
            layer_route_indices_unified = torch.concat(curr_route_indices, dim=0)

            # Routing Layer
            if layer_id < len(self.cigtLayers) - 1:
                # Calculate routing weights for the next layer
                p_n_given_x_soft, routing_activations = self.blockEndLayers[layer_id](layer_outputs_unified,
                                                                                      layer_labels_unified,
                                                                                      temperature,
                                                                                      balance_coefficient_list[
                                                                                          layer_id])
                # Calculate the hard routing matrix
                p_n_given_x_hard = self.routingManager.get_hard_routing_matrix(
                    model=self,
                    layer_id=layer_id,
                    p_n_given_x_soft=p_n_given_x_soft)
                layer_outputs.append({"net": layer_outputs_unified,
                                      "labels": layer_labels_unified,
                                      "sample_indices": layer_sample_indices_unified,
                                      "route_indices": layer_route_indices_unified,
                                      "routing_matrix_hard": p_n_given_x_hard,
                                      "routing_matrices_soft": p_n_given_x_soft,
                                      "routing_activations": routing_activations})
            # Loss Layer
            else:
                if self.lossCalculationKind == "SingleLogitSingleLoss":
                    list_of_logits = self.calculate_logits(p_n_given_x_hard=None,
                                                           loss_block_outputs=[layer_outputs_unified])
                # Calculate logits with all block separately
                elif self.lossCalculationKind == "MultipleLogitsMultipleLosses" \
                        or self.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
                    list_of_logits = self.calculate_logits(p_n_given_x_hard=None,
                                                           loss_block_outputs=curr_layer_outputs)
                else:
                    raise ValueError("Unknown logit calculation method: {0}".format(self.lossCalculationKind))

                logits_unified = torch.concat(list_of_logits, dim=0)
                layer_outputs.append({"net": layer_outputs_unified,
                                      "labels": layer_labels_unified,
                                      "sample_indices": layer_sample_indices_unified,
                                      "route_indices": layer_route_indices_unified,
                                      "labels_masked": labels_masked,
                                      "list_of_logits": list_of_logits,
                                      "logits_unified": logits_unified})

        return layer_outputs

    def train_single_epoch(self, epoch_id, train_loader):
        self.eval()
        for i, (input_, target) in enumerate(train_loader):
            outputs_1 = self.forward(x=input_, labels=target, temperature=0.1)
            outputs_2 = self.forward_v2(x=input_, labels=target, temperature=0.1)
