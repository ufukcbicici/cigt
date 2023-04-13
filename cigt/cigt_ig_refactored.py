import os
from collections import OrderedDict, Counter

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_soft_routing import CigtIgSoftRouting
from cigt.cigt_model import conv3x3, BasicBlock, Sequential_ext
from cigt.cigt_soft_routing import CigtSoftRouting
from cigt.cutout_augmentation import CutoutPIL
from cigt.moe_layer import MoeLayer
from cigt.resnet_cigt_constants import ResnetCigtConstants
from cigt.routing_layers.hard_routing_layer import HardRoutingLayer
from randaugment import RandAugment
from torchvision import transforms
import torchvision.datasets as datasets

from cigt.routing_layers.info_gain_routing_layer import InfoGainRoutingLayer
from cigt.routing_layers.soft_routing_layer import SoftRoutingLayer


class CigtIgHardRoutingX(nn.Module):
    def __init__(self, run_id, model_definition, num_classes):
        super().__init__()
        self.runId = run_id
        self.modelDefinition = model_definition
        self.imageSize = (32, 32)
        self.numClasses = num_classes
        self.classCount = 10
        self.useDataParallelism = ResnetCigtConstants.data_parallelism
        self.hardRoutingAlgorithmTypes = {"InformationGainRouting", "RandomRouting", "EnforcedRouting"}
        self.hardRoutingAlgorithmKind = ResnetCigtConstants.hard_routing_algorithm_kind
        self.enforcedRoutingMatrices = []
        self.lossCalculationTypes = {"SingleLogitSingleLoss", "MultipleLogitsMultipleLosses"}
        self.lossCalculationKind = ResnetCigtConstants.loss_calculation_kind
        self.lossLayers = None
        self.modelFilesRootPath = None
        self.routingStrategyName = ResnetCigtConstants.routing_strategy_name
        self.useStraightThrough = ResnetCigtConstants.use_straight_through
        self.decisionNonLinearity = ResnetCigtConstants.decision_non_linearity
        self.warmUpPeriod = ResnetCigtConstants.warm_up_period
        self.optimizerType = ResnetCigtConstants.optimizer_type
        self.learningRateSchedule = ResnetCigtConstants.learning_schedule
        self.initialLr = ResnetCigtConstants.initial_lr
        self.resnetConfigList = ResnetCigtConstants.resnet_config_list
        self.firstConvKernelSize = ResnetCigtConstants.first_conv_kernel_size
        self.firstConvOutputDim = ResnetCigtConstants.first_conv_output_dim
        self.firstConvStride = ResnetCigtConstants.first_conv_stride
        self.bnMomentum = ResnetCigtConstants.bn_momentum
        self.batchNormType = ResnetCigtConstants.batch_norm_type
        self.applyMaskToBatchNorm = ResnetCigtConstants.apply_mask_to_batch_norm
        self.doubleStrideLayers = ResnetCigtConstants.double_stride_layers
        self.batchSize = ResnetCigtConstants.batch_size
        self.inputDims = ResnetCigtConstants.input_dims
        self.advancedAugmentation = ResnetCigtConstants.advanced_augmentation
        self.decisionDimensions = ResnetCigtConstants.decision_dimensions
        self.decisionAveragePoolingStrides = ResnetCigtConstants.decision_average_pooling_strides
        self.routerLayersCount = ResnetCigtConstants.router_layers_count
        self.isInWarmUp = True
        self.temperatureController = ResnetCigtConstants.softmax_decay_controller
        self.decisionLossCoeff = ResnetCigtConstants.decision_loss_coeff
        self.informationGainBalanceCoeffList = ResnetCigtConstants.information_gain_balance_coeff_list
        self.classificationWd = ResnetCigtConstants.classification_wd
        self.decisionWd = ResnetCigtConstants.decision_wd
        self.epochCount = ResnetCigtConstants.epoch_count
        self.boostLearningRatesLayerWise = ResnetCigtConstants.boost_learning_rates_layer_wise
        self.multipleCeLosses = ResnetCigtConstants.multiple_ce_losses
        self.perSampleEntropyBalance = ResnetCigtConstants.per_sample_entropy_balance
        self.evaluationPeriod = ResnetCigtConstants.evaluation_period
        self.pathCounts = [1]
        self.pathCounts.extend([d_["path_count"] for d_ in self.resnetConfigList][1:])
        self.blockParametersList = self.interpret_config_list()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print("Device:{0}".format(self.device))
        # Train and test time augmentations
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        if not self.advancedAugmentation:
            print("WILL BE USING ONLY CROP AND HORIZONTAL FLIP AUGMENTATION")
            self.transformTrain = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.transformTest = transforms.Compose([
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            print("WILL BE USING RANDOM AUGMENTATION")
            self.transformTrain = transforms.Compose([
                transforms.Resize(self.imageSize),
                CutoutPIL(cutout_factor=0.5),
                RandAugment(),
                transforms.ToTensor(),
            ])
            self.transformTest = transforms.Compose([
                transforms.Resize(self.imageSize),
                transforms.ToTensor(),
            ])

        # Initial layer
        self.in_planes = self.firstConvOutputDim
        self.conv1 = conv3x3(3, self.firstConvOutputDim, self.firstConvStride)
        self.bn1 = nn.BatchNorm2d(self.firstConvOutputDim)
        if self.useDataParallelism:
            self.conv1 = nn.DataParallel(self.conv1)
            self.bn1 = nn.DataParallel(self.bn1)

        # Build Cigt Blocks
        self.cigtLayers = nn.ModuleList()
        self.blockEndLayers = nn.ModuleList()
        self.create_cigt_blocks()

        # MoE Loss Layer
        self.moeLossLayer = MoeLayer()
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.nllLoss = nn.NLLLoss()
        self.crossEntropyLoss = nn.CrossEntropyLoss()
        self.identityLayer = nn.Identity()
        self.crossEntropyLosses = [nn.CrossEntropyLoss() for _ in range(self.pathCounts[-1])]

        self.numOfTrainingIterations = 0
        self.warmUpFinalIteration = 0

        self.layerCoefficients = []
        self.modelOptimizer = self.create_optimizer()

    # OK
    def add_explanation(self, name_of_param, value, explanation, kv_rows):
        explanation += "{0}:{1}\n".format(name_of_param, value)
        kv_rows.append((self.runId, name_of_param, "{0}".format(value)))
        return explanation

    # OK
    def get_explanation_string(self):
        kv_rows = []
        explanation = ""
        explanation = self.add_explanation(name_of_param="Model Definition",
                                           value=self.modelDefinition,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Batch Size", value=self.batchSize,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Path Counts", value=self.pathCounts,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Routing Strategy", value=self.routingStrategyName,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Use Straight Through", value=self.useStraightThrough,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Decision Nonlinearity", value=self.decisionNonLinearity,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Warm Up Period", value=self.warmUpPeriod,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Optimizer Type", value=self.optimizerType,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Data Parallelism", value=self.useDataParallelism,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Lr Settings",
                                           value=self.learningRateSchedule,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Initial Lr", value=self.initialLr,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.temperatureController.get_explanation(network=self,
                                                                 explanation=explanation,
                                                                 kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Decision Loss Coeff", value=self.decisionLossCoeff,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Batch Norm Decay", value=self.bnMomentum,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Classification Wd", value=self.classificationWd,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Decision Wd", value=self.decisionWd,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Information Gain Balance Coefficient List",
                                           value=self.informationGainBalanceCoeffList,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="firstConvKernelSize", value=self.firstConvKernelSize,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="firstConvStride", value=self.firstConvStride,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="firstConvOutputDim", value=self.firstConvOutputDim,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="decisionAveragePoolingStrides",
                                           value=self.decisionAveragePoolingStrides,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="decisionDimensions", value=self.decisionDimensions,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="boostLearningRatesLayerWise",
                                           value=self.boostLearningRatesLayerWise,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="multiple_ce_losses",
                                           value=self.boostLearningRatesLayerWise,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="applyMaskToBatchNorm", value=self.applyMaskToBatchNorm,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="advancedAugmentation", value=self.advancedAugmentation,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="evaluationPeriod", value=self.evaluationPeriod,
                                           explanation=explanation, kv_rows=kv_rows)
        # explanation = self.add_explanation(name_of_param="startMovingAveragesFromZero",
        #                                    value=self.startMovingAveragesFromZero,
        #                                    explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="batchNormType", value=self.batchNormType,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="doubleStrideLayers", value=self.doubleStrideLayers,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="hardRoutingAlgorithmKind",
                                           value=self.hardRoutingAlgorithmKind,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="lossCalculationKind", value=self.lossCalculationKind,
                                           explanation=explanation, kv_rows=kv_rows)
        # Explanation for block configurations
        block_params = [(block_id, block_config_list)
                        for block_id, block_config_list in self.blockParametersList]
        block_params = sorted(block_params, key=lambda t__: t__[0])

        layer_id = 0
        for t_ in block_params:
            block_id = t_[0]
            block_config_list = t_[1]
            for block_config_dict in block_config_list:
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} in_dimension".format(layer_id),
                                                   value=block_config_dict["in_dimension"],
                                                   explanation=explanation, kv_rows=kv_rows)
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} input_path_count".format(layer_id),
                                                   value=block_config_dict["input_path_count"],
                                                   explanation=explanation, kv_rows=kv_rows)
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} layer_id".format(layer_id),
                                                   value=layer_id,
                                                   explanation=explanation, kv_rows=kv_rows)
                assert block_id == block_config_dict["block_id"]
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} block_id".format(layer_id),
                                                   value=block_config_dict["block_id"],
                                                   explanation=explanation, kv_rows=kv_rows)
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} out_dimension".format(layer_id),
                                                   value=block_config_dict["out_dimension"],
                                                   explanation=explanation, kv_rows=kv_rows)
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} output_path_count".format(layer_id),
                                                   value=block_config_dict["output_path_count"],
                                                   explanation=explanation, kv_rows=kv_rows)
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} stride".format(layer_id),
                                                   value=block_config_dict["stride"],
                                                   explanation=explanation, kv_rows=kv_rows)
                layer_id += 1

        DbLogger.write_into_table(rows=kv_rows, table="run_parameters")
        return explanation

    # OK
    def interpret_config_list(self):
        block_list = []
        # Unravel the configuration information into a complete block by block list.
        for block_id, block_config_dict in enumerate(self.resnetConfigList):
            path_count = block_config_dict["path_count"]
            for idx, d_ in enumerate(block_config_dict["layer_structure"]):
                for idy in range(d_["layer_count"]):
                    block_list.append((block_id, path_count, d_["feature_map_count"]))

        block_parameters_dict = {}
        for layer_id, layer_info in enumerate(block_list):
            block_id = layer_info[0]
            path_count = layer_info[1]
            feature_map_count = layer_info[2]
            if block_id not in block_parameters_dict:
                block_parameters_dict[block_id] = []
            block_options = {}
            if layer_id == 0:
                block_options["in_dimension"] = self.firstConvOutputDim
                block_options["input_path_count"] = 1
            else:
                path_count_prev = block_list[layer_id - 1][1]
                feature_map_count_prev = block_list[layer_id - 1][2]
                block_options["in_dimension"] = feature_map_count_prev
                block_options["input_path_count"] = path_count_prev
            block_options["layer_id"] = layer_id
            block_options["block_id"] = block_id
            block_options["out_dimension"] = feature_map_count
            block_options["output_path_count"] = path_count
            if layer_id in self.doubleStrideLayers:
                block_options["stride"] = 2
            else:
                block_options["stride"] = 1
            block_parameters_dict[block_id].append(block_options)
        block_parameters_list = sorted([(k, v) for k, v in block_parameters_dict.items()], key=lambda tpl: tpl[0])
        return block_parameters_list

    # OK
    def get_routing_layer(self, cigt_layer_id, input_feature_map_size, input_feature_map_count):
        routing_layer = SoftRoutingLayer(
            feature_dim=self.decisionDimensions[cigt_layer_id],
            avg_pool_stride=self.decisionAveragePoolingStrides[cigt_layer_id],
            path_count=self.pathCounts[cigt_layer_id + 1],
            class_count=self.numClasses,
            input_feature_map_size=input_feature_map_size,
            input_feature_map_count=input_feature_map_count,
            device=self.device)
        return routing_layer

    # OK
    def create_cigt_blocks(self):
        curr_input_shape = (self.batchSize, *self.inputDims)
        feature_edge_size = curr_input_shape[-1]
        for cigt_layer_id, cigt_layer_info in self.blockParametersList:
            path_count_in_layer = self.pathCounts[cigt_layer_id]
            cigt_layer_blocks = nn.ModuleList()
            for path_id in range(path_count_in_layer):
                layers = []
                for inner_block_info in cigt_layer_info:
                    block = BasicBlock(in_planes=inner_block_info["in_dimension"],
                                       planes=inner_block_info["out_dimension"],
                                       stride=inner_block_info["stride"])
                    layers.append(block)
                block_obj = Sequential_ext(*layers)
                if self.useDataParallelism:
                    block_obj = nn.DataParallel(block_obj)
                # block_obj.name = "block_{0}_{1}".format(cigt_layer_id, path_id)
                cigt_layer_blocks.append(block_obj)
            self.cigtLayers.append(cigt_layer_blocks)
            # Block end layers: Routing layers for inner layers, loss layer for the last one.
            if cigt_layer_id < len(self.blockParametersList) - 1:
                for inner_block_info in cigt_layer_info:
                    feature_edge_size = int(feature_edge_size / inner_block_info["stride"])
                routing_layer = self.get_routing_layer(cigt_layer_id=cigt_layer_id,
                                                       input_feature_map_size=feature_edge_size,
                                                       input_feature_map_count=cigt_layer_info[-1]["out_dimension"])
                if self.useDataParallelism:
                    routing_layer = nn.DataParallel(routing_layer)
                self.blockEndLayers.append(routing_layer)
        # if cigt_layer_id == len(self.blockParametersList) - 1:
        self.get_loss_layer()

    # OK
    def get_loss_layer(self):
        self.lossLayers = nn.ModuleList()
        final_feature_dimension = self.blockParametersList[-1][-1][-1]["out_dimension"]
        if self.lossCalculationKind == "SingleLogitSingleLoss":
            end_module = nn.Sequential(OrderedDict([
                ('avg_pool', torch.nn.AvgPool2d(kernel_size=8)),
                ('flatten', torch.nn.Flatten()),
                ('logits', torch.nn.Linear(in_features=final_feature_dimension, out_features=self.numClasses))
            ]))
            self.lossLayers.append(end_module)
        elif self.lossCalculationKind == "MultipleLogitsMultipleLosses":
            for block_id in range(self.pathCounts[-1]):
                end_module = nn.Sequential(OrderedDict([
                    ('avg_pool_{0}'.format(block_id), torch.nn.AvgPool2d(kernel_size=8)),
                    ('flatten_{0}'.format(block_id), torch.nn.Flatten()),
                    ('logits_{0}'.format(block_id), torch.nn.Linear(
                        in_features=final_feature_dimension, out_features=self.numClasses))
                ]))
                self.lossLayers.append(end_module)

    # OK
    def create_optimizer(self):
        paths = []
        for pc in self.pathCounts:
            paths.append([i_ for i_ in range(pc)])
        path_variaties = Utilities.get_cartesian_product(list_of_lists=paths)

        for idx in range(len(self.pathCounts)):
            cnt = len([tpl for tpl in path_variaties if tpl[idx] == 0])
            self.layerCoefficients.append(len(path_variaties) / cnt)

        # Create parameter groups per CIGT layer and shared parameters
        shared_parameters = []
        parameters_per_cigt_layers = []
        for idx in range(len(self.pathCounts)):
            parameters_per_cigt_layers.append([])

        for name, param in self.named_parameters():
            if "cigtLayers" not in name:
                shared_parameters.append(param)
            else:
                param_name_splitted = name.split(".")
                layer_id = int(param_name_splitted[1])
                assert 0 <= layer_id <= len(self.pathCounts) - 1
                parameters_per_cigt_layers[layer_id].append(param)
        num_shared_parameters = len(shared_parameters)
        num_cigt_layer_parameters = sum([len(arr) for arr in parameters_per_cigt_layers])
        num_all_parameters = len([tpl for tpl in self.named_parameters()])
        assert num_shared_parameters + num_cigt_layer_parameters == num_all_parameters

        parameter_groups = []
        # Add parameter groups with respect to their cigt layers
        for layer_id in range(len(self.pathCounts)):
            parameter_groups.append(
                {'params': parameters_per_cigt_layers[layer_id],
                 # 'lr': self.initialLr * self.layerCoefficients[layer_id],
                 'lr': self.initialLr,
                 'weight_decay': self.classificationWd})

        # Shared parameters, always the group
        parameter_groups.append(
            {'params': shared_parameters,
             'lr': self.initialLr,
             'weight_decay': self.classificationWd})

        if self.optimizerType == "SGD":
            model_optimizer = optim.SGD(parameter_groups, momentum=0.9)
        elif self.optimizerType == "Adam":
            model_optimizer = optim.Adam(parameter_groups)
        else:
            raise ValueError("{0} is not supported as optimizer.".format(self.optimizerType))
        return model_optimizer

    # OK
    def get_hard_routing_matrix(self, layer_id, p_n_given_x_soft):
        if self.hardRoutingAlgorithmKind == "InformationGainRouting":
            p_n_given_x_hard = torch.zeros_like(p_n_given_x_soft)
            arg_max_entries = torch.argmax(p_n_given_x_soft, dim=1)
            p_n_given_x_hard[torch.arange(p_n_given_x_hard.shape[0]), arg_max_entries] = 1.0
        elif self.hardRoutingAlgorithmKind == "RandomRouting":
            random_routing_matrix = torch.rand(size=p_n_given_x_soft.shape)
            arg_max_entries = torch.argmax(random_routing_matrix, dim=1)
            random_routing_matrix_hard = torch.zeros_like(p_n_given_x_soft)
            random_routing_matrix_hard[torch.arange(random_routing_matrix_hard.shape[0]), arg_max_entries] = 1.0
            p_n_given_x_hard = random_routing_matrix_hard
        elif self.hardRoutingAlgorithmKind == "EnforcedRouting":
            enforced_routing_matrix = self.enforcedRoutingMatrices[layer_id]
            p_n_given_x_hard = enforced_routing_matrix
        else:
            raise ValueError("Unknown routing algorithm: {0}".format(self.hardRoutingAlgorithmKind))

        return p_n_given_x_hard

    # OK
    def calculate_logits(self, p_n_given_x_hard, loss_block_outputs):
        list_of_logits = []
        # Unify all last layer block outputs into a single output and calculate logits with only that output
        if self.lossCalculationKind == "SingleLogitSingleLoss":
            out = self.weighted_sum_of_tensors(routing_matrix=p_n_given_x_hard, tensors=loss_block_outputs)
            logits = self.lossLayers[0](out)
            list_of_logits.append(logits)
        # Calculate logits with all block separately
        elif self.lossCalculationKind == "MultipleLogitsMultipleLosses":
            for idx, block_x in enumerate(loss_block_outputs):
                logits = self.lossLayers[idx](block_x)
                list_of_logits.append(logits)
        else:
            raise ValueError("Unknown logit calculation method: {0}".format(self.lossCalculationKind))
        return list_of_logits

    # OK
    def weighted_sum_of_tensors(self, routing_matrix, tensors):
        block_output_shape = tensors[0].shape
        weighted_tensors = []
        for block_id in range(routing_matrix.shape[1]):
            probs_exp = self.identityLayer(routing_matrix[:, block_id])
            for _ in range(len(block_output_shape) - len(probs_exp.shape)):
                probs_exp = torch.unsqueeze(probs_exp, -1)
            block_output_weighted = probs_exp * tensors[block_id]
            weighted_tensors.append(block_output_weighted)
        weighted_tensors = torch.stack(weighted_tensors, dim=1)
        weighted_sum_tensor = torch.sum(weighted_tensors, dim=1)
        return weighted_sum_tensor

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
                p_n_given_x_soft = self.blockEndLayers[layer_id](out,
                                                                 labels,
                                                                 temperature,
                                                                 balance_coefficient_list[layer_id])
                routing_matrices_soft.append(p_n_given_x_soft)
                # Calculate the hard routing matrix
                p_n_given_x_hard = self.get_hard_routing_matrix(layer_id=layer_id, p_n_given_x_soft=p_n_given_x_soft)
                routing_matrices_hard.append(p_n_given_x_hard)
            # Logits layer
            else:
                list_of_logits = self.calculate_logits(p_n_given_x_hard=routing_matrices_hard[-1],
                                                       loss_block_outputs=block_outputs[-1])

        return routing_matrices_hard, routing_matrices_soft, block_outputs, list_of_logits

    def measure_accuracy(self, probs, target):
        """Computes the precision@k for the specified values of k"""
        pred = torch.argmax(probs, dim=1)
        correct_vector = pred.eq(target).to(torch.float)
        acc = torch.mean(correct_vector)
        return acc.cpu().numpy().item()

    def calculate_classification_loss_and_accuracy(self, list_of_logits, routing_matrices, target_var):
        if self.lossCalculationKind == "SingleLogitSingleLoss":
            classification_loss = self.crossEntropyLoss(list_of_logits[0], target_var)
            batch_accuracy = self.measure_accuracy(list_of_logits[0].detach().cpu(), target_var.cpu())
        elif self.lossCalculationKind == "MultipleLogitsMultipleLosses":
            # Independently calculate loss for every block, by selecting the samples that are routed into these blocks.
            classification_loss = 0.0
            batch_accuracy = 0.0
            for idx, logit in enumerate(list_of_logits):
                sample_selection_vector = routing_matrices[-1][:, idx].to(torch.bool)
                selected_logits_1d = torch.masked_select(list_of_logits[idx],
                                                         torch.unsqueeze(sample_selection_vector, dim=1))
                selected_labels = torch.masked_select(target_var, sample_selection_vector)
                # Reshape back into 2d
                new_shape = (selected_logits_1d.shape[0] // list_of_logits[idx].shape[1], list_of_logits[idx].shape[1])
                # print("Block {0} Count:{1}".format(idx, new_shape[0]))
                if selected_logits_1d.shape[0] == 0:
                    continue
                selected_logits = torch.reshape(selected_logits_1d, new_shape)
                # The following are for testing the torch indexing logic
                # non_zero_indices = np.nonzero(sample_selection_vector.cpu().numpy())[0]
                # for i_, j_ in enumerate(non_zero_indices):
                #     assert np.array_equal(selected_logits[i_].cpu().numpy(),
                #                           list_of_logits[idx][j_].cpu().numpy())
                #     assert selected_labels[i_] == target_var[j_]

                block_classification_loss = self.crossEntropyLosses[idx](selected_logits, selected_labels)
                classification_loss += block_classification_loss
                block_accuracy = self.measure_accuracy(selected_logits.detach().cpu(), selected_labels.cpu())
                batch_coefficient = (new_shape[0] / target_var.shape[0])
                batch_accuracy += batch_coefficient * block_accuracy
        else:
            raise ValueError("Unknown loss calculation method:{0}".format(self.lossCalculationKind))
        return classification_loss, batch_accuracy

    def calculate_entropy(self, prob_distribution, eps=1e-30):
        log_prob = torch.log(prob_distribution + eps)
        # is_inf = tf.is_inf(log_prob)
        # zero_tensor = tf.zeros_like(log_prob)
        # log_prob = tf.where(is_inf, x=zero_tensor, y=log_prob)
        prob_log_prob = prob_distribution * log_prob
        entropy = -1.0 * torch.sum(prob_log_prob)
        return entropy, log_prob

    def calculate_information_gain_losses(self, routing_matrices, labels, balance_coefficient_list):
        information_gain_list = []
        for layer_id, p_n_given_x in enumerate(routing_matrices[1:]):
            weight_vector = torch.ones(size=(p_n_given_x.shape[0],),
                                       dtype=torch.float32,
                                       device=self.device)
            # # probability_vector = tf.cast(weight_vector / tf.reduce_sum(weight_vector), dtype=activations.dtype)
            sample_count = torch.sum(weight_vector)
            probability_vector = torch.div(weight_vector, sample_count)
            batch_size = p_n_given_x.shape[0]
            node_degree = p_n_given_x.shape[1]
            joint_distribution = torch.ones(size=(batch_size, self.classCount, node_degree),
                                            dtype=p_n_given_x.dtype,
                                            device=self.device)

            # Calculate p(x)
            joint_distribution = joint_distribution * torch.unsqueeze(torch.unsqueeze(
                probability_vector, dim=-1), dim=-1)
            # Calculate p(c|x) * p(x) = p(x,c)
            p_c_given_x = torch.nn.functional.one_hot(labels, self.classCount)
            joint_distribution = joint_distribution * torch.unsqueeze(p_c_given_x, dim=2)
            p_xcn = joint_distribution * torch.unsqueeze(p_n_given_x, dim=1)

            # Calculate p(c,n)
            marginal_p_cn = torch.sum(p_xcn, dim=0)
            # Calculate p(n)
            marginal_p_n = torch.sum(marginal_p_cn, dim=0)
            # Calculate p(c)
            marginal_p_c = torch.sum(marginal_p_cn, dim=1)
            # Calculate entropies
            entropy_p_cn, log_prob_p_cn = self.calculate_entropy(prob_distribution=marginal_p_cn)
            entropy_p_n, log_prob_p_n = self.calculate_entropy(prob_distribution=marginal_p_n)
            entropy_p_c, log_prob_p_c = self.calculate_entropy(prob_distribution=marginal_p_c)
            # Calculate the information gain
            balance_coefficient = balance_coefficient_list[layer_id]
            information_gain = (balance_coefficient * entropy_p_n) + entropy_p_c - entropy_p_cn
            information_gain_list.append(information_gain)
        return information_gain_list

    def calculate_branch_statistics(self,
                                    run_id, iteration, dataset_type, routing_probability_matrices, labels,
                                    write_to_db):
        kv_rows = []
        for block_id, routing_probability_matrix in enumerate(routing_probability_matrices):
            path_count = routing_probability_matrix.shape[1]
            selected_paths = np.argmax(routing_probability_matrix, axis=1)
            path_counter = Counter(selected_paths)
            print("Path Distributions Data Type:{0} Block ID:{1} Iteration:{2} Path Distribution:{3}".format(
                dataset_type, block_id, iteration, path_counter))
            p_n = np.mean(routing_probability_matrix, axis=0)
            print("Block:{0} Route Probabilties:{1}".format(block_id, p_n))
            kv_rows.append((run_id,
                            iteration,
                            "Path Distributions Data Type:{0} Block ID:{1} Path Distribution".format(
                                dataset_type, block_id),
                            "{0}".format(path_counter)))
            for path_id in range(path_count):
                path_labels = labels[selected_paths == path_id]
                label_counter = Counter(path_labels)
                str_ = \
                    "Path Distributions Data Type:{0} Block ID:{1} Path ID:{2} Iteration:{3} Label Distribution:{4}" \
                        .format(dataset_type, block_id, path_id, iteration, label_counter)
                print(str_)
                kv_rows.append((run_id,
                                iteration,
                                "Path Distributions Data Type:{0} Block ID:{1} Path ID:{2} Label Distribution".format(
                                    dataset_type, block_id, path_id),
                                "{0}".format(label_counter)))
        if write_to_db:
            DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 after 150 and 250 epochs"""
        lr = self.initialLr
        # if epoch >= 150:
        #     lr = 0.1 * lr
        # if epoch >= 250:
        #     lr = 0.1 * lr
        # learning_schedule = [(150, 0.1), (250, 0.01)]

        # Calculate base learning rate
        lower_bounds = [0]
        lower_bounds.extend([tpl[0] for tpl in self.learningRateSchedule])
        upper_bounds = [tpl[0] for tpl in self.learningRateSchedule]
        upper_bounds.append(np.inf)
        bounds = np.stack([lower_bounds, upper_bounds], axis=1)
        lr_coeffs = [1.0]
        lr_coeffs.extend([tpl[1] for tpl in self.learningRateSchedule])
        lower_comparison = bounds[:, 0] <= epoch
        upper_comparison = epoch < bounds[:, 1]
        bounds_binary = np.stack([lower_comparison, upper_comparison], axis=1)
        res = np.all(bounds_binary, axis=1)
        idx = np.argmax(res)
        lr_coeff = lr_coeffs[idx]
        base_lr = lr * lr_coeff

        assert len(self.modelOptimizer.param_groups) == len(self.pathCounts) + 1

        # Cigt layers with boosted lrs.
        for layer_id in range(len(self.pathCounts)):
            if self.boostLearningRatesLayerWise:
                self.modelOptimizer.param_groups[layer_id]['lr'] = self.layerCoefficients[layer_id] * base_lr
            else:
                self.modelOptimizer.param_groups[layer_id]['lr'] = base_lr
        assert len(self.pathCounts) == len(self.modelOptimizer.param_groups) - 1
        # Shared parameters
        self.modelOptimizer.param_groups[-1]['lr'] = base_lr

        if not self.boostLearningRatesLayerWise:
            for p_group in self.modelOptimizer.param_groups:
                assert p_group["lr"] == base_lr

    def adjust_warmup(self, epoch):
        if self.isInWarmUp:
            if epoch >= self.warmUpPeriod:
                print("Warmup is ending!")
                self.isInWarmUp = False
                self.warmUpFinalIteration = self.numOfTrainingIterations
                self.hardRoutingAlgorithmKind = "InformationGainRouting"
            else:
                print("Still in warm up!")
                self.isInWarmUp = True
                self.hardRoutingAlgorithmKind = "RandomRouting"
        else:
            print("Warmup has ended!")
        print("Epoch:{0} Routing Kind:{1}".format(epoch, self.hardRoutingAlgorithmKind))

    def train_single_epoch(self, epoch_id, train_loader):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_c = AverageMeter()
        losses_t = AverageMeter()
        losses_t_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
        grad_magnitude = AverageMeter()
        accuracy_avg = AverageMeter()

        # Switch to train mode
        self.train()

        for i, (input_, target) in enumerate(train_loader):
            time_begin = time.time()
            print("*************Epoch:{0} Iteration:{1}*************".format(
                epoch_id, self.numOfTrainingIterations))

            # Print learning rates
            for layer_id in range(len(self.pathCounts)):
                print("Cigt layer {0} learning rate:{1}".format(
                    layer_id, self.modelOptimizer.param_groups[layer_id]['lr']))
            assert len(self.pathCounts) == len(self.modelOptimizer.param_groups) - 1
            # Shared parameters
            print("Shared parameters learning rate:{0}".format(self.modelOptimizer.param_groups[-1]['lr']))

            self.modelOptimizer.zero_grad()
            with torch.set_grad_enabled(True):
                input_var = torch.autograd.Variable(input_).to(self.device)
                target_var = torch.autograd.Variable(target).to(self.device)
                batch_size = input_var.size(0)

                if not self.isInWarmUp:
                    decay_t = self.numOfTrainingIterations - self.warmUpFinalIteration
                    self.temperatureController.update(iteration=decay_t)
                    decision_loss_coeff = self.decisionLossCoeff
                else:
                    decision_loss_coeff = 0.0
                temperature = self.temperatureController.get_value()
                print("temperature:{0}".format(temperature))
                print("decision_loss_coeff:{0}".format(decision_loss_coeff))

                # Cigt moe output, information gain losses
                routing_matrices_hard, routing_matrices_soft, block_outputs, list_of_logits = self(
                    input_var, target_var, temperature)
                classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                    list_of_logits,
                    routing_matrices_hard,
                    target_var)
                information_gain_losses = self.calculate_information_gain_losses(
                    routing_matrices=routing_matrices_soft, labels=target_var,
                    balance_coefficient_list=self.informationGainBalanceCoeffList)
                total_routing_loss = 0.0
                for t_loss in information_gain_losses:
                    total_routing_loss += t_loss
                total_routing_loss = -1.0 * decision_loss_coeff * total_routing_loss
                total_loss = classification_loss + total_routing_loss
                # print("len(list_of_logits)={0}".format(len(list_of_logits)))
                # print("multipleCeLosses:{0}".format(self.multipleCeLosses))
                total_loss.backward()
                self.modelOptimizer.step()

            time_end = time.time()
            # measure accuracy and record loss
            print("Epoch:{0} Iteration:{1}".format(epoch_id, self.numOfTrainingIterations))

            losses.update(total_loss.detach().cpu().numpy().item(), 1)
            losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
            accuracy_avg.update(batch_accuracy, batch_size)
            batch_time.update((time_end - time_begin), 1)
            losses_t.update(total_routing_loss.detach().cpu().numpy().item(), 1)
            for lid in range(len(self.pathCounts) - 1):
                losses_t_layer_wise[lid].update(information_gain_losses[lid].detach().cpu().numpy().item(), 1)

            print("batch_accuracy:{0}".format(batch_accuracy))
            print("decision_loss_coeff:{0}".format(decision_loss_coeff))
            print("total_loss:{0}".format(losses.avg))
            print("classification_loss:{0}".format(losses_c.avg))
            print("routing_loss:{0}".format(losses_t.avg))
            for lid in range(len(self.pathCounts) - 1):
                print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))
            print("accuracy_avg:{0}".format(accuracy_avg.avg))
            print("batch_time:{0}".format(batch_time.avg))
            print("grad_magnitude:{0}".format(grad_magnitude.avg))
            print("*************Epoch:{0} Iteration:{1}*************".format(
                epoch_id, self.numOfTrainingIterations))
            self.numOfTrainingIterations += 1
        # print("AVERAGE GRAD MAGNITUDE FOR EPOCH:{0}".format(grad_magnitude.avg))

        print("*************Epoch:{0} Ending Measurements*************".format(epoch_id))
        print("decision_loss_coeff:{0}".format(decision_loss_coeff))
        print("total_loss:{0}".format(losses.avg))
        print("classification_loss:{0}".format(losses_c.avg))
        print("routing_loss:{0}".format(losses_t.avg))
        for lid in range(len(self.pathCounts) - 1):
            print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))
        print("accuracy_avg:{0}".format(accuracy_avg.avg))
        print("batch_time:{0}".format(batch_time.avg))
        print("grad_magnitude:{0}".format(grad_magnitude.avg))
        print("*************Epoch:{0} Ending Measurements*************".format(epoch_id))
        return batch_time.avg

    def save_cigt_model(self, epoch):
        db_name = DbLogger.log_db_path.split("/")[-1].split(".")[0]
        checkpoint_file_root = os.path.join(self.modelFilesRootPath, "{0}_{1}".format(db_name, self.runId))
        checkpoint_file_path = checkpoint_file_root + "_epoch{0}.pth".format(epoch)
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.modelOptimizer.state_dict()
        }, checkpoint_file_path)

    def fit(self):
        original_decision_coeff = self.decisionLossCoeff
        self.to(self.device)
        torch.manual_seed(1)
        best_performance = 0.0

        # Cifar 10 Dataset
        kwargs = {'num_workers': 2, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=self.transformTrain),
            batch_size=self.batchSize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=self.transformTest),
            batch_size=self.batchSize, shuffle=False, **kwargs)

        print("Type of optimizer:{0}".format(self.modelOptimizer))
        # self.validate(loader=train_loader, data_kind="train", epoch=0)
        # self.validate(loader=test_loader, data_kind="test", epoch=0)

        total_epoch_count = self.epochCount + self.warmUpPeriod
        for epoch in range(0, total_epoch_count):
            self.adjust_learning_rate(epoch)
            self.adjust_warmup(epoch)

            # train for one epoch
            train_mean_batch_time = self.train_single_epoch(epoch_id=epoch, train_loader=train_loader)

            if epoch % self.evaluationPeriod == 0 or epoch >= (total_epoch_count - 10):
                print("***************Epoch {0} End, Training Evaluation***************".format(epoch))
                train_accuracy = self.validate(loader=train_loader, epoch=epoch, data_kind="train")
                print("***************Epoch {0} End, Test Evaluation***************".format(epoch))
                test_accuracy = self.validate(loader=test_loader, epoch=epoch, data_kind="test")

                if test_accuracy > best_performance:
                    self.save_cigt_model(epoch=epoch)
                    best_performance = test_accuracy

                DbLogger.write_into_table(
                    rows=[(self.runId,
                           self.numOfTrainingIterations,
                           epoch,
                           train_accuracy,
                           0.0,
                           test_accuracy,
                           train_mean_batch_time,
                           0.0,
                           0.0,
                           "YYY")], table=DbLogger.logsTable)

    def validate(self, loader, epoch, data_kind):
        """Perform validation on the validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_c = AverageMeter()
        losses_t = AverageMeter()
        losses_t_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
        accuracy_avg = AverageMeter()
        list_of_labels = []
        list_of_routing_probability_matrices = []
        for _ in range(len(self.pathCounts) - 1):
            list_of_routing_probability_matrices.append([])

        # Temperature of Gumble Softmax
        # We simply keep it fixed
        temperature = self.temperatureController.get_value()

        # switch to evaluate mode
        self.eval()
        self.hardRoutingAlgorithmKind = "InformationGainRouting"

        for i, (input_, target) in enumerate(loader):
            time_begin = time.time()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input_).to(self.device)
                target_var = torch.autograd.Variable(target).to(self.device)
                batch_size = input_var.size(0)

                # Cigt moe output, information gain losses
                routing_matrices_hard, routing_matrices_soft, block_outputs, list_of_logits = self(
                    input_var, target_var, temperature)
                classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                    list_of_logits,
                    routing_matrices_hard,
                    target_var)
                information_gain_losses = self.calculate_information_gain_losses(
                    routing_matrices=routing_matrices_soft, labels=target_var,
                    balance_coefficient_list=self.informationGainBalanceCoeffList)
                total_routing_loss = 0.0
                for t_loss in information_gain_losses:
                    total_routing_loss += t_loss
                total_routing_loss = -1.0 * self.decisionLossCoeff * total_routing_loss
                total_loss = classification_loss + total_routing_loss

                # print("len(list_of_logits)={0}".format(len(list_of_logits)))
                # print("multipleCeLosses:{0}".format(self.multipleCeLosses))
                time_end = time.time()

                list_of_labels.append(target_var.cpu().numpy())
                for idx_, matr_ in enumerate(routing_matrices_soft[1:]):
                    list_of_routing_probability_matrices[idx_].append(matr_.detach().cpu().numpy())

                # measure accuracy and record loss
                losses.update(total_loss.detach().cpu().numpy().item(), 1)
                losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
                accuracy_avg.update(batch_accuracy, batch_size)
                batch_time.update((time_end - time_begin), 1)
                losses_t.update(total_routing_loss.detach().cpu().numpy().item(), 1)
                for lid in range(len(self.pathCounts) - 1):
                    losses_t_layer_wise[lid].update(information_gain_losses[lid].detach().cpu().numpy().item(), 1)

        kv_rows = []
        list_of_labels = np.concatenate(list_of_labels, axis=0)
        for idx_ in range(len(list_of_routing_probability_matrices)):
            list_of_routing_probability_matrices[idx_] = np.concatenate(
                list_of_routing_probability_matrices[idx_], axis=0)

        self.calculate_branch_statistics(
            run_id=self.runId,
            iteration=self.numOfTrainingIterations,
            dataset_type=data_kind,
            labels=list_of_labels,
            routing_probability_matrices=list_of_routing_probability_matrices,
            write_to_db=True)

        print("total_loss:{0}".format(losses.avg))
        kv_rows.append((self.runId,
                        self.numOfTrainingIterations,
                        "{0} Epoch {1} total_loss".format(data_kind, epoch),
                        "{0}".format(losses.avg)))

        print("accuracy_avg:{0}".format(accuracy_avg.avg))
        kv_rows.append((self.runId,
                        self.numOfTrainingIterations,
                        "{0} Epoch {1} Accuracy".format(data_kind, epoch),
                        "{0}".format(accuracy_avg.avg)))

        print("batch_time:{0}".format(batch_time.avg))
        kv_rows.append((self.runId,
                        self.numOfTrainingIterations,
                        "{0} Epoch {1} batch_time".format(data_kind, epoch),
                        "{0}".format(batch_time.avg)))

        print("classification_loss:{0}".format(losses_c.avg))
        kv_rows.append((self.runId,
                        self.numOfTrainingIterations,
                        "{0} Epoch {1} classification_loss".format(data_kind, epoch),
                        "{0}".format(losses_c.avg)))

        print("routing_loss:{0}".format(losses_t.avg))
        kv_rows.append((self.runId,
                        self.numOfTrainingIterations,
                        "{0} Epoch {1} routing_loss".format(data_kind, epoch),
                        "{0}".format(losses_t.avg)))

        for lid in range(len(self.pathCounts) - 1):
            print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))
            kv_rows.append((self.runId,
                            self.numOfTrainingIterations,
                            "{0} Epoch {1} Layer {2} routing_loss".format(data_kind, epoch, lid),
                            "{0}".format(losses_t_layer_wise[lid].avg)))

        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
        return accuracy_avg.avg

    # def random_fine_tuning(self):
    #     self.isInWarmUp = False
    #     original_decision_coeff = self.decisionLossCoeff
    #     self.to(self.device)
    #     torch.manual_seed(1)
    #     best_performance = 0.0
    #
    #     # Cifar 10 Dataset
    #     kwargs = {'num_workers': 2, 'pin_memory': True}
    #     train_loader = torch.utils.data.DataLoader(
    #         datasets.CIFAR10('../data', train=True, download=True, transform=self.transformTrain),
    #         batch_size=self.batchSize, shuffle=True, **kwargs)
    #     val_loader = torch.utils.data.DataLoader(
    #         datasets.CIFAR10('../data', train=False, transform=self.transformTest),
    #         batch_size=self.batchSize, shuffle=False, **kwargs)
    #
    #     print("Type of optimizer:{0}".format(self.modelOptimizer))
    #
    #     total_epoch_count = self.epochCount
    #     for epoch in range(0, total_epoch_count):
    #         self.adjust_learning_rate(epoch)
    #         print("***************Random Fine Tuning "
    #               "Epoch {0} End, Test Evaluation***************".format(epoch))
    #         # test_accuracy = self.validate(loader=val_loader, epoch=epoch, data_kind="test")
    #
    #         # train for one epoch, disabling information gain, randomly routing samples
    #         self.randomFineTuning = True
    #         self.decisionLossCoeff = 0.0
    #         train_mean_batch_time = self.train_single_epoch(epoch_id=epoch, train_loader=train_loader)
    #
    #         if epoch % self.evaluationPeriod == 0 or epoch >= (total_epoch_count - 10):
    #             self.randomFineTuning = False
    #             self.decisionLossCoeff = original_decision_coeff
    #             print("***************Epoch {0} End, Training Evaluation***************".format(epoch))
    #             train_accuracy = self.validate(loader=train_loader, epoch=epoch, data_kind="train")
    #             print("***************Epoch {0} End, Test Evaluation***************".format(epoch))
    #             test_accuracy = self.validate(loader=val_loader, epoch=epoch, data_kind="test")
    #
    #             if test_accuracy > best_performance:
    #                 self.save_cigt_model(epoch=epoch)
    #                 best_performance = test_accuracy
    #
    #             DbLogger.write_into_table(
    #                 rows=[(self.runId,
    #                        self.numOfTrainingIterations,
    #                        epoch,
    #                        train_accuracy,
    #                        0.0,
    #                        test_accuracy,
    #                        train_mean_batch_time,
    #                        0.0,
    #                        0.0,
    #                        "YYY")], table=DbLogger.logsTable)
