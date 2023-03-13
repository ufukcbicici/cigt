from collections import OrderedDict

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from randaugment import RandAugment
from torchvision import transforms
import torchvision.datasets as datasets
import time
import numpy as np

from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cutout_augmentation import CutoutPIL
from cigt.routing_layers.info_gain_routing_layer import InfoGainRoutingLayer
from cigt.moe_layer import MoeLayer
from cigt.resnet_cigt_constants import ResnetCigtConstants


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + out
        out = F.relu(out)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        return out


class Sequential_ext(nn.Module):
    """A Sequential container extended to also propagate the gating information
    that is needed in the target rate loss.
    """

    def __init__(self, *args):
        super(Sequential_ext, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input_):
        x = input_
        for i, module in enumerate(self._modules.values()):
            x = module(x)
            # print(x.max())
        return x


class Cigt(nn.Module):
    def __init__(self, run_id, model_definition, num_classes=10):
        super().__init__()
        self.runId = run_id
        self.modelDefinition = model_definition
        self.imageSize = (32, 32)
        self.numClasses = num_classes
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

        # Build Cigt Blocks
        self.cigtLayers = nn.ModuleList()
        self.blockEndLayers = nn.ModuleList()
        self.create_cigt_blocks()

        # MoE Loss Layer
        self.moeLossLayer = MoeLayer()
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.nllLoss = nn.NLLLoss()
        self.crossEntropyLoss = nn.CrossEntropyLoss()

        self.numOfTrainingIterations = 0
        self.warmUpFinalIteration = 0

        self.layerCoefficients = []
        self.modelOptimizer = self.create_optimizer()

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
                if cigt_layer_id == len(self.blockParametersList) - 1:
                    last_dim = cigt_layer_info[-1]["out_dimension"]
                    layers.append(torch.nn.AvgPool2d(kernel_size=8))
                    layers.append(torch.nn.Flatten())
                    layers.append(torch.nn.Linear(in_features=last_dim, out_features=self.numClasses))
                block_obj = Sequential_ext(*layers)
                # block_obj.name = "block_{0}_{1}".format(cigt_layer_id, path_id)
                cigt_layer_blocks.append(block_obj)
            # cigt_layer_blocks[0].eval()
            # cigt_layer_output_dummy = cigt_layer_blocks[0]()
            self.cigtLayers.append(cigt_layer_blocks)
            # Block end layers: Routing layers for inner layers, loss layer for the last one.
            if cigt_layer_id < len(self.blockParametersList) - 1:
                for inner_block_info in cigt_layer_info:
                    feature_edge_size = int(feature_edge_size / inner_block_info["stride"])
                routing_layer = InfoGainRoutingLayer(
                    feature_dim=self.decisionDimensions[cigt_layer_id],
                    avg_pool_stride=self.decisionAveragePoolingStrides[cigt_layer_id],
                    path_count=self.pathCounts[cigt_layer_id + 1],
                    class_count=self.numClasses,
                    input_feature_map_size=feature_edge_size,
                    input_feature_map_count=cigt_layer_info[-1]["out_dimension"],
                    device=self.device)
                self.blockEndLayers.append(routing_layer)

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

    def forward(self, x, labels, temperature):
        moe_probs = 0.0
        balance_coefficient_list = self.informationGainBalanceCoeffList
        # Classification loss
        classification_loss = 0.0
        # Information gain losses
        information_gain_losses = torch.zeros(size=(len(self.cigtLayers) - 1,), dtype=torch.float32, device=self.device)
        # Routing Matrices
        routing_matrices = [torch.ones(size=(x.shape[0], 1), dtype=torch.float32, device=self.device)]
        # Initial layer
        out = F.relu(self.bn1(self.conv1(x)))
        prev_layer_outputs = {(): out}

        # Execute Cigt layers
        for layer_id, cigt_layer_blocks in enumerate(self.cigtLayers):
            curr_layer_outputs = {}
            for path_so_far, out in prev_layer_outputs.items():
                for block_id, block_obj in enumerate(cigt_layer_blocks):
                    block_output = block_obj(out)
                    curr_path = (*path_so_far, block_id)
                    curr_layer_outputs[curr_path] = block_output

            # Routing Layer
            if layer_id < len(self.cigtLayers) - 1:
                information_gain, p_n_given_x = \
                    self.blockEndLayers[layer_id](curr_layer_outputs,
                                                  routing_matrices, labels, temperature,
                                                  balance_coefficient_list[layer_id])
                information_gain_losses[layer_id] = information_gain
                # If in warm up, send into all blocks
                if self.isInWarmUp:
                    routing_matrix = torch.ones_like(p_n_given_x)
                    row_normalizing_constants = torch.reciprocal(torch.sum(routing_matrix, dim=1, keepdim=True))
                    routing_matrix = routing_matrix * row_normalizing_constants
                    routing_matrices.append(routing_matrix)
                else:
                    routing_matrix = torch.zeros_like(p_n_given_x)
                    arg_max_entries = torch.argmax(p_n_given_x, dim=1)
                    routing_matrix[torch.arange(routing_matrix.shape[0]), arg_max_entries] = 1.0
                    routing_matrices.append(routing_matrix)
            # Classification Loss Layer
            else:
                moe_probs = self.moeLossLayer(curr_layer_outputs, routing_matrices)

            prev_layer_outputs = curr_layer_outputs
        # total_loss = classification_loss + decision_loss_coefficient * torch.sum(information_gain_losses)
        return moe_probs, information_gain_losses

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

        # return adjusted_lr

    def measure_accuracy(self, probs, target):
        """Computes the precision@k for the specified values of k"""
        pred = torch.argmax(probs, dim=1)
        correct_vector = pred.eq(target).to(torch.float)
        acc = torch.mean(correct_vector)
        return acc.cpu().numpy().item()

    def add_explanation(self, name_of_param, value, explanation, kv_rows):
        explanation += "{0}:{1}\n".format(name_of_param, value)
        kv_rows.append((self.runId, name_of_param, "{0}".format(value)))
        return explanation

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
                print("Temperature:{0}".format(temperature))

                # Cigt moe output, information gain losses
                moe_probs, information_gain_losses = self(input_var, target_var, temperature)
                # print(moe_probs.min())
                diff_arr = moe_probs - np.finfo(float).eps
                diff_arr[diff_arr > 0.0] = 0.0
                moe_probs = moe_probs - diff_arr
                # print(moe_probs.min())
                log_moe_probs = torch.log(moe_probs)
                classification_loss = self.nllLoss(log_moe_probs, target_var)
                routing_loss = decision_loss_coeff * torch.sum(information_gain_losses)
                total_loss = routing_loss

                # compute gradient and do SGD step
                qq = [tpl for tpl in self.named_parameters() if tpl[0] == 'blockEndLayers.0.fc2.weight'][0]
                # print("Grad Magnitude Before:{0}".format(torch.linalg.norm(qq[1].grad)))
                total_loss.backward()
                print("Grad Magnitude After:{0}".format(torch.linalg.norm(qq[1].grad)))
                grad_magnitude.update(torch.linalg.norm(qq[1].grad).cpu().numpy(), 1)
                self.modelOptimizer.step()

            time_end = time.time()
            # measure accuracy and record loss
            print("Epoch:{0} Iteration:{1}".format(epoch_id, self.numOfTrainingIterations))
            batch_accuracy = self.measure_accuracy(moe_probs.detach().cpu(), target.cpu())
            losses.update(total_loss.detach().cpu().numpy().item(), 1)
            losses_c.update(classification_loss.detach().cpu().numpy().item(), batch_size)
            losses_t.update(routing_loss.detach().cpu().numpy().item(), 1)
            for lid in range(len(self.pathCounts) - 1):
                losses_t_layer_wise[lid].update(information_gain_losses.detach().cpu().numpy()[lid].item(), 1)
            accuracy_avg.update(batch_accuracy, batch_size)
            batch_time.update((time_end - time_begin), 1)

            print("batch_accuracy:{0}".format(batch_accuracy))
            # print("total_loss:{0}".format(total_loss.detach().cpu().numpy().item()))
            # print("classification_loss:{0}".format(classification_loss.detach().cpu().numpy().item()))
            # print("routing_loss:{0}".format(routing_loss.detach().cpu().numpy().item()))
            # print("accuracy_avg:{0}".format(accuracy_avg.val))
            # print("batch_time:{0}".format(time_end - time_begin))
            print("decision_loss_coeff:{0}".format(decision_loss_coeff))
            print("total_loss:{0}".format(losses.val))
            print("classification_loss:{0}".format(losses_c.val))
            print("routing_loss:{0}".format(losses_t.val))
            for lid in range(len(self.pathCounts) - 1):
                print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))
            print("accuracy_avg:{0}".format(accuracy_avg.val))
            print("batch_time:{0}".format(batch_time.val))
            print("grad_magnitude:{0}".format(grad_magnitude.avg))
            print("*************Epoch:{0} Iteration:{1}*************".format(
                epoch_id, self.numOfTrainingIterations))
            self.numOfTrainingIterations += 1
        print("AVERAGE GRAD MAGNITUDE FOR EPOCH:{0}".format(grad_magnitude.avg))
        return batch_time.avg

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

    def validate(self, loader, epoch, data_kind):
        """Perform validation on the validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_c = AverageMeter()
        losses_t = AverageMeter()
        losses_t_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
        accuracy_avg = AverageMeter()

        # Temperature of Gumble Softmax
        # We simply keep it fixed
        temperature = 1.0

        # switch to evaluate mode
        self.eval()

        for i, (input_, target) in enumerate(loader):
            time_begin = time.time()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input_).to(self.device)
                target_var = torch.autograd.Variable(target).to(self.device)
                batch_size = input_var.size(0)

                # Cigt moe output, information gain losses
                moe_probs, information_gain_losses = self(input_var, target_var, 1.0)
                log_moe_probs = torch.log(moe_probs)
                classification_loss = self.nllLoss(log_moe_probs, target_var)
                routing_loss = torch.sum(information_gain_losses)
                total_loss = classification_loss + routing_loss
                time_end = time.time()

                # measure accuracy and record loss
                batch_accuracy = self.measure_accuracy(moe_probs.detach().cpu(), target.cpu())
                losses.update(total_loss.detach().cpu().numpy().item(), 1)
                losses_c.update(classification_loss.detach().cpu().numpy().item(), batch_size)
                losses_t.update(routing_loss.detach().cpu().numpy().item(), 1)
                for lid in range(len(self.pathCounts) - 1):
                    losses_t_layer_wise[lid].update(information_gain_losses.detach().cpu().numpy()[lid].item(), 1)
                accuracy_avg.update(batch_accuracy, batch_size)
                batch_time.update((time_end - time_begin), 1)

        kv_rows = []

        print("total_loss:{0}".format(losses.avg))
        kv_rows.append((self.runId,
                        self.numOfTrainingIterations,
                        "{0} Epoch {1} total_loss".format(data_kind, epoch),
                        "{0}".format(losses.avg)))

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
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
        return accuracy_avg.avg

    def save_cigt_model(self, epoch):
        db_name = DbLogger.log_db_path.split("/")[-1].split(".")[0]
        checkpoint_file_root = os.path.join(self.modelFilesRootPath, "{0}_{1}".format(db_name, self.runId))
        checkpoint_file_path = checkpoint_file_root + "_epoch{0}.pth".format(epoch)
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.modelOptimizer.state_dict()
        }, checkpoint_file_path)

    def fit(self):
        self.to(self.device)
        torch.manual_seed(1)
        best_performance = 0.0

        # Cifar 10 Dataset
        kwargs = {'num_workers': 2, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=self.transformTrain),
            batch_size=self.batchSize, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=self.transformTest),
            batch_size=self.batchSize, shuffle=False, **kwargs)

        print("Type of optimizer:{0}".format(self.modelOptimizer))

        total_epoch_count = self.epochCount + self.warmUpPeriod
        for epoch in range(0, total_epoch_count):
            self.adjust_learning_rate(epoch)

            if epoch >= self.warmUpPeriod and self.isInWarmUp:
                print("Warmup is ending!")
                self.isInWarmUp = False
                self.warmUpFinalIteration = self.numOfTrainingIterations

            print("***************Epoch {0} End, Test Evaluation***************".format(epoch))
            # test_accuracy = self.validate(loader=val_loader, epoch=epoch, data_kind="test")

            # train for one epoch
            train_mean_batch_time = self.train_single_epoch(epoch_id=epoch, train_loader=train_loader)

            if epoch % self.evaluationPeriod == 0 or epoch >= (total_epoch_count - 10):
                print("***************Epoch {0} End, Training Evaluation***************".format(epoch))
                train_accuracy = self.validate(loader=train_loader, epoch=epoch, data_kind="train")
                print("***************Epoch {0} End, Test Evaluation***************".format(epoch))
                test_accuracy = self.validate(loader=val_loader, epoch=epoch, data_kind="test")

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
