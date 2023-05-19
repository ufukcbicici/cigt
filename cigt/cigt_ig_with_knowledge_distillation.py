from cigt.cigt_ig_refactored import CigtIgHardRoutingX
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


class CigtIgWithKnowledgeDistillation(CigtIgHardRoutingX):
    def __init__(self, run_id, model_definition, num_classes, teacher_model):
        self.useKdForRouting = ResnetCigtConstants.use_kd_for_routing
        self.teacherTemperature = ResnetCigtConstants.kd_teacher_temperature
        self.teacherAlpha = ResnetCigtConstants.kd_loss_alpha
        super().__init__(run_id, model_definition, num_classes)
        self.teacherModel = teacher_model

    def calculate_kd_loss_and_accuracy(self, list_of_logits, teacher_logits, routing_matrices, target_var):
        if self.lossCalculationKind == "SingleLogitSingleLoss":
            classification_loss = self.crossEntropyLoss(list_of_logits[0], target_var)
            batch_accuracy = self.measure_accuracy(list_of_logits[0].detach().cpu(), target_var.cpu())
        elif self.lossCalculationKind in {"MultipleLogitsMultipleLosses", "MultipleLogitsMultipleLossesAveraged"}:
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
            if self.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
                classification_loss = classification_loss / len(list_of_logits)
        else:
            raise ValueError("Unknown loss calculation method:{0}".format(self.lossCalculationKind))
        return classification_loss, batch_accuracy

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

                decision_loss_coeff = self.adjust_decision_loss_coeff()
                temperature = self.adjust_temperature()

                print("temperature:{0}".format(temperature))
                print("decision_loss_coeff:{0}".format(decision_loss_coeff))

                # Teacher output
                with torch.no_grad():
                    teacher_rm_hard, teacher_rm_soft, \
                    teacher_bo, teacher_list_of_logits, teacher_ra = self.teacherModel(input_var,
                                                                                       target_var, temperature)
                # Student output
                routing_matrices_hard, routing_matrices_soft, \
                block_outputs, list_of_logits, routing_activations_list = self(input_var, target_var, temperature)

                classification_loss, batch_accuracy = self.calculate_kd_loss_and_accuracy(
                    list_of_logits,
                    teacher_list_of_logits,
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
