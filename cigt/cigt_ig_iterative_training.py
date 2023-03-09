import torch
import time
import numpy as np
import torch
import os
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_different_losses import CigtIgDifferentLosses
from cigt.cigt_ig_soft_routing import CigtIgSoftRouting
from cigt.cigt_soft_routing import CigtSoftRouting
from cigt.resnet_cigt_constants import ResnetCigtConstants
from cigt.routing_layers.info_gain_routing_layer import InfoGainRoutingLayer
from cigt.routing_layers.soft_routing_module import SoftRoutingModule
from convnet_aig import BasicBlock, Sequential_ext


class CigtIgIterativeTraining(CigtIgDifferentLosses):
    def __init__(self, run_id, model_definition):
        self.currentLoss = None
        super().__init__(run_id, model_definition)

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

        assert len(self.modelOptimizer["classification_optimizer"].param_groups) == len(self.pathCounts) + 1

        # Cigt layers with boosted lrs.
        for layer_id in range(len(self.pathCounts)):
            if self.boostLearningRatesLayerWise:
                self.modelOptimizer[
                    "classification_optimizer"].param_groups[layer_id]['lr'] \
                    = self.layerCoefficients[layer_id] * base_lr
            else:
                self.modelOptimizer["classification_optimizer"].param_groups[layer_id]['lr'] = base_lr
        assert len(self.pathCounts) == len(self.modelOptimizer["classification_optimizer"].param_groups) - 1
        # Shared parameters
        self.modelOptimizer["classification_optimizer"].param_groups[-1]['lr'] = base_lr
        # Routing layer parameters
        self.modelOptimizer["routing_optimizer"].param_groups[0]['lr'] = base_lr

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
                    layer_id, self.modelOptimizer["classification_optimizer"].param_groups[layer_id]['lr']))
            print("Shared layer learning rate:{0}".format(
                self.modelOptimizer["classification_optimizer"].param_groups[-1]['lr']))
            print("Routing layers learning rate:{0}".format(
                self.modelOptimizer["routing_optimizer"].param_groups[0]['lr']
            ))

            assert len(self.pathCounts) == len(self.modelOptimizer["classification_optimizer"].param_groups) - 1

            self.modelOptimizer[self.currentLoss].zero_grad()
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
                list_of_logits, routing_matrices = self(input_var, target_var, temperature)
                classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                    list_of_logits,
                    routing_matrices,
                    target_var)
                information_gain_losses = self.calculate_information_gain_losses(
                    routing_matrices=routing_matrices, labels=target_var,
                    balance_coefficient=self.informationGainBalanceCoeff)
                total_routing_loss = 0.0
                for t_loss in information_gain_losses:
                    total_routing_loss += t_loss
                total_routing_loss = -1.0 * self.decisionLossCoeff * total_routing_loss
                # total_loss = classification_loss + total_routing_loss
                # print("len(list_of_logits)={0}".format(len(list_of_logits)))
                # print("multipleCeLosses:{0}".format(self.multipleCeLosses))
                if self.currentLoss == "classification_optimizer":
                    print("Optimizing classification")
                    classification_loss.backward()
                    self.modelOptimizer["classification_optimizer"].step()
                elif self.currentLoss == "routing_optimizer":
                    print("Optimizing routing")
                    total_routing_loss.backward()
                    self.modelOptimizer["routing_optimizer"].step()

            time_end = time.time()
            # measure accuracy and record loss
            print("Epoch:{0} Iteration:{1}".format(epoch_id, self.numOfTrainingIterations))

            losses.update((classification_loss + total_routing_loss).detach().cpu().numpy().item(), 1)
            losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
            accuracy_avg.update(batch_accuracy, batch_size)
            batch_time.update((time_end - time_begin), 1)
            losses_t.update(total_routing_loss.detach().cpu().numpy().item(), 1)
            for lid in range(len(self.pathCounts) - 1):
                losses_t_layer_wise[lid].update(information_gain_losses[lid].detach().cpu().numpy().item(), 1)

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

    def save_cigt_model(self, epoch):
        db_name = DbLogger.log_db_path.split("/")[-1].split(".")[0]
        checkpoint_file_root = os.path.join(
            "/clusterusers/can.bicici@boun.edu.tr/cigt", "{0}_{1}".format(db_name, self.runId))
        checkpoint_file_path = checkpoint_file_root + "_epoch{0}.pth".format(epoch)
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict_classification_optimizer":
                self.modelOptimizer["classification_optimizer"].state_dict(),
            "optimizer_state_dict_routing_optimizer":
                self.modelOptimizer["routing_optimizer"].state_dict()
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

        # Iteration structure
        iterations_list = [{"warm_up": True,
                            "epoch_count": self.warmUpPeriod,
                            "loss": "classification_optimizer"}]
        for i_ in range(ResnetCigtConstants.outer_loop_count):
            iterations_list.append({"warm_up": False,
                                    "epoch_count": ResnetCigtConstants.single_loss_epoch_count,
                                    "loss": "routing_optimizer"})
            iterations_list.append({"warm_up": False,
                                    "epoch_count": ResnetCigtConstants.single_loss_epoch_count,
                                    "loss": "classification_optimizer"})

        total_epoch_count = 0
        for outer_iteration_config in iterations_list:
            print("New outer iteration!:{0}".format(outer_iteration_config))
            self.isInWarmUp = outer_iteration_config["warm_up"]
            inner_epoch_count = outer_iteration_config["epoch_count"]
            self.currentLoss = outer_iteration_config["loss"]
            for _ep in range(0, inner_epoch_count):
                self.adjust_learning_rate(total_epoch_count)
                # train for one epoch
                train_mean_batch_time = self.train_single_epoch(epoch_id=total_epoch_count,
                                                                train_loader=train_loader)
                # validations
                print("***************Epoch {0} End, Training Evaluation***************".format(total_epoch_count))
                train_accuracy = self.validate(loader=train_loader,
                                               epoch=total_epoch_count,
                                               data_kind="train")
                print("***************Epoch {0} End, Test Evaluation***************".format(total_epoch_count))
                test_accuracy = self.validate(loader=val_loader,
                                              epoch=total_epoch_count,
                                              data_kind="test")
                if test_accuracy > best_performance:
                    self.save_cigt_model(epoch=total_epoch_count)
                    best_performance = test_accuracy

                DbLogger.write_into_table(
                    rows=[(self.runId,
                           self.numOfTrainingIterations,
                           total_epoch_count,
                           train_accuracy,
                           0.0,
                           test_accuracy,
                           train_mean_batch_time,
                           0.0,
                           0.0,
                           "YYY")], table=DbLogger.logsTable)
                total_epoch_count += 1

            if self.isInWarmUp:
                self.warmUpFinalIteration = self.numOfTrainingIterations
