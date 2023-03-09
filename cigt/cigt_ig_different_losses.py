import torch
import time
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_soft_routing import CigtIgSoftRouting
from cigt.cigt_soft_routing import CigtSoftRouting
from cigt.routing_layers.info_gain_routing_layer import InfoGainRoutingLayer
from cigt.routing_layers.soft_routing_module import SoftRoutingModule
from convnet_aig import BasicBlock, Sequential_ext


class CigtIgDifferentLosses(CigtIgSoftRouting):
    def __init__(self, run_id, model_definition):
        self.classCount = 10
        super().__init__(run_id, model_definition)

    def get_routing_layer(self, cigt_layer_id, input_feature_map_size, input_feature_map_count):
        h_blocks = self.blockParametersList[cigt_layer_id][1][-self.routerLayersCount:]
        routing_layer = SoftRoutingModule(
            blocks=h_blocks,
            feature_dim=self.decisionDimensions[cigt_layer_id],
            avg_pool_stride=self.decisionAveragePoolingStrides[cigt_layer_id],
            path_count=self.pathCounts[cigt_layer_id + 1],
            class_count=self.numClasses,
            input_feature_map_size=input_feature_map_size,
            input_feature_map_count=input_feature_map_count,
            device=self.device)
        return routing_layer

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
        shared_parameters_dict = {}
        routing_layer_parameters = []
        routing_layer_parameters_dict = {}
        parameters_per_cigt_layers = []
        parameters_per_cigt_layers_dict = {}
        for idx in range(len(self.pathCounts)):
            parameters_per_cigt_layers.append([])

        for name, param in self.named_parameters():
            if "cigtLayers" in name:
                parameters_per_cigt_layers_dict[name] = param
                param_name_splitted = name.split(".")
                layer_id = int(param_name_splitted[1])
                assert 0 <= layer_id <= len(self.pathCounts) - 1
                parameters_per_cigt_layers[layer_id].append(param)
            elif "blockEndLayers" in name:
                routing_layer_parameters.append(param)
                routing_layer_parameters_dict[name] = param
            else:
                shared_parameters.append(param)
                shared_parameters_dict[name] = param

        num_shared_parameters = len(shared_parameters)
        num_routing_parameters = len(routing_layer_parameters)
        num_cigt_layer_parameters = sum([len(arr) for arr in parameters_per_cigt_layers])
        num_all_parameters = len([tpl for tpl in self.named_parameters()])
        assert num_shared_parameters + num_routing_parameters + num_cigt_layer_parameters == num_all_parameters

        parameter_groups_classification = []
        # Add parameter groups with respect to their cigt layers
        for layer_id in range(len(self.pathCounts)):
            parameter_groups_classification.append(
                {'params': parameters_per_cigt_layers[layer_id],
                 'lr': self.initialLr * self.layerCoefficients[layer_id],
                 'weight_decay': self.classificationWd})

        # Shared parameters, always the last group
        parameter_groups_classification.append(
            {'params': shared_parameters,
             'lr': self.initialLr,
             'weight_decay': self.classificationWd})

        # Parameters for the routing layers; this will be optimized separately.
        parameter_groups_routing = [{'params': routing_layer_parameters,
                                     'lr': self.initialLr,
                                     'weight_decay': self.classificationWd}]

        if self.optimizerType == "SGD":
            classification_optimizer = optim.SGD(parameter_groups_classification, momentum=0.9)
            routing_optimizer = optim.SGD(parameter_groups_routing, momentum=0.9)
        elif self.optimizerType == "Adam":
            classification_optimizer = optim.Adam(parameter_groups_classification)
            routing_optimizer = optim.Adam(parameter_groups_routing)
        else:
            raise ValueError("{0} is not supported as optimizer.".format(self.optimizerType))
        return {"classification_optimizer": classification_optimizer,
                "routing_optimizer": routing_optimizer}

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

            self.modelOptimizer["classification_optimizer"].zero_grad()
            self.modelOptimizer["routing_optimizer"].zero_grad()
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
                classification_loss.backward(retain_graph=True)
                total_routing_loss.backward(retain_graph=True)
                self.modelOptimizer["classification_optimizer"].step()
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
