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


class CigtVarianceRouting(CigtIgSoftRouting):
    def __init__(self, run_id, model_definition):
        self.currentLoss = None
        super().__init__(run_id, model_definition)

    def calculate_information_gain_losses(self, routing_matrices, labels, balance_coefficient_list):
        information_gain_list = []
        for p_n_given_x in routing_matrices[1:]:
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

            # Routing loss
            mean_prob_p_cn = 1.0 / (marginal_p_cn.shape[0] * marginal_p_cn.shape[1])
            A_ = torch.pow(marginal_p_cn - mean_prob_p_cn, 2.0)
            routing_loss = torch.mean(A_)

            # Label balance loss
            mean_prob_p_c = 1.0 / (marginal_p_c.shape[0])
            B_ = torch.pow(marginal_p_c - mean_prob_p_c, 2.0)
            label_balance_loss = torch.mean(B_)

            # Path balance loss
            mean_prob_p_n = 1.0 / (marginal_p_n.shape[0])
            C_ = torch.pow(marginal_p_n - mean_prob_p_n, 2.0)
            path_balance_loss = torch.mean(C_)
            information_gain_list.append({"routing_loss": routing_loss,
                                          "label_balance_loss": label_balance_loss,
                                          "path_balance_loss": path_balance_loss})
        return information_gain_list

    def train_single_epoch(self, epoch_id, train_loader):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_c = AverageMeter()
        losses_t = AverageMeter()
        losses_routing_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
        losses_label_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
        losses_path_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
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
                list_of_logits, routing_matrices = self(input_var, target_var, temperature)
                classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                    list_of_logits,
                    routing_matrices,
                    target_var)
                information_gain_losses = self.calculate_information_gain_losses(
                    routing_matrices=routing_matrices, labels=target_var,
                    balance_coefficient=self.informationGainBalanceCoeff)

                route_lambda = 1.0
                label_lambda = 1.0
                path_lambda = 1.0
                total_routing_loss = 0.0
                # CORRECT!
                for lid, loss_dict in enumerate(information_gain_losses):
                    total_routing_loss -= route_lambda * loss_dict["routing_loss"]
                    losses_routing_layer_wise[lid].update(
                        loss_dict["routing_loss"].detach().cpu().numpy(), 1)

                    total_routing_loss += label_lambda * loss_dict["label_balance_loss"]
                    losses_label_layer_wise[lid].update(
                        loss_dict["label_balance_loss"].detach().cpu().numpy(), 1)

                    total_routing_loss += path_lambda * loss_dict["path_balance_loss"]
                    losses_path_layer_wise[lid].update(
                        loss_dict["path_balance_loss"].detach().cpu().numpy(), 1)

                total_loss = classification_loss + total_routing_loss
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

            print("decision_loss_coeff:{0}".format(decision_loss_coeff))
            print("total_loss:{0}".format(losses.val))
            print("classification_loss:{0}".format(losses_c.val))
            print("routing_loss:{0}".format(losses_t.val))
            for lid in range(len(self.pathCounts) - 1):
                print("Layer {0} routing_loss:{1}".format(lid, losses_routing_layer_wise[lid].avg))
                print("Layer {0} label_balance_loss:{1}".format(lid, losses_label_layer_wise[lid].avg))
                print("Layer {0} path_balance_loss:{1}".format(lid, losses_path_layer_wise[lid].avg))

            print("accuracy_avg:{0}".format(accuracy_avg.val))
            print("batch_time:{0}".format(batch_time.val))
            print("grad_magnitude:{0}".format(grad_magnitude.avg))
            print("*************Epoch:{0} Iteration:{1}*************".format(
                epoch_id, self.numOfTrainingIterations))
            self.numOfTrainingIterations += 1
        print("AVERAGE GRAD MAGNITUDE FOR EPOCH:{0}".format(grad_magnitude.avg))
        return batch_time.avg

    def validate(self, loader, epoch, data_kind):
        """Perform validation on the validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_c = AverageMeter()
        losses_t = AverageMeter()
        losses_routing_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
        losses_label_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
        losses_path_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
        grad_magnitude = AverageMeter()
        accuracy_avg = AverageMeter()
        list_of_labels = []
        list_of_routing_probability_matrices = []
        for _ in range(len(self.pathCounts) - 1):
            list_of_routing_probability_matrices.append([])

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
                list_of_logits, routing_matrices = self(input_var, target_var, 1.0)
                classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                    list_of_logits,
                    routing_matrices,
                    target_var)
                information_gain_losses = self.calculate_information_gain_losses(
                    routing_matrices=routing_matrices, labels=target_var,
                    balance_coefficient=self.informationGainBalanceCoeff)
                route_lambda = 1.0
                label_lambda = 1.0
                path_lambda = 1.0
                total_routing_loss = 0.0
                # CORRECT!
                for lid, loss_dict in enumerate(information_gain_losses):
                    total_routing_loss -= route_lambda * loss_dict["routing_loss"]
                    losses_routing_layer_wise[lid].update(
                        loss_dict["routing_loss"].detach().cpu().numpy(), 1)

                    total_routing_loss += label_lambda * loss_dict["label_balance_loss"]
                    losses_label_layer_wise[lid].update(
                        loss_dict["label_balance_loss"].detach().cpu().numpy(), 1)

                    total_routing_loss += path_lambda * loss_dict["path_balance_loss"]
                    losses_path_layer_wise[lid].update(
                        loss_dict["path_balance_loss"].detach().cpu().numpy(), 1)

                time_end = time.time()
                total_loss = classification_loss + total_routing_loss

                list_of_labels.append(target_var.cpu().numpy())
                for idx_, matr_ in enumerate(routing_matrices[1:]):
                    list_of_routing_probability_matrices[idx_].append(matr_.detach().cpu().numpy())

                # measure accuracy and record loss
                losses.update(total_loss.detach().cpu().numpy().item(), 1)
                losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
                accuracy_avg.update(batch_accuracy, batch_size)
                batch_time.update((time_end - time_begin), 1)
                losses_t.update(total_routing_loss.detach().cpu().numpy().item(), 1)
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
            print("Layer {0} routing_loss:{1}".format(lid, losses_routing_layer_wise[lid].avg))
            kv_rows.append((self.runId,
                            self.numOfTrainingIterations,
                            "{0} Epoch {1} Layer {2} routing_loss".format(data_kind, epoch, lid),
                            "{0}".format(losses_routing_layer_wise[lid].avg)))

            print("Layer {0} label_balance_loss:{1}".format(lid, losses_label_layer_wise[lid].avg))
            kv_rows.append((self.runId,
                            self.numOfTrainingIterations,
                            "{0} Epoch {1} Layer {2} label_balance_loss".format(data_kind, epoch, lid),
                            "{0}".format(losses_label_layer_wise[lid].avg)))

            print("Layer {0} path_balance_loss:{1}".format(lid, losses_path_layer_wise[lid].avg))
            kv_rows.append((self.runId,
                            self.numOfTrainingIterations,
                            "{0} Epoch {1} Layer {2} path_balance_loss".format(data_kind, epoch, lid),
                            "{0}".format(losses_path_layer_wise[lid].avg)))

        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
        return accuracy_avg.avg
