import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from cigt.cigt_model import Cigt, BasicBlock, Sequential_ext
from cigt.routing_layers.soft_routing_layer import SoftRoutingLayer
from collections import OrderedDict, Counter


class CigtSoftRouting(Cigt):
    def __init__(self, run_id, model_definition):
        # self.finalPoolingLayer = None  # torch.nn.AvgPool2d(kernel_size=8)
        # self.finalFlattenLayer = None  # torch.nn.Flatten()
        # self.finalFeatureDim = None
        # self.finalLogitsLayer = None
        self.lossLayers = None
        super().__init__(run_id, model_definition)
        self.identityLayer = nn.Identity()
        self.crossEntropyLosses = [nn.CrossEntropyLoss(reduction='none') for _ in range(self.pathCounts[-1])]

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

    def get_loss_layer(self):
        self.lossLayers = nn.ModuleList()
        final_feature_dimension = self.blockParametersList[-1][-1][-1]["out_dimension"]
        if not self.multipleCeLosses:
            end_module = nn.Sequential(OrderedDict([
                ('avg_pool', torch.nn.AvgPool2d(kernel_size=8)),
                ('flatten', torch.nn.Flatten()),
                ('logits', torch.nn.Linear(in_features=final_feature_dimension, out_features=self.numClasses))
            ]))
            self.lossLayers.append(end_module)
        else:
            for block_id in range(self.pathCounts[-1]):
                end_module = nn.Sequential(OrderedDict([
                    ('avg_pool_{0}'.format(block_id), torch.nn.AvgPool2d(kernel_size=8)),
                    ('flatten_{0}'.format(block_id), torch.nn.Flatten()),
                    ('logits_{0}'.format(block_id), torch.nn.Linear(
                        in_features=final_feature_dimension, out_features=self.numClasses))
                ]))
                self.lossLayers.append(end_module)

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
                self.blockEndLayers.append(routing_layer)
        # if cigt_layer_id == len(self.blockParametersList) - 1:
        self.get_loss_layer()

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
        moe_probs = 0.0
        balance_coefficient = self.informationGainBalanceCoeff
        # Classification loss
        classification_loss = 0.0
        # Information gain losses
        information_gain_losses = torch.zeros(size=(len(self.cigtLayers) - 1,), dtype=torch.float32, device=self.device)
        # Routing Matrices
        routing_matrices = []
        # Initial layer
        out = F.relu(self.bn1(self.conv1(x)))
        routing_matrices.append(torch.ones(size=(x.shape[0], 1), dtype=torch.float32, device=self.device))
        # prev_layer_outputs = {(): out}
        list_of_logits = []

        for layer_id, cigt_layer_blocks in enumerate(self.cigtLayers):
            curr_layer_outputs = []
            for block_id, block_obj in enumerate(cigt_layer_blocks):
                block_output = block_obj(out)
                curr_layer_outputs.append(block_output)

            # Routing Layer
            if layer_id < len(self.cigtLayers) - 1:
                # Weighted sum of block outputs
                out = self.weighted_sum_of_tensors(routing_matrix=routing_matrices[-1],
                                                   tensors=curr_layer_outputs)
                # Calculate routing weights for the next layer
                routing_weights = self.blockEndLayers[layer_id](out,
                                                                labels,
                                                                temperature,
                                                                balance_coefficient)
                routing_matrices.append(routing_weights)
            # Logits
            else:
                if not self.multipleCeLosses:
                    # Weighted sum of block outputs
                    out = self.weighted_sum_of_tensors(routing_matrix=routing_matrices[-1],
                                                       tensors=curr_layer_outputs)
                    logits = self.lossLayers[0](out)
                    list_of_logits.append(logits)
                else:
                    for idx, block_x in enumerate(curr_layer_outputs):
                        logits = self.lossLayers[idx](block_x)
                        list_of_logits.append(logits)
        return list_of_logits, routing_matrices

    def calculate_classification_loss_and_accuracy(self, list_of_logits, routing_matrices, target_var):
        if not self.multipleCeLosses:
            classification_loss = self.crossEntropyLoss(list_of_logits[0], target_var)
            batch_accuracy = self.measure_accuracy(list_of_logits[0].detach().cpu(), target_var.cpu())
        else:
            ce_losses = []
            probs = []
            for idx, logit in enumerate(list_of_logits):
                ce_loss = self.crossEntropyLosses[idx](logit, target_var)
                ce_losses.append(ce_loss)
                probs.append(torch.softmax(logit, dim=1))
            weighted_ce_losses = self.weighted_sum_of_tensors(routing_matrix=routing_matrices[-1],
                                                              tensors=ce_losses)
            weighted_probs = self.weighted_sum_of_tensors(routing_matrix=routing_matrices[-1],
                                                          tensors=probs)
            weighted_probs = weighted_probs.detach().cpu()
            classification_loss = torch.mean(weighted_ce_losses)
            batch_accuracy = self.measure_accuracy(weighted_probs, target_var.cpu())
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
                total_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(list_of_logits,
                                                                                             routing_matrices,
                                                                                             target_var)
                print("len(list_of_logits)={0}".format(len(list_of_logits)))
                print("multipleCeLosses:{0}".format(self.multipleCeLosses))
                total_loss.backward()
                self.modelOptimizer.step()

            time_end = time.time()
            # measure accuracy and record loss
            print("Epoch:{0} Iteration:{1}".format(epoch_id, self.numOfTrainingIterations))

            losses.update(total_loss.detach().cpu().numpy().item(), 1)
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

        for i, (input_, target) in enumerate(loader):
            time_begin = time.time()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input_).to(self.device)
                target_var = torch.autograd.Variable(target).to(self.device)
                batch_size = input_var.size(0)

                # Cigt moe output, information gain losses
                # logits = self(input_var, target_var, 1.0)
                # total_loss = self.crossEntropyLoss(logits, target_var)
                list_of_logits, routing_matrices = self(input_var, target_var, temperature)
                total_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(list_of_logits,
                                                                                             routing_matrices,
                                                                                             target_var)
                list_of_labels.append(target_var.cpu().numpy())
                for idx_, matr_ in enumerate(routing_matrices[1:]):
                    list_of_routing_probability_matrices[idx_].append(matr_.detach().cpu().numpy())

                time_end = time.time()

                # measure accuracy and record loss
                losses.update(total_loss.detach().cpu().numpy().item(), 1)
                accuracy_avg.update(batch_accuracy, batch_size)
                batch_time.update((time_end - time_begin), 1)

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
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
        return accuracy_avg.avg
