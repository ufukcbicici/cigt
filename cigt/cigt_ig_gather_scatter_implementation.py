from collections import OrderedDict, Counter
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets

from auxillary.db_logger import DbLogger
from auxillary.average_meter import AverageMeter
from auxillary.utilities import Utilities
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
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
    def __init__(self, configs, run_id, model_definition, num_classes):
        super().__init__(configs, run_id, model_definition, num_classes)

    @staticmethod
    def divide_tensor_wrt_routing_matrix(tens, routing_matrix):
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

    def calculate_logits(self, p_n_given_x_hard, loss_block_outputs):
        list_of_logits = []
        for idx, block_x in enumerate(loss_block_outputs):
            logits = self.lossLayers[idx](block_x)
            list_of_logits.append(logits)
        return list_of_logits

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
                # Calculate routing weights for the next layer
                p_n_given_x_soft, routing_activations = self.blockEndLayers[layer_id](layer_outputs_unified,
                                                                                      layer_labels_unified,
                                                                                      temperature,
                                                                                      balance_coefficient_list[
                                                                                          layer_id])
                # Calculate the hard routing matrix
                p_n_given_x_hard = self.get_hard_routing_matrix(layer_id=layer_id, p_n_given_x_soft=p_n_given_x_soft)
                layer_outputs.append({"net": layer_outputs_unified,
                                      "labels": layer_labels_unified,
                                      "sample_indices": layer_sample_indices_unified,
                                      "block_indices": layer_block_indices_unified,
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
                                      "block_indices": layer_block_indices_unified,
                                      "labels_masked": labels_masked,
                                      "list_of_logits": list_of_logits,
                                      "logits_unified": logits_unified})

        return layer_outputs

    def forward_v2(self, x, labels, temperature):
        sample_indices = torch.arange(0, labels.shape[0], device=self.device)
        balance_coefficient_list = self.informationGainBalanceCoeffList
        result_buffers = []
        # Create buffers for holding the results.
        # Initial layer
        net = self.preprocess_input(x=x)
        routing_matrices_soft = torch.ones(size=(x.shape[0], 1),
                                           dtype=torch.float32,
                                           device=self.device)
        routing_matrices_hard = torch.ones(size=(x.shape[0], 1),
                                           dtype=torch.float32,
                                           device=self.device)
        routing_activations = torch.ones(size=(x.shape[0], 1),
                                         dtype=torch.float32,
                                         device=self.device)
        block_outputs_dict = {(): net}
        routing_matrices_soft_dict = {(): routing_matrices_soft}
        routing_matrices_hard_dict = {(): routing_matrices_hard}
        routing_activations_dict = {(): routing_activations}
        logits_dict = {}

        for layer_id, cigt_layer_blocks in enumerate(self.cigtLayers):
            past_route_combinations = Utilities.create_route_combinations(shape_=self.pathCounts[:layer_id])
            for block_id, block_obj in enumerate(cigt_layer_blocks):
                for route_combination in past_route_combinations:
                    block_input = block_outputs_dict[route_combination]
                    block_output = block_obj(block_input)
                    output_id = tuple([*route_combination, block_id])
                    block_outputs_dict[output_id] = block_output
                    if layer_id < len(self.cigtLayers) - 1:
                        p_n_given_x_soft, routing_activations = self.blockEndLayers[layer_id](block_output,
                                                                                              labels,
                                                                                              temperature,
                                                                                              balance_coefficient_list[
                                                                                                  layer_id])
                        # Calculate the hard routing matrix
                        p_n_given_x_hard = self.get_hard_routing_matrix(layer_id=layer_id,
                                                                        p_n_given_x_soft=p_n_given_x_soft)
                        routing_matrices_soft_dict[output_id] = p_n_given_x_soft
                        routing_matrices_hard_dict[output_id] = p_n_given_x_hard
                        routing_activations_dict[output_id] = routing_activations
                    else:
                        assert self.lossCalculationKind in {"MultipleLogitsMultipleLosses",
                                                            "MultipleLogitsMultipleLossesAveraged"}
                        logits = self.lossLayers[block_id](block_output)
                        logits_dict[output_id] = logits

        return block_outputs_dict, routing_matrices_soft_dict, \
            routing_matrices_hard_dict, routing_activations_dict, logits_dict

    def calculate_classification_loss_and_accuracy(self, list_of_logits, routing_matrices, target_var):
        assert isinstance(list_of_logits, list) and routing_matrices is None and isinstance(target_var, list)
        assert len(list_of_logits) == len(target_var)
        classification_loss = 0.0
        batch_accuracy = 0.0
        total_sample_count = sum([arr.shape[0] for arr in list_of_logits])
        for idx in range(len(list_of_logits)):
            block_logits = list_of_logits[idx]
            block_targets = target_var[idx]
            if block_logits.shape[0] == 0:
                continue
            block_classification_loss = self.calculate_classification_loss_from_logits(
                criterion=self.classificationLosses[idx], logits=block_logits, labels=block_targets)
            classification_loss += block_classification_loss
            block_accuracy = self.measure_accuracy(block_logits.detach().cpu(), block_targets.cpu())
            batch_coefficient = (block_logits.shape[0] / total_sample_count)
            batch_accuracy += batch_coefficient * block_accuracy
        if self.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
            classification_loss = classification_loss / len(list_of_logits)
        return classification_loss, batch_accuracy

    def calculate_information_gain_losses(self, routing_matrices, labels, balance_coefficient_list):
        assert len(routing_matrices) == len(labels) and len(routing_matrices) == len(balance_coefficient_list)
        information_gain_list = []
        for layer_id in range(len(routing_matrices)):
            p_n_given_x = routing_matrices[layer_id]
            labels_for_layer = labels[layer_id]
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
            p_c_given_x = torch.nn.functional.one_hot(labels_for_layer, self.classCount)
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

                # Cigt Classification Loss and Accuracy Calculation
                layer_outputs = self(input_var, target_var, temperature)
                if self.lossCalculationKind == "SingleLogitSingleLoss":
                    classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                        list_of_logits=layer_outputs[-1]["list_of_logits"],
                        routing_matrices=None,
                        target_var=[layer_outputs[-1]["labels"]])
                # Calculate logits with all block separately
                elif self.lossCalculationKind == "MultipleLogitsMultipleLosses" \
                        or self.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
                    classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                        list_of_logits=layer_outputs[-1]["list_of_logits"],
                        routing_matrices=None,
                        target_var=layer_outputs[-1]["labels_masked"])
                else:
                    raise ValueError("Unknown logit calculation method: {0}".format(self.lossCalculationKind))
                # Calculate the information gain losses, with respect to each routing layer
                routing_matrices_soft = [od["routing_matrices_soft"] for od in layer_outputs[1:-1]]
                labels_per_routing_layer = [od["labels"] for od in layer_outputs[1:-1]]
                information_gain_losses = self.calculate_information_gain_losses(
                    routing_matrices=routing_matrices_soft, labels=labels_per_routing_layer,
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

    def calculate_branch_statistics(self,
                                    run_id, iteration, dataset_type, routing_probability_matrices, labels,
                                    write_to_db):
        assert isinstance(routing_probability_matrices, list) and isinstance(labels, list)
        assert len(routing_probability_matrices) == len(labels)
        kv_rows = []
        for block_id in range(len(routing_probability_matrices)):
            routing_probability_matrix = routing_probability_matrices[block_id]
            label_vec = labels[block_id]
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
                path_labels = label_vec[selected_paths == path_id]
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

    def validate(self, loader, epoch, data_kind, temperature=None,
                 enforced_hard_routing_kind=None, print_avg_measurements=False, return_network_outputs=False):
        """Perform validation on the validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_c = AverageMeter()
        losses_t = AverageMeter()
        losses_t_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
        accuracy_avg = AverageMeter()
        list_of_original_inputs = []
        list_of_labels = []
        list_of_routing_probability_matrices = []
        list_of_routing_activations = []
        list_of_logits_complete = []
        list_of_logits_unified = []
        for _ in range(len(self.pathCounts) - 1):
            list_of_labels.append([])
            list_of_routing_probability_matrices.append([])
            list_of_routing_activations.append([])
        for _ in range(len(self.lossLayers)):
            list_of_logits_complete.append([])

        # Temperature of Gumble Softmax
        # We simply keep it fixed
        if temperature is None:
            temperature = self.temperatureController.get_value()

        # switch to evaluate mode
        self.eval()
        if enforced_hard_routing_kind is None:
            self.hardRoutingAlgorithmKind = "InformationGainRouting"
        else:
            assert enforced_hard_routing_kind in self.hardRoutingAlgorithmTypes
            self.hardRoutingAlgorithmKind = enforced_hard_routing_kind

        for i, (input_, target) in enumerate(loader):
            time_begin = time.time()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input_).to(self.device)
                target_var = torch.autograd.Variable(target).to(self.device)
                batch_size = input_var.size(0)
                # Cigt Classification Loss and Accuracy Calculation
                layer_outputs = self(input_var, target_var, temperature)
                if self.lossCalculationKind == "SingleLogitSingleLoss":
                    classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                        list_of_logits=layer_outputs[-1]["list_of_logits"],
                        routing_matrices=None,
                        target_var=[layer_outputs[-1]["labels"]])
                # Calculate logits with all block separately
                elif self.lossCalculationKind == "MultipleLogitsMultipleLosses" \
                        or self.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
                    classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                        list_of_logits=layer_outputs[-1]["list_of_logits"],
                        routing_matrices=None,
                        target_var=layer_outputs[-1]["labels_masked"])
                else:
                    raise ValueError("Unknown logit calculation method: {0}".format(self.lossCalculationKind))

                # Calculate the information gain losses, with respect to each routing layer
                routing_matrices_soft = [od["routing_matrices_soft"] for od in layer_outputs[1:-1]]
                routing_activations_list = [od["routing_activations"] for od in layer_outputs[1:-1]]
                labels_per_routing_layer = [od["labels"] for od in layer_outputs[1:-1]]
                information_gain_losses = self.calculate_information_gain_losses(
                    routing_matrices=routing_matrices_soft, labels=labels_per_routing_layer,
                    balance_coefficient_list=self.informationGainBalanceCoeffList)
                total_routing_loss = 0.0
                for t_loss in information_gain_losses:
                    total_routing_loss += t_loss
                total_routing_loss = -1.0 * self.decisionLossCoeff * total_routing_loss
                total_loss = classification_loss + total_routing_loss

                time_end = time.time()

                for idx_, matr_ in enumerate(labels_per_routing_layer):
                    list_of_labels[idx_].append(matr_.detach().cpu().numpy())
                for idx_, matr_ in enumerate(routing_matrices_soft):
                    list_of_routing_probability_matrices[idx_].append(matr_.detach().cpu().numpy())
                for idx_, matr_ in enumerate(routing_activations_list):
                    list_of_routing_activations[idx_].append(matr_.detach().cpu().numpy())
                for idx_, matr_ in enumerate(layer_outputs[-1]["list_of_logits"]):
                    list_of_logits_complete[idx_].append(matr_.detach().cpu().numpy())
                list_of_logits_unified.append(layer_outputs[-1]["logits_unified"].detach().cpu().numpy())
                list_of_original_inputs.append(input_.cpu().numpy())

                # measure accuracy and record loss
                losses.update(total_loss.detach().cpu().numpy().item(), 1)
                losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
                accuracy_avg.update(batch_accuracy, batch_size)
                batch_time.update((time_end - time_begin), 1)
                losses_t.update(total_routing_loss.detach().cpu().numpy().item(), 1)
                for lid in range(len(self.pathCounts) - 1):
                    losses_t_layer_wise[lid].update(information_gain_losses[lid].detach().cpu().numpy().item(), 1)

        kv_rows = []
        for idx_ in range(len(list_of_labels)):
            list_of_labels[idx_] = np.concatenate(list_of_labels[idx_], axis=0)
        for idx_ in range(len(list_of_routing_probability_matrices)):
            list_of_routing_probability_matrices[idx_] = np.concatenate(
                list_of_routing_probability_matrices[idx_], axis=0)
        for idx_ in range(len(list_of_routing_activations)):
            list_of_routing_activations[idx_] = np.concatenate(list_of_routing_activations[idx_], axis=0)
        for idx_ in range(len(list_of_logits_complete)):
            list_of_logits_complete[idx_] = np.concatenate(list_of_logits_complete[idx_], axis=0)
        list_of_logits_unified = np.concatenate(list_of_logits_unified, axis=0)
        list_of_original_inputs = np.concatenate(list_of_original_inputs, axis=0)

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
        self.hardRoutingAlgorithmKind = self.afterWarmupRoutingAlgorithmKind
        if not return_network_outputs:
            return accuracy_avg.avg
        else:
            res_dict = {
                "accuracy": accuracy_avg.avg,
                "list_of_labels": list_of_labels,
                "list_of_routing_probability_matrices": list_of_routing_probability_matrices,
                "list_of_routing_activations": list_of_routing_activations,
                "list_of_logits_complete": list_of_logits_complete,
                "list_of_logits_unified": list_of_logits_unified,
                "list_of_original_inputs": list_of_original_inputs
            }
            return res_dict

    def validate_v2(self, loader, temperature=None, enforced_hard_routing_kind=None):
        if temperature is None:
            temperature = self.temperatureController.get_value()

        # switch to evaluate mode
        self.eval()
        if enforced_hard_routing_kind is None:
            self.hardRoutingAlgorithmKind = "InformationGainRouting"
        else:
            assert enforced_hard_routing_kind in self.hardRoutingAlgorithmTypes
            self.hardRoutingAlgorithmKind = enforced_hard_routing_kind

        block_outputs_complete = {}
        routing_matrices_soft_complete = {}
        routing_matrices_hard_complete = {}
        routing_activations_complete = {}
        logits_complete = {}

        for i, (input_, target) in enumerate(loader):
            with torch.no_grad():
                input_var = torch.autograd.Variable(input_).to(self.device)
                target_var = torch.autograd.Variable(target).to(self.device)
                batch_size = input_var.size(0)
                # Cigt Classification Loss and Accuracy Calculation
                block_outputs_dict, routing_matrices_soft_dict, \
                    routing_matrices_hard_dict, routing_activations_dict, logits_dict = \
                    self.forward_v2(input_var, target_var, temperature)
                Utilities.append_to_dictionary(destination_dictionary=block_outputs_complete,
                                               source_dictionary=block_outputs_dict)
                Utilities.append_to_dictionary(destination_dictionary=routing_matrices_soft_complete,
                                               source_dictionary=routing_matrices_soft_dict)
                Utilities.append_to_dictionary(destination_dictionary=routing_matrices_hard_complete,
                                               source_dictionary=routing_matrices_hard_dict)
                Utilities.append_to_dictionary(destination_dictionary=routing_activations_complete,
                                               source_dictionary=routing_activations_dict)
                Utilities.append_to_dictionary(destination_dictionary=logits_complete,
                                               source_dictionary=logits_dict)

        block_outputs_complete = Utilities.concat_all_arrays_in_dictionary(source_dictionary=block_outputs_complete)
        routing_matrices_soft_complete = Utilities.concat_all_arrays_in_dictionary(
            source_dictionary=routing_matrices_soft_complete)
        routing_matrices_hard_complete = Utilities.concat_all_arrays_in_dictionary(
            source_dictionary=routing_matrices_hard_complete)
        routing_activations_complete = Utilities.concat_all_arrays_in_dictionary(
            source_dictionary=routing_activations_complete)
        logits_complete = Utilities.concat_all_arrays_in_dictionary(
            source_dictionary=logits_complete)

        return block_outputs_complete, routing_matrices_soft_complete, \
            routing_matrices_hard_complete, routing_activations_complete, logits_complete

