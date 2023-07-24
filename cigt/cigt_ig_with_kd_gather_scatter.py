from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs


class CigtIgWithKdGatherScatter(CigtIgGatherScatterImplementation):
    def __init__(self, run_id, model_definition, num_classes, teacher_model):
        self.idealRoutingErrorRatio = FashionLenetCigtConfigs.ideal_routing_error_ratio
        self.useKdForRouting = FashionLenetCigtConfigs.use_kd_for_routing
        self.teacherTemperature = FashionLenetCigtConfigs.kd_teacher_temperature
        self.teacherAlpha = FashionLenetCigtConfigs.kd_loss_alpha
        super().__init__(run_id, model_definition, num_classes)
        self.teacherModel = teacher_model

    def get_explanation_string(self):
        explanation = super().get_explanation_string()
        kv_rows = []
        explanation = self.add_explanation(name_of_param="Ideal Routing Error Ratio",
                                           value=self.idealRoutingErrorRatio,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Use KD for Routing", value=self.useKdForRouting,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Teacher Temperature", value=self.teacherTemperature,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="Teacher Alpha", value=self.teacherAlpha,
                                           explanation=explanation, kv_rows=kv_rows)
        DbLogger.write_into_table(rows=kv_rows, table="run_parameters")
        return explanation

    def calculate_classification_loss_and_accuracy(self, list_of_logits, routing_matrices, target_var):
        alpha = self.teacherAlpha
        T = self.teacherTemperature
        if self.lossCalculationKind == "SingleLogitSingleLoss":
            student_logits = list_of_logits["student"][0]
            teacher_logits = list_of_logits["teacher"][0]

            KD_loss = nn.KLDivLoss()(F.log_softmax(student_logits / T, dim=1),
                                     F.softmax(teacher_logits / T, dim=1)) * (alpha * T * T) + \
                      F.cross_entropy(student_logits, target_var) * (1.0 - alpha)
            batch_accuracy = self.measure_accuracy(student_logits.detach().cpu(), target_var.cpu())
        elif self.lossCalculationKind in {"MultipleLogitsMultipleLosses", "MultipleLogitsMultipleLossesAveraged"}:
            # Independently calculate loss for every block, by selecting the samples that are routed into these blocks.
            KD_loss = 0.0
            batch_accuracy = 0.0
            student_logits = list_of_logits["student"]
            teacher_logits = list_of_logits["teacher"][0]
            for idx, logit in enumerate(student_logits):
                sample_selection_vector = routing_matrices[-1][:, idx].to(torch.bool)
                assert student_logits[idx].shape == teacher_logits.shape
                selected_student_logits_1d = torch.masked_select(student_logits[idx],
                                                                 torch.unsqueeze(sample_selection_vector, dim=1))
                selected_labels = torch.masked_select(target_var, sample_selection_vector)
                selected_teacher_logits_1d = torch.masked_select(teacher_logits,
                                                                 torch.unsqueeze(sample_selection_vector, dim=1))
                # Reshape back into 2d
                new_shape = (selected_student_logits_1d.shape[0] // student_logits[idx].shape[1],
                             student_logits[idx].shape[1])
                # print("Block {0} Count:{1}".format(idx, new_shape[0]))
                if selected_student_logits_1d.shape[0] == 0:
                    continue
                selected_student_logits = torch.reshape(selected_student_logits_1d, new_shape)
                selected_teacher_logits = torch.reshape(selected_teacher_logits_1d, new_shape)
                # The following are for testing the torch indexing logic
                # non_zero_indices = np.nonzero(sample_selection_vector.cpu().numpy())[0]
                # for i_, j_ in enumerate(non_zero_indices):
                #     assert np.array_equal(selected_logits[i_].cpu().numpy(),
                #                           list_of_logits[idx][j_].cpu().numpy())
                #     assert selected_labels[i_] == target_var[j_]

                # block_classification_loss = self.crossEntropyLosses[idx](selected_logits, selected_labels)
                # classification_loss += block_classification_loss
                block_kd_loss = nn.KLDivLoss()(F.log_softmax(selected_student_logits / T, dim=1),
                                         F.softmax(selected_teacher_logits / T, dim=1)) * (alpha * T * T) + \
                          F.cross_entropy(selected_student_logits, selected_labels) * (1.0 - alpha)
                KD_loss += block_kd_loss
                block_accuracy = self.measure_accuracy(selected_student_logits.detach().cpu(), selected_labels.cpu())
                batch_coefficient = (new_shape[0] / target_var.shape[0])
                batch_accuracy += batch_coefficient * block_accuracy
            if self.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
                KD_loss = KD_loss / len(student_logits)
        else:
            raise ValueError("Unknown loss calculation method:{0}".format(self.lossCalculationKind))
        return KD_loss, batch_accuracy

    def calculate_information_gain_losses(self, routing_matrices, labels, balance_coefficient_list):
        information_gain_list = []
        for layer_id, p_n_given_x in enumerate(routing_matrices[1:]):
            information_gain = 0.0
            for loss_kind in ["student", "teacher"]:
                if loss_kind == "teacher" and not self.useKdForRouting:
                    continue
                elif loss_kind == "teacher" and self.useKdForRouting:
                    loss_weight = self.teacherAlpha
                elif loss_kind == "student" and not self.useKdForRouting:
                    loss_weight = 1.0
                elif loss_kind == "student" and self.useKdForRouting:
                    loss_weight = 1.0 - self.teacherAlpha
                else:
                    raise ValueError("Error in values: loss_kind:{0} self.useKdForRouting:{1}".format(
                        loss_kind, self.useKdForRouting))

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
                if loss_kind == "student":
                    p_c_given_x = torch.nn.functional.one_hot(labels["student"], self.classCount)
                else:
                    p_c_given_x = F.softmax(labels["teacher"][0] / self.teacherTemperature, dim=1)

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
                ig_loss_list_by_type = (balance_coefficient * entropy_p_n) + entropy_p_c - entropy_p_cn
                information_gain += (loss_weight * ig_loss_list_by_type)
                print("Layer:{0} {1} IG:{2}".format(layer_id, loss_kind, ig_loss_list_by_type))
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

        # Switch to train mode, teacher mode should be in eval
        self.train()
        self.teacherModel.eval()

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
                block_outputs, student_list_of_logits, routing_activations_list = self(input_var, target_var,
                                                                                       temperature)
                KD_Loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                    {"student": student_list_of_logits, "teacher": teacher_list_of_logits},
                    routing_matrices_hard,
                    target_var)
                information_gain_losses = self.calculate_information_gain_losses(
                    routing_matrices=routing_matrices_soft,
                    labels={"student": target_var, "teacher": teacher_list_of_logits},
                    balance_coefficient_list=self.informationGainBalanceCoeffList)
                total_routing_loss = 0.0
                for t_loss in information_gain_losses:
                    total_routing_loss += t_loss
                total_routing_loss = -1.0 * decision_loss_coeff * total_routing_loss
                total_loss = KD_Loss + total_routing_loss
                # print("len(list_of_logits)={0}".format(len(list_of_logits)))
                # print("multipleCeLosses:{0}".format(self.multipleCeLosses))
                total_loss.backward()
                self.modelOptimizer.step()

            time_end = time.time()
            # measure accuracy and record loss
            print("Epoch:{0} Iteration:{1}".format(epoch_id, self.numOfTrainingIterations))

            losses.update(total_loss.detach().cpu().numpy().item(), 1)
            losses_c.update(KD_Loss.detach().cpu().numpy().item(), 1)
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

    def validate(self, loader, epoch, data_kind, temperature=None,
                 enforced_hard_routing_kind=None, print_avg_measurements=False, return_network_outputs=False):
        """Perform validation on the validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_c = AverageMeter()
        losses_t = AverageMeter()
        losses_t_layer_wise = [AverageMeter() for _ in range(len(self.pathCounts) - 1)]
        accuracy_avg = AverageMeter()
        list_of_labels = []
        list_of_routing_probability_matrices = []
        list_of_routing_activations = []
        list_of_logits_complete = []
        for _ in range(len(self.pathCounts) - 1):
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
        self.teacherModel.eval()

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

                # Teacher output
                with torch.no_grad():
                    teacher_rm_hard, teacher_rm_soft, \
                    teacher_bo, teacher_list_of_logits, teacher_ra = self.teacherModel(input_var,
                                                                                       target_var, temperature)
                # Student output
                routing_matrices_hard, routing_matrices_soft, \
                block_outputs, student_list_of_logits, routing_activations_list = self(input_var, target_var,
                                                                                       temperature)

                KD_Loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                    {"student": student_list_of_logits, "teacher": teacher_list_of_logits},
                    routing_matrices_hard,
                    target_var)
                information_gain_losses = self.calculate_information_gain_losses(
                    routing_matrices=routing_matrices_soft,
                    labels={"student": target_var, "teacher": teacher_list_of_logits},
                    balance_coefficient_list=self.informationGainBalanceCoeffList)

                total_routing_loss = 0.0
                for t_loss in information_gain_losses:
                    total_routing_loss += t_loss
                total_routing_loss = -1.0 * self.decisionLossCoeff * total_routing_loss
                total_loss = KD_Loss + total_routing_loss

                # print("len(list_of_logits)={0}".format(len(list_of_logits)))
                # print("multipleCeLosses:{0}".format(self.multipleCeLosses))
                time_end = time.time()

                list_of_labels.append(target_var.cpu().numpy())
                for idx_, matr_ in enumerate(routing_matrices_soft[1:]):
                    list_of_routing_probability_matrices[idx_].append(matr_.detach().cpu().numpy())
                for idx_, matr_ in enumerate(routing_activations_list):
                    list_of_routing_activations[idx_].append(matr_.detach().cpu().numpy())
                for idx_, matr_ in enumerate(student_list_of_logits):
                    list_of_logits_complete[idx_].append(matr_.detach().cpu().numpy())

                # measure accuracy and record loss
                losses.update(total_loss.detach().cpu().numpy().item(), 1)
                losses_c.update(KD_Loss.detach().cpu().numpy().item(), 1)
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
        for idx_ in range(len(list_of_routing_activations)):
            list_of_routing_activations[idx_] = np.concatenate(list_of_routing_activations[idx_], axis=0)
        for idx_ in range(len(list_of_logits_complete)):
            list_of_logits_complete[idx_] = np.concatenate(list_of_logits_complete[idx_], axis=0)

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
                "list_of_logits_complete": list_of_logits_complete
            }
            return res_dict
