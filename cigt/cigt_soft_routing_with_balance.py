import torch
import time
import numpy as np

from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from cigt.cigt_soft_routing import CigtSoftRouting


class CigtSoftRoutingWithBalance(CigtSoftRouting):
    def __init__(self, run_id, model_definition):
        super().__init__(run_id, model_definition)

    def calculate_entropy(self, prob_distribution, eps=1e-30):
        log_prob = torch.log(prob_distribution + eps)
        prob_log_prob = prob_distribution * log_prob
        entropy = -1.0 * torch.sum(prob_log_prob)
        return entropy

    def calculate_routing_entropies(self, routing_matrices, print_matrices):
        entropy_losses = []
        if not self.perSampleEntropyBalance:
            for block_id, routing_matrix in enumerate(routing_matrices[1:]):
                p_n = torch.mean(routing_matrix, dim=0)
                routing_entropy = self.calculate_entropy(prob_distribution=p_n)
                if print_matrices:
                    print("***********Block {0} Routing Matrix***********".format(block_id))
                    print(routing_matrix.detach().cpu().numpy()[0:5, :])
                    print("***********Block {0} Routing Matrix***********".format(block_id))
                # entropy_losses.append(routing_entropy)
                print("Block:{0} Routing Probability p(n):{1} Entropy:{2}".format(
                    block_id,
                    p_n.detach().cpu().numpy(),
                    routing_entropy.detach().cpu().numpy()))
                entropy_losses.append(routing_entropy)
            return entropy_losses
        else:
            eps = 1e-30
            for block_id, routing_matrix in enumerate(routing_matrices[1:]):
                log_prob = torch.log(routing_matrix + eps)
                prob_log_prob = routing_matrix * log_prob
                entropy_vector = -1.0 * torch.sum(prob_log_prob, dim=1)
                if print_matrices:
                    print("***********Block {0} Routing Matrix***********".format(block_id))
                    print(routing_matrix.detach().cpu().numpy()[0:5, :])
                    print("***********Block {0} Routing Matrix***********".format(block_id))
                    print("entropy_vector:{0}".format(entropy_vector.detach().cpu().numpy()[0:5]))
                mean_routing_entropy = torch.mean(entropy_vector)
                print("Block:{0} Mean Entropy:{1}".format(
                    block_id,
                    mean_routing_entropy.detach().cpu().numpy()))
                entropy_losses.append(mean_routing_entropy)
            return entropy_losses

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
                classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                    list_of_logits,
                    routing_matrices,
                    target_var)
                # Calculate routing entropies
                entropy_losses = self.calculate_routing_entropies(routing_matrices=routing_matrices,
                                                                  print_matrices=False)
                total_entropy_loss = 0.0
                for e_loss in entropy_losses:
                    total_entropy_loss += e_loss
                total_entropy_loss = -1.0 * self.decisionLossCoeff * total_entropy_loss
                total_loss = classification_loss + total_entropy_loss
                print("len(list_of_logits)={0}".format(len(list_of_logits)))
                print("multipleCeLosses:{0}".format(self.multipleCeLosses))
                total_loss.backward()
                self.modelOptimizer.step()

            time_end = time.time()
            # measure accuracy and record loss
            print("Epoch:{0} Iteration:{1}".format(epoch_id, self.numOfTrainingIterations))

            losses.update(total_loss.detach().cpu().numpy().item(), 1)
            losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
            accuracy_avg.update(batch_accuracy, batch_size)
            batch_time.update((time_end - time_begin), 1)
            losses_t.update(total_entropy_loss.detach().cpu().numpy().item(), 1)
            for lid in range(len(self.pathCounts) - 1):
                losses_t_layer_wise[lid].update(entropy_losses[lid].detach().cpu().numpy().item(), 1)

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
                # logits = self(input_var, target_var, 1.0)
                # total_loss = self.crossEntropyLoss(logits, target_var)
                # list_of_logits, routing_matrices = self(input_var, target_var, temperature)
                # classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                #     list_of_logits,
                #     routing_matrices,
                #     target_var)
                # total_entropy_loss = self.calculate_routing_entropies(routing_matrices=routing_matrices)
                # total_loss = classification_loss + total_entropy_loss

                # Cigt moe output, information gain losses
                list_of_logits, routing_matrices = self(input_var, target_var, temperature)
                classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                    list_of_logits,
                    routing_matrices,
                    target_var)
                # Calculate routing entropies
                entropy_losses = self.calculate_routing_entropies(routing_matrices=routing_matrices,
                                                                  print_matrices=True)
                total_entropy_loss = 0.0
                for e_loss in entropy_losses:
                    total_entropy_loss += e_loss
                total_entropy_loss = -1.0 * self.decisionLossCoeff * total_entropy_loss
                total_loss = classification_loss + total_entropy_loss
                print("len(list_of_logits)={0}".format(len(list_of_logits)))
                print("multipleCeLosses:{0}".format(self.multipleCeLosses))
                time_end = time.time()

                list_of_labels.append(target_var.cpu().numpy())
                for idx_, matr_ in enumerate(routing_matrices[1:]):
                    list_of_routing_probability_matrices[idx_].append(matr_.detach().cpu().numpy())

                # measure accuracy and record loss
                losses.update(total_loss.detach().cpu().numpy().item(), 1)
                losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
                accuracy_avg.update(batch_accuracy, batch_size)
                batch_time.update((time_end - time_begin), 1)
                losses_t.update(total_entropy_loss.detach().cpu().numpy().item(), 1)
                for lid in range(len(self.pathCounts) - 1):
                    losses_t_layer_wise[lid].update(entropy_losses[lid].detach().cpu().numpy().item(), 1)

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
