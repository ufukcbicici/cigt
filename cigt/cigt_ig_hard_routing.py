import torch
import time
import numpy as np
import torch.nn.functional as F
from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from cigt.cigt_ig_soft_routing import CigtIgSoftRouting
from cigt.cigt_soft_routing import CigtSoftRouting
from cigt.routing_layers.hard_routing_layer import HardRoutingLayer
from randaugment import RandAugment
from torchvision import transforms
import torchvision.datasets as datasets


class CigtIgHardRouting(CigtIgSoftRouting):
    def __init__(self, run_id, model_definition):
        self.classCount = 10
        self.randomFineTuning = False
        self.enforcedRouting = False
        self.enforcedRoutingMatrices = []
        super().__init__(run_id, model_definition)

    def forward(self, x, labels, temperature):
        moe_probs = 0.0
        balance_coefficient_list = self.informationGainBalanceCoeffList
        # Classification loss
        classification_loss = 0.0
        # Information gain losses
        information_gain_losses = torch.zeros(size=(len(self.cigtLayers) - 1,), dtype=torch.float32, device=self.device)
        # Routing Matrices
        routing_matrices_hard = []
        routing_matrices_soft = []
        # Initial layer
        out = F.relu(self.bn1(self.conv1(x)))
        routing_matrices_hard.append(torch.ones(size=(x.shape[0], 1), dtype=torch.float32, device=self.device))
        routing_matrices_soft.append(torch.ones(size=(x.shape[0], 1), dtype=torch.float32, device=self.device))
        # prev_layer_outputs = {(): out}
        list_of_last_features = []
        list_of_logits = []

        for layer_id, cigt_layer_blocks in enumerate(self.cigtLayers):
            curr_layer_outputs = []
            for block_id, block_obj in enumerate(cigt_layer_blocks):
                block_output = block_obj(out)
                curr_layer_outputs.append(block_output)

            # Routing Layer
            if layer_id < len(self.cigtLayers) - 1:
                # Weighted sum of block outputs
                out = self.weighted_sum_of_tensors(routing_matrix=routing_matrices_hard[-1],
                                                   tensors=curr_layer_outputs)
                # Calculate routing weights for the next layer
                p_n_given_x_soft = self.blockEndLayers[layer_id](out,
                                                                 labels,
                                                                 temperature,
                                                                 balance_coefficient_list[layer_id])
                if
                # Calculate the hard routing matrix
                p_n_given_x_hard = torch.zeros_like(p_n_given_x_soft)
                arg_max_entries = torch.argmax(p_n_given_x_soft, dim=1)
                p_n_given_x_hard[torch.arange(p_n_given_x_hard.shape[0]), arg_max_entries] = 1.0

                routing_matrices_soft.append(p_n_given_x_soft)

                if not self.enforcedRouting:
                    routing_matrices_hard.append(p_n_given_x_hard)
                else:
                    enforced_routing_matrix = self.enforcedRoutingMatrices[layer_id]
                    routing_matrices_hard.append(enforced_routing_matrix)
            # Logits
            else:
                if not self.multipleCeLosses:
                    # Weighted sum of block outputs
                    out = self.weighted_sum_of_tensors(routing_matrix=routing_matrices_hard[-1],
                                                       tensors=curr_layer_outputs)
                    logits = self.lossLayers[0](out)
                    list_of_logits.append(logits)
                    for idx, block_x in enumerate(curr_layer_outputs):
                        list_of_last_features.append(block_x)
                else:
                    for idx, block_x in enumerate(curr_layer_outputs):
                        logits = self.lossLayers[idx](block_x)
                        list_of_logits.append(logits)
        return list_of_logits, routing_matrices_hard, routing_matrices_soft, list_of_last_features

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
                list_of_logits, routing_matrices_hard, \
                routing_matrices_soft, list_of_last_features = self(input_var, target_var, temperature)
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
                list_of_logits, routing_matrices_hard, routing_matrices_soft, list_of_last_features = self(
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

    def random_fine_tuning(self):
        self.isInWarmUp = False
        self.randomFineTuning = True
        self.decisionLossCoeff = 0.0

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
        total_epoch_count = self.epochCount
        for epoch in range(0, total_epoch_count):
            self.adjust_learning_rate(epoch)
            print("***************Random Fine Tuning "
                  "Epoch {0} End, Test Evaluation***************".format(epoch))
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
