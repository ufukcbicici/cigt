from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
import torch
import time
import numpy as np
from collections import Counter


class CigtIgGsRouting(CigtIgGatherScatterImplementation):
    def __init__(self, configs, run_id, model_definition, num_classes):
        super().__init__(configs, run_id, model_definition, num_classes)
        self.zSampleCount = configs.z_sample_count

    # OK
    def warm_up_optimization_step(self, layer_outputs, decision_loss_coeff):
        # Cigt Classification Loss and Accuracy Calculation
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
        return total_loss, classification_loss, total_routing_loss, batch_accuracy, information_gain_losses

    # OK
    def gs_optimization_step(self,
                             target_var,
                             decision_loss_coeff,
                             routing_matrices_hard,
                             routing_matrices_soft,
                             list_of_logits):
        # Call unscattered version
        classification_loss, batch_accuracy = super(CigtIgGatherScatterImplementation,
                                                    self).calculate_classification_loss_and_accuracy(
            list_of_logits,
            routing_matrices_hard,
            target_var)
        # Call unscattered version
        information_gain_losses = super(CigtIgGatherScatterImplementation, self).calculate_information_gain_losses(
            routing_matrices=routing_matrices_soft, labels=target_var,
            balance_coefficient_list=self.informationGainBalanceCoeffList)
        total_routing_loss = 0.0
        for t_loss in information_gain_losses:
            total_routing_loss += t_loss
        total_routing_loss = -1.0 * decision_loss_coeff * total_routing_loss
        total_loss = classification_loss + total_routing_loss
        return total_loss, classification_loss, total_routing_loss, batch_accuracy, information_gain_losses

    # OK
    def gs_routing(self, routing_activations, temperature, z_sample_count):
        print("z_sample_count={0}".format(z_sample_count))
        logits = torch.softmax(routing_activations, dim=1)
        eps = 1e-20
        samples_shape = [logits.shape[0], logits.shape[1], z_sample_count]
        U_ = torch.rand(size=samples_shape, device=self.device)
        G_ = -torch.log(-torch.log(U_ + eps) + eps)
        log_logits = torch.log(logits + eps)
        log_logits = torch.unsqueeze(log_logits, dim=-1)
        gumbel_logits = log_logits + G_
        gumbel_logits_tempered = gumbel_logits / temperature
        z_samples = torch.softmax(gumbel_logits_tempered, dim=1)
        # Experiment
        # argmax_array = torch.argmax(z_samples, dim=1)
        # counters = [Counter(argmax_array[idx].numpy()) for idx in range(argmax_array.shape[0])]
        # z_expected = torch.mean(z_samples, dim=-1)
        # return z_expected
        return z_samples

    # OK
    def forward_gs(self, x, labels, temperature):
        balance_coefficient_list = self.informationGainBalanceCoeffList
        # Routing Matrices
        routing_matrices_hard = []
        routing_matrices_soft = []
        # Initial layer
        out = self.preprocess_input(x=x)
        routing_matrices_hard.append(torch.ones(size=(x.shape[0], 1), dtype=torch.float32, device=self.device))
        routing_matrices_soft.append(torch.ones(size=(x.shape[0], 1), dtype=torch.float32, device=self.device))
        block_outputs = []
        routing_activations_list = []
        list_of_logits = None

        for layer_id, cigt_layer_blocks in enumerate(self.cigtLayers):
            block_outputs.append([])
            for block_id, block_obj in enumerate(cigt_layer_blocks):
                block_output = block_obj(out)
                block_outputs[-1].append(block_output)

            # Routing Layer
            if layer_id < len(self.cigtLayers) - 1:
                # Weighted sum of block outputs
                out = self.weighted_sum_of_tensors(routing_matrix=routing_matrices_hard[-1],
                                                   tensors=block_outputs[-1])
                # Calculate routing weights for the next layer
                _, routing_activations = self.blockEndLayers[layer_id](out,
                                                                       labels,
                                                                       temperature,
                                                                       balance_coefficient_list[
                                                                           layer_id])
                z_samples = self.gs_routing(routing_activations=routing_activations,
                                            temperature=temperature,
                                            z_sample_count=self.zSampleCount)
                z_expected = torch.mean(z_samples, dim=-1)
                # Straight through trick
                _, ind = z_expected.max(dim=-1)
                z_hard = torch.zeros_like(z_expected)
                z_hard[(torch.arange(z_hard.shape[0]), ind)] = 1.0
                # assert np.array_equal(z_hard.cpu().numpy(), z_hard2.cpu().numpy())
                p_n_given_x_soft = z_expected
                p_n_given_x_hard = (z_hard - z_expected).detach() + z_expected
                routing_matrices_soft.append(p_n_given_x_soft)
                routing_activations_list.append(routing_activations)
                # Calculate the hard routing matrix
                p_n_given_x_hard = self.routingManager.get_hard_routing_matrix(model=self,
                                                                               layer_id=layer_id,
                                                                               p_n_given_x_soft=p_n_given_x_hard)
                routing_matrices_hard.append(p_n_given_x_hard)
            # Logits layer
            else:
                list_of_logits = super(CigtIgGatherScatterImplementation,
                                       self).calculate_logits(p_n_given_x_hard=routing_matrices_hard[-1],
                                                              loss_block_outputs=block_outputs[-1])

        return routing_matrices_hard, routing_matrices_soft, block_outputs, list_of_logits, routing_activations_list

    # OK
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

                decision_loss_coeff = self.routingManager.adjust_decision_loss_coeff(model=self)
                temperature = self.routingManager.adjust_temperature(model=self)

                print("temperature:{0}".format(temperature))
                print("decision_loss_coeff:{0}".format(decision_loss_coeff))

                # Gather scatter type optimization
                if self.isInWarmUp:
                    print("Warm up step!")
                    layer_outputs = self(input_var, target_var, temperature)
                    total_loss, classification_loss, total_routing_loss, batch_accuracy, information_gain_losses = \
                        self.warm_up_optimization_step(
                            layer_outputs=layer_outputs,
                            decision_loss_coeff=decision_loss_coeff)
                    total_loss.backward()
                    self.modelOptimizer.step()
                # GS based optimization
                else:
                    print("Gumbel Softmax Step!")
                    routing_matrices_hard, routing_matrices_soft, \
                        block_outputs, list_of_logits, routing_activations_list = \
                        self.forward_gs(input_var, target_var, temperature)
                    total_loss, classification_loss, total_routing_loss, batch_accuracy, \
                        information_gain_losses = self.gs_optimization_step(routing_matrices_hard=routing_matrices_hard,
                                                                            routing_matrices_soft=routing_matrices_soft,
                                                                            list_of_logits=list_of_logits,
                                                                            target_var=target_var,
                                                                            decision_loss_coeff=decision_loss_coeff)
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

    # OK
    def validate(self, loader, epoch, data_kind, temperature=None, print_avg_measurements=False,
                 return_network_outputs=False,
                 verbose=False):
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

        for i, (input_, target) in enumerate(loader):
            time_begin = time.time()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input_).to(self.device)
                target_var = torch.autograd.Variable(target).to(self.device)
                batch_size = input_var.size(0)

                # Cigt moe output, information gain losses
                routing_matrices_hard, routing_matrices_soft, \
                    block_outputs, list_of_logits, routing_activations_list = \
                    self.forward_gs(input_var, target_var, temperature)
                classification_loss, batch_accuracy = super(CigtIgGatherScatterImplementation,
                                                            self).calculate_classification_loss_and_accuracy(
                    list_of_logits,
                    routing_matrices_hard,
                    target_var)
                information_gain_losses = super(CigtIgGatherScatterImplementation,
                                                self).calculate_information_gain_losses(
                    routing_matrices=routing_matrices_soft,
                    labels=target_var,
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
                for idx_, matr_ in enumerate(routing_activations_list):
                    list_of_routing_activations[idx_].append(matr_.detach().cpu().numpy())
                for idx_, matr_ in enumerate(list_of_logits):
                    list_of_logits_complete[idx_].append(matr_.detach().cpu().numpy())
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
        list_of_labels = np.concatenate(list_of_labels, axis=0)
        for idx_ in range(len(list_of_routing_probability_matrices)):
            list_of_routing_probability_matrices[idx_] = np.concatenate(
                list_of_routing_probability_matrices[idx_], axis=0)
        for idx_ in range(len(list_of_routing_activations)):
            list_of_routing_activations[idx_] = np.concatenate(list_of_routing_activations[idx_], axis=0)
        for idx_ in range(len(list_of_logits_complete)):
            list_of_logits_complete[idx_] = np.concatenate(list_of_logits_complete[idx_], axis=0)
        list_of_original_inputs = np.concatenate(list_of_original_inputs, axis=0)

        super(CigtIgGatherScatterImplementation,
              self).calculate_branch_statistics(
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
        if not return_network_outputs:
            return accuracy_avg.avg
        else:
            res_dict = {
                "accuracy": accuracy_avg.avg,
                "list_of_labels": list_of_labels,
                "list_of_routing_probability_matrices": list_of_routing_probability_matrices,
                "list_of_routing_activations": list_of_routing_activations,
                "list_of_logits_complete": list_of_logits_complete,
                "list_of_original_inputs": list_of_original_inputs
            }
            return res_dict
