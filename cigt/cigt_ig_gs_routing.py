from auxillary.average_meter import AverageMeter
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
import torch
import time


class CigtIgGsRouting(CigtIgGatherScatterImplementation):
    def __init__(self, configs, run_id, model_definition, num_classes):
        super().__init__(configs, run_id, model_definition, num_classes)

    def warm_up_ops(self, layer_outputs):
        pass

    def gs_routing(self, routing_activations, temperature, z_sample_count):
        logits = torch.softmax(routing_activations, dim=1)
        eps = 1e-20
        samples_shape = torch.stack([logits.shape[0], logits.shape[1], z_sample_count], dim=0)
        U_ = torch.rand(size=samples_shape)
        G_ = -torch.math.log(-torch.math.log(U_ + eps) + eps)
        log_logits = torch.math.log(logits + eps)
        log_logits = torch.unsqueeze(log_logits, dim=-1)
        gumbel_logits = log_logits + G_
        gumbel_logits_tempered = gumbel_logits / temperature
        z_samples = torch.softmax(gumbel_logits_tempered, dim=1)
        return z_samples

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
                out = self.weighted_sum_of_tensors(routing_matrix=routing_matrices_soft[-1],
                                                   tensors=block_outputs[-1])
                # Calculate routing weights for the next layer
                _, routing_activations = self.blockEndLayers[layer_id](out,
                                                                       labels,
                                                                       temperature,
                                                                       balance_coefficient_list[
                                                                           layer_id])
                z_samples = self.gs_routing(routing_activations=routing_activations,
                                            temperature=temperature,
                                            z_sample_count=100)

                # routing_matrices_soft.append(p_n_given_x_soft)
                # routing_activations_list.append(routing_activations)

                # # Calculate the hard routing matrix
                # p_n_given_x_hard = self.routingManager.get_hard_routing_matrix(model=self,
                #                                                                layer_id=layer_id,
                #                                                                p_n_given_x_soft=p_n_given_x_soft)
        #         # routing_matrices_hard.append(p_n_given_x_hard)
        #     # Logits layer
        #     else:
        #         list_of_logits = self.calculate_logits(p_n_given_x_hard=routing_matrices_hard[-1],
        #                                                loss_block_outputs=block_outputs[-1])
        #
        # return routing_matrices_hard, routing_matrices_soft, block_outputs, list_of_logits, routing_activations_list

    # for layer_id, cigt_layer_blocks in enumerate(self.cigtLayers):
    #     block_outputs.append([])
    #     for block_id, block_obj in enumerate(cigt_layer_blocks):
    #         block_output = block_obj(out)
    #         block_outputs[-1].append(block_output)
    #
    #     # Routing Layer
    #     if layer_id < len(self.cigtLayers) - 1:
    #         # Weighted sum of block outputs
    #         out = self.weighted_sum_of_tensors(routing_matrix=routing_matrices_hard[-1],
    #                                            tensors=block_outputs[-1])
    #         # Calculate routing weights for the next layer
    #         p_n_given_x_soft, routing_activations = self.blockEndLayers[layer_id](out,
    #                                                                               labels,
    #                                                                               temperature,
    #                                                                               balance_coefficient_list[
    #                                                                                   layer_id])
    #         routing_matrices_soft.append(p_n_given_x_soft)
    #         routing_activations_list.append(routing_activations)
    #         # Calculate the hard routing matrix
    #         p_n_given_x_hard = self.routingManager.get_hard_routing_matrix(model=self,
    #                                                                        layer_id=layer_id,
    #                                                                        p_n_given_x_soft=p_n_given_x_soft)
    #         routing_matrices_hard.append(p_n_given_x_hard)
    #     # Logits layer
    #     else:
    #         list_of_logits = self.calculate_logits(p_n_given_x_hard=routing_matrices_hard[-1],
    #                                                loss_block_outputs=block_outputs[-1])
    #
    # return routing_matrices_hard, routing_matrices_soft, block_outputs, list_of_logits, routing_activations_list

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

                if self.isInWarmUp:
                    layer_outputs = self(input_var, target_var, temperature)
                    self.warm_up_ops(layer_outputs=layer_outputs)
                else:
                    self.forward_gs(input_var, target_var, temperature)

        #
        #         # Cigt Classification Loss and Accuracy Calculation
        #         layer_outputs = self(input_var, target_var, temperature)
        #         if self.lossCalculationKind == "SingleLogitSingleLoss":
        #             classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
        #                 list_of_logits=layer_outputs[-1]["list_of_logits"],
        #                 routing_matrices=None,
        #                 target_var=[layer_outputs[-1]["labels"]])
        #         # Calculate logits with all block separately
        #         elif self.lossCalculationKind == "MultipleLogitsMultipleLosses" \
        #                 or self.lossCalculationKind == "MultipleLogitsMultipleLossesAveraged":
        #             classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
        #                 list_of_logits=layer_outputs[-1]["list_of_logits"],
        #                 routing_matrices=None,
        #                 target_var=layer_outputs[-1]["labels_masked"])
        #         else:
        #             raise ValueError("Unknown logit calculation method: {0}".format(self.lossCalculationKind))
        #         # Calculate the information gain losses, with respect to each routing layer
        #         routing_matrices_soft = [od["routing_matrices_soft"] for od in layer_outputs[1:-1]]
        #         labels_per_routing_layer = [od["labels"] for od in layer_outputs[1:-1]]
        #         information_gain_losses = self.calculate_information_gain_losses(
        #             routing_matrices=routing_matrices_soft, labels=labels_per_routing_layer,
        #             balance_coefficient_list=self.informationGainBalanceCoeffList)
        #         total_routing_loss = 0.0
        #         for t_loss in information_gain_losses:
        #             total_routing_loss += t_loss
        #         total_routing_loss = -1.0 * decision_loss_coeff * total_routing_loss
        #         total_loss = classification_loss + total_routing_loss
        #         # print("len(list_of_logits)={0}".format(len(list_of_logits)))
        #         # print("multipleCeLosses:{0}".format(self.multipleCeLosses))
        #         total_loss.backward()
        #         self.modelOptimizer.step()
        #
        #     time_end = time.time()
        #     # measure accuracy and record loss
        #     print("Epoch:{0} Iteration:{1}".format(epoch_id, self.numOfTrainingIterations))
        #
        #     losses.update(total_loss.detach().cpu().numpy().item(), 1)
        #     losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
        #     accuracy_avg.update(batch_accuracy, batch_size)
        #     batch_time.update((time_end - time_begin), 1)
        #     losses_t.update(total_routing_loss.detach().cpu().numpy().item(), 1)
        #     for lid in range(len(self.pathCounts) - 1):
        #         losses_t_layer_wise[lid].update(information_gain_losses[lid].detach().cpu().numpy().item(), 1)
        #
        #     print("batch_accuracy:{0}".format(batch_accuracy))
        #     print("decision_loss_coeff:{0}".format(decision_loss_coeff))
        #     print("total_loss:{0}".format(losses.avg))
        #     print("classification_loss:{0}".format(losses_c.avg))
        #     print("routing_loss:{0}".format(losses_t.avg))
        #     for lid in range(len(self.pathCounts) - 1):
        #         print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))
        #     print("accuracy_avg:{0}".format(accuracy_avg.avg))
        #     print("batch_time:{0}".format(batch_time.avg))
        #     print("grad_magnitude:{0}".format(grad_magnitude.avg))
        #     print("*************Epoch:{0} Iteration:{1}*************".format(
        #         epoch_id, self.numOfTrainingIterations))
        #     self.numOfTrainingIterations += 1
        # # print("AVERAGE GRAD MAGNITUDE FOR EPOCH:{0}".format(grad_magnitude.avg))
        #
        # print("*************Epoch:{0} Ending Measurements*************".format(epoch_id))
        # print("decision_loss_coeff:{0}".format(decision_loss_coeff))
        # print("total_loss:{0}".format(losses.avg))
        # print("classification_loss:{0}".format(losses_c.avg))
        # print("routing_loss:{0}".format(losses_t.avg))
        # for lid in range(len(self.pathCounts) - 1):
        #     print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))
        # print("accuracy_avg:{0}".format(accuracy_avg.avg))
        # print("batch_time:{0}".format(batch_time.avg))
        # print("grad_magnitude:{0}".format(grad_magnitude.avg))
        # print("*************Epoch:{0} Ending Measurements*************".format(epoch_id))
        # return batch_time.avg
