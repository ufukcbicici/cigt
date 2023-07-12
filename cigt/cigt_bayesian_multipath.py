from collections import OrderedDict, Counter
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets
from torch import optim

from auxillary.db_logger import DbLogger
from auxillary.average_meter import AverageMeter
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cigt_model import conv3x3, BasicBlock, Sequential_ext
from cigt.cigt_constants import CigtConstants


class CigtBayesianMultipath(CigtIgGatherScatterImplementation):
    def __init__(self, run_id, model_definition, num_classes):
        super().__init__(run_id, model_definition, num_classes)
        self.temperatureOptimizationEpochCount = CigtConstants.temperature_optimization_epoch_count
        self.softmaxTemperatures = nn.Parameter(torch.Tensor([1.0] * (len(self.pathCounts) - 1)))

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
                          "block_indices": torch.zeros(size=(x.shape[0], ),
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
                block_indices_arr = block_id * torch.ones(size=(block_output.shape[0], ),
                                                          dtype=torch.int64, device=self.device)
                curr_block_indices.append(block_indices_arr)

            layer_outputs_unified = torch.concat(curr_layer_outputs, dim=0)
            layer_labels_unified = torch.concat(labels_masked, dim=0)
            layer_sample_indices_unified = torch.concat(sample_indices_masked, dim=0)
            layer_block_indices_unified = torch.concat(curr_block_indices, dim=0)

            # Routing Layer
            if layer_id < len(self.cigtLayers) - 1:
                layer_temperature = self.softmaxTemperatures[layer_id]
                # Calculate routing weights for the next layer
                p_n_given_x_soft, routing_activations = self.blockEndLayers[layer_id](layer_outputs_unified,
                                                                                      layer_labels_unified,
                                                                                      layer_temperature,
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

                layer_outputs.append({"net": layer_outputs_unified,
                                      "labels": layer_labels_unified,
                                      "sample_indices": layer_sample_indices_unified,
                                      "block_indices": layer_block_indices_unified,
                                      "labels_masked": labels_masked,
                                      "list_of_logits": list_of_logits})

        return layer_outputs

    def calculate_entropy_variances(self, routing_matrices):
        eps = 1e-30
        variances_list = []
        for layer_id in range(len(routing_matrices)):
            p_n_given_x = routing_matrices[layer_id]
            log_p_n_given_x = torch.log(p_n_given_x + eps)
            prob_log_prob = p_n_given_x * log_p_n_given_x
            entropies = -1.0 * torch.sum(prob_log_prob, dim=1)
            layer_entropy_variance = torch.var(entropies)
            variances_list.append(layer_entropy_variance)
        return variances_list

    def eval_variances(self, data_loader):
        variance_avg = AverageMeter()
        self.eval()
        for i, (input_, target) in enumerate(data_loader):
            input_var = torch.autograd.Variable(input_).to(self.device)
            target_var = torch.autograd.Variable(target).to(self.device)
            batch_size = input_var.size(0)
            with torch.no_grad():
                layer_outputs = self(input_var, target_var, None)
                routing_matrices_soft = [od["routing_matrices_soft"] for od in layer_outputs[1:-1]]
                variances_list = self.calculate_entropy_variances(routing_matrices=routing_matrices_soft)
                total_var = torch.tensor(0.0, device=self.device)
                for var_layer in variances_list:
                    total_var += var_layer
                variance_avg.update(total_var.detach().cpu().numpy().item(), batch_size)
        return variance_avg.avg

    def fit_temperatures_with_respect_to_variances(self):
        self.to(self.device)
        torch.manual_seed(1)
        variance_avg = AverageMeter()

        # Cifar 10 Dataset
        kwargs = {'num_workers': 2, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=self.transformTrain),
            batch_size=self.batchSize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=self.transformTest),
            batch_size=self.batchSize, shuffle=False, **kwargs)
        temperature_optimizer_lr = 0.0001

        temperature_optimizer = optim.Adam([{'params': self.softmaxTemperatures,
                                             'lr': temperature_optimizer_lr, 'weight_decay': 0.0}])
        # We don't want to update batch normalization layer statistics.
        self.eval()
        for epoch in range(0, self.temperatureOptimizationEpochCount):
            print("X")

            for i, (input_, target) in enumerate(train_loader):
                temperature_optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    input_var = torch.autograd.Variable(input_).to(self.device)
                    target_var = torch.autograd.Variable(target).to(self.device)
                    batch_size = input_var.size(0)

                    # Cigt Classification Loss and Accuracy Calculation
                    layer_outputs = self(input_var, target_var, None)
                    routing_matrices_soft = [od["routing_matrices_soft"] for od in layer_outputs[1:-1]]
                    variances_list = self.calculate_entropy_variances(routing_matrices=routing_matrices_soft)
                    total_var = torch.tensor(0.0, device=self.device)
                    for var_layer in variances_list:
                        total_var += var_layer
                    neg_total_variances = -1.0 * total_var
                    neg_total_variances.backward()
                    temperature_optimizer.step()
                    variance_avg.update(total_var.detach().cpu().numpy().item(), batch_size)
                    print("Epoch{0} Iteration{1} Variances:{2} Softmax Temperatures:{3}".format(
                        epoch, i, variance_avg.avg, self.softmaxTemperatures))
            train_variance = self.eval_variances(data_loader=train_loader)
            test_variance = self.eval_variances(data_loader=test_loader)
            print("Epoch{0} Train Variance:{1}".format(epoch, train_variance))
            print("Epoch{0} Test Variance:{1}".format(epoch, test_variance))
