import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from cigt.cigt_ig_soft_routing import CigtIgSoftRouting
from cigt.cigt_model import conv3x3
from cigt.cigt_soft_routing import CigtSoftRouting
from cigt.cutout_augmentation import CutoutPIL
from cigt.moe_layer import MoeLayer
from cigt.resnet_cigt_constants import ResnetCigtConstants
from cigt.routing_layers.hard_routing_layer import HardRoutingLayer
from randaugment import RandAugment
from torchvision import transforms
import torchvision.datasets as datasets


class CigtIgHardRouting(nn.Module):
    def __init__(self, run_id, model_definition, num_classes):
        super().__init__()
        self.runId = run_id
        self.modelDefinition = model_definition
        self.imageSize = (32, 32)
        self.numClasses = num_classes
        self.modelFilesRootPath = None
        self.routingStrategyName = ResnetCigtConstants.routing_strategy_name
        self.useStraightThrough = ResnetCigtConstants.use_straight_through
        self.decisionNonLinearity = ResnetCigtConstants.decision_non_linearity
        self.warmUpPeriod = ResnetCigtConstants.warm_up_period
        self.optimizerType = ResnetCigtConstants.optimizer_type
        self.learningRateSchedule = ResnetCigtConstants.learning_schedule
        self.initialLr = ResnetCigtConstants.initial_lr
        self.resnetConfigList = ResnetCigtConstants.resnet_config_list
        self.firstConvKernelSize = ResnetCigtConstants.first_conv_kernel_size
        self.firstConvOutputDim = ResnetCigtConstants.first_conv_output_dim
        self.firstConvStride = ResnetCigtConstants.first_conv_stride
        self.bnMomentum = ResnetCigtConstants.bn_momentum
        self.batchNormType = ResnetCigtConstants.batch_norm_type
        self.applyMaskToBatchNorm = ResnetCigtConstants.apply_mask_to_batch_norm
        self.doubleStrideLayers = ResnetCigtConstants.double_stride_layers
        self.batchSize = ResnetCigtConstants.batch_size
        self.inputDims = ResnetCigtConstants.input_dims
        self.advancedAugmentation = ResnetCigtConstants.advanced_augmentation
        self.decisionDimensions = ResnetCigtConstants.decision_dimensions
        self.decisionAveragePoolingStrides = ResnetCigtConstants.decision_average_pooling_strides
        self.routerLayersCount = ResnetCigtConstants.router_layers_count
        self.isInWarmUp = True
        self.temperatureController = ResnetCigtConstants.softmax_decay_controller
        self.decisionLossCoeff = ResnetCigtConstants.decision_loss_coeff
        self.informationGainBalanceCoeffList = ResnetCigtConstants.information_gain_balance_coeff_list
        self.classificationWd = ResnetCigtConstants.classification_wd
        self.decisionWd = ResnetCigtConstants.decision_wd
        self.epochCount = ResnetCigtConstants.epoch_count
        self.boostLearningRatesLayerWise = ResnetCigtConstants.boost_learning_rates_layer_wise
        self.multipleCeLosses = ResnetCigtConstants.multiple_ce_losses
        self.perSampleEntropyBalance = ResnetCigtConstants.per_sample_entropy_balance
        self.evaluationPeriod = ResnetCigtConstants.evaluation_period
        self.pathCounts = [1]
        self.pathCounts.extend([d_["path_count"] for d_ in self.resnetConfigList][1:])
        self.blockParametersList = self.interpret_config_list()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print("Device:{0}".format(self.device))
        # Train and test time augmentations
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        if not self.advancedAugmentation:
            print("WILL BE USING ONLY CROP AND HORIZONTAL FLIP AUGMENTATION")
            self.transformTrain = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.transformTest = transforms.Compose([
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            print("WILL BE USING RANDOM AUGMENTATION")
            self.transformTrain = transforms.Compose([
                transforms.Resize(self.imageSize),
                CutoutPIL(cutout_factor=0.5),
                RandAugment(),
                transforms.ToTensor(),
            ])
            self.transformTest = transforms.Compose([
                transforms.Resize(self.imageSize),
                transforms.ToTensor(),
            ])

        # Initial layer
        self.in_planes = self.firstConvOutputDim
        self.conv1 = conv3x3(3, self.firstConvOutputDim, self.firstConvStride)
        self.bn1 = nn.BatchNorm2d(self.firstConvOutputDim)

        # Build Cigt Blocks
        self.cigtLayers = nn.ModuleList()
        self.blockEndLayers = nn.ModuleList()
        self.create_cigt_blocks()

        # MoE Loss Layer
        self.moeLossLayer = MoeLayer()
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.nllLoss = nn.NLLLoss()
        self.crossEntropyLoss = nn.CrossEntropyLoss()

        self.numOfTrainingIterations = 0
        self.warmUpFinalIteration = 0

        self.layerCoefficients = []
        self.modelOptimizer = self.create_optimizer()

        self.classCount = 10
        self.hardRoutingAlgorithmTypes = {"InformationGainRouting", "RandomRouting", "EnforcedRouting"}
        self.hardRoutingAlgorithmKind = None
        self.enforcedRoutingMatrices = []

        self.lossCalculationTypes = {"SingleLogitSingleLoss", "MultipleLogitsMultipleLosses"}
        self.lossCalculationKind = None

    def get_hard_routing_matrix(self, layer_id, p_n_given_x_soft):
        if self.hardRoutingAlgorithmKind == "InformationGainRouting":
            p_n_given_x_hard = torch.zeros_like(p_n_given_x_soft)
            arg_max_entries = torch.argmax(p_n_given_x_soft, dim=1)
            p_n_given_x_hard[torch.arange(p_n_given_x_hard.shape[0]), arg_max_entries] = 1.0
        elif self.hardRoutingAlgorithmKind == "RandomRouting":
            random_routing_matrix = torch.rand(size=p_n_given_x_soft.shape)
            arg_max_entries = torch.argmax(random_routing_matrix, dim=1)
            random_routing_matrix_hard = torch.zeros_like(p_n_given_x_soft)
            random_routing_matrix_hard[torch.arange(random_routing_matrix_hard.shape[0]), arg_max_entries] = 1.0
            p_n_given_x_hard = random_routing_matrix_hard
        elif self.hardRoutingAlgorithmKind == "EnforcedRouting":
            enforced_routing_matrix = self.enforcedRoutingMatrices[layer_id]
            p_n_given_x_hard = enforced_routing_matrix
        else:
            raise ValueError("Unknown routing algorithm: {0}".format(self.hardRoutingAlgorithmKind))

        return p_n_given_x_hard

    def calculate_logits(self, p_n_given_x_hard, loss_block_outputs):
        list_of_logits = []
        # Unify all last layer block outputs into a single output and calculate logits with only that output
        if self.lossCalculationKind == "SingleLogitSingleLoss":
            out = self.weighted_sum_of_tensors(routing_matrix=p_n_given_x_hard, tensors=loss_block_outputs)
            logits = self.lossLayers[0](out)
            list_of_logits.append(logits)
        # Calculate logits with all block separately
        elif self.lossCalculationKind == "MultipleLogitsMultipleLosses":
            for idx, block_x in enumerate(loss_block_outputs):
                logits = self.lossLayers[idx](block_x)
                list_of_logits.append(logits)
        else:
            raise ValueError("Unknown logit calculation method: {0}".format(self.lossCalculationKind))
        return list_of_logits

    def forward(self, x, labels, temperature):
        balance_coefficient_list = self.informationGainBalanceCoeffList
        # Routing Matrices
        routing_matrices_hard = []
        routing_matrices_soft = []
        # Initial layer
        out = F.relu(self.bn1(self.conv1(x)))
        routing_matrices_hard.append(torch.ones(size=(x.shape[0], 1), dtype=torch.float32, device=self.device))
        routing_matrices_soft.append(torch.ones(size=(x.shape[0], 1), dtype=torch.float32, device=self.device))
        block_outputs = []
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
                p_n_given_x_soft = self.blockEndLayers[layer_id](out,
                                                                 labels,
                                                                 temperature,
                                                                 balance_coefficient_list[layer_id])
                routing_matrices_soft.append(p_n_given_x_soft)
                # Calculate the hard routing matrix
                p_n_given_x_hard = self.get_hard_routing_matrix(layer_id=layer_id, p_n_given_x_soft=p_n_given_x_soft)
                routing_matrices_hard.append(p_n_given_x_hard)
            # Logits layer
            else:
                list_of_logits = self.calculate_logits(p_n_given_x_hard=routing_matrices_hard[-1],
                                                       loss_block_outputs=block_outputs[-1])

        return routing_matrices_hard, routing_matrices_soft, block_outputs, list_of_logits

    def measure_accuracy(self, probs, target):
        """Computes the precision@k for the specified values of k"""
        pred = torch.argmax(probs, dim=1)
        correct_vector = pred.eq(target).to(torch.float)
        acc = torch.mean(correct_vector)
        return acc.cpu().numpy().item()

    def calculate_classification_loss_and_accuracy(self, list_of_logits, routing_matrices, target_var):
        if self.lossCalculationKind == "SingleLogitSingleLoss":
            classification_loss = self.crossEntropyLoss(list_of_logits[0], target_var)
            batch_accuracy = self.measure_accuracy(list_of_logits[0].detach().cpu(), target_var.cpu())
        elif self.lossCalculationKind == "MultipleLogitsMultipleLosses":
            # Independently calculate loss for every block, by selecting the samples that are routed into these blocks.
            for idx, logit in enumerate(list_of_logits):
                sample_selection_vector = routing_matrices[:, idx]







        # if not self.multipleCeLosses:
        #     classification_loss = self.crossEntropyLoss(list_of_logits[0], target_var)
        #     batch_accuracy = self.measure_accuracy(list_of_logits[0].detach().cpu(), target_var.cpu())
        # else:
        #     ce_losses = []
        #     probs = []
        #     for idx, logit in enumerate(list_of_logits):
        #         ce_loss = self.crossEntropyLosses[idx](logit, target_var)
        #         ce_losses.append(ce_loss)
        #         probs.append(torch.softmax(logit, dim=1))
        #     weighted_ce_losses = self.weighted_sum_of_tensors(routing_matrix=routing_matrices[-1],
        #                                                       tensors=ce_losses)
        #     weighted_probs = self.weighted_sum_of_tensors(routing_matrix=routing_matrices[-1],
        #                                                   tensors=probs)
        #     weighted_probs = weighted_probs.detach().cpu()
        #     classification_loss = torch.mean(weighted_ce_losses)
        #     batch_accuracy = self.measure_accuracy(weighted_probs, target_var.cpu())
        # return classification_loss, batch_accuracy

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

                # Run the Cigt model, get the hard routing matrices, the soft matrices, the output of every block and
                # each of the logits
                routing_matrices_hard, routing_matrices_soft, \
                    block_outputs, list_of_logits = self(input_var, target_var, temperature)
                classification_loss, batch_accuracy = self.calculate_classification_loss_and_accuracy(
                    list_of_logits,
                    routing_matrices_hard,
                    target_var)
        #         information_gain_losses = self.calculate_information_gain_losses(
        #             routing_matrices=routing_matrices_soft, labels=target_var,
        #             balance_coefficient_list=self.informationGainBalanceCoeffList)
        #         total_routing_loss = 0.0
        #         for t_loss in information_gain_losses:
        #             total_routing_loss += t_loss
        #         total_routing_loss = -1.0 * decision_loss_coeff * total_routing_loss
        #         total_loss = classification_loss + total_routing_loss
        #         print("len(list_of_logits)={0}".format(len(list_of_logits)))
        #         print("multipleCeLosses:{0}".format(self.multipleCeLosses))
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
        original_decision_coeff = self.decisionLossCoeff
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

            # train for one epoch, disabling information gain, randomly routing samples
            self.randomFineTuning = True
            self.decisionLossCoeff = 0.0
            train_mean_batch_time = self.train_single_epoch(epoch_id=epoch, train_loader=train_loader)

            if epoch % self.evaluationPeriod == 0 or epoch >= (total_epoch_count - 10):
                self.randomFineTuning = False
                self.decisionLossCoeff = original_decision_coeff
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
