from cigt.cigt_reinforce_v2 import CigtReinforceV2
from collections import OrderedDict
from collections import Counter

import torch
import time
import inspect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from auxillary.db_logger import DbLogger
from auxillary.average_meter import AverageMeter
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.custom_layers.basic_block_with_cbam import BasicBlockWithCbam
from tqdm import tqdm


class CigtReinforcePreprocessedDatasets(CigtReinforceV2):
    def __init__(self, configs, run_id, model_definition, num_classes, model_mac_info, is_debug_mode,
                 train_dataset, test_dataset):
        super().__init__(configs, run_id, model_definition, num_classes, model_mac_info, is_debug_mode)
        self.trainDataset = train_dataset
        self.testDataset = test_dataset

    # The same policy network, but without the starting max pooling layers.
    def create_policy_networks(self):
        for layer_id, path_count in enumerate(self.pathCounts[1:]):
            layers = OrderedDict()
            action_space_size = path_count
            single_path_feature_count = self.layerConfigList[layer_id]["layer_structure"][-1]["feature_map_count"]
            current_route_combinations = self.pathCounts[:(layer_id + 1)]
            input_feature_count = np.prod(current_route_combinations) * single_path_feature_count
            input_layer = nn.Conv2d(
                kernel_size=1,
                in_channels=input_feature_count,
                out_channels=self.policyNetworksCbamFeatureMapCount
            )
            layers["policy_gradients_input_block_{0}".format(layer_id)] = input_layer

            for cid in range(self.policyNetworksCbamLayerCount):
                block = BasicBlockWithCbam(in_planes=self.policyNetworksCbamFeatureMapCount,
                                           planes=self.policyNetworksCbamFeatureMapCount,
                                           stride=1,
                                           cbam_reduction_ratio=self.policyNetworksCbamReductionRatio,
                                           norm_type=self.batchNormType)
                layers["policy_gradients_block_{0}_cbam_layer_{1}".format(layer_id, cid)] = block

            layers["policy_gradients_block_{0}_avg_pool".format(layer_id)] = nn.AvgPool2d(
                self.decisionAveragePoolingStrides[layer_id],
                stride=self.decisionAveragePoolingStrides[layer_id])
            layers["policy_gradients_block_{0}_flatten".format(layer_id)] = nn.Flatten()
            # layers["policy_gradients_block_{0}_relu".format(layer_id)] = nn.ReLU()
            layers["policy_gradients_block_{0}_feature_fc".format(layer_id)] = nn.LazyLinear(
                out_features=self.decisionDimensions[layer_id])
            layers["policy_gradients_block_{0}_action_space_fc".format(layer_id)] = nn.Linear(
                in_features=self.decisionDimensions[layer_id], out_features=action_space_size)
            layers["policy_gradients_block_{0}_softmax".format(layer_id)] = nn.Softmax(dim=1)

            policy_gradient_network_backbone = nn.Sequential(layers)
            self.policyNetworks.append(policy_gradient_network_backbone)

    def move_cigt_outputs_to_device(self, cigt_outputs):
        d_ = {}
        for field_name in cigt_outputs.keys():
            d_[field_name] = {}
            for k, v in cigt_outputs[field_name]:
                d_[k] = v.to(self.device)
        return d_

    def get_cigt_outputs(self, x, y):
        assert isinstance(x, dict)
        assert y is None
        cigt_outputs = x
        batch_size = cigt_outputs["block_outputs_dict"][(0, )].shape[0]
        return cigt_outputs, batch_size

    def execute_forward_with_random_input(self):
        max_batch_size = np.prod(self.pathCounts) * self.batchSize
        self.enforcedRoutingMatrices = []
        for path_count in self.pathCounts[1:]:
            self.enforcedRoutingMatrices.append(torch.ones(size=(max_batch_size, path_count),
                                                           dtype=torch.int64).to(self.device))
        # for cigt_output in self.trainDataset:
        #     print("X")
        #     break
        cigt_outputs = next(iter(self.testDataset))
        cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)

        self.eval()
        self.forward_with_policies(x=cigt_outputs, y=None, training=False, greedy_actions=True)
        self.enforcedRoutingMatrices = []

    def validate(self, loader, epoch, data_kind, temperature=None, print_avg_measurements=False,
                 return_network_outputs=False,
                 verbose=False):
        batch_time = AverageMeter()
        mean_reward_for_batch_avg = AverageMeter()
        macs_per_batch_avg = AverageMeter()
        accuracy_per_batch_avg = AverageMeter()
        mean_state_network_loss_avg = AverageMeter()
        mean_policy_value_avg = AverageMeter()
        mean_policy_value_no_baseline_avg = AverageMeter()

        if temperature is None:
            temperature = 1.0
        if verbose is False:
            verbose_loader = enumerate(loader)
        else:
            verbose_loader = tqdm(enumerate(loader))

        for i, cigt_outputs in verbose_loader:
            time_begin = time.time()
            cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
            with torch.no_grad():
                # input_var = torch.autograd.Variable(input_).to(self.device)
                # target_var = torch.autograd.Variable(target).to(self.device)
                batch_size = cigt_outputs["block_outputs_dict"][(0, )].shape[0]
                outputs = self.forward_with_policies(x=cigt_outputs, y=None, training=False, greedy_actions=False)

                # Mean reward from the network execution.
                mean_reward_for_batch = torch.mean(outputs["reward_array"])
                mean_reward_for_batch = mean_reward_for_batch.detach().cpu().numpy().item()
                mean_reward_for_batch_avg.update(mean_reward_for_batch, batch_size)
                # Mean accuracy for the batch.
                accuracy_per_batch = torch.mean(outputs["correctness_vec"]).detach().cpu().numpy().item()
                accuracy_per_batch_avg.update(accuracy_per_batch, batch_size)
                # Mean mac for the batch.
                macs_per_batch = torch.mean(outputs["mac_vec"]).detach().cpu().numpy().item()
                macs_per_batch_avg.update(macs_per_batch, batch_size)
        return {
            "mean_reward_for_batch_avg": mean_reward_for_batch_avg.avg,
            "accuracy_per_batch_avg": accuracy_per_batch_avg.avg,
            "macs_per_batch_avg": macs_per_batch_avg.avg}

    def fit_policy_network(self, train_loader, test_loader):
        self.to(self.device)
        torch.manual_seed(1)
        best_performance = 0.0
        num_of_total_iterations = self.policyNetworkTotalNumOfEpochs * len(train_loader)

        # Run a forward pass first to initialize each LazyXXX layer.
        self.execute_forward_with_random_input()

        # Test with enforced actions set to 0. The accuracy should be the naive IG accuracy.
        self.toggle_allways_ig_routing(enable=True)
        validation_dict = self.validate(loader=test_loader, epoch=-1, data_kind="test", temperature=1.0)
        self.toggle_allways_ig_routing(enable=False)
        print("test_ig_accuracy_avg:{0} test_ig_mac_avg:{1}".format(validation_dict["accuracy_per_batch_avg"],
                                                                    validation_dict["macs_per_batch_avg"]))

        # Create the model optimizer, we should have every parameter initialized right now.
        self.policyGradientsModelOptimizer = self.create_optimizer()

        temp_warm_up_state = self.isInWarmUp
        temp_random_routing_ratio = self.routingRandomizationRatio
        self.isInWarmUp = False
        self.routingRandomizationRatio = -1.0
        # Train the policy network for one epoch
        iteration_id = 0
        for epoch_id in range(0, self.policyNetworkTotalNumOfEpochs):
            for i, cigt_outputs in enumerate(train_loader):
                print("*************Policy Network Training Epoch:{0} Iteration:{1}*************".format(
                    epoch_id, self.iteration_id))
                cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)

                # Adjust the learning rate
                self.adjust_learning_rate_polynomial(iteration=self.iteration_id,
                                                     num_of_total_iterations=num_of_total_iterations)
                # Print learning rates
                print("Policy Network Lr:{0}".format(self.policyGradientsModelOptimizer.param_groups[0]["lr"]))
                self.policyGradientsModelOptimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    batch_size = cigt_outputs["block_outputs_dict"][(0,)].shape[0]
                    outputs = self.forward_with_policies(x=cigt_outputs, y=None, training=True, greedy_actions=False)
                    cumulative_rewards = self.calculate_cumulative_rewards(rewards_array=outputs["reward_array"])
                    self.update_baselines(cumulative_rewards=cumulative_rewards)
                    print("Baseline Values:{0}".format(self.baselinesPerLayer))
                    policy_loss = self.calculate_policy_loss(cumulative_rewards=cumulative_rewards,
                                                             log_policy_probs=outputs["log_probs_trajectory"])
                    entropy_loss = torch.Tensor(outputs["policy_entropies"])
                    entropy_loss = -torch.sum(entropy_loss)
                    total_loss = policy_loss + (self.policyNetworksEntropyLossCoeff * entropy_loss)

                    # Step
                    if self.isDebugMode:
                        grad_check = [param.grad is None or np.array_equal(param.grad.cpu().numpy(),
                                                                           np.zeros_like(param.grad.cpu().numpy()))
                                      for param in self.policyGradientsModelOptimizer.param_groups[0]["params"]]
                        # print(self.policyGradientsModelOptimizer.param_groups[0]["params"][0].grad)
                        # print(grad_check)
                        assert all(grad_check)
                    total_loss.backward()
                    if self.isDebugMode:
                        grad_check = [isinstance(param.grad, torch.Tensor) for param in
                                      self.policyGradientsModelOptimizer.param_groups[0]["params"]]
                        assert all(grad_check)
                    self.policyGradientsModelOptimizer.step()

                    print("Epoch:{0} Iteration:{1} Reward:{2} Policy Loss:{3} Entropy Loss:{4}".format(
                        epoch_id,
                        self.iteration_id,
                        torch.mean(outputs["reward_array"]).detach().cpu().numpy(),
                        policy_loss.detach().cpu().numpy(),
                        entropy_loss.detach().cpu().numpy()))
                self.iteration_id += 1

            # Validation
            if epoch_id % self.policyNetworksEvaluationPeriod == 0 or \
                    epoch_id >= (self.policyNetworkTotalNumOfEpochs - 10):
                print("***************Db:{0} RunId:{1} Epoch {2} End, Training Evaluation***************".format(
                    DbLogger.log_db_path, self.runId, epoch_id))
                train_dict = self.validate(loader=train_loader, epoch=epoch_id, data_kind="train", temperature=1.0)
                print("train_reward:{0} train_accuracy:{1} train_mac_avg:{2}".format(
                    train_dict["mean_reward_for_batch_avg"],
                    train_dict["accuracy_per_batch_avg"],
                    train_dict["macs_per_batch_avg"]))
                print("***************Db:{0} RunId:{1} Epoch {2} End, Test Evaluation***************".format(
                    DbLogger.log_db_path, self.runId, epoch_id))
                test_dict = self.validate(loader=test_loader, epoch=epoch_id, data_kind="test", temperature=1.0)
                print("test_reward:{0} test_accuracy:{1} test_mac_avg:{2}".format(
                    test_dict["mean_reward_for_batch_avg"],
                    test_dict["accuracy_per_batch_avg"],
                    test_dict["macs_per_batch_avg"]))
                self.toggle_allways_ig_routing(enable=True)
                validation_dict = self.validate(loader=test_loader, epoch=-1, data_kind="test", temperature=1.0)
                self.toggle_allways_ig_routing(enable=False)
                print("test_ig_accuracy:{0} test_mac_ig_avg:{1}".format(
                    validation_dict["accuracy_per_batch_avg"], validation_dict["macs_per_batch_avg"]))
                # self.save_cigt_model(epoch=epoch_id)

                DbLogger.write_into_table(
                    rows=[(self.runId,
                           self.iteration_id,
                           epoch_id,
                           train_dict["accuracy_per_batch_avg"],
                           train_dict["macs_per_batch_avg"],
                           test_dict["accuracy_per_batch_avg"],
                           test_dict["macs_per_batch_avg"],
                           0.0,
                           0.0,
                           "YYY")], table=DbLogger.logsTable)

    # def forward_with_policies(self, x, y, training, greedy_actions):
    #     cigt_outputs = x
    #     # cigt_outputs = self.forward_v2(x=x, labels=y, temperature=1.0)
    #     if training:
    #         self.train()
    #     else:
    #         self.eval()
    #
    #     policy_entropies = []
    #     log_probs_trajectory = []
    #     actions_trajectory = []
    #     correctness_vec = None
    #     mac_vec = None
    #     reward_array = None
    #     paths_history = [{idx: {(0,)} for idx in range(x.shape[0])}]
    #
    #     for layer_id in range(len(self.pathCounts)):
    #         if layer_id < len(self.pathCounts) - 1:
    #             # Get sparse input arrays for the policy networks
    #             # OK FOR PREPROCESSED DATASET IMPLEMENTATION!!!
    #             pg_sparse_input = self.prepare_policy_network_input_f(
    #                 batch_size=x.shape[0],
    #                 layer_id=layer_id,
    #                 current_paths_dict=paths_history[layer_id],
    #                 cigt_outputs=cigt_outputs
    #             )
    #             # Execute this layers policy network, get log action probs, actions and policy entropies.
    #             # OK FOR PREPROCESSED DATASET IMPLEMENTATION!!!
    #             mean_policy_entropy, log_probs_selected, probs_selected, action_probs, actions = \
    #                 self.run_policy_networks(layer_id=layer_id, pn_input=pg_sparse_input)
    #             policy_entropies.append(mean_policy_entropy)
    #             log_probs_trajectory.append(log_probs_selected)
    #             actions_trajectory.append(actions)
    #
    #             # Extend the trajectories for each sample based on the actions selected.
    #             # OK FOR PREPROCESSED DATASET IMPLEMENTATION!!!
    #             new_paths_dict = self.extend_sample_trajectories_wrt_actions(actions=actions,
    #                                                                          cigt_outputs=cigt_outputs,
    #                                                                          current_paths_dict=paths_history[layer_id],
    #                                                                          layer_id=layer_id)
    #             paths_history.append(new_paths_dict)
    #         else:
    #             # OK FOR PREPROCESSED DATASET IMPLEMENTATION!!!
    #             reward_array, correctness_vec, mac_vec = \
    #                 self.calculate_rewards(cigt_outputs=cigt_outputs, complete_path_history=paths_history)
    #
    #     for lid, actions_arr in enumerate(actions_trajectory):
    #         print("Layer{0} Actions:{1}".format(lid, Counter(actions_arr)))
    #
    #     return {
    #         "policy_entropies": policy_entropies,
    #         "log_probs_trajectory": log_probs_trajectory,
    #         "actions_trajectory": actions_trajectory,
    #         "paths_history": paths_history,
    #         "reward_array": reward_array,
    #         "correctness_vec": correctness_vec,
    #         "mac_vec": mac_vec}