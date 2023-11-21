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
            for k, v in cigt_outputs[field_name].items():
                d_[field_name][k] = v.to(self.device)
        return d_

    def get_cigt_outputs(self, x, y):
        assert isinstance(x, dict)
        assert y is None
        cigt_outputs = x
        batch_size = cigt_outputs["block_outputs_dict"][(0,)].shape[0]
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
        self.forward_with_policies(x=cigt_outputs, y=None, greedy_actions=True)
        self.enforcedRoutingMatrices = []

    def validate(self, loader, epoch, data_kind, temperature=None, print_avg_measurements=False,
                 return_network_outputs=False,
                 verbose=False, repeat_count=1):
        self.eval()
        if temperature is None:
            temperature = 1.0
        # if verbose is False:
        #     verbose_loader = enumerate(loader)
        # else:
        #     verbose_loader = tqdm(enumerate(loader))

        results_array = []
        counters = [Counter() for _ in range(len(self.pathCounts) - 1)]

        for repeat_id in range(repeat_count):
            mean_reward_for_batch_avg = AverageMeter()
            macs_per_batch_avg = AverageMeter()
            accuracy_per_batch_avg = AverageMeter()
            print("Repeat Id:{0}".format(repeat_id))
            print("Device:{0}".format(self.device))
            for i__, cigt_outputs in tqdm(enumerate(loader)):
                time_begin = time.time()
                cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
                # print(cigt_outputs["block_outputs_dict"][(0,)].shape[0])
                with torch.no_grad():
                    # input_var = torch.autograd.Variable(input_).to(self.device)
                    # target_var = torch.autograd.Variable(target).to(self.device)
                    batch_size = cigt_outputs["block_outputs_dict"][(0,)].shape[0]
                    outputs = self.forward_with_policies(x=cigt_outputs, y=None, greedy_actions=False)

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
                    # Action trajectory
                    actions_trajectory = outputs["actions_trajectory"]
                    for lid, actions_arr in enumerate(actions_trajectory):
                        counters[lid].update(actions_arr)

            results_dict = {
                "mean_reward_for_batch_avg": mean_reward_for_batch_avg.avg,
                "accuracy_per_batch_avg": accuracy_per_batch_avg.avg,
                "macs_per_batch_avg": macs_per_batch_avg.avg}
            results_array.append(results_dict)

        results_accumulated = {}
        for d_ in results_array:
            for k, v in d_.items():
                if k not in results_accumulated:
                    results_accumulated[k] = []
                results_accumulated[k].append(v)

        for k in results_accumulated:
            results_accumulated[k] = np.mean(np.array(results_accumulated[k]))

        for lid, counter in enumerate(counters):
            results_accumulated["counter_{0}".format(lid)] = counter

        return results_accumulated

    def validate_with_expectation(self, loader, temperature=None):
        self.eval()
        if temperature is None:
            temperature = 1.0
        action_space = Utilities.create_route_combinations(shape_=self.pathCounts[1:])
        action_probabilities_dict = {}
        correctness_vectors_dict = {}
        mac_vectors_dict = {}
        reward_vectors_dict = {}
        policy_probabilities_dict = {}
        for actions in action_space:
            action_probabilities_dict[actions] = []
            correctness_vectors_dict[actions] = []
            mac_vectors_dict[actions] = []
            reward_vectors_dict[actions] = []
            policy_probabilities_dict[actions] = []
            for _ in range(len(self.pathCounts) - 1):
                policy_probabilities_dict[actions].append([])

        print("Device:{0}".format(self.device))
        for i__, cigt_outputs in tqdm(enumerate(loader)):
            time_begin = time.time()
            cigt_outputs = self.move_cigt_outputs_to_device(cigt_outputs=cigt_outputs)
            for actions in action_space:
                # Prepare actions arrays
                self.policyNetworksEnforcedActions = []
                max_branch_count = np.prod(self.pathCounts)
                for layer_id, action in enumerate(actions):
                    layer_actions = action * torch.ones(size=(max_branch_count * self.batchSize,),
                                                        dtype=torch.int64).to(self.device)
                    self.policyNetworksEnforcedActions.append(layer_actions)
                # Run on the dataset with the predefined action arrays
                with torch.no_grad():
                    # input_var = torch.autograd.Variable(input_).to(self.device)
                    # target_var = torch.autograd.Variable(target).to(self.device)
                    batch_size = cigt_outputs["block_outputs_dict"][(0,)].shape[0]
                    outputs = self.forward_with_policies(x=cigt_outputs, y=None, greedy_actions=False)
                    # Record probabilities
                    action_trajectory_probabilities = outputs["probs_trajectory"]
                    action_trajectory_probabilities = torch.stack(action_trajectory_probabilities, dim=1)
                    action_trajectory_probabilities = torch.prod(action_trajectory_probabilities, dim=1)
                    action_probabilities_dict[actions].append(action_trajectory_probabilities.detach().cpu().numpy())
                    # Record rewards
                    reward_vectors_dict[actions].append(outputs["reward_array"].detach().cpu().numpy())
                    # Record accuracies
                    correctness_vectors_dict[actions].append(outputs["correctness_vec"].detach().cpu().numpy())
                    # Record mac values
                    mac_vectors_dict[actions].append(outputs["mac_vec"].detach().cpu().numpy())
                    # Record full policy distributions
                    for lid, arr in enumerate(outputs["full_action_probs_trajectory"]):
                        policy_probabilities_dict[actions][lid].append(arr.detach().cpu().numpy())
        self.policyNetworksEnforcedActions = []

        # Merge all results
        for actions in action_space:
            action_probabilities_dict[actions] = np.concatenate(action_probabilities_dict[actions])
            # Record rewards
            reward_vectors_dict[actions] = np.concatenate(reward_vectors_dict[actions])
            # Record accuracies
            correctness_vectors_dict[actions] = np.concatenate(correctness_vectors_dict[actions])
            # Record mac values
            mac_vectors_dict[actions] = np.concatenate(mac_vectors_dict[actions])
            # Record full policy distributions
            for lid, arr in enumerate(policy_probabilities_dict[actions]):
                policy_probabilities_dict[actions][lid] = np.concatenate(arr, axis=0)

        action_probabilities_matrix = []
        reward_vectors_matrix = []
        correctness_vectors_matrix = []
        mac_vectors_matrix = []
        for actions in action_space:
            action_probabilities_matrix.append(action_probabilities_dict[actions])
            reward_vectors_matrix.append(reward_vectors_dict[actions])
            correctness_vectors_matrix.append(correctness_vectors_dict[actions])
            mac_vectors_matrix.append(mac_vectors_dict[actions])

        action_probabilities_matrix = np.stack(action_probabilities_matrix, axis=1)
        reward_vectors_matrix = np.stack(reward_vectors_matrix, axis=1)
        correctness_vectors_matrix = np.stack(correctness_vectors_matrix, axis=1)
        mac_vectors_matrix = np.stack(mac_vectors_matrix, axis=1)

        sum_prob = np.sum(action_probabilities_matrix, axis=1)
        assert np.allclose(sum_prob, np.ones_like(sum_prob))

        expected_reward = np.mean(np.sum(action_probabilities_matrix * reward_vectors_matrix, axis=1))
        expected_accuracy = np.mean(np.sum(action_probabilities_matrix * correctness_vectors_matrix, axis=1))
        expected_mac = np.mean(np.sum(action_probabilities_matrix * mac_vectors_matrix, axis=1))

        prob_dict = {}
        for lid in range(len(self.pathCounts) - 1):
            for actions in action_space:
                if actions[:lid] not in prob_dict:
                    prob_dict[actions[:lid]] = []
                prob_dict[actions[:lid]].append(policy_probabilities_dict[actions][lid])
        mean_policy_distributions = {}
        for k in prob_dict.keys():
            stacked_probs = np.stack(prob_dict[k], axis=0)
            mean_probs = np.mean(stacked_probs, axis=0)
            diff_arr = stacked_probs - np.expand_dims(mean_probs, axis=0)
            assert np.abs(diff_arr - np.zeros_like(diff_arr)).max() < 1e-6
            mean_policy_distribution = np.mean(mean_probs, axis=0)
            print("Trajectory {0} Mean Policy Distribution:{1}".format(k, mean_policy_distribution))
            mean_policy_distributions[k] = mean_policy_distribution

        results_dict = {
            "expected_reward": expected_reward,
            "expected_accuracy": expected_accuracy,
            "expected_mac": expected_mac,
            "mean_policy_distributions": mean_policy_distributions}
        self.policyNetworksEnforcedActions = []
        return results_dict

    def evaluate_datasets(self, train_loader, test_loader, epoch):
        # Test with enforced actions set to 0. The accuracy should be the naive IG accuracy.
        self.toggle_allways_ig_routing(enable=True)
        validation_dict = self.validate(loader=test_loader, epoch=-1, data_kind="test", temperature=1.0,
                                        repeat_count=1, verbose=False)
        self.toggle_allways_ig_routing(enable=False)
        print("test_ig_accuracy_avg:{0} test_ig_mac_avg:{1}".format(validation_dict["accuracy_per_batch_avg"],
                                                                    validation_dict["macs_per_batch_avg"]))
        outputs = {}
        kv_rows = []
        for data_type, data_loader in [("Test", test_loader), ("Train", train_loader)]:
            print("***************Db:{0} RunId:{1} Epoch {2} End, {3} Evaluation***************".format(
                DbLogger.log_db_path, self.runId, epoch, data_type))
            outputs_dict = self.validate_with_expectation(loader=data_loader, temperature=1.0)
            outputs[data_type] = outputs_dict
            print("{0} Reward:{1}".format(data_type, outputs_dict["expected_reward"]))
            print("{0} Accuracy:{1}".format(data_type, outputs_dict["expected_accuracy"]))
            print("{0} Mac:{1}".format(data_type, outputs_dict["expected_mac"]))
            for k, dist in outputs_dict["mean_policy_distributions"].items():
                print("{0}-{1} mean_policy_distributions:{2}".format(data_type, k, dist))
                kv_rows.append((self.runId,
                                epoch,
                                "{0}-{1} mean_policy_distributions".format(data_type, k),
                                "{0}".format(dist)))
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)

        DbLogger.write_into_table(
            rows=[(self.runId,
                   self.iteration_id,
                   epoch,
                   outputs["Train"]["expected_reward"].item(),
                   outputs["Train"]["expected_accuracy"].item(),
                   outputs["Train"]["expected_mac"].item(),
                   # 0.0,
                   # 0.0,
                   # 0.0,
                   outputs["Test"]["expected_reward"].item(),
                   outputs["Test"]["expected_accuracy"].item(),
                   outputs["Test"]["expected_mac"].item(),
                   "YYY")], table=DbLogger.logsTable)

    def fit_policy_network(self, train_loader, test_loader):
        self.to(self.device)
        torch.manual_seed(1)
        best_performance = 0.0
        num_of_total_iterations = self.policyNetworkTotalNumOfEpochs * len(train_loader)

        # Run a forward pass first to initialize each LazyXXX layer.
        self.execute_forward_with_random_input()

        # Run validation in the beginning
        self.evaluate_datasets(train_loader=train_loader, test_loader=test_loader, epoch=-1)

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
                self.train()
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
                    outputs = self.forward_with_policies(x=cigt_outputs, y=None, greedy_actions=False)
                    cumulative_rewards = self.calculate_cumulative_rewards(rewards_array=outputs["reward_array"])
                    self.update_baselines(cumulative_rewards=cumulative_rewards)
                    print("Baseline Values:{0}".format(self.baselinesPerLayer))
                    policy_loss = self.calculate_policy_loss(cumulative_rewards=cumulative_rewards,
                                                             log_policy_probs=outputs["log_probs_trajectory"])
                    entropy_loss = torch.Tensor(outputs["policy_entropies"])
                    entropy_loss = -torch.sum(entropy_loss)
                    total_loss = policy_loss + (self.policyNetworksEntropyLossCoeff * entropy_loss)

                    actions_trajectory = outputs["actions_trajectory"]

                    for lid, actions_arr in enumerate(actions_trajectory):
                        print("Layer{0} Actions:{1}".format(lid, Counter(actions_arr)))

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
                self.evaluate_datasets(train_loader=train_loader, test_loader=test_loader, epoch=epoch_id)

