# from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
import os
import torch
from randaugment import RandAugment
from torchvision import transforms, datasets

from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cigt_output_dataset import CigtOutputDataset
from cigt.cigt_qlearning_end2end import CigtQlearningEnd2End
from cigt.cutout_augmentation import CutoutPIL
from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from configs.cifar10_resnet_cigt_configs import adjust_to_batch_size
from configs.q_cigt_cifar_10_end_to_end_configs import QCigtCifar10Configs


# random.seed(53)
# np.random.seed(61)

def get_cifar_datasets():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    print("QCigtCifar10Configs.advanced_augmentation:{0}".format(
        QCigtCifar10Configs.advanced_augmentation
    ))
    if not QCigtCifar10Configs.advanced_augmentation:
        print("WILL BE USING ONLY CROP AND HORIZONTAL FLIP AUGMENTATION")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        print("WILL BE USING RANDOM AUGMENTATION")
        transform_train = transforms.Compose([
            transforms.Resize(QCigtCifar10Configs.input_dims[1:]),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(QCigtCifar10Configs.input_dims[1:]),
            transforms.ToTensor(),
        ])

    # Cifar 10 Dataset
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
        batch_size=QCigtCifar10Configs.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=QCigtCifar10Configs.batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


if __name__ == "__main__":
    print("X")

    # target = 0.6
    # x = torch.tensor([1.], requires_grad=True)
    # y = torch.tensor([1.], requires_grad=True)
    # target_proxy = torch.zeros(size=x.shape, dtype=x.dtype)
    # with torch.set_grad_enabled(True):
    #     fx = torch.sin(x)
    #     fy = torch.cos(y)
    #     # with torch.set_grad_enabled(False):
    #     #     target_proxy[0] = 0.3 * x + 0.3 * y
    #     s = fx + fy
    #     s = torch.max(s)
    #     a = torch.argmax(s)
    #     loss = torch.pow(a - target, 2.0)
    # loss.backward()
    # print("")
    # 5e-4,
    # 0.0005
    QCigtCifar10Configs.layer_config_list = [
        {"path_count": 1,
         "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
        {"path_count": 2,
         "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
                             {"layer_count": 18, "feature_map_count": 16}]},
        {"path_count": 4,
         "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]

    QCigtCifar10Configs.policy_networks_cbam_layer_count = 3
    QCigtCifar10Configs.policy_networks_cbam_feature_map_count = 32
    QCigtCifar10Configs.policy_networks_cbam_reduction_ratio = 4
    QCigtCifar10Configs.policy_networks_cbam_layer_input_reduction_ratio = 4
    QCigtCifar10Configs.policy_networks_cbam_end_avg_pool_strode = 2
    QCigtCifar10Configs.policy_networks_use_lstm = True
    QCigtCifar10Configs.policy_networks_lstm_dimension = 128
    QCigtCifar10Configs.policy_networks_lstm_num_layers = 1
    QCigtCifar10Configs.policy_networks_lstm_bidirectional = False
    QCigtCifar10Configs.policy_networks_total_num_of_epochs = 250
    QCigtCifar10Configs.policy_networks_initial_lr = 0.0001
    QCigtCifar10Configs.policy_networks_backbone_lr_coefficient = 0.1
    QCigtCifar10Configs.policy_networks_polynomial_scheduler_power = 1.0
    QCigtCifar10Configs.policy_networks_wd = 0.0001
    QCigtCifar10Configs.policy_networks_mac_lambda = 0.05
    QCigtCifar10Configs.policy_networks_discount_factor = 0.99
    QCigtCifar10Configs.policy_networks_logit_temperature = 1.0
    QCigtCifar10Configs.policy_networks_apply_reward_whitening = False
    QCigtCifar10Configs.policy_networks_evaluation_period = 5
    QCigtCifar10Configs.policy_networks_use_moving_average_baseline = True
    QCigtCifar10Configs.policy_networks_baseline_momentum = 0.99
    QCigtCifar10Configs.policy_networks_policy_entropy_loss_coeff = 0.0
    QCigtCifar10Configs.policy_networks_epsilon_decay_coeff = 1.0
    QCigtCifar10Configs.policy_networks_last_eval_start = 5
    QCigtCifar10Configs.policy_networks_train_only_action_heads = False
    QCigtCifar10Configs.policy_networks_no_improvement_stop_count = 20
    QCigtCifar10Configs.policy_networks_initial_lr = 0.0001
    QCigtCifar10Configs.policy_networks_backbone_lr_coefficient = 0.1

    QCigtCifar10Configs.classification_wd = 0.0004
    QCigtCifar10Configs.information_gain_balance_coeff_list = [5.0, 5.0]
    QCigtCifar10Configs.loss_calculation_kind = "MultipleLogitsMultipleLosses"
    QCigtCifar10Configs.enable_information_gain_during_warm_up = True
    QCigtCifar10Configs.enable_strict_routing_randomization = False
    QCigtCifar10Configs.routing_randomization_ratio = 0.5
    QCigtCifar10Configs.warm_up_kind = "RandomRouting"
    QCigtCifar10Configs.decision_drop_probability = 0.5
    QCigtCifar10Configs.number_of_cbam_layers_in_routing_layers = 3
    QCigtCifar10Configs.cbam_reduction_ratio = 4
    QCigtCifar10Configs.cbam_layer_input_reduction_ratio = 4
    QCigtCifar10Configs.apply_relu_dropout_to_decision_layer = False
    QCigtCifar10Configs.decision_dimensions = [128, 128]
    QCigtCifar10Configs.apply_mask_to_batch_norm = False
    QCigtCifar10Configs.advanced_augmentation = True
    QCigtCifar10Configs.use_focal_loss = False
    QCigtCifar10Configs.focal_loss_gamma = 2.0
    QCigtCifar10Configs.batch_norm_type = "BatchNorm"
    QCigtCifar10Configs.data_parallelism = False
    QCigtCifar10Configs.policy_networks_evaluation_period = 5

    QCigtCifar10Configs.softmax_decay_controller = StepWiseDecayAlgorithm(
        decay_name="Stepwise",
        initial_value=QCigtCifar10Configs.softmax_decay_initial,
        decay_coefficient=QCigtCifar10Configs.softmax_decay_coefficient,
        decay_period=QCigtCifar10Configs.softmax_decay_period,
        decay_min_limit=QCigtCifar10Configs.softmax_decay_min_limit)

    train_loader, test_loader = get_cifar_datasets()
    DbLogger.log_db_path = DbLogger.hpc_docker1

    model_mac = CigtIgGatherScatterImplementation(
        run_id=-1,
        model_definition="Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
        num_classes=10,
        configs=QCigtCifar10Configs)
    model_mac.to(model_mac.device)
    model_mac.execute_forward_with_random_input()
    mac_counts_per_block = CigtIgHardRoutingX.calculate_mac(model=model_mac)
    model_mac = None

    print("Start!")
    run_id = DbLogger.get_run_id()

    print("Running run_id:{0}".format(run_id))
    print("Writing into DB:{0}".format(DbLogger.log_db_path))
    print("Running with mac_lambda:{0}".format(QCigtCifar10Configs.policy_networks_mac_lambda))
    print("Running with wd_coeff:{0}".format(QCigtCifar10Configs.policy_networks_wd))

    model = CigtQlearningEnd2End(
        configs=QCigtCifar10Configs,
        model_definition="Q Learning CIGT",
        num_classes=10,
        run_id=run_id,
        model_mac_info=mac_counts_per_block,
        is_debug_mode=False,
        precalculated_datasets_dict=None)
    chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..",
                             "checkpoints/cigtlogger2_75_epoch1575.pth")
    data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..",
                             "cigtlogger2_75_epoch1575_data")
    checkpoint = torch.load(chck_path, map_location=model.device)
    model_load_results = model.load_state_dict(state_dict=checkpoint["model_state_dict"], strict=False)
    assert all([elem.startswith("policyNetworks") for elem in model_load_results.missing_keys])
    model.to(model.device)

    model.modelFilesRootPath = QCigtCifar10Configs.model_file_root_path_hpc_docker
    explanation = model.get_explanation_string()
    DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

    model.execute_forward_with_random_input()
    model.fit_policy_network(train_loader=train_loader, test_loader=test_loader)

    kv_rows = [(run_id,
                model.policyNetworkTotalNumOfEpochs,
                "Training Status",
                "Training Finished!!!")]
    DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)


    # # mac_lambda_list = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15]
    # # wd_list = [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01] * 5
    # # mac_lambda_list = sorted(mac_lambda_list)
    # # wd_list = sorted(wd_list)
    # # param_grid = Utilities.get_cartesian_product(list_of_lists=[mac_lambda_list, wd_list])
    #
    #
    # # Skip already processed parameter combinations
    # # get_parameters_histogram(parameters_used=["policyNetworksWd", "policyNetworksMacLambda"])
    #
    # for iteration_id, params in enumerate(param_grid):
    #     print("Iteration:{0}".format(iteration_id))
    #     run_id = DbLogger.get_run_id()
    #     # mac_lambda = params[0]
    #     # wd_coeff = params[1]
    #     # QCigtCifar10Configs.policy_networks_mac_lambda = mac_lambda
    #     # QCigtCifar10Configs.policy_networks_wd = wd_coeff
    #
    #     print("Running run_id:{0}".format(run_id))
    #     print("Writing into DB:{0}".format(DbLogger.log_db_path))
    #     print("Running with mac_lambda:{0}".format(QCigtCifar10Configs.policy_networks_mac_lambda))
    #     print("Running with wd_coeff:{0}".format(QCigtCifar10Configs.policy_networks_wd))
    #
    #     model = CigtQlearningEnd2End(
    #         configs=QCigtCifar10Configs,
    #         model_definition="Q Learning CIGT",
    #         num_classes=10,
    #         run_id=run_id,
    #         model_mac_info=mac_counts_per_block,
    #         is_debug_mode=False,
    #         precalculated_datasets_dict=None)
    #     chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..",
    #                              "checkpoints/cigtlogger2_75_epoch1575.pth")
    #     data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..",
    #                              "cigtlogger2_75_epoch1575_data")
    #     checkpoint = torch.load(chck_path, map_location=model.device)
    #     model_load_results = model.load_state_dict(state_dict=checkpoint["model_state_dict"], strict=False)
    #     assert all([elem.startswith("policyNetworks") for elem in model_load_results.missing_keys])
    #     model.to(model.device)
    #
    #     model.modelFilesRootPath = QCigtCifar10Configs.model_file_root_path_hpc_docker
    #     explanation = model.get_explanation_string()
    #     DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
    #
    #     model.execute_forward_with_random_input()
    #     model.fit_policy_network(train_loader=train_loader, test_loader=test_loader)
    #
    #     kv_rows = [(run_id,
    #                 model.policyNetworkTotalNumOfEpochs,
    #                 "Training Status",
    #                 "Training Finished!!!")]
    #     DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
