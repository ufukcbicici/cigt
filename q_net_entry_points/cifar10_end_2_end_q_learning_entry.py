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
from configs.cifar10_resnet_cigt_configs import Cifar10ResnetCigtConfigs


# random.seed(53)
# np.random.seed(61)

def get_cifar_datasets():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if not Cifar10ResnetCigtConfigs.advanced_augmentation:
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
            transforms.Resize(Cifar10ResnetCigtConfigs.input_dims[1:]),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(Cifar10ResnetCigtConfigs.input_dims[1:]),
            transforms.ToTensor(),
        ])

    # Cifar 10 Dataset
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
        batch_size=Cifar10ResnetCigtConfigs.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=Cifar10ResnetCigtConfigs.batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


if __name__ == "__main__":
    print("X")
    # 5e-4,
    # 0.0005
    Cifar10ResnetCigtConfigs.layer_config_list = [
        {"path_count": 1,
         "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
        {"path_count": 2,
         "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
                             {"layer_count": 18, "feature_map_count": 16}]},
        {"path_count": 4,
         "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]

    Cifar10ResnetCigtConfigs.backbone = "ResNet"
    Cifar10ResnetCigtConfigs.input_dims = (3, 32, 32)
    Cifar10ResnetCigtConfigs.batch_size = 800

    # Thin Baseline
    # CigtConstants.layer_config_list = [
    #     {"path_count": 1,
    #      "layer_structure": [{"layer_count": 9, "feature_map_count": 16},
    #                          {"layer_count": 9, "feature_map_count": 12},
    #                          {"layer_count": 18, "feature_map_count": 16},
    #                          {"layer_count": 18, "feature_map_count": 16}]}]

    # Thick Baseline
    # CigtConstants.layer_config_list = [
    #     {"path_count": 1,
    #      "layer_structure": [{"layer_count": 18, "feature_map_count": 16},
    #                          {"layer_count": 18, "feature_map_count": 32},
    #                          {"layer_count": 18, "feature_map_count": 64}]}]

    Cifar10ResnetCigtConfigs.classification_wd = 0.0005
    Cifar10ResnetCigtConfigs.information_gain_balance_coeff_list = [5.0, 5.0]
    Cifar10ResnetCigtConfigs.loss_calculation_kind = "MultipleLogitsMultipleLosses"
    Cifar10ResnetCigtConfigs.enable_information_gain_during_warm_up = True
    Cifar10ResnetCigtConfigs.enable_strict_routing_randomization = False
    Cifar10ResnetCigtConfigs.routing_randomization_ratio = 0.5
    Cifar10ResnetCigtConfigs.warm_up_kind = "RandomRouting"
    Cifar10ResnetCigtConfigs.decision_drop_probability = 0.5
    Cifar10ResnetCigtConfigs.number_of_cbam_layers_in_routing_layers = 3
    Cifar10ResnetCigtConfigs.cbam_reduction_ratio = 4
    Cifar10ResnetCigtConfigs.cbam_layer_input_reduction_ratio = 4
    Cifar10ResnetCigtConfigs.apply_relu_dropout_to_decision_layer = False
    Cifar10ResnetCigtConfigs.decision_dimensions = [128, 128]
    Cifar10ResnetCigtConfigs.apply_mask_to_batch_norm = False
    Cifar10ResnetCigtConfigs.advanced_augmentation = True
    Cifar10ResnetCigtConfigs.use_focal_loss = False
    Cifar10ResnetCigtConfigs.focal_loss_gamma = 2.0
    Cifar10ResnetCigtConfigs.batch_norm_type = "BatchNorm"
    Cifar10ResnetCigtConfigs.data_parallelism = False

    Cifar10ResnetCigtConfigs.softmax_decay_controller = StepWiseDecayAlgorithm(
        decay_name="Stepwise",
        initial_value=Cifar10ResnetCigtConfigs.softmax_decay_initial,
        decay_coefficient=Cifar10ResnetCigtConfigs.softmax_decay_coefficient,
        decay_period=Cifar10ResnetCigtConfigs.softmax_decay_period,
        decay_min_limit=Cifar10ResnetCigtConfigs.softmax_decay_min_limit)

    train_loader, test_loader = get_cifar_datasets()
    DbLogger.log_db_path = DbLogger.paperspace

    model_mac = CigtIgGatherScatterImplementation(
        run_id=-1,
        model_definition="Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
        num_classes=10,
        configs=Cifar10ResnetCigtConfigs)
    model_mac.to(model_mac.device)
    model_mac.execute_forward_with_random_input()
    mac_counts_per_block = CigtIgHardRoutingX.calculate_mac(model=model_mac)
    model_mac = None

    print("Start!")

    mac_lambda_list = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15]
    wd_list = [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01] * 5
    mac_lambda_list = sorted(mac_lambda_list)
    wd_list = sorted(wd_list)
    param_grid = Utilities.get_cartesian_product(list_of_lists=[mac_lambda_list, wd_list])
    Cifar10ResnetCigtConfigs.policy_networks_evaluation_period = 1

    # Skip already processed parameter combinations
    # get_parameters_histogram(parameters_used=["policyNetworksWd", "policyNetworksMacLambda"])

    for params in param_grid:
        run_id = DbLogger.get_run_id()
        mac_lambda = params[0]
        wd_coeff = params[1]
        Cifar10ResnetCigtConfigs.policy_networks_mac_lambda = mac_lambda
        Cifar10ResnetCigtConfigs.policy_networks_wd = wd_coeff

        print("Running run_id:{0}".format(run_id))
        print("Writing into DB:{0}".format(DbLogger.log_db_path))
        print("Running with mac_lambda:{0}".format(Cifar10ResnetCigtConfigs.policy_networks_mac_lambda))
        print("Running with wd_coeff:{0}".format(Cifar10ResnetCigtConfigs.policy_networks_wd))

        model = CigtQlearningEnd2End(
            configs=Cifar10ResnetCigtConfigs,
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

        model.modelFilesRootPath = Cifar10ResnetCigtConfigs.model_file_root_path_paperspace
        explanation = model.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        model.execute_forward_with_random_input()
        model.fit_policy_network(train_loader=train_loader, test_loader=test_loader)

        kv_rows = [(run_id,
                    model.policyNetworkTotalNumOfEpochs,
                    "Training Status",
                    "Training Finished!!!")]
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)
