import os

from randaugment import RandAugment
from torchvision import datasets
from torchvision.transforms import transforms

from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cutout_augmentation import CutoutPIL
from cigt.multipath_evaluator import MultipathEvaluator
from cigt.multipath_inference_bayesian import MultiplePathBayesianOptimizer
# from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
import torch

from cigt.multipath_inference_cross_entropy import MultipathInferenceCrossEntropy
from cigt.multipath_inference_cross_entropy_v2 import MultipathInferenceCrossEntropyV2
from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from configs.cifar10_resnet_cigt_configs import Cifar10ResnetCigtConfigs
from multipath_cross_entropy_optimization_with_validation_scoring import MultipathInferenceCrossEntropyValidationScoring
from multipath_inference_cross_entropy_free_target import MultipathInferenceCrossEntropyFreeTarget

# random.seed(53)
# np.random.seed(61)

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

    kwargs = {'num_workers': 0, 'pin_memory': True}
    heavyweight_augmentation = transforms.Compose([
        transforms.Resize((32, 32)),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor(),
    ])
    lightweight_augmentation = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_loader_hard = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', download=True, train=True, transform=heavyweight_augmentation),
        batch_size=Cifar10ResnetCigtConfigs.batch_size, shuffle=False, **kwargs)
    test_loader_light = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', download=True, train=False, transform=lightweight_augmentation),
        batch_size=Cifar10ResnetCigtConfigs.batch_size, shuffle=False, **kwargs)

    # chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "checkpoints/cigtlogger2_75_epoch1575.pth")
    # data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "cigtlogger2_75_epoch1575_data")

    # chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "checkpoints/dblogger2_94_epoch1390.pth")
    # data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "dblogger2_94_epoch1390_data")

    chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "checkpoints/cigtlogger2_73_epoch1660.pth")
    data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "cigtlogger2_73_epoch1660_data")

    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    DbLogger.log_db_path = DbLogger.jr_cigt

    model = CigtIgGatherScatterImplementation(
        run_id=-2,
        model_definition="Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
        num_classes=10,
        configs=Cifar10ResnetCigtConfigs)
    model.to(model.device)
    model.execute_forward_with_random_input()

    model_mac = CigtIgGatherScatterImplementation(
        run_id=-1,
        model_definition="Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
        num_classes=10,
        configs=Cifar10ResnetCigtConfigs)
    model_mac.to(model_mac.device)
    model_mac.execute_forward_with_random_input()

    # explanation = model.get_explanation_string()
    # DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
    checkpoint = torch.load(chck_path, map_location=model.device)
    model_load_results = model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    model_load_results_mac = model_mac.load_state_dict(state_dict=checkpoint["model_state_dict"])

    # total_parameter_count = model.get_total_parameter_count()
    mac_counts_per_block = CigtIgHardRoutingX.calculate_mac(model=model_mac)
    model_mac = None
    # accuracy = model.validate(loader=test_loader_light, epoch=0, data_kind="test", temperature=0.1)

    multipath_evaluator = MultipathEvaluator(
        data_root_path=data_path,
        model=model,
        train_dataset=train_loader_hard,
        test_dataset=test_loader_light,
        mac_counts_per_block=mac_counts_per_block,
        evaluate_network_first=False,
        train_dataset_repeat_count=1
    )

    run_id = DbLogger.get_run_id()
    mp_cross_entropy_optimizer = MultipathInferenceCrossEntropyValidationScoring(
        run_id=run_id,
        mac_lambda=0.0,
        max_probabilities=[0.5, 0.25],
        multipath_evaluator=multipath_evaluator,
        n_iter=100,
        quantile_interval=(0.0, 0.05),
        num_of_components=1,
        single_threshold_for_each_layer=False,
        num_samples_each_iteration=1000,
        num_jobs=1,
        covariance_type="diag",
        path_counts=model.pathCounts,
        maximum_iterations_without_improvement=25,
        validation_ratio=0.2)

    mp_cross_entropy_optimizer.histogram_analysis(
        path_to_saved_output=os.path.join(data_path, "cross_entropy_histogram_analysis.sav"),
        repeat_count=100,
        bin_size=10000)

    # FOR GRID SEARCH Method 1
    # mac_lambda_list = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    # max_probabilities_list = [[0.5, 0.25], [1.0, 1.0]]
    # quantile_intervals_list = [(0.0, 0.05)]
    # n_components_list = [1, 2, 3, 5]
    # covariance_types_list = ["diag", "full"]
    # covariance_types_list = sorted(covariance_types_list)
    # single_threshold_for_each_layer_list = [False] * 3
    # single_threshold_for_each_layer_list = sorted(single_threshold_for_each_layer_list)
    #
    # param_grid = Utilities.get_cartesian_product(list_of_lists=[mac_lambda_list,
    #                                                             max_probabilities_list,
    #                                                             quantile_intervals_list,
    #                                                             n_components_list,
    #                                                             covariance_types_list,
    #                                                             single_threshold_for_each_layer_list])
    # for params in param_grid:
    #     mac_lambda = params[0]
    #     max_probabilities = params[1]
    #     quantile_interval = params[2]
    #     num_of_components = params[3]
    #     covariance_type = params[4]
    #     single_threshold_for_each_layer = params[5]
    #
    #     run_id = DbLogger.get_run_id()
    #
    #     mp_cross_entropy_optimizer = MultipathInferenceCrossEntropyV2(
    #         run_id=run_id,
    #         mac_lambda=mac_lambda,
    #         max_probabilities=max_probabilities,
    #         multipath_evaluator=multipath_evaluator,
    #         n_iter=100,
    #         quantile_interval=quantile_interval,
    #         num_of_components=num_of_components,
    #         single_threshold_for_each_layer=single_threshold_for_each_layer,
    #         num_samples_each_iteration=10000,
    #         num_jobs=1,
    #         covariance_type=covariance_type,
    #         path_counts=model.pathCounts,
    #         maximum_iterations_without_improvement=25)
    #
    #     mp_cross_entropy_optimizer.fit()

    # FOR GRID SEARCH Method 2
    # accuracy_target_list = [0.5, 0.25]
    # mac_lambda_list = [0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75]
    # max_probabilities_list = [[0.5, 0.25], [1.0, 1.0]]
    # quantile_intervals_list = [(0.0, 0.05)]
    # n_components_list = [1, 2, 3, 5]
    # covariance_types_list = ["diag", "full"]
    # covariance_types_list = sorted(covariance_types_list)
    # single_threshold_for_each_layer_list = [False] * 3
    # single_threshold_for_each_layer_list = sorted(single_threshold_for_each_layer_list)
    #
    # param_grid = Utilities.get_cartesian_product(list_of_lists=[accuracy_target_list,
    #                                                             mac_lambda_list,
    #                                                             max_probabilities_list,
    #                                                             quantile_intervals_list,
    #                                                             n_components_list,
    #                                                             covariance_types_list,
    #                                                             single_threshold_for_each_layer_list])
    # for params in param_grid:
    #     accuracy_target = params[0]
    #     mac_lambda = params[1]
    #     max_probabilities = params[2]
    #     quantile_interval = params[3]
    #     num_of_components = params[4]
    #     covariance_type = params[5]
    #     single_threshold_for_each_layer = params[6]
    #
    #     run_id = DbLogger.get_run_id()
    #
    #     mp_cross_entropy_optimizer = MultipathInferenceCrossEntropyFreeTarget(
    #         run_id=run_id,
    #         mac_lambda=mac_lambda,
    #         max_probabilities=max_probabilities,
    #         multipath_evaluator=multipath_evaluator,
    #         n_iter=100,
    #         quantile_interval=quantile_interval,
    #         num_of_components=num_of_components,
    #         single_threshold_for_each_layer=single_threshold_for_each_layer,
    #         num_samples_each_iteration=10000,
    #         num_jobs=1,
    #         covariance_type=covariance_type,
    #         path_counts=model.pathCounts,
    #         maximum_iterations_without_improvement=25,
    #         accuracy_target_normalized=accuracy_target,
    #         path_to_saved_output="cross_entropy_histogram_analysis.sav")
    #
    #     mp_cross_entropy_optimizer.fit()
