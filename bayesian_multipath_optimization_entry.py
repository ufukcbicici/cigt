import os

from randaugment import RandAugment
from torchvision import datasets
from torchvision.transforms import transforms

from auxillary.db_logger import DbLogger
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cutout_augmentation import CutoutPIL
from cigt.multipath_inference_bayesian import MultiplePathBayesianOptimizer
# from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
import torch

from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from configs.cifar10_resnet_cigt_configs import Cifar10ResnetCigtConfigs

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

    chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                             "checkpoints/dblogger2_94_epoch1390.pth")
    data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "dblogger2_94_epoch1390_data")

    DbLogger.log_db_path = DbLogger.home_asus

    run_id = DbLogger.get_run_id()
    model = CigtIgGatherScatterImplementation(
        run_id=run_id,
        model_definition="Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
        num_classes=10,
        configs=Cifar10ResnetCigtConfigs)
    model.execute_forward_with_random_input()
    model_mac = CigtIgGatherScatterImplementation(
        run_id=run_id,
        model_definition="Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
        num_classes=10,
        configs=Cifar10ResnetCigtConfigs)
    model_mac.execute_forward_with_random_input()

    explanation = model.get_explanation_string()
    DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
    checkpoint = torch.load(chck_path, map_location="cpu")
    model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    model_mac.load_state_dict(state_dict=checkpoint["model_state_dict"])

    # total_parameter_count = model.get_total_parameter_count()
    mac_counts_per_block = CigtIgHardRoutingX.calculate_mac(model=model_mac)
    model_mac = None
    # accuracy = model.validate(loader=test_loader_light, epoch=0, data_kind="test", temperature=0.1)

    mp_bayesian_optimizer = MultiplePathBayesianOptimizer(
        data_root_path=data_path,
        train_dataset=train_loader_hard,
        test_dataset=test_loader_light,
        xi=0.01,
        n_iter=1000,
        init_points=500,
        train_dataset_repeat_count=10,
        evaluate_network_first=True,
        model=model,
        mac_counts_per_block=mac_counts_per_block)

    # # weight_decay = 5 * [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
    # weight_decay = 10 * [0.0005]
    # weight_decay = sorted(weight_decay)
    #
    # param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
    #
    # for param_tpl in param_grid:
    #     ResnetCigtConstants.resnet_config_list = [
    #         {"path_count": 1,
    #          "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
    #         {"path_count": 2,
    #          "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
    #                              {"layer_count": 18, "feature_map_count": 16}]},
    #         {"path_count": 4,
    #          "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
    #     ResnetCigtConstants.classification_wd = param_tpl[0]
    #     ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
    #     ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
    #     ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
    #     ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
    #     ResnetCigtConstants.decision_drop_probability = 0.5
    #     ResnetCigtConstants.number_of_cbam_layers_in_routing_layers = 3
    #     ResnetCigtConstants.cbam_reduction_ratio = 4
    #     ResnetCigtConstants.cbam_layer_input_reduction_ratio = 4
    #     ResnetCigtConstants.apply_relu_dropout_to_decision_layer = False
    #     ResnetCigtConstants.decision_dimensions = [128, 128]
    #     ResnetCigtConstants.apply_mask_to_batch_norm = False
    #     ResnetCigtConstants.advanced_augmentation = True
    #     ResnetCigtConstants.use_focal_loss = False
    #     ResnetCigtConstants.focal_loss_gamma = 2.0
    #     # ResnetCigtConstants.use_kd_for_routing = False
    #     # ResnetCigtConstants.kd_teacher_temperature = 10.0
    #     # ResnetCigtConstants.kd_loss_alpha = 0.95
    #
    #     ResnetCigtConstants.softmax_decay_controller = StepWiseDecayAlgorithm(
    #         decay_name="Stepwise",
    #         initial_value=ResnetCigtConstants.softmax_decay_initial,
    #         decay_coefficient=ResnetCigtConstants.softmax_decay_coefficient,
    #         decay_period=ResnetCigtConstants.softmax_decay_period,
    #         decay_min_limit=ResnetCigtConstants.softmax_decay_min_limit)
    #
    #     run_id = DbLogger.get_run_id()

    # ResnetCigtConstants.loss_calculation_kind = "SingleLogitSingleLoss"
    # teacher_model = CigtIdealRouting(
    #     run_id=run_id,
    #     model_definition="Cigt Ideal Routing - [1,2,4] - SingleLogitSingleLoss - Wd:0.0005",
    #     num_classes=10,
    #     class_to_route_mappings=
    #     [[(5, 6, 4, 7, 3, 2), (1, 8, 9, 0)],
    #      [(6, 0), (1, 8, 9), (5, 7, 3), (4, 2)]])

    # Cifar 10 Dataset
    # kwargs = {'num_workers': 2, 'pin_memory': True}
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10('../data', train=False, transform=teacher_model.transformTest),
    #     batch_size=ResnetCigtConstants.batch_size, shuffle=False, **kwargs)
    # teacher_model.validate(loader=test_loader, epoch=0, data_kind="test")

    # ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
    # model = CigtIgWithKnowledgeDistillation(
    #     run_id=run_id,
    #     model_definition="KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = False - kd_teacher_temperature = 10.0 - kd_loss_alpha = 0.95",
    #     num_classes=10, teacher_model=teacher_model)
    # model.modelFilesRootPath = ResnetCigtConstants.model_file_root_path_hpc
    # explanation = model.get_explanation_string()
    # DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

    # model = CigtMaskedRouting(
    #     run_id=run_id,
    #     model_definition="Masked Cigt - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 "
    #                      "Epoch Warm "
    #                      "up with: RandomRoutingButInformationGainOptimizationEnabled - "
    #                      "InformationGainRoutingWithRandomization - apply_mask_to_batch_norm: True",
    #     num_classes=10)

    # chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
    #                                  "checkpoints/dblogger_141_epoch1370.pth")
    # _141_checkpoint = torch.load(chck_path, map_location="cpu")
    # model.load_state_dict(state_dict=_141_checkpoint["model_state_dict"])

    # model = CigtBayesianMultipath(
    #     run_id=run_id,
    #     model_definition="Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
    #     num_classes=10)
    #
    # chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
    #                                  "checkpoints/dblogger2_94_epoch1390.pth")
    # cbam_checkpoint = torch.load(chck_path, map_location="cpu")
    # model.load_state_dict(state_dict=cbam_checkpoint["model_state_dict"], strict=False)
    #
    # model.modelFilesRootPath = ResnetCigtConstants.model_file_root_path_tetam_tuna
    # explanation = model.get_explanation_string()
    # DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
    #
    # heavyweight_augmentation = transforms.Compose([
    #     transforms.Resize(model.imageSize),
    #     CutoutPIL(cutout_factor=0.5),
    #     RandAugment(),
    #     transforms.ToTensor(),
    # ])
    # lightweight_augmentation = transforms.Compose([
    #     transforms.Resize(model.imageSize),
    #     transforms.ToTensor(),
    # ])
    #
    # # Cifar 10 Dataset
    # kwargs = {'num_workers': 2, 'pin_memory': True}
    # train_loader_hard = torch.utils.data.DataLoader(
    #     datasets.CIFAR10('../data', train=True, transform=heavyweight_augmentation),
    #     batch_size=ResnetCigtConstants.batch_size, shuffle=False, **kwargs)
    #
    # train_loader_light = torch.utils.data.DataLoader(
    #     datasets.CIFAR10('../data', train=True, transform=lightweight_augmentation),
    #     batch_size=ResnetCigtConstants.batch_size, shuffle=False, **kwargs)
    #
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10('../data', train=False, transform=lightweight_augmentation),
    #     batch_size=ResnetCigtConstants.batch_size, shuffle=False, **kwargs)
    #
    # # model.validate(loader=test_loader, epoch=0, data_kind="test", temperature=0.1)
    # model.validate(loader=test_loader, epoch=0, data_kind="test", temperature=0.1)
    # model.validate(loader=train_loader_hard, data_kind="train", epoch=0, temperature=0.1)
    # model.validate(loader=train_loader_light, data_kind="train", epoch=0, temperature=0.1)
    # print("X")
    #
    # # model.fit_temperatures_with_respect_to_variances()
    # # model.validate(loader=test_loader, epoch=0, data_kind="test", temperature=0.1)
