import torch
from randaugment import RandAugment
from torchvision import datasets
from torchvision.transforms import transforms

from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cutout_augmentation import CutoutPIL
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs

from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm

# random.seed(53)
# np.random.seed(61)

if __name__ == "__main__":
    print("X")
    # 5e-4,
    # 0.0005
    DbLogger.log_db_path = DbLogger.tetam_cigt_db
    # weight_decay = 5 * [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
    p_dropout = 10 * [0.3]
    p_dropout = sorted(p_dropout)
    param_grid = Utilities.get_cartesian_product(list_of_lists=[p_dropout])

    # Thick Baseline
    # CigtConstants.layer_config_list = [
    #     {"path_count": 1,
    #      "layer_structure": [{"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 5,
    #                           "use_max_pool": True, "use_batch_normalization": False},
    #                          {"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 5,
    #                           "use_max_pool": True, "use_batch_normalization": False},
    #                          {"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 1,
    #                           "use_max_pool": True, "use_batch_normalization": False},
    #                          {"layer_type": "flatten"},
    #                          {"layer_type": "fc", "dimension": 128, "use_dropout": True,
    #                           "use_batch_normalization": False},
    #                          {"layer_type": "fc", "dimension": 64, "use_dropout": True,
    #                           "use_batch_normalization": False}]}]

    FashionLenetCigtConfigs.backbone = "LeNet"
    FashionLenetCigtConfigs.input_dims = (1, 28, 28)
    # CIGT-[1,2,4]
    FashionLenetCigtConfigs.layer_config_list = [
        {"path_count": 1,
         "layer_structure": [{"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 5,
                              "use_max_pool": True, "use_batch_normalization": False}]},
        {"path_count": 2,
         "layer_structure": [{"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 5,
                              "use_max_pool": True, "use_batch_normalization": False}]},
        {"path_count": 4,
         "layer_structure": [{"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 1,
                              "use_max_pool": True, "use_batch_normalization": False},
                             {"layer_type": "flatten"},
                             {"layer_type": "fc", "dimension": 128, "use_dropout": True,
                              "use_batch_normalization": False},
                             {"layer_type": "fc", "dimension": 64, "use_dropout": True,
                              "use_batch_normalization": False}]}]
    # These are especially important for the LeNet-CIGT
    FashionLenetCigtConfigs.classification_wd = 0.0
    FashionLenetCigtConfigs.information_gain_balance_coeff_list = [2.5, 2.5]
    FashionLenetCigtConfigs.classification_drop_probability = 0.3
    FashionLenetCigtConfigs.apply_relu_dropout_to_decision_layer = False
    FashionLenetCigtConfigs.decision_drop_probability = 0.0
    FashionLenetCigtConfigs.decision_loss_coeff = 0.7
    FashionLenetCigtConfigs.decision_dimensions = [128, 128]
    FashionLenetCigtConfigs.softmax_decay_controller = StepWiseDecayAlgorithm(
        decay_name="Stepwise",
        initial_value=FashionLenetCigtConfigs.softmax_decay_initial,
        decay_coefficient=FashionLenetCigtConfigs.softmax_decay_coefficient,
        decay_period=FashionLenetCigtConfigs.softmax_decay_period,
        decay_min_limit=FashionLenetCigtConfigs.softmax_decay_min_limit)

    # The rest can be left like they are
    FashionLenetCigtConfigs.loss_calculation_kind = "MultipleLogitsMultipleLosses"
    FashionLenetCigtConfigs.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
    FashionLenetCigtConfigs.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
    FashionLenetCigtConfigs.number_of_cbam_layers_in_routing_layers = 0
    FashionLenetCigtConfigs.cbam_reduction_ratio = 4
    FashionLenetCigtConfigs.cbam_layer_input_reduction_ratio = 4
    FashionLenetCigtConfigs.apply_mask_to_batch_norm = False
    FashionLenetCigtConfigs.advanced_augmentation = True
    FashionLenetCigtConfigs.use_focal_loss = False
    FashionLenetCigtConfigs.focal_loss_gamma = 2.0
    FashionLenetCigtConfigs.batch_norm_type = "BatchNorm"
    FashionLenetCigtConfigs.data_parallelism = False

    kwargs = {'num_workers': 0, 'pin_memory': True}
    heavyweight_augmentation = transforms.Compose([
        # transforms.Resize((32, 32)),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor(),
    ])
    lightweight_augmentation = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', download=True, train=True, transform=lightweight_augmentation),
        batch_size=FashionLenetCigtConfigs.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', download=True, train=False, transform=lightweight_augmentation),
        batch_size=FashionLenetCigtConfigs.batch_size, shuffle=False, **kwargs)

    # chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
    #                          "checkpoints/dblogger2_94_epoch1390.pth")
    # data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "dblogger2_94_epoch1390_data")

    run_id = DbLogger.get_run_id()
    model = CigtIgGatherScatterImplementation(
        configs=FashionLenetCigtConfigs,
        run_id=run_id,
        model_definition="Gather Scatter LeNet Cigt - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
        num_classes=10)
    model.modelFilesRootPath = FashionLenetCigtConfigs.model_file_root_path_tetam

    explanation = model.get_explanation_string()
    DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
    # model.execute_forward_with_random_input()

    model.fit(train_loader=train_loader, test_loader=test_loader)


    # checkpoint = torch.load(chck_path, map_location="cpu")
    # model.load_state_dict(state_dict=checkpoint["model_state_dict"])

    # total_parameter_count = model.get_total_parameter_count()
    # mac_counts_per_block = CigtIgHardRoutingX.calculate_mac(model=model)
    print("X")
    # accuracy = model.validate(loader=test_loader_light, epoch=0, data_kind="test", temperature=0.1)

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
