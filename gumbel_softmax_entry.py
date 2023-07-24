from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_gumbel_softmax_routing import CigtGumbelSoftmaxRouting
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs

from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm

if __name__ == "__main__":
    print("X")
    # 5e-4,
    # 0.0005
    DbLogger.log_db_path = DbLogger.jr_cigt
    # weight_decay = 5 * [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
    weight_decay = 10 * [0.0005]
    weight_decay = sorted(weight_decay)

    param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])

    for param_tpl in param_grid:
        FashionLenetCigtConfigs.layer_config_list = [
            {"path_count": 1,
             "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
            {"path_count": 2,
             "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
                                 {"layer_count": 18, "feature_map_count": 16}]},
            {"path_count": 4,
             "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
        FashionLenetCigtConfigs.classification_wd = param_tpl[0]
        FashionLenetCigtConfigs.information_gain_balance_coeff_list = [5.0, 5.0]
        FashionLenetCigtConfigs.loss_calculation_kind = "MultipleLogitsMultipleLosses"
        FashionLenetCigtConfigs.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
        FashionLenetCigtConfigs.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
        FashionLenetCigtConfigs.decision_drop_probability = 0.5
        FashionLenetCigtConfigs.number_of_cbam_layers_in_routing_layers = 6
        FashionLenetCigtConfigs.cbam_reduction_ratio = 4
        FashionLenetCigtConfigs.cbam_layer_input_reduction_ratio = 0
        FashionLenetCigtConfigs.apply_relu_dropout_to_decision_layer = False
        FashionLenetCigtConfigs.decision_dimensions = [128, 128]
        FashionLenetCigtConfigs.apply_mask_to_batch_norm = False
        FashionLenetCigtConfigs.advanced_augmentation = True
        FashionLenetCigtConfigs.use_focal_loss = False
        FashionLenetCigtConfigs.focal_loss_gamma = 2.0
        FashionLenetCigtConfigs.batch_norm_type = "BatchNorm"
        FashionLenetCigtConfigs.decision_non_linearity = "Softmax"
        FashionLenetCigtConfigs.use_straight_through = True
        FashionLenetCigtConfigs.z_sample_count = 1000
        # ResnetCigtConstants.use_kd_for_routing = False
        # ResnetCigtConstants.kd_teacher_temperature = 10.0
        # ResnetCigtConstants.kd_loss_alpha = 0.95

        FashionLenetCigtConfigs.softmax_decay_controller = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
            initial_value=FashionLenetCigtConfigs.softmax_decay_initial,
            decay_coefficient=FashionLenetCigtConfigs.softmax_decay_coefficient,
            decay_period=FashionLenetCigtConfigs.softmax_decay_period,
            decay_min_limit=FashionLenetCigtConfigs.softmax_decay_min_limit)

        run_id = DbLogger.get_run_id()

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

        model = CigtGumbelSoftmaxRouting(
            run_id=run_id,
            model_definition="Gather Scatter Cigt Gumbel Softmax With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
            num_classes=10)

        # model = CigtIdealRouting()

        model.modelFilesRootPath = FashionLenetCigtConfigs.model_file_root_path_jr
        explanation = model.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        model.fit()
