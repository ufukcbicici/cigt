import numpy as np
import os

from torchvision import datasets

from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_constant_routing_weights import CigtConstantRoutingWeights
from cigt.cigt_ideal_routing import CigtIdealRouting
from cigt.cigt_ig_different_losses import CigtIgDifferentLosses
from cigt.cigt_gumbel_softmax_routing import CigtGumbelSoftmaxRouting
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_hard_routing import CigtIgHardRouting
from cigt.cigt_ig_hard_routing_with_random_batches import CigtIgHardRoutingWithRandomBatches
from cigt.cigt_ig_iterative_training import CigtIgIterativeTraining
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cigt_ig_soft_routing import CigtIgSoftRouting
from cigt.cigt_ig_with_knowledge_distillation import CigtIgWithKnowledgeDistillation
from cigt.cigt_masked_routing import CigtMaskedRouting
from cigt.cigt_model import Cigt
from cigt.cigt_soft_routing import CigtSoftRouting
from cigt.cigt_soft_routing_with_balance import CigtSoftRoutingWithBalance
from cigt.cigt_variance_routing import CigtVarianceRouting
from cigt.cigt_constants import CigtConstants
import torch

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
        CigtConstants.layer_config_list = [
            {"path_count": 1,
             "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
            {"path_count": 2,
             "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
                                 {"layer_count": 18, "feature_map_count": 16}]},
            {"path_count": 4,
             "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
        CigtConstants.classification_wd = param_tpl[0]
        CigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
        CigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
        CigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
        CigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
        CigtConstants.decision_drop_probability = 0.5
        CigtConstants.number_of_cbam_layers_in_routing_layers = 6
        CigtConstants.cbam_reduction_ratio = 4
        CigtConstants.cbam_layer_input_reduction_ratio = 0
        CigtConstants.apply_relu_dropout_to_decision_layer = False
        CigtConstants.decision_dimensions = [128, 128]
        CigtConstants.apply_mask_to_batch_norm = False
        CigtConstants.advanced_augmentation = True
        CigtConstants.use_focal_loss = False
        CigtConstants.focal_loss_gamma = 2.0
        CigtConstants.batch_norm_type = "BatchNorm"
        CigtConstants.decision_non_linearity = "Softmax"
        CigtConstants.use_straight_through = True
        CigtConstants.z_sample_count = 1000
        # ResnetCigtConstants.use_kd_for_routing = False
        # ResnetCigtConstants.kd_teacher_temperature = 10.0
        # ResnetCigtConstants.kd_loss_alpha = 0.95

        CigtConstants.softmax_decay_controller = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
            initial_value=CigtConstants.softmax_decay_initial,
            decay_coefficient=CigtConstants.softmax_decay_coefficient,
            decay_period=CigtConstants.softmax_decay_period,
            decay_min_limit=CigtConstants.softmax_decay_min_limit)

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

        model.modelFilesRootPath = CigtConstants.model_file_root_path_jr
        explanation = model.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        model.fit()