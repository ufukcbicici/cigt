import numpy as np
import os
from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_constant_routing_weights import CigtConstantRoutingWeights
from cigt.cigt_ideal_routing import CigtIdealRouting
from cigt.cigt_ig_different_losses import CigtIgDifferentLosses
from cigt.cigt_ig_hard_routing import CigtIgHardRouting
from cigt.cigt_ig_hard_routing_with_random_batches import CigtIgHardRoutingWithRandomBatches
from cigt.cigt_ig_iterative_training import CigtIgIterativeTraining
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cigt_ig_soft_routing import CigtIgSoftRouting
from cigt.cigt_model import Cigt
from cigt.cigt_soft_routing import CigtSoftRouting
from cigt.cigt_soft_routing_with_balance import CigtSoftRoutingWithBalance
from cigt.cigt_variance_routing import CigtVarianceRouting
from cigt.resnet_cigt_constants import ResnetCigtConstants
import torch

from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm

if __name__ == "__main__":

    # 5e-4,
    # 0.0005
    DbLogger.log_db_path = DbLogger.hpc_db2
    # weight_decay = 5 * [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
    weight_decay = 5 * [0.0007]
    weight_decay = sorted(weight_decay)

    param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])

    for param_tpl in param_grid:
        ResnetCigtConstants.classification_wd = param_tpl[0]
        ResnetCigtConstants.information_gain_balance_coeff_list = [5.0, 5.0]
        ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
        ResnetCigtConstants.after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
        ResnetCigtConstants.warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
        ResnetCigtConstants.softmax_decay_controller = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
            initial_value=ResnetCigtConstants.softmax_decay_initial,
            decay_coefficient=ResnetCigtConstants.softmax_decay_coefficient,
            decay_period=ResnetCigtConstants.softmax_decay_period,
            decay_min_limit=ResnetCigtConstants.softmax_decay_min_limit)

        run_id = DbLogger.get_run_id()

        model = CigtIgHardRoutingX(
            run_id=run_id,
            model_definition="Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0007 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
            num_classes=10)
        model.modelFilesRootPath = ResnetCigtConstants.model_file_root_path_hpc
        explanation = model.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        # checkpoint_pth = os.path.join(os.path.split(os.path.abspath(__file__))[0], "random_cigtlogger2_29_epoch350.pth")
        # checkpoint = torch.load(checkpoint_pth, map_location="cpu")
        # model.load_state_dict(state_dict=checkpoint["model_state_dict"])

        # print("Before Reset")
        # for layer_id, block_end_module in enumerate(model.blockEndLayers):
        #     print("Block {0}".format(layer_id))
        #     for name, param in block_end_module.named_parameters():
        #         print("Param name:{0} Weight:{1}".format(name, np.linalg.norm(param.detach().numpy())))

        # for block_end_module in model.blockEndLayers:
        #     block_end_module.fc1.reset_parameters()
        #     block_end_module.fc2.reset_parameters()
        #     block_end_module.igBatchNorm.reset_parameters()

        # print("After Reset")
        # for layer_id, block_end_module in enumerate(model.blockEndLayers):
        #     print("Block {0}".format(layer_id))
        #     for name, param in block_end_module.named_parameters():
        #         print("Param name:{0} Weight:{1}".format(name, np.linalg.norm(param.detach().numpy())))

        model.fit()

        print("XXXXXXXXXXXXXXXX")

        # with tf.device("GPU"):
        #     run_id = DbLogger.get_run_id()
        #
        #     print("Entering ResnetCigt", flush=True)
        #     resnet_cigt = ResnetCigt(run_id=run_id, model_definition="Resnet-110 [2,4] Cigt Debug: What if info gain is always 0?")
        #
        #     explanation = resnet_cigt.get_explanation_string()
        #     DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
        #
        #     # training_accuracy, training_info_gain_list = resnet_cigt.evaluate(
        #     #     x=cifar10.testDataset, epoch_id=0, dataset_type="test")
        #
        #     print("Entering resnet_cigt.fit", flush=True)
        #     resnet_cigt.fit(x=cifar10.trainDataset,
        #                     validation_data=cifar10.testDataset,
        #                     epochs=ResnetCigtConstants.epoch_count)

    # ResnetCigt.create_default_config_json()
    # print("X")
