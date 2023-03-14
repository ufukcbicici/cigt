import numpy as np
from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_constant_routing_weights import CigtConstantRoutingWeights
from cigt.cigt_ideal_routing import CigtIdealRouting
from cigt.cigt_ig_different_losses import CigtIgDifferentLosses
from cigt.cigt_ig_hard_routing import CigtIgHardRouting
from cigt.cigt_ig_hard_routing_with_random_batches import CigtIgHardRoutingWithRandomBatches
from cigt.cigt_ig_iterative_training import CigtIgIterativeTraining
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
    weight_decay = 5 * [0.00085]
    weight_decay = sorted(weight_decay)

    param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])

    for param_tpl in param_grid:
        ResnetCigtConstants.classification_wd = param_tpl[0]
        ResnetCigtConstants.softmax_decay_controller = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
            initial_value=ResnetCigtConstants.softmax_decay_initial,
            decay_coefficient=ResnetCigtConstants.softmax_decay_coefficient,
            decay_period=ResnetCigtConstants.softmax_decay_period,
            decay_min_limit=ResnetCigtConstants.softmax_decay_min_limit)

        run_id = DbLogger.get_run_id()
        # model = CigtSoftRouting(run_id=run_id, model_definition="Resnet CIGT equal and constant routing variables.")
        # model = CigtSoftRoutingWithBalance(
        #     run_id=run_id,
        #     model_definition="Resnet Soft Routing - No MoE Layer - Per Sample Entropy Regularization - Linear Routing Transformation.")
        # model = CigtConstantRoutingWeights(run_id=run_id,
        #                                    model_definition="Resnet Soft Routing - Equal Probabilities.")

        # model = CigtIgDifferentLosses(run_id=run_id,
        #                               model_definition="Resnet Soft Routing - IG Routing - Different Losses.")

        # model = CigtIgIterativeTraining(run_id=run_id,
        #                                 model_definition="Resnet Soft Routing - IG Routing - Iterative Training.")
        # model = CigtIgSoftRouting(run_id=run_id, model_definition="Resnet Soft Routing - IG Routing - Only IG Training.")
        # model = CigtVarianceRouting(run_id=run_id, model_definition="Resnet Soft Routing - Variance Routing.")

        # model = CigtIgHardRoutingWithRandomBatches(
        #     run_id=run_id,
        #     model_definition="Resnet Hard Routing - 1,2,4. Random Routing Regularization. Batch Size 1024.")

        model = CigtIgHardRouting(
            run_id=run_id,
            model_definition="Resnet Hard Routing - 1,2,2. Batch Size 1024. - Classification Wd:0.00085")
        model.modelFilesRootPath = ResnetCigtConstants.model_file_root_path_hpc
        explanation = model.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        # checkpoint_pth = "C://Users//asus//Desktop//ConvAig//convnet-aig//cigt//dblogger2_45_epoch165.pth"
        # checkpoint = torch.load(checkpoint_pth, map_location="cpu")
        # model.load_state_dict(state_dict=checkpoint["model_state_dict"])

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
