import os
import torch
from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_hard_routing import CigtIgHardRouting
from cigt.resnet_cigt_constants import ResnetCigtConstants

from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm

if __name__ == "__main__":

    # 5e-4,
    # 0.0005
    DbLogger.log_db_path = DbLogger.home_asus
    # weight_decay = 5 * [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
    weight_decay = [0.0005]
    weight_decay = sorted(weight_decay)

    param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])

    for param_tpl in param_grid:
        ResnetCigtConstants.epoch_count = 350
        ResnetCigtConstants.initial_lr = 0.001
        ResnetCigtConstants.learning_schedule = [(200, 0.1)]
        ResnetCigtConstants.optimizer_type = "SGD"
        ResnetCigtConstants.classification_wd = param_tpl[0]
        ResnetCigtConstants.softmax_decay_controller = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
            initial_value=ResnetCigtConstants.softmax_decay_initial,
            decay_coefficient=ResnetCigtConstants.softmax_decay_coefficient,
            decay_period=ResnetCigtConstants.softmax_decay_period,
            decay_min_limit=ResnetCigtConstants.softmax_decay_min_limit)

        run_id = DbLogger.get_run_id()
        trained_model = CigtIgHardRouting(
            run_id=run_id,
            model_definition="Resnet 1,2,4 - Random routing fine tuning")
        trained_model.modelFilesRootPath = ResnetCigtConstants.model_file_root_path_tetam_tuna

        checkpoint_pth = os.path.join(os.path.split(os.path.abspath(__file__))[0], "cigtlogger_14_epoch1180.pth")
        checkpoint = torch.load(checkpoint_pth, map_location="cpu")
        trained_model.load_state_dict(state_dict=checkpoint["model_state_dict"])

        explanation = trained_model.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        trained_model.random_fine_tuning()

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
