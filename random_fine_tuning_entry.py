import os
import torch

from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_hard_routing import CigtIgHardRouting
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs

from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm

if __name__ == "__main__":
    # 5e-4,
    # 0.0005
    DbLogger.log_db_path = DbLogger.tetam_tuna_cigt_db2
    # weight_decay = 5 * [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
    weight_decay = [0.0005]
    weight_decay = sorted(weight_decay)

    param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])

    for param_tpl in param_grid:
        FashionLenetCigtConfigs.epoch_count = 1400
        FashionLenetCigtConfigs.initial_lr = 0.1
        # ResnetCigtConstants.learning_schedule = [(300, 0.1)]
        FashionLenetCigtConfigs.learning_schedule = [(600, 0.1), (1000, 0.01)]
        FashionLenetCigtConfigs.optimizer_type = "Adam"
        FashionLenetCigtConfigs.classification_wd = param_tpl[0]
        FashionLenetCigtConfigs.softmax_decay_initial = 0.1
        FashionLenetCigtConfigs.advanced_augmentation = False
        FashionLenetCigtConfigs.evaluation_period = 1
        FashionLenetCigtConfigs.softmax_decay_controller = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
            initial_value=FashionLenetCigtConfigs.softmax_decay_initial,
            decay_coefficient=FashionLenetCigtConfigs.softmax_decay_coefficient,
            decay_period=FashionLenetCigtConfigs.softmax_decay_period,
            decay_min_limit=FashionLenetCigtConfigs.softmax_decay_min_limit)

        run_id = DbLogger.get_run_id()
        trained_model = CigtIgHardRouting(
            run_id=run_id,
            model_definition="Resnet 1,2,4 - Random routing fine tuning")
        trained_model.modelFilesRootPath = FashionLenetCigtConfigs.model_file_root_path_tetam_tuna

        checkpoint_pth = os.path.join(os.path.split(os.path.abspath(__file__))[0], "cigtlogger_14_epoch1180.pth")
        checkpoint = torch.load(checkpoint_pth, map_location="cpu")
        trained_model.load_state_dict(state_dict=checkpoint["model_state_dict"])

        explanation = trained_model.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        # trained_model.fit()
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
