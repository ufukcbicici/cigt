import os

from randaugment import RandAugment
from torchvision import datasets
from torchvision.transforms import transforms

from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cigt_output_dataset import CigtOutputDataset
from cigt.cigt_q_learning import CigtQLearning
from cigt.cigt_reinforce_multipath import CigtReinforceMultipath
from cigt.cigt_reinforce_preprocessed_datasets import CigtReinforcePreprocessedDatasets
from cigt.cigt_reinforce_v2 import CigtReinforceV2
from cigt.cutout_augmentation import CutoutPIL
from cigt.multipath_evaluator import MultipathEvaluator
from cigt.multipath_inference_bayesian import MultiplePathBayesianOptimizer
# from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
import torch

from cigt.multipath_inference_cross_entropy import MultipathInferenceCrossEntropy
from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
from load_trained_models.load_trained_fmnist_model import load_trained_fmnist_model


# random.seed(53)
# np.random.seed(61)

def get_parameters_histogram(parameters_used):
    parameters_used = sorted(parameters_used)
    detection_histogram = {}
    worked_ids = DbLogger.read_query(query=
                                     "SELECT RunId FROM run_meta_data WHERE Explanation LIKE \"%Q Learning CIGT%\"")
    worked_ids_str = "(" + ",".join([str(tpl[0]) for tpl in worked_ids]) + ")"
    completed_runs = DbLogger.read_query(query="SELECT RunId FROM run_kv_store "
                                               "WHERE Key == \"Training Status\" "
                                               "AND Value == \"Training Finished!!!\" "
                                               "AND RunID In {0}".format(worked_ids_str))
    parameters_str = "(" + ",".join(["'{0}'".format(str_) for str_ in parameters_used]) + ")"
    completed_ids_str = "(" + ",".join([str(tpl[0]) for tpl in completed_runs]) + ")"
    parameters_queried = DbLogger.read_query(query="SELECT RunId, Parameter, Value FROM run_parameters "
                                                   "WHERE Parameter In {0}"
                                                   "AND RunId In {1}".format(parameters_str, completed_ids_str))
    print("X")


if __name__ == "__main__":
    chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..",
                             "checkpoints/cigtlogger2_160_epoch145.pth")
    data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..",
                             "cigtlogger2_160_epoch145_data")
    DbLogger.log_db_path = DbLogger.jr_cigt

    FashionLenetCigtConfigs.batch_size = 1024
    FashionLenetCigtConfigs.policy_networks_cbam_layer_input_reduction_ratio = 2
    fmnist_model, mac_counts_per_block = load_trained_fmnist_model(checkpoint_path=chck_path)
    fmnist_model = None

    kwargs = {'num_workers': 0, 'pin_memory': True}
    test_cigt_output_dataset = CigtOutputDataset(
        input_reduction_factor=FashionLenetCigtConfigs.policy_networks_cbam_layer_input_reduction_ratio)
    test_cigt_output_dataset.load_from_file(file_path=os.path.join(data_path, "qnet_test_cigt_dataset.sav"))
    test_loader = torch.utils.data.DataLoader(test_cigt_output_dataset,
                                              batch_size=FashionLenetCigtConfigs.batch_size, shuffle=False, **kwargs)

    train_cigt_output_dataset = CigtOutputDataset(
        input_reduction_factor=FashionLenetCigtConfigs.policy_networks_cbam_layer_input_reduction_ratio)
    train_cigt_output_dataset.load_from_file(file_path=os.path.join(data_path, "qnet_train_cigt_dataset3.sav"))
    train_loader = torch.utils.data.DataLoader(train_cigt_output_dataset,
                                               batch_size=FashionLenetCigtConfigs.batch_size, shuffle=True, **kwargs)


    print("Start!")
    # mac_lambda_list = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15]
    # wd_list = [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01] * 5
    mac_lambda_list = [0.0]
    wd_list = [0.0]
    mac_lambda_list = sorted(mac_lambda_list)
    wd_list = sorted(wd_list)
    param_grid = Utilities.get_cartesian_product(list_of_lists=[mac_lambda_list, wd_list])
    FashionLenetCigtConfigs.policy_networks_evaluation_period = 1

    for params in param_grid:
        run_id = DbLogger.get_run_id()
        mac_lambda = params[0]
        wd_coeff = params[1]
        FashionLenetCigtConfigs.policy_networks_mac_lambda = mac_lambda
        FashionLenetCigtConfigs.policy_networks_wd = wd_coeff

        print("Running run_id:{0}".format(run_id))
        print("Writing into DB:{0}".format(DbLogger.log_db_path))
        print("Running with mac_lambda:{0}".format(FashionLenetCigtConfigs.policy_networks_mac_lambda))
        print("Running with wd_coeff:{0}".format(FashionLenetCigtConfigs.policy_networks_wd))

        model = CigtQLearning(
            configs=FashionLenetCigtConfigs,
            model_definition="Fashion MNIST Q Learning CIGT",
            num_classes=10,
            run_id=run_id,
            model_mac_info=mac_counts_per_block,
            is_debug_mode=False,
            precalculated_datasets_dict={"train_dataset": train_loader, "test_dataset": test_loader})
        model.to(model.device)

        model.modelFilesRootPath = FashionLenetCigtConfigs.model_file_root_path_jr
        explanation = model.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        model.execute_forward_with_random_input()
        model.fit_policy_network(train_loader=train_loader, test_loader=test_loader)

        kv_rows = [(run_id,
                    model.policyNetworkTotalNumOfEpochs,
                    "Training Status",
                    "Training Finished!!!")]
        DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore)

        # print("compare_q_net_input_calculation_types - Comparison with the test set.")
        # model.compare_q_net_input_calculation_types(dataset=test_loader)
        # print("compare_q_net_input_calculation_types - Comparison with the training set.")
        # model.compare_q_net_input_calculation_types(dataset=train_loader)

        # print("Comparison with the test set.")
        # model.compare_trajectory_evaluation_methods(dataset=test_loader, repeat_count=1000)
        # print("Comparison with the training set.")
        # model.compare_trajectory_evaluation_methods(dataset=train_loader, repeat_count=1000)

        # print("Comparison of optimal q table calculation: Test set")
        # model.compare_q_table_calculation_types(dataset=test_loader)
        # print("Comparison of optimal q table calculation: Training set")
        # model.compare_q_table_calculation_types(dataset=train_loader)

        # ig_accuracy, ig_mac, ig_time = \
        #     model.validate_with_single_action_trajectory(loader=test_loader, action_trajectory=(0, 0))
        # print("Ig Accuracy:{0} Ig Mac:{1} Ig Mean Validation Time:{2}".format(
        #     ig_accuracy, ig_mac, ig_time))
        #
        # print("Test set run")
        # expected_accuracy_test, expected_mac_test, expected_time_test = \
        #     model.validate_with_expectation(loader=test_loader)
        # print("Test Accuracy:{0} Test Mac:{1} Test Mean Validation Time:{2}".format(
        #     expected_accuracy_test, expected_mac_test, expected_time_test))
        #
        # print("Training set run")
        # expected_accuracy_training, expected_mac_training, expected_time_training = \
        #     model.validate_with_expectation(loader=train_loader)
        # print("Training Accuracy:{0} Training Mac:{1} Training Mean Validation Time:{2}".format(
        #     expected_accuracy_training, expected_mac_training, expected_time_training))
        #
        # print("Successfully finished!")

        # model.execute_forward_with_random_input()
        # model.fit_policy_network(train_loader=train_loader, test_loader=test_loader)
        #
        # model.modelFilesRootPath = Cifar10ResnetCigtConfigs.model_file_root_path_jr
        # explanation = model.get_explanation_string()
        # DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
        #
        # model.fit_policy_network(train_loader=train_loader, test_loader=test_loader)
