import os

from randaugment import RandAugment
from torchvision import datasets
from torchvision.transforms import transforms

from auxillary.db_logger import DbLogger
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cigt_output_dataset import CigtOutputDataset
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
    test_cigt_output_dataset = CigtOutputDataset(configs=Cifar10ResnetCigtConfigs)
    test_cigt_output_dataset.load_from_file(file_path="test_cigt_dataset.sav")
    test_loader = torch.utils.data.DataLoader(test_cigt_output_dataset,
                                              batch_size=Cifar10ResnetCigtConfigs.batch_size, shuffle=False, **kwargs)

    train_cigt_output_dataset = CigtOutputDataset(configs=Cifar10ResnetCigtConfigs)
    train_cigt_output_dataset.load_from_file(file_path="train_cigt_dataset.sav")
    train_loader = torch.utils.data.DataLoader(train_cigt_output_dataset,
                                               batch_size=Cifar10ResnetCigtConfigs.batch_size, shuffle=True, **kwargs)

    DbLogger.log_db_path = DbLogger.paperspace

    print("Start!")
    model_mac = CigtIgGatherScatterImplementation(
        run_id=-1,
        model_definition="Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
        num_classes=10,
        configs=Cifar10ResnetCigtConfigs)
    model_mac.to(model_mac.device)
    model_mac.execute_forward_with_random_input()
    mac_counts_per_block = CigtIgHardRoutingX.calculate_mac(model=model_mac)
    model_mac = None
    run_id = DbLogger.get_run_id()

    model = CigtReinforcePreprocessedDatasets(
        configs=Cifar10ResnetCigtConfigs,
        model_definition="Reinforce Multipath CIGT",
        num_classes=10,
        run_id=run_id,
        model_mac_info=mac_counts_per_block,
        is_debug_mode=True,
        train_dataset=train_loader,
        test_dataset=test_loader)
    model.to(model.device)
    model.execute_forward_with_random_input()

    model.fit_policy_network(train_loader=train_loader, test_loader=test_loader)

    # kwargs = {'num_workers': 0, 'pin_memory': True}
    # heavyweight_augmentation = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     CutoutPIL(cutout_factor=0.5),
    #     RandAugment(),
    #     transforms.ToTensor(),
    # ])
    # lightweight_augmentation = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    # ])
    # train_loader_hard = torch.utils.data.DataLoader(
    #     datasets.CIFAR10('../data', download=True, train=True, transform=heavyweight_augmentation),
    #     batch_size=Cifar10ResnetCigtConfigs.batch_size, shuffle=False, **kwargs)
    # test_loader_light = torch.utils.data.DataLoader(
    #     datasets.CIFAR10('../data', download=True, train=False, transform=lightweight_augmentation),
    #     batch_size=Cifar10ResnetCigtConfigs.batch_size, shuffle=False, **kwargs)
    #
    # chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "checkpoints/cigtlogger2_75_epoch1575.pth")
    # # chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "checkpoints/dblogger_331_epoch11.pth")
    # data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "cigtlogger2_75_epoch1575")
    #
    # DbLogger.log_db_path = DbLogger.paperspace
    #
    # model_mac = CigtIgGatherScatterImplementation(
    #     run_id=-1,
    #     model_definition="Gather Scatter Cigt With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:4  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:3 - MultipleLogitsMultipleLosses - Wd:0.0006 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
    #     num_classes=10,
    #     configs=Cifar10ResnetCigtConfigs)
    # model_mac.to(model_mac.device)
    # model_mac.execute_forward_with_random_input()
    # mac_counts_per_block = CigtIgHardRoutingX.calculate_mac(model=model_mac)
    # model_mac = None
    #
    # run_id = DbLogger.get_run_id()
    # model = CigtReinforceV2(
    #     configs=Cifar10ResnetCigtConfigs,
    #     model_definition="Reinforce Multipath CIGT",
    #     num_classes=10,
    #     run_id=run_id,
    #     model_mac_info=mac_counts_per_block,
    #     is_debug_mode=True)
    # model.to(model.device)
    # model.execute_forward_with_random_input()
    # checkpoint = torch.load(chck_path, map_location=model.device)
    # load_result = model.load_state_dict(state_dict=checkpoint["model_state_dict"], strict=False)
    # for param_name in load_result.missing_keys:
    #     block_check1 = [param_name.startswith("policyNetworks.{0}".format(block_id))
    #                     or param_name.startswith("valueNetworks.{0}".format(block_id))
    #                     for block_id in range(len(model.pathCounts[1:]))]
    #     assert any(block_check1)
    #     block_check2 = ["block_{0}".format(block_id) in param_name for block_id in range(len(model.pathCounts[1:]))]
    #     assert any(block_check2)
    #
    # model.modelFilesRootPath = Cifar10ResnetCigtConfigs.model_file_root_path_paperspace
    # explanation = model.get_explanation_string()
    # DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
    #
    # # model.validate(loader=train_loader, data_kind="train", epoch=0, temperature=0.1)
    # # model.validate(loader=test_loader, data_kind="test", epoch=0, temperature=0.1)
    #
    # # train_dict = model.validate(loader=train_loader_hard, epoch=0, data_kind="train", temperature=1.0)
    # # test_dict = model.validate(loader=test_loader_light, epoch=0, data_kind="test", temperature=1.0)
    # model.fit_policy_network(train_loader=train_loader_hard, test_loader=test_loader_light)
