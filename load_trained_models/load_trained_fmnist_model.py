import os

# from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
import torch
from randaugment import RandAugment
from torchvision import datasets
from torchvision.transforms import transforms

from auxillary.db_logger import DbLogger
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cigt_output_dataset import CigtOutputDataset
from cigt.cutout_augmentation import CutoutPIL
from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from configs.cifar10_resnet_cigt_configs import Cifar10ResnetCigtConfigs
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs


def load_trained_fmnist_model(checkpoint_path, run_id=-1):
    X = 0.2
    Y = 5.0
    Z = 1.0
    W = 0.001
    U = 0.999
    V = 0.5

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
    FashionLenetCigtConfigs.classification_drop_probability = X
    FashionLenetCigtConfigs.information_gain_balance_coeff_list = [Y] * (
            len(FashionLenetCigtConfigs.layer_config_list) - 1)
    FashionLenetCigtConfigs.decision_loss_coeff = Z
    FashionLenetCigtConfigs.initial_lr = W
    FashionLenetCigtConfigs.softmax_decay_initial = 25.0
    FashionLenetCigtConfigs.softmax_decay_coefficient = U
    FashionLenetCigtConfigs.softmax_decay_period = 1
    FashionLenetCigtConfigs.softmax_decay_min_limit = 0.01
    FashionLenetCigtConfigs.softmax_decay_controller = StepWiseDecayAlgorithm(
        decay_name="Stepwise",
        initial_value=FashionLenetCigtConfigs.softmax_decay_initial,
        decay_coefficient=FashionLenetCigtConfigs.softmax_decay_coefficient,
        decay_period=FashionLenetCigtConfigs.softmax_decay_period,
        decay_min_limit=FashionLenetCigtConfigs.softmax_decay_min_limit)
    FashionLenetCigtConfigs.routing_randomization_ratio = V

    FashionLenetCigtConfigs.classification_wd = 0.0
    FashionLenetCigtConfigs.apply_relu_dropout_to_decision_layer = False
    FashionLenetCigtConfigs.decision_drop_probability = 0.0
    FashionLenetCigtConfigs.decision_dimensions = [128, 128]

    FashionLenetCigtConfigs.enable_information_gain_during_warm_up = False
    FashionLenetCigtConfigs.enable_strict_routing_randomization = False
    FashionLenetCigtConfigs.warm_up_kind = "FullRouting"

    # The rest can be left like they are
    FashionLenetCigtConfigs.loss_calculation_kind = "MultipleLogitsMultipleLosses"
    FashionLenetCigtConfigs.number_of_cbam_layers_in_routing_layers = 0
    FashionLenetCigtConfigs.cbam_reduction_ratio = 4
    FashionLenetCigtConfigs.cbam_layer_input_reduction_ratio = 4
    FashionLenetCigtConfigs.apply_mask_to_batch_norm = False
    FashionLenetCigtConfigs.advanced_augmentation = False
    FashionLenetCigtConfigs.use_focal_loss = False
    FashionLenetCigtConfigs.focal_loss_gamma = 2.0
    FashionLenetCigtConfigs.batch_norm_type = "BatchNorm"
    FashionLenetCigtConfigs.data_parallelism = False

    model = CigtIgGatherScatterImplementation(
        configs=FashionLenetCigtConfigs,
        run_id=run_id,
        model_definition="Fashion MNIST LeNet Bayesian Search enable_information_gain_during_warm_up = False - enable_strict_routing_randomization = False - warm_up_kind = FullRouting",
        num_classes=10)
    model.to(model.device)
    model.execute_forward_with_random_input()

    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model_load_results = model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    param_count = model.get_total_parameter_count()

    model_mac = CigtIgGatherScatterImplementation(
        run_id=-1,
        model_definition="Fashion MNIST LeNet Bayesian Search enable_information_gain_during_warm_up = False - enable_strict_routing_randomization = False - warm_up_kind = FullRouting",
        num_classes=10,
        configs=FashionLenetCigtConfigs)
    model_mac.to(model_mac.device)
    model_mac.execute_forward_with_random_input()

    model_load_results_mac = model_mac.load_state_dict(state_dict=checkpoint["model_state_dict"])
    mac_counts_per_block = CigtIgHardRoutingX.calculate_mac(model=model_mac)

    return model, mac_counts_per_block
