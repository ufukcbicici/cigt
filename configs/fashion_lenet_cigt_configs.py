from math import ceil, floor

# from auxillary.parameters import DiscreteParameter
# from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
# DONT TOUCH THIS!!!
base_batch_size = 125


def adjust_to_batch_size(original_value, target_batch_size):
    adjusted_value = int(original_value // (base_batch_size / target_batch_size))
    return adjusted_value


class FashionLenetCigtConfigs:
    # Standart Parameters
    backbone = "LeNet"
    input_dims = (1, 28, 28)
    class_count = 10
    batch_size = 1000
    warm_up_period = adjust_to_batch_size(original_value=25, target_batch_size=batch_size)
    epoch_count = adjust_to_batch_size(original_value=125, target_batch_size=batch_size)
    temperature_optimization_epoch_count = 1000
    data_parallelism = True
    classification_wd = 0.0
    decision_wd = 0.0
    softmax_decay_initial = 25.0
    softmax_decay_coefficient = 0.9999
    softmax_decay_period = 1
    softmax_decay_min_limit = 0.1
    softmax_decay_controllers = {}
    information_gain_balance_coeff_list = [1.0, 1.0]
    decision_drop_probability = 0.5
    classification_drop_probability = 0.0
    batch_norm_type = "InstanceNorm"
    apply_relu_dropout_to_decision_layer = False
    apply_mask_to_batch_norm = False
    number_of_cbam_layers_in_routing_layers = 3
    cbam_reduction_ratio = 4
    cbam_layer_input_reduction_ratio = 2
    start_moving_averages_from_zero = False
    boost_learning_rates_layer_wise = False
    multiple_ce_losses = False
    per_sample_entropy_balance = True
    advanced_augmentation = True
    use_focal_loss = True
    focal_loss_gamma = 2.0
    validation_period = adjust_to_batch_size(original_value=5, target_batch_size=batch_size)
    # assert batch_norm_type in {"StandardBatchNormalization",
    #                            "CigtBatchNormalization",
    #                            "CigtProbabilisticBatchNormalization"}
    bn_momentum = 0.9
    evaluation_period = adjust_to_batch_size(original_value=5, target_batch_size=batch_size)
    measurement_start = 11
    decision_dimensions = [128, 128]
    decision_average_pooling_strides = [2, 1]
    initial_lr = 0.01

    decision_loss_coeff = 1.0
    optimizer_type = "SGD"
    decision_non_linearity = "Softmax"
    z_sample_count = 1000
    save_model = True
    routing_strategy_name = "Approximate_Training"
    use_straight_through = True
    first_conv_kernel_size = 3
    first_conv_output_dim = 16
    first_conv_stride = 1
    learning_schedule = [
        (adjust_to_batch_size(original_value=32, target_batch_size=batch_size) + warm_up_period, 0.5),
        (adjust_to_batch_size(original_value=64, target_batch_size=batch_size) + warm_up_period, 0.25),
        (adjust_to_batch_size(original_value=86, target_batch_size=batch_size) + warm_up_period, 0.025)]
    hard_routing_algorithm_kind = "InformationGainRouting"
    after_warmup_routing_algorithm_kind = "InformationGainRoutingWithRandomization"
    routing_randomization_ratio = 0.5
    warmup_routing_algorithm_kind = "RandomRoutingButInformationGainOptimizationEnabled"
    loss_calculation_kind = "MultipleLogitsMultipleLosses"

    model_file_root_path_hpc = "/clusterusers/can.bicici@boun.edu.tr/cigt"
    model_file_root_path_asus = "C://Users//asus//Desktop//ConvAig//convnet-aig//checkpoints"
    model_file_root_path_tetam = "/cta/users/ucbicici/cigt"
    model_file_root_path_tetam_tuna = "/cta/users/hmeral/cigt"
    model_file_root_path_jr = "/"



    # Resnet110 Thick Baseline
    # layer_config_list = [
    #     {"path_count": 1,
    #      "layer_structure": [{"layer_count": 18, "feature_map_count": 16},
    #                          {"layer_count": 18, "feature_map_count": 32},
    #                          {"layer_count": 18, "feature_map_count": 64}]}]

    # Resnet110 Thin Baseline
    # layer_config_list = [
    #     {"path_count": 1,
    #      "layer_structure": [{"layer_count": 9, "feature_map_count": 16},
    #                          {"layer_count": 9, "feature_map_count": 12},
    #                          {"layer_count": 18, "feature_map_count": 16},
    #                          {"layer_count": 18, "feature_map_count": 16}]}]

    # [2, 4] Cigt
    router_layers_count = 3
    extra_routing_epochs = 100
    single_loss_epoch_count = 50
    outer_loop_count = 8

    random_batch_ratio = 0.5

    random_classification_loss_weight = 1.0
    layer_config_list = [
        {"path_count": 1,
         "layer_structure": [{"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 5,
                              "use_max_pool": True, "use_batch_normalization": False}]},
        {"path_count": 2,
         "layer_structure": [{"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 5,
                              "use_max_pool": True, "use_batch_normalization": False}]},
        {"path_count": 4,
         "layer_structure": [{"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 1,
                              "use_max_pool": True, "use_batch_normalization": False},
                             {"layer_type": "fc", "dimension": 128, "use_batch_normalization": False},
                             {"layer_type": "fc", "dimension": 64, "use_batch_normalization": False}]}]

    double_stride_layers = {18, 36}

    softmax_decay_controller = StepWiseDecayAlgorithm(
        decay_name="Stepwise",
        initial_value=softmax_decay_initial,
        decay_coefficient=softmax_decay_coefficient,
        decay_period=softmax_decay_period,
        decay_min_limit=softmax_decay_min_limit)

    # Knowledge Distillation Parameters
    ideal_routing_error_ratio = 0.05
    use_kd_for_routing = False
    kd_teacher_temperature = 6.0
    kd_loss_alpha = 0.5
