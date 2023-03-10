from math import ceil, floor

# from auxillary.parameters import DiscreteParameter
# from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm


class ResnetCigtConstants:
    # Standart Parameters
    input_dims = (3, 32, 32)
    class_count = 10
    batch_size = 1024
    epoch_count = 1400
    classification_wd = 0.0
    decision_wd = 0.0
    softmax_decay_initial = 25.0
    softmax_decay_coefficient = 0.9999
    softmax_decay_period = 1
    softmax_decay_min_limit = 0.1
    softmax_decay_controllers = {}
    information_gain_balance_coeff_list = [1.0, 5.0]
    decision_drop_probability = 0.0
    classification_drop_probability = 0.0
    batch_norm_type = "StandardBatchNormalization"
    apply_mask_to_batch_norm = False
    start_moving_averages_from_zero = False
    boost_learning_rates_layer_wise = False
    multiple_ce_losses = False
    per_sample_entropy_balance = True
    advanced_augmentation = False
    validation_period = 1
    # assert batch_norm_type in {"StandardBatchNormalization",
    #                            "CigtBatchNormalization",
    #                            "CigtProbabilisticBatchNormalization"}
    bn_momentum = 0.9
    evaluation_period = 10
    measurement_start = 11
    decision_dimensions = [128, 128]
    decision_average_pooling_strides = [4, 2]
    initial_lr = 0.1
    iteration_count_per_epoch = floor(50000 / batch_size) + 1 if 50000 % batch_size != 0 else 50000 / batch_size

    decision_loss_coeff = 1.0
    optimizer_type = "SGD"
    decision_non_linearity = "Softmax"
    save_model = False
    warm_up_period = 0
    routing_strategy_name = "Approximate_Training"
    use_straight_through = True
    first_conv_kernel_size = 3
    first_conv_output_dim = 16
    first_conv_stride = 1
    learning_schedule = [(600 + warm_up_period, 0.1), (1000 + warm_up_period, 0.01)]

    model_file_root_path_hpc = "/clusterusers/can.bicici@boun.edu.tr/cigt"
    model_file_root_path_tetam = "/cta/users/ucbicici/cigt"
    model_file_root_path_tetam_tuna = "/cta/users/hmeral/cigt"

    # Thick Baseline
    # resnet_config_list = [
    #     {"path_count": 1,
    #      "layer_structure": [{"layer_count": 18, "feature_map_count": 16},
    #                          {"layer_count": 18, "feature_map_count": 32},
    #                          {"layer_count": 18, "feature_map_count": 64}]}]

    # Thin Baseline
    # resnet_config_list = [
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

    resnet_config_list = [
        {"path_count": 1,
         "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
        {"path_count": 2,
         "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
                             {"layer_count": 18, "feature_map_count": 16}]},
        {"path_count": 4,
         "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]

    double_stride_layers = {18, 36}

    softmax_decay_controller = StepWiseDecayAlgorithm(
        decay_name="Stepwise",
        initial_value=softmax_decay_initial,
        decay_coefficient=softmax_decay_coefficient,
        decay_period=softmax_decay_period,
        decay_min_limit=softmax_decay_min_limit)
