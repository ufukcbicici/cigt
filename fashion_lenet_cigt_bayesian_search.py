import torch
from randaugment import RandAugment
from torchvision import transforms, datasets

from auxillary.bayesian_optimizer import BayesianOptimizer
from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from cigt.cutout_augmentation import CutoutPIL
from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
from configs.fashion_lenet_cigt_configs import adjust_to_batch_size


class FashionMnistLenetCigtBayesianOptimizer(BayesianOptimizer):
    def __init__(self, xi, init_points, n_iter):
        super().__init__(xi, init_points, n_iter)
        self.optimization_bounds_continuous = {
            "classification_dropout_probability": (0.0, 1.0),
            "information_gain_balance_coefficient": (1.0, 10.0),
            "decision_loss_coefficient": (0.01, 1.0),
            "lr_initial_rate": (0.0, 0.05),
            "temperature_decay_rate": (0.9998, 0.99995),
            "random_routing_ratio": (0.0, 1.0)
        }

    def cost_function(self, **kwargs):
        # lr_initial_rate,
        # hyperbolic_exponent):
        X = kwargs["classification_dropout_probability"]
        Y = kwargs["information_gain_balance_coefficient"]
        Z = kwargs["decision_loss_coefficient"]
        W = kwargs["lr_initial_rate"]
        U = kwargs["temperature_decay_rate"]
        V = kwargs["random_routing_ratio"]

        print("classification_dropout_probability={0}".format(kwargs["classification_dropout_probability"]))
        print("information_gain_balance_coefficient={0}".format(kwargs["information_gain_balance_coefficient"]))
        print("decision_loss_coefficient={0}".format(kwargs["decision_loss_coefficient"]))
        print("lr_initial_rate={0}".format(kwargs["lr_initial_rate"]))
        print("temperature_decay_rate={0}".format(kwargs["temperature_decay_rate"]))
        print("random_routing_ratio={0}".format(kwargs["random_routing_ratio"]))

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

        FashionLenetCigtConfigs.classification_wd = 0.0
        FashionLenetCigtConfigs.apply_relu_dropout_to_decision_layer = False
        FashionLenetCigtConfigs.decision_drop_probability = 0.0
        FashionLenetCigtConfigs.decision_dimensions = [128, 128]

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

        kwargs = {'num_workers': 2, 'pin_memory': True}
        heavyweight_augmentation = transforms.Compose([
            # transforms.Resize((32, 32)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
        ])
        lightweight_augmentation = transforms.Compose([
            # transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', download=True, train=True, transform=lightweight_augmentation),
            batch_size=FashionLenetCigtConfigs.batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', download=True, train=False, transform=lightweight_augmentation),
            batch_size=FashionLenetCigtConfigs.batch_size, shuffle=False, **kwargs)









































        # FashionNetConstants.softmax_decay_initial = 25.0
        # FashionNetConstants.softmax_decay_coefficient = U
        # FashionNetConstants.softmax_decay_period = 1
        # FashionNetConstants.softmax_decay_min_limit = 0.01
        #
        # fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
        # softmax_decay_controller = StepWiseDecayAlgorithm(
        #     decay_name="Stepwise",
        #     initial_value=FashionNetConstants.softmax_decay_initial,
        #     decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
        #     decay_period=FashionNetConstants.softmax_decay_period,
        #     decay_min_limit=FashionNetConstants.softmax_decay_min_limit)
        #
        # learning_rate_calculator = DiscreteParameter(name="lr_calculator",
        #                                              value=W,
        #                                              schedule=[(15000 + 12000, (1.0 / 2.0) * W),
        #                                                        (30000 + 12000, (1.0 / 4.0) * W),
        #                                                        (40000 + 12000, (1.0 / 40.0) * W)])
        # print(learning_rate_calculator)
        #
        # with tf.device("GPU"):
        #     run_id = DbLogger.get_run_id()
        #     fashion_cigt = LenetCigt(batch_size=125,
        #                              input_dims=(28, 28, 1),
        #                              filter_counts=[32, 64, 128],
        #                              kernel_sizes=[5, 5, 1],
        #                              hidden_layers=[512, 256],
        #                              decision_drop_probability=0.0,
        #                              classification_drop_probability=X,
        #                              decision_wd=0.0,
        #                              classification_wd=0.0,
        #                              decision_dimensions=[128, 128],
        #                              class_count=10,
        #                              information_gain_balance_coeff=Y,
        #                              softmax_decay_controller=softmax_decay_controller,
        #                              learning_rate_schedule=learning_rate_calculator,
        #                              decision_loss_coeff=Z,
        #                              path_counts=[2, 4],
        #                              bn_momentum=0.9,
        #                              warm_up_period=25,
        #                              routing_strategy_name="Full_Training",
        #                              run_id=run_id,
        #                              evaluation_period=10,
        #                              measurement_start=25,
        #                              use_straight_through=True,
        #                              optimizer_type="SGD",
        #                              decision_non_linearity="Softmax",
        #                              save_model=True,
        #                              model_definition="Lenet CIGT - Gumbel Softmax with E[Z] based routing - "
        #                                               "Softmax and Straight Through Bayesian Optimization - "
        #                                               "Optimizing Lr Initial Rate and Temperature Decay Rate")
        #
        #     explanation = fashion_cigt.get_explanation_string()
        #     DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
        #     score = fashion_cigt.fit(x=fashion_mnist.trainDataTf, validation_data=fashion_mnist.testDataTf,
        #                              epochs=FashionNetConstants.epoch_count)
        #
        # return score
