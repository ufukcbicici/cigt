import torch
import os
from randaugment import RandAugment
from torchvision import transforms, datasets

from auxillary.bayesian_optimizer import BayesianOptimizer
from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cutout_augmentation import CutoutPIL
from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
from configs.fashion_lenet_cigt_configs import adjust_to_batch_size


class FashionMnistLenetCigtBayesianOptimizer(BayesianOptimizer):
    def __init__(self, init_points, n_iter):
        super().__init__(init_points, n_iter)
        self.optimization_bounds_continuous = {
            "classification_dropout_probability": (0.0, 1.0),
            "information_gain_balance_coefficient": (1.0, 10.0),
            "decision_loss_coefficient": (0.01, 1.0),
            "lr_initial_rate": (0.0, 0.05),
            "temperature_decay_rate": (0.9998, 0.99995),
            "random_routing_ratio": (0.0, 1.0)
        }

        # enable_information_gain_during_warm_up = True
        # enable_strict_routing_randomization = False
        # routing_randomization_ratio = 0.5
        # warm_up_kind = "RandomRouting"

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
                                  "use_max_pool": True, "use_batch_normalization": True}]},
            {"path_count": 2,
             "layer_structure": [{"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 5,
                                  "use_max_pool": True, "use_batch_normalization": True}]},
            {"path_count": 4,
             "layer_structure": [{"layer_type": "conv", "feature_map_count": 32, "strides": 1, "kernel_size": 1,
                                  "use_max_pool": True, "use_batch_normalization": True},
                                 {"layer_type": "flatten"},
                                 {"layer_type": "fc", "dimension": 128, "use_dropout": True,
                                  "use_batch_normalization": True},
                                 {"layer_type": "fc", "dimension": 64, "use_dropout": True,
                                  "use_batch_normalization": True}]}]

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

        FashionLenetCigtConfigs.enable_information_gain_during_warm_up = True
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

        kwargs = {'num_workers': 0, 'pin_memory': True}
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

        model_definition = "Fashion MNIST LeNet Bayesian Search enable_information_gain_during_warm_up = {0} - " \
                           "enable_strict_routing_randomization = {1} - warm_up_kind = {2}".format(
            FashionLenetCigtConfigs.enable_information_gain_during_warm_up,
            FashionLenetCigtConfigs.enable_strict_routing_randomization,
            FashionLenetCigtConfigs.warm_up_kind
        )

        run_id = DbLogger.get_run_id()
        model = CigtIgGatherScatterImplementation(
            configs=FashionLenetCigtConfigs,
            run_id=run_id,
            model_definition=model_definition,
            num_classes=10)
        model.modelFilesRootPath = FashionLenetCigtConfigs.model_file_root_path_hpc_docker

        explanation = model.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

        best_performance = model.fit(train_loader=train_loader, test_loader=test_loader)
        return best_performance


# --SELECT RunID FROM run_meta_data WHERE Explanation LIKE "%Fashion MNIST LeNet Bayesian Search enable_information_gain_during_warm_up = False - enable_strict_routing_randomization = False - warm_up_kind = FullRouting%";


if __name__ == "__main__":
    DbLogger.log_db_path = DbLogger.hpc_docker3
    bayesian_optimizer = FashionMnistLenetCigtBayesianOptimizer(init_points=50, n_iter=150)
    bayesian_optimizer.fit(log_file_root_path=os.path.split(os.path.abspath(__file__))[0],
                           log_file_name="TFF_fashion_lenet_0")
