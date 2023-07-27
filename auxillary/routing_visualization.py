import torch
import numpy as np
import cv2
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


def convert_string_to_dict(dict_as_str):
    kv_strings = dict_as_str.split(",")
    d_ = {}
    for kv in kv_strings:
        k = kv.split(":")[0].replace(" ", "")
        v = kv.split(":")[1].replace(" ", "")
        label = int(k)
        label_freq = int(v)
        d_[label] = label_freq
    return d_


def plot_mode_images_v3(dataset_, class_names, node_name,
                        image_width, image_height, label_distribution, mode_threshold,
                        sample_count_per_class):
    extent_size = 4
    column_margin = 32
    _w = image_width
    _h = image_height

    # Calculate the mode label distribution
    label_distribution_sorted = sorted([(k, v) for k, v in label_distribution.items()],
                                       key=lambda tpl: tpl[1], reverse=True)
    total_sample_count = sum([tpl[1] for tpl in label_distribution_sorted])
    label_distribution_freqs = [(k, v / total_sample_count) for k, v in label_distribution_sorted]
    cum_probability = 0.0
    mode_labels = []
    for k, v in label_distribution_freqs:
        cum_probability += v
        mode_labels.append((k, v))
        if cum_probability > mode_threshold:
            break

    # Empty canvas sizes
    img_width = (sample_count_per_class + 3) * extent_size + (sample_count_per_class + 2) * _w
    img_height = ((len(mode_labels) + 1) * extent_size + len(mode_labels) * _h) + _h

    canvas = np.ones(shape=(img_height, img_width, 3), dtype=np.uint8)
    canvas[:] = 255

    max_probability_mass = mode_labels[0][1]
    curr_class_idx = 0

    text_size = cv2.getTextSize(node_name, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, thickness=0)
    text_width = text_size[0][0]
    cv2.putText(canvas, node_name, (img_width // 2 - text_width // 2, _h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color=(0, 0, 0), thickness=0, lineType=cv2.LINE_AA)

    top = _h + extent_size
    for tpl in mode_labels:
        col_left = extent_size
        label_id = tpl[0]
        probability_mass = tpl[1]
        indices = np.random.choice(dataset_[label_id].shape[0], sample_count_per_class, replace=False)
        for col_idx in range(sample_count_per_class + 2):
            if col_idx == 0:
                cv2.putText(canvas, "{0}".format(class_names[label_id]), (col_left, top + _h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color=(0, 0, 0), thickness=0, lineType=cv2.LINE_AA)
            # Draw sample images
            elif 1 <= col_idx <= sample_count_per_class:
                img = dataset_[label_id][indices[col_idx - 1]]
                img = np.transpose(img, axes=(1, 2, 0))
                resized_img = cv2.resize(img, (_w, _h), interpolation=cv2.INTER_AREA)
                # Convert to BGR
                if resized_img.shape[-1] == 1:
                    resized_img = np.stack([resized_img, resized_img, resized_img], axis=0)
                b = np.copy(resized_img[:, :, 2])
                r = np.copy(resized_img[:, :, 0])
                resized_img[:, :, 0] = b
                resized_img[:, :, 2] = r
                canvas[top: top + _h, col_left: col_left + _w, :] = resized_img
            # Draw distribution info
            else:
                cv2.putText(canvas, "%{0:.2f}".format(probability_mass * 100.0), (col_left, int(top + _h * 0.35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 0), thickness=0, lineType=cv2.LINE_AA)
                top_left = (col_left, int(top + _h * 0.75))
                relative_prob = probability_mass
                bottom_right = (col_left + int(relative_prob * _w + 0.5), int(top + _h * 0.95))
                cv2.rectangle(canvas, top_left, bottom_right, (255, 0, 0), -1)

            col_left += (_w + extent_size)
        top += (_h + extent_size)
    cv2.imwrite("{0}_histogram.png".format(node_name.lower()), canvas)


def plot_leaf_distribution(dataset_name, block_id, sample_distribution_matrix, class_names):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    node_ids = [i_ for i_ in range(sample_distribution_matrix.shape[1])]
    label_ids = [class_names[i_] for i_ in range(sample_distribution_matrix.shape[0])]

    ax.set_xticks(np.arange(sample_distribution_matrix.shape[1]))
    ax.set_yticks(np.arange(sample_distribution_matrix.shape[0]))
    ax.set_xticklabels(node_ids)
    ax.set_yticklabels(label_ids)
    ax.tick_params(labelsize=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(label_ids)):
        for j in range(len(node_ids)):
            text = ax.text(j, i, int(sample_distribution_matrix[i, j]),
                           ha="center", va="center", color="w", fontsize=15)

    ax.set_title("Class distribution on block {0}".format(block_id), fontsize=18)
    fig.tight_layout()
    plt.show()
    # fig.savefig("Leaf_distribution_Ids_({0}-{1}).png".format(idx, idx + max_class_count))
    fig.savefig("{0}_block_{1}_sample_distribution.png".format(dataset_name, block_id))


def print_node(test_loader, node_dicts, dataset_name, class_names):
    dataset = {}
    for x_, y_ in test_loader:
        x_np = (x_.numpy() * 255).astype(np.uint8)
        y_np = y_.numpy()
        for i in range(x_np.shape[0]):
            if y_np[i] not in dataset:
                dataset[y_np[i]] = []
            dataset[y_np[i]].append(x_np[i])
    total_sample_count = 0
    for idx in dataset.keys():
        dataset[idx] = np.stack(dataset[idx], axis=0)
        total_sample_count += dataset[idx].shape[0]

    for block_id, dicts in enumerate(node_dicts):
        # Sample distributions in units in the current block
        sample_distribution_matrix = np.zeros(shape=(len(class_names), len(dicts)), dtype=np.int32)
        # Individual histograms for every unit
        for unit_id, dict_as_str in enumerate(dicts):
            lbl_distribution_dict = convert_string_to_dict(dict_as_str=dict_as_str)
            node_name = "{0} BLOCK ({1},{2})".format(dataset_name, block_id + 1, unit_id)
            plot_mode_images_v3(dataset_=dataset, class_names=class_names,
                                image_width=32, image_height=32, label_distribution=lbl_distribution_dict,
                                mode_threshold=0.85, sample_count_per_class=10, node_name=node_name)

            for label_id, freq in lbl_distribution_dict.items():
                sample_distribution_matrix[label_id, unit_id] += int(total_sample_count * freq)
        plot_leaf_distribution(dataset_name=dataset_name,
                               sample_distribution_matrix=sample_distribution_matrix,
                               block_id=block_id,
                               class_names=class_names)


if __name__ == "__main__":
    print("X")

    cifar10_names = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }
    transform_test = transforms.Compose([transforms.ToTensor()])
    kwargs = {'num_workers': 0, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../data', train=False, transform=transform_test),
        batch_size=1024, shuffle=False, **kwargs)

    # CIFAR 10
    cifar10_block_1_0 = "5: 976, 6: 974, 4: 972, 7: 966, 3: 920, 2: 806, 0: 24, 8: 10, 9: 8, 1: 1"
    cifar10_block_1_1 = "1: 999, 9: 992, 8: 990, 0: 976, 2: 194, 3: 80, 7: 34, 4: 28, 6: 26, 5: 24"

    cifar10_block_2_0 = "7: 970, 5: 956, 3: 884, 2: 48, 4: 41, 0: 11, 6: 8, 9: 2, 8: 1, 1: 1"
    cifar10_block_2_1 = "6: 986, 4: 946, 2: 140, 3: 74, 5: 30, 7: 15, 0: 11, 8: 2, 9: 1"
    cifar10_block_2_2 = "8: 977, 0: 968, 2: 810, 3: 33, 9: 13, 5: 12, 7: 12, 4: 12, 6: 6, 1: 5"
    cifar10_block_2_3 = "1: 994, 9: 984, 8: 20, 0: 10, 3: 9, 7: 3, 2: 2, 5: 2, 4: 1"

    distributions = [[cifar10_block_1_0, cifar10_block_1_1],
                     [cifar10_block_2_0, cifar10_block_2_1, cifar10_block_2_2, cifar10_block_2_3]]
    print_node(test_loader=test_loader, dataset_name="CIFAR 10", class_names=cifar10_names,
               node_dicts=distributions)

    # 5e-4,
    # 0.0005
    # DbLogger.log_db_path = DbLogger.jr_cigt
    # # weight_decay = 5 * [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
    # weight_decay = 10 * [0.0005]
    # weight_decay = sorted(weight_decay)
    #
    # param_grid = Utilities.get_cartesian_product(list_of_lists=[weight_decay])
    #
    # for param_tpl in param_grid:
    #     Cifar10ResnetCigtConfigs.layer_config_list = [
    #         {"path_count": 1,
    #          "layer_structure": [{"layer_count": 9, "feature_map_count": 16}]},
    #         {"path_count": 2,
    #          "layer_structure": [{"layer_count": 9, "feature_map_count": 12},
    #                              {"layer_count": 18, "feature_map_count": 16}]},
    #         {"path_count": 4,
    #          "layer_structure": [{"layer_count": 18, "feature_map_count": 16}]}]
    #     Cifar10ResnetCigtConfigs.classification_wd = param_tpl[0]
    #     Cifar10ResnetCigtConfigs.information_gain_balance_coeff_list = [5.0, 5.0]
    #     Cifar10ResnetCigtConfigs.loss_calculation_kind = "MultipleLogitsMultipleLosses"
    #     Cifar10ResnetCigtConfigs.enable_information_gain_during_warm_up = False
    #     Cifar10ResnetCigtConfigs.enable_strict_routing_randomization = True
    #     Cifar10ResnetCigtConfigs.routing_randomization_ratio = 0.5
    #     Cifar10ResnetCigtConfigs.warm_up_kind = "FullRouting"
    #     Cifar10ResnetCigtConfigs.decision_drop_probability = 0.5
    #     Cifar10ResnetCigtConfigs.number_of_cbam_layers_in_routing_layers = 6
    #     Cifar10ResnetCigtConfigs.cbam_reduction_ratio = 4
    #     Cifar10ResnetCigtConfigs.cbam_layer_input_reduction_ratio = 0
    #     Cifar10ResnetCigtConfigs.apply_relu_dropout_to_decision_layer = False
    #     Cifar10ResnetCigtConfigs.decision_dimensions = [128, 128]
    #     Cifar10ResnetCigtConfigs.apply_mask_to_batch_norm = False
    #     Cifar10ResnetCigtConfigs.advanced_augmentation = True
    #     Cifar10ResnetCigtConfigs.use_focal_loss = False
    #     Cifar10ResnetCigtConfigs.focal_loss_gamma = 2.0
    #     Cifar10ResnetCigtConfigs.batch_norm_type = "BatchNorm"
    #     Cifar10ResnetCigtConfigs.data_parallelism = False
    #     # ResnetCigtConstants.use_kd_for_routing = False
    #     # ResnetCigtConstants.kd_teacher_temperature = 10.0
    #     # ResnetCigtConstants.kd_loss_alpha = 0.95
    #
    #     Cifar10ResnetCigtConfigs.softmax_decay_controller = StepWiseDecayAlgorithm(
    #         decay_name="Stepwise",
    #         initial_value=Cifar10ResnetCigtConfigs.softmax_decay_initial,
    #         decay_coefficient=Cifar10ResnetCigtConfigs.softmax_decay_coefficient,
    #         decay_period=Cifar10ResnetCigtConfigs.softmax_decay_period,
    #         decay_min_limit=Cifar10ResnetCigtConfigs.softmax_decay_min_limit)
    #
    #     run_id = DbLogger.get_run_id()
    #     normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #     if not Cifar10ResnetCigtConfigs.advanced_augmentation:
    #         print("WILL BE USING ONLY CROP AND HORIZONTAL FLIP AUGMENTATION")
    #         transform_train = transforms.Compose([
    #             transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
    #         transform_test = transforms.Compose([
    #             transforms.ToTensor(),
    #             normalize
    #         ])
    #     else:
    #         print("WILL BE USING RANDOM AUGMENTATION")
    #         transform_train = transforms.Compose([
    #             transforms.Resize(Cifar10ResnetCigtConfigs.input_dims[1:]),
    #             CutoutPIL(cutout_factor=0.5),
    #             RandAugment(),
    #             transforms.ToTensor(),
    #         ])
    #         transform_test = transforms.Compose([
    #             transforms.Resize(Cifar10ResnetCigtConfigs.input_dims[1:]),
    #             transforms.ToTensor(),
    #         ])
    #
    #     # Cifar 10 Dataset
    #     kwargs = {'num_workers': 0, 'pin_memory': True}
    #     train_loader = torch.utils.data.DataLoader(
    #         datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
    #         batch_size=Cifar10ResnetCigtConfigs.batch_size, shuffle=True, **kwargs)
    #     test_loader = torch.utils.data.DataLoader(
    #         datasets.CIFAR10('../data', train=False, transform=transform_test),
    #         batch_size=Cifar10ResnetCigtConfigs.batch_size, shuffle=False, **kwargs)
    #
    #
    #     # kwargs = {'num_workers': 2, 'pin_memory': True}
    #     # test_loader = torch.utils.data.DataLoader(
    #     #     datasets.CIFAR10('../data', train=False, transform=teacher_model.transformTest),
    #     #     batch_size=ResnetCigtConstants.batch_size, shuffle=False, **kwargs)
    #
    #
    #     # teacher_model.validate(loader=test_loader, epoch=0, data_kind="test")
    #
    #     # ResnetCigtConstants.loss_calculation_kind = "SingleLogitSingleLoss"
    #     # teacher_model = CigtIdealRouting(
    #     #     run_id=run_id,
    #     #     model_definition="Cigt Ideal Routing - [1,2,4] - SingleLogitSingleLoss - Wd:0.0005",
    #     #     num_classes=10,
    #     #     class_to_route_mappings=
    #     #     [[(5, 6, 4, 7, 3, 2), (1, 8, 9, 0)],
    #     #      [(6, 0), (1, 8, 9), (5, 7, 3), (4, 2)]])
    #
    #     # Cifar 10 Dataset
    #     # kwargs = {'num_workers': 2, 'pin_memory': True}
    #     # test_loader = torch.utils.data.DataLoader(
    #     #     datasets.CIFAR10('../data', train=False, transform=teacher_model.transformTest),
    #     #     batch_size=ResnetCigtConstants.batch_size, shuffle=False, **kwargs)
    #     # teacher_model.validate(loader=test_loader, epoch=0, data_kind="test")
    #
    #     # ResnetCigtConstants.loss_calculation_kind = "MultipleLogitsMultipleLosses"
    #     # model = CigtIgWithKnowledgeDistillation(
    #     #     run_id=run_id,
    #     #     model_definition="KD Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Wd:0.0005 - use_kd_for_routing = False - kd_teacher_temperature = 10.0 - kd_loss_alpha = 0.95",
    #     #     num_classes=10, teacher_model=teacher_model)
    #     # model.modelFilesRootPath = ResnetCigtConstants.model_file_root_path_hpc
    #     # explanation = model.get_explanation_string()
    #     # DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
    #
    #     # model = CigtMaskedRouting(
    #     #     run_id=run_id,
    #     #     model_definition="Masked Cigt - [1,2,4] - [5.0, 5.0] - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 "
    #     #                      "Epoch Warm "
    #     #                      "up with: RandomRoutingButInformationGainOptimizationEnabled - "
    #     #                      "InformationGainRoutingWithRandomization - apply_mask_to_batch_norm: True",
    #     #     num_classes=10)
    #
    #     # chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],
    #     #                                  "checkpoints/dblogger_141_epoch1370.pth")
    #     # _141_checkpoint = torch.load(chck_path, map_location="cpu")
    #     # model.load_state_dict(state_dict=_141_checkpoint["model_state_dict"])
    #
    #     model = CigtIgGatherScatterImplementation(
    #         run_id=run_id,
    #         model_definition="Vanilla With CBAM Routers With Random Augmentation - cbam_layer_input_reduction_ratio:0  - [1,2,4] - [5.0, 5.0] - number_of_cbam_layers_in_routing_layers:6 - MultipleLogitsMultipleLosses - Wd:0.0005 - 350 Epoch Warm up with: RandomRoutingButInformationGainOptimizationEnabled - InformationGainRoutingWithRandomization",
    #         num_classes=10,
    #         configs=Cifar10ResnetCigtConfigs)
    #
    #     # model = CigtIdealRouting()
    #
    #     model.modelFilesRootPath = Cifar10ResnetCigtConfigs.model_file_root_path_jr
    #     explanation = model.get_explanation_string()
    #     DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
    #
    #     model.fit(train_loader=train_loader, test_loader=test_loader)
