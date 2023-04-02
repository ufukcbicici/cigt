import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import torchvision.datasets as datasets
import time
import numpy as np
import shutil
from tqdm import tqdm
from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_soft_routing import CigtIgSoftRouting

from auxillary.db_logger import DbLogger
from cigt.cigt_ig_hard_routing import CigtIgHardRouting

device = "cpu"
temperature = 0.1


def execute_routing(model_, data_loader, data_kind, routing_matrices, data_file_path_):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_c = AverageMeter()
    losses_t = AverageMeter()
    losses_t_layer_wise = [AverageMeter() for _ in range(len(model_.pathCounts) - 1)]
    accuracy_avg = AverageMeter()
    list_of_labels = []
    list_of_routing_probability_matrices = []
    for _ in range(len(model_.pathCounts) - 1):
        list_of_routing_probability_matrices.append([])
    list_of_last_features_complete = []
    for _ in range(model_.pathCounts[-1]):
        list_of_last_features_complete.append([])

    model_.eval()
    for i, (input_, target) in tqdm(enumerate(data_loader)):
        actual_batch_size = input_.shape[0]
        time_begin = time.time()
        if routing_matrices is None:
            model_.enforcedRouting = False
        else:
            model_.enforcedRouting = True
            model_.enforcedRoutingMatrices = [mat[0:actual_batch_size, :] for mat in routing_matrices]

        with torch.no_grad():
            input_var = torch.autograd.Variable(input_).to(device)
            target_var = torch.autograd.Variable(target).to(device)
            batch_size = input_var.size(0)

            # Cigt moe output, information gain losses
            list_of_logits, routing_matrices_hard, \
            routing_matrices_soft, list_of_last_features = model_(
                input_var, target_var, temperature)
            classification_loss, batch_accuracy = \
                model_.calculate_classification_loss_and_accuracy(
                    list_of_logits,
                    routing_matrices_hard,
                    target_var)
            information_gain_losses = model_.calculate_information_gain_losses(
                routing_matrices=routing_matrices_soft, labels=target_var,
                balance_coefficient_list=model_.informationGainBalanceCoeffList)
            total_routing_loss = 0.0
            for t_loss in information_gain_losses:
                total_routing_loss += t_loss
            total_routing_loss = -1.0 * model_.decisionLossCoeff * total_routing_loss
            total_loss = classification_loss + total_routing_loss

            # print("len(list_of_logits)={0}".format(len(list_of_logits)))
            # print("multipleCeLosses:{0}".format(self.multipleCeLosses))
            time_end = time.time()

            list_of_labels.append(target_var.cpu().numpy())
            for idx_, matr_ in enumerate(routing_matrices_soft[1:]):
                list_of_routing_probability_matrices[idx_].append(matr_.detach().cpu().numpy())

            for idx_, feat_arr in enumerate(list_of_last_features):
                list_of_last_features_complete[idx_].append(feat_arr.detach().cpu().numpy())

            # measure accuracy and record loss
            losses.update(total_loss.detach().cpu().numpy().item(), 1)
            losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
            accuracy_avg.update(batch_accuracy, batch_size)
            batch_time.update((time_end - time_begin), 1)
            losses_t.update(total_routing_loss.detach().cpu().numpy().item(), 1)
            for lid in range(len(model_.pathCounts) - 1):
                losses_t_layer_wise[lid].update(information_gain_losses[lid].detach().cpu().numpy().item(), 1)
    kv_rows = []
    list_of_labels = np.concatenate(list_of_labels, axis=0)
    for idx_ in range(len(list_of_routing_probability_matrices)):
        list_of_routing_probability_matrices[idx_] = np.concatenate(
            list_of_routing_probability_matrices[idx_], axis=0)

    for idx_ in range(len(list_of_last_features_complete)):
        list_of_last_features_complete[idx_] = np.concatenate(
            list_of_last_features_complete[idx_], axis=0)

    model_.calculate_branch_statistics(
        run_id=model_.runId,
        iteration=0,
        dataset_type=data_kind,
        labels=list_of_labels,
        routing_probability_matrices=list_of_routing_probability_matrices,
        write_to_db=True)

    print("total_loss:{0}".format(losses.avg))
    print("accuracy_avg:{0}".format(accuracy_avg.avg))
    print("batch_time:{0}".format(batch_time.avg))
    print("classification_loss:{0}".format(losses_c.avg))
    print("routing_loss:{0}".format(losses_t.avg))
    for lid in range(len(model_.pathCounts) - 1):
        print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))

    if data_file_path_ is not None:
        data_dict = {
            "last_layer_features": list_of_last_features_complete,
            "list_of_labels": list_of_labels,
            "routing_matrices": list_of_routing_probability_matrices,
        }
        print(list_of_labels[0:100])
        Utilities.pickle_save_to_file(file_content=data_dict, path=data_file_path_)


def execute_enforced_routing(model_, data_loader, data_kind, data_root_path_):
    # All path configurations
    list_of_path_choices = []
    for path_count in model_.pathCounts[1:]:
        list_of_path_choices.append([i_ for i_ in range(path_count)])
    route_combinations = Utilities.get_cartesian_product(list_of_lists=list_of_path_choices)
    for selected_routes in route_combinations:
        print("Executing route selection:{0}".format(selected_routes))
        data_file_path = os.path.join(data_root_path_,
                                      "{0}_{1}_data.sav".format(data_kind, selected_routes))
        if os.path.isfile(data_file_path):
            continue
        routing_matrices = []
        for layer_id, route_id in enumerate(selected_routes):
            routing_matrix = torch.zeros(size=(data_loader.batch_size, model_.pathCounts[layer_id + 1]),
                                         dtype=torch.float32)
            routing_matrix[:, route_id] = 1.0
            routing_matrices.append(routing_matrix)

        execute_routing(model_=model_, data_kind=data_kind, data_loader=data_loader,
                        routing_matrices=routing_matrices,
                        data_file_path_=data_file_path)

        print("X")


def record_model_outputs(model_, pretrained_model_path):
    # Get the checkpoint name
    checkpoint_name = os.path.split(pretrained_model_path)[1].split(".")[0]

    # Cifar 10 Dataset
    kwargs = {'num_workers': 2, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
        batch_size=1024, shuffle=False, **kwargs)
    train_loader_test_time_augmentation = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform_test),
        batch_size=1024, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=1024, shuffle=False, **kwargs)

    model_outputs_root_directory_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], checkpoint_name)
    if not os.path.isdir(model_outputs_root_directory_path):
        os.mkdir(model_outputs_root_directory_path)

    # Information gain - training
    # data_file_path = os.path.join(model_outputs_root_directory_path, "{0}_data.sav".format("train_ig"))
    # if not os.path.isfile(data_file_path):
    #     execute_routing(model_=model, data_kind="train", data_loader=train_loader,
    #                     routing_matrices=None,
    #                     data_file_path_=data_file_path)
    # Information gain - training - with test time augmentation
    data_file_path = os.path.join(model_outputs_root_directory_path, "{0}_data.sav".format(
        "train_ig_test_time_augmentation"))
    if not os.path.isfile(data_file_path):
        execute_routing(model_=model_, data_kind="train", data_loader=train_loader_test_time_augmentation,
                        routing_matrices=None,
                        data_file_path_=data_file_path)
    # Information gain - test
    data_file_path = os.path.join(model_outputs_root_directory_path, "{0}_data.sav".format("test_ig"))
    if not os.path.isfile(data_file_path):
        execute_routing(model_=model_, data_kind="test", data_loader=test_loader,
                        routing_matrices=None,
                        data_file_path_=data_file_path)

    execute_enforced_routing(model_=model_, data_loader=train_loader_test_time_augmentation,
                             data_kind="train",
                             data_root_path_=model_outputs_root_directory_path)
    print("X")


def load_ig_routing_outputs(pretrained_model_path, is_train):
    checkpoint_name = os.path.split(pretrained_model_path)[1].split(".")[0]
    model_outputs_root_directory_path = os.path.join(os.path.split(os.path.abspath(
        __file__))[0], checkpoint_name)
    assert os.path.isdir(model_outputs_root_directory_path)

    # Load regular IG routing results
    if is_train:
        data_file_path = os.path.join(model_outputs_root_directory_path, "{0}_data.sav".format(
            "train_ig_test_time_augmentation"))
    else:
        data_file_path = os.path.join(model_outputs_root_directory_path, "{0}_data.sav".format(
            "test_ig"))
    ig_routing_results = Utilities.pickle_load_from_file(path=data_file_path)
    ig_routes_arr = []
    for routing_matrix in ig_routing_results["routing_matrices"]:
        ig_routes_arr.append(np.argmax(routing_matrix, axis=1))
    ig_routes_arr = np.stack(ig_routes_arr, axis=1)
    routes_to_samples_map = {}
    for sample_id in range(ig_routes_arr.shape[0]):
        route_tpl = tuple(ig_routes_arr[sample_id])
        if route_tpl not in routes_to_samples_map:
            routes_to_samples_map[route_tpl] = []
        routes_to_samples_map[route_tpl].append(sample_id)

    return ig_routing_results, ig_routes_arr, routes_to_samples_map


def compare_model_outputs_for_consistency(model_, pretrained_model_path):
    checkpoint_name = os.path.split(pretrained_model_path)[1].split(".")[0]
    model_outputs_root_directory_path = os.path.join(os.path.split(os.path.abspath(
        __file__))[0], checkpoint_name)
    assert os.path.isdir(model_outputs_root_directory_path)
    ig_routing_results, ig_routes_arr, routes_to_samples_map = load_ig_routing_outputs(
        pretrained_model_path=pretrained_model_path, is_train=True)

    # Load all path configurations
    list_of_path_choices = []
    for path_count in model_.pathCounts[1:]:
        list_of_path_choices.append([i_ for i_ in range(path_count)])
    route_combinations = Utilities.get_cartesian_product(list_of_lists=list_of_path_choices)

    # enforced_data = {}
    for selected_routes in route_combinations:
        print("Executing route selection:{0}".format(selected_routes))
        data_file_path = os.path.join(model_outputs_root_directory_path,
                                      "{0}_{1}_data.sav".format("train", selected_routes))
        enforced_data = Utilities.pickle_load_from_file(path=data_file_path)
        if selected_routes not in routes_to_samples_map:
            continue
        samples_for_route = routes_to_samples_map[selected_routes]
        assert np.array_equal(ig_routing_results["list_of_labels"][samples_for_route],
                              enforced_data["list_of_labels"][samples_for_route])
        ig_features = ig_routing_results["last_layer_features"][selected_routes[-1]][samples_for_route]
        enforced_features = enforced_data["last_layer_features"][selected_routes[-1]][samples_for_route]
        assert np.allclose(ig_features, enforced_features)
    print("Model outputs are consistent!!!")


# Algorithm 1:
# Let each train sample to go its corresponding leaf.
# Train classifiers on the routed data.
# Let each test sample to go its corresponding leaf.
# Test accuracies. Hope: Ensemble effect can reduce overfitting?
def algorithm_1(model_, pretrained_model_path):
    checkpoint_name = os.path.split(pretrained_model_path)[1].split(".")[0]
    model_outputs_root_directory_path = os.path.join(os.path.split(os.path.abspath(
        __file__))[0], checkpoint_name)
    assert os.path.isdir(model_outputs_root_directory_path)
    ig_routing_results_train, ig_routes_arr_train, routes_to_samples_map_train = load_ig_routing_outputs(
        pretrained_model_path=pretrained_model_path, is_train=True)
    ig_routing_results_test, ig_routes_arr_test, routes_to_samples_map_est = load_ig_routing_outputs(
        pretrained_model_path=pretrained_model_path, is_train=False)

    avg_pool = torch.nn.AvgPool2d(kernel_size=8)
    flatten = torch.nn.Flatten()

    ig_final_features_train = []
    for idx, features in enumerate(ig_routing_results_train["last_layer_features"]):
        pooled_features = avg_pool(torch.from_numpy(features))
        flattened_features = flatten(pooled_features)
        ig_final_features_train.append(flattened_features)

    ig_final_features_test = []
    for idx, features in enumerate(ig_routing_results_test["last_layer_features"]):
        pooled_features = avg_pool(torch.from_numpy(features))
        flattened_features = flatten(pooled_features)
        ig_final_features_test.append(flattened_features)

    for block_id in range(len(ig_final_features_train)):
        print("Training Block {0}".format(block_id))
        # Get all samples that reach into this final block
        # train_block_X = ig_final_features_train[idx][ig_routes_arr_train[:, -1] == idx]
        # train_block_y = ig_routing_results_train["list_of_labels"][ig_routes_arr_train[:, -1] == idx]
        train_block_X = ig_final_features_train[block_id]
        train_block_y = ig_routing_results_train["list_of_labels"]
        train_block_X = train_block_X.numpy()

        test_block_X = ig_final_features_test[block_id][ig_routes_arr_test[:, -1] == block_id]
        test_block_y = ig_routing_results_test["list_of_labels"][ig_routes_arr_test[:, -1] == block_id]
        test_block_X = test_block_X.numpy()

        # MLP
        alpha_list = [0.00005 * i__ for i__ in range(200)]
        param_grid = \
            {
                "pca__n_components": [None],
                "mlp__hidden_layer_sizes": [(16, )],
                "mlp__activation": ["relu"],
                "mlp__solver": ["adam"],
                # "mlp__learning_rate": ["adaptive"],
                "mlp__alpha": alpha_list,
                "mlp__max_iter": [10000],
                "mlp__early_stopping": [True],
                "mlp__n_iter_no_change": [100]
            }
        standard_scaler = StandardScaler()
        pca = PCA()
        mlp = MLPClassifier()
        pipe = Pipeline(steps=[("scaler", standard_scaler),
                               ('pca', pca),
                               ('mlp', mlp)])
        search = GridSearchCV(pipe, param_grid, n_jobs=8, cv=10, verbose=10,
                              scoring=["accuracy", "f1_weighted", "f1_micro", "f1_macro",
                                       "balanced_accuracy"],
                              refit="accuracy")
        search.fit(train_block_X, train_block_y)

        y_pred = {"training": search.best_estimator_.predict(train_block_X),
                  "test": search.best_estimator_.predict(test_block_X)}
        print("*************Training*************")
        print(classification_report(y_pred=y_pred["training"], y_true=train_block_y))
        print("*************Test*************")
        print(classification_report(y_pred=y_pred["test"], y_true=test_block_y))

        # Save the search results
        model_path = os.path.join(model_outputs_root_directory_path, "block_{0}_model.sav".format(block_id))
        Utilities.pickle_save_to_file(path=model_path, file_content=search)
        print("X")


if __name__ == "__main__":
    DbLogger.log_db_path = DbLogger.hpc_db
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    run_id = DbLogger.get_run_id()
    trained_model = CigtIgHardRouting(
        run_id=run_id,
        model_definition="Resnet Hard Routing - Only Routing - 1.2.4. Batch Size 1024.")

    explanation = trained_model.get_explanation_string()
    DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

    checkpoint_pth = os.path.join(os.path.split(os.path.abspath(__file__))[0], "cigtlogger_14_epoch1180.pth")
    checkpoint = torch.load(checkpoint_pth, map_location="cpu")
    trained_model.load_state_dict(state_dict=checkpoint["model_state_dict"])

    torch.manual_seed(1)
    best_performance = 0.0

    record_model_outputs(model_=trained_model, pretrained_model_path=checkpoint_pth)

    compare_model_outputs_for_consistency(model_=trained_model,
                                          pretrained_model_path=checkpoint_pth)

    algorithm_1(model_=trained_model, pretrained_model_path=checkpoint_pth)

    # execute_enforced_routing(model_=model, train_data=train_loader, test_data=val_loader)

    # """Perform validation on the validation set"""
    #
    # # Temperature of Gumble Softmax
    # # We simply keep it fixed
    # temperature = 0.1
    # device = "cpu"
    # # switch to evaluate mode
    # model.eval()
    #
    # for data_kind, data_loader in [("training", train_loader), ("test", val_loader)]:
    #     data_file_path = os.path.join(
    #         os.path.split(os.path.abspath(__file__))[0], "{0}_data.sav".format(data_kind))
    #     if not os.path.isfile(data_file_path):
    #         batch_time = AverageMeter()
    #         losses = AverageMeter()
    #         losses_c = AverageMeter()
    #         losses_t = AverageMeter()
    #         losses_t_layer_wise = [AverageMeter() for _ in range(len(model.pathCounts) - 1)]
    #         accuracy_avg = AverageMeter()
    #         list_of_labels = []
    #         list_of_routing_probability_matrices = []
    #         for _ in range(len(model.pathCounts) - 1):
    #             list_of_routing_probability_matrices.append([])
    #         list_of_last_features_complete = []
    #         for _ in range(model.pathCounts[-1]):
    #             list_of_last_features_complete.append([])
    #
    #         for i, (input_, target) in tqdm(enumerate(data_loader)):
    #             time_begin = time.time()
    #             with torch.no_grad():
    #                 input_var = torch.autograd.Variable(input_).to(device)
    #                 target_var = torch.autograd.Variable(target).to(device)
    #                 batch_size = input_var.size(0)
    #
    #                 # Cigt moe output, information gain losses
    #                 list_of_logits, routing_matrices_hard, routing_matrices_soft, list_of_last_features = model(
    #                     input_var, target_var, temperature)
    #                 classification_loss, batch_accuracy = model.calculate_classification_loss_and_accuracy(
    #                     list_of_logits,
    #                     routing_matrices_hard,
    #                     target_var)
    #                 information_gain_losses = model.calculate_information_gain_losses(
    #                     routing_matrices=routing_matrices_soft, labels=target_var,
    #                     balance_coefficient_list=model.informationGainBalanceCoeffList)
    #                 total_routing_loss = 0.0
    #                 for t_loss in information_gain_losses:
    #                     total_routing_loss += t_loss
    #                 total_routing_loss = -1.0 * model.decisionLossCoeff * total_routing_loss
    #                 total_loss = classification_loss + total_routing_loss
    #
    #                 # print("len(list_of_logits)={0}".format(len(list_of_logits)))
    #                 # print("multipleCeLosses:{0}".format(self.multipleCeLosses))
    #                 time_end = time.time()
    #
    #                 list_of_labels.append(target_var.cpu().numpy())
    #                 for idx_, matr_ in enumerate(routing_matrices_soft[1:]):
    #                     list_of_routing_probability_matrices[idx_].append(matr_.detach().cpu().numpy())
    #
    #                 for idx_, feat_arr in enumerate(list_of_last_features):
    #                     list_of_last_features_complete[idx_].append(feat_arr.detach().cpu().numpy())
    #
    #                 # measure accuracy and record loss
    #                 losses.update(total_loss.detach().cpu().numpy().item(), 1)
    #                 losses_c.update(classification_loss.detach().cpu().numpy().item(), 1)
    #                 accuracy_avg.update(batch_accuracy, batch_size)
    #                 batch_time.update((time_end - time_begin), 1)
    #                 losses_t.update(total_routing_loss.detach().cpu().numpy().item(), 1)
    #                 for lid in range(len(model.pathCounts) - 1):
    #                     losses_t_layer_wise[lid].update(information_gain_losses[lid].detach().cpu().numpy().item(), 1)
    #         kv_rows = []
    #         list_of_labels = np.concatenate(list_of_labels, axis=0)
    #         for idx_ in range(len(list_of_routing_probability_matrices)):
    #             list_of_routing_probability_matrices[idx_] = np.concatenate(
    #                 list_of_routing_probability_matrices[idx_], axis=0)
    #
    #         for idx_ in range(len(list_of_last_features_complete)):
    #             list_of_last_features_complete[idx_] = np.concatenate(
    #                 list_of_last_features_complete[idx_], axis=0)
    #
    #         model.calculate_branch_statistics(
    #             run_id=model.runId,
    #             iteration=0,
    #             dataset_type=data_kind,
    #             labels=list_of_labels,
    #             routing_probability_matrices=list_of_routing_probability_matrices,
    #             write_to_db=True)
    #
    #         print("total_loss:{0}".format(losses.avg))
    #         print("accuracy_avg:{0}".format(accuracy_avg.avg))
    #         print("batch_time:{0}".format(batch_time.avg))
    #         print("classification_loss:{0}".format(losses_c.avg))
    #         print("routing_loss:{0}".format(losses_t.avg))
    #         for lid in range(len(model.pathCounts) - 1):
    #             print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))
    #
    #         data_dict = {
    #             "last_layer_features": list_of_last_features_complete,
    #             "routing_matrices": list_of_routing_probability_matrices
    #         }
    #         Utilities.pickle_save_to_file(file_content=data_dict, path=data_file_path)
    #     print("X")
    #
    # training_files = Utilities.pickle_load_from_file(
    #     path=os.path.join(
    #         os.path.split(os.path.abspath(__file__))[0],
    #         "training_data.sav".format(data_kind)))
    #
    # test_files = Utilities.pickle_load_from_file(
    #     path=os.path.join(
    #         os.path.split(os.path.abspath(__file__))[0],
    #         "test_data.sav".format(data_kind)))
