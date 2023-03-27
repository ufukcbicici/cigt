import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
import time
import numpy as np
from tqdm import tqdm
from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from auxillary.utilities import Utilities
from cigt.cigt_ig_soft_routing import CigtIgSoftRouting

from auxillary.db_logger import DbLogger
from cigt.cigt_ig_hard_routing import CigtIgHardRouting

# Algorithm 1:
# Let each train sample to go its corresponding leaf.
# Train classifiers on the routed data.
# Let each test sample to go its corresponding leaf.
# Test accuracies. Hope: Ensemble effect can reduce overfitting?

device = "cpu"
temperature = 0.1


def execute_routing(model_, data_loader, data_kind, routing_matrices):
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

    for i, (input_, target) in tqdm(enumerate(data_loader)):
        time_begin = time.time()
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

    model.calculate_branch_statistics(
        run_id=model.runId,
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
    for lid in range(len(model.pathCounts) - 1):
        print("Layer {0} routing loss:{1}".format(lid, losses_t_layer_wise[lid].avg))

    # data_dict = {
    #     "last_layer_features": list_of_last_features_complete,
    #     "routing_matrices": list_of_routing_probability_matrices
    # }
    # Utilities.pickle_save_to_file(file_content=data_dict, path=data_file_path)


def execute_enforced_routing(model_, train_data, test_data):
    # All path configurations
    list_of_path_choices = []
    for path_count in model_.pathCounts[1:]:
        list_of_path_choices.append([i_ for i_ in range(path_count)])
    route_combinations = Utilities.get_cartesian_product(list_of_lists=list_of_path_choices)
    print("X")


if __name__ == "__main__":
    DbLogger.log_db_path = DbLogger.home_asus
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
    model = CigtIgHardRouting(
        run_id=run_id,
        model_definition="Resnet Hard Routing - Only Routing - 1.2.4. Batch Size 1024.")

    explanation = model.get_explanation_string()
    DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

    checkpoint_pth = "C://Users//asus//Desktop//ConvAig//convnet-aig//cigtlogger_14_epoch1180.pth"
    checkpoint = torch.load(checkpoint_pth, map_location="cpu")
    model.load_state_dict(state_dict=checkpoint["model_state_dict"])

    torch.manual_seed(1)
    best_performance = 0.0

    # Cifar 10 Dataset
    kwargs = {'num_workers': 2, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
        batch_size=1024, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=1024, shuffle=False, **kwargs)

    execute_routing(model_=model, data_kind="train",
                    data_loader=train_loader, routing_matrices=None)

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
