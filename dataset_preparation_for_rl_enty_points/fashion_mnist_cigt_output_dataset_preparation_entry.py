import os

# from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
import torch
from randaugment import RandAugment
from torchvision import datasets
from torchvision.transforms import transforms

from auxillary.db_logger import DbLogger
from cigt.cigt_ig_gather_scatter_implementation import CigtIgGatherScatterImplementation
from cigt.cigt_output_dataset import CigtOutputDataset
from cigt.cutout_augmentation import CutoutPIL
from cigt.cutout_augmentation_gray import CutoutPILGray
from cigt.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from configs.cifar10_resnet_cigt_configs import Cifar10ResnetCigtConfigs
from configs.fashion_lenet_cigt_configs import FashionLenetCigtConfigs
from load_trained_models.load_trained_fmnist_model import load_trained_fmnist_model

# random.seed(53)
# np.random.seed(61)

if __name__ == "__main__":
    chck_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..",
                             "checkpoints/cigtlogger2_170_epoch141_data.pth")
    data_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..",
                             "cigtlogger2_170_epoch141_data")
    DbLogger.log_db_path = DbLogger.hpc_docker2

    # print("DB FILES")
    # print(os.path.isfile(DbLogger.hpc_docker1))
    # print(os.path.isfile(DbLogger.hpc_docker2))

    fmnist_model, mac_counts_per_block = load_trained_fmnist_model(checkpoint_path=chck_path)

    # Load datasets
    kwargs = {'num_workers': 0, 'pin_memory': True}
    heavyweight_augmentation = transforms.Compose([
        # transforms.Resize((32, 32)),
        CutoutPILGray(cutout_factor=0.25),
        # RandAugment(),
        transforms.ToTensor(),
    ])
    lightweight_augmentation = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../../data', download=True, train=True, transform=heavyweight_augmentation),
        batch_size=FashionLenetCigtConfigs.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../../data', download=True, train=False, transform=lightweight_augmentation),
        batch_size=FashionLenetCigtConfigs.batch_size, shuffle=False, **kwargs)

    train_acc = fmnist_model.validate(data_kind="test", epoch=0, loader=train_loader, temperature=0.1,
                                      verbose=True)
    test_acc = fmnist_model.validate(data_kind="test", epoch=0, loader=test_loader, temperature=0.1,
                                     verbose=True)
    print("Standard train accuracy:{0}".format(train_acc))
    print("Standard test accuracy:{0}".format(test_acc))

    test_cigt_output_dataset = CigtOutputDataset(input_reduction_factor=2)
    # test_cigt_output_dataset.load_from_file(file_path="test_cigt_dataset.sav")
    test_cigt_output_dataset.load_from_model(model=fmnist_model, data_loader=test_loader, repeat_count=1,
                                             list_of_fields=["block_outputs_dict",
                                                             "routing_matrices_soft_dict",
                                                             "labels_dict",
                                                             "logits_dict"])
    test_cigt_output_dataset.save(file_path=os.path.join(data_path, "qnet_test_cigt_dataset.sav"))

    for repeat_count in [3]:
        train_cigt_output_dataset = CigtOutputDataset(input_reduction_factor=2)
        # train_cigt_output_dataset.load_from_file(file_path="train_cigt_dataset.sav")
        train_cigt_output_dataset.load_from_model(model=fmnist_model,
                                                  data_loader=train_loader, repeat_count=repeat_count,
                                                  list_of_fields=["block_outputs_dict",
                                                                  "routing_matrices_soft_dict",
                                                                  "labels_dict",
                                                                  "logits_dict"])
        train_cigt_output_dataset.save(file_path=
                                       os.path.join(data_path, "qnet_train_cigt_dataset{0}.sav".format(repeat_count)))
