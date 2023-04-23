import os
import torch
import numpy as np
from auxillary.db_logger import DbLogger
from torchvision import transforms
import torchvision.datasets as datasets

from auxillary.rump_dataset import RumpDataset
from auxillary.similar_dataset_division_algorithm import SimilarDatasetDivisionAlgorithm
from cigt.cigt_ig_refactored import CigtIgHardRoutingX


class MultiplePathOptimizer(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset






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

    # rump_dataset = RumpDataset(source_dataset=test_loader, indices=np.arange(0, 5000))
    # rump_loader = torch.utils.data.DataLoader(rump_dataset, batch_size=1024, shuffle=False, **kwargs)

    run_id = DbLogger.get_run_id()
    trained_model = CigtIgHardRoutingX(
        run_id=run_id,
        model_definition="Cigt - [1,2,4] - MultipleLogitsMultipleLosses - Multiple Path Inference",
        num_classes=10)

    explanation = trained_model.get_explanation_string()
    DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)

    checkpoint_pth = os.path.join(os.path.split(os.path.abspath(__file__))[0], "cigtlogger_32_epoch1119.pth")
    checkpoint = torch.load(checkpoint_pth, map_location="cpu")
    trained_model.load_state_dict(state_dict=checkpoint["model_state_dict"])

    SimilarDatasetDivisionAlgorithm.run(model=trained_model, parent_loader=test_loader,
                                        save_path=os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                                               "multiple_inference_data.sav"),
                                        division_ratio=0.5, max_accuracy_dif=0.002)

    # trained_model.validate(loader=rump_loader, temperature=0.1, epoch=0, data_kind="test")
    # print("X")
