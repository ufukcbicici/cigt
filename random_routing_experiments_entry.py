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
from cigt.cigt_ig_refactored import CigtIgHardRoutingX
from cigt.cigt_ig_soft_routing import CigtIgSoftRouting

from auxillary.db_logger import DbLogger
from cigt.cigt_ig_hard_routing import CigtIgHardRouting

device = "cpu"
temperature = 0.1

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

    trained_model0 = CigtIgHardRoutingX(
        run_id=run_id,
        model_definition="0",
        num_classes=10)
    trained_model350 = CigtIgHardRoutingX(
        run_id=run_id,
        model_definition="350",
        num_classes=10)

    checkpoint0 = torch.load(os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                          "random_cigtlogger2_29_epoch0.pth"), map_location="cpu")
    trained_model0.load_state_dict(state_dict=checkpoint0["model_state_dict"])

    checkpoint350 = torch.load(os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                            "random_cigtlogger2_29_epoch350.pth"), map_location="cpu")
    trained_model350.load_state_dict(state_dict=checkpoint350["model_state_dict"])

    print("X")
