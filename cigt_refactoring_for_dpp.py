import os
from collections import Counter

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from focal_loss.focal_loss import FocalLoss
from collections import Counter
from auxillary.average_meter import AverageMeter
from auxillary.db_logger import DbLogger
from auxillary.lenet_config_interpreter import LenetConfigInterpreter
from auxillary.resnet_config_interpreter import ResnetConfigInterpreter
from auxillary.utilities import Utilities
from cigt.cigt_model import conv3x3
from cigt.moe_layer import MoeLayer
# from configs.lenet_cigt_configs import LenetCigtConfigs
from cigt.routing_layers.cbam_routing_layer import CbamRoutingLayer
from torchvision import transforms

from cigt.routing_layers.soft_routing_layer import SoftRoutingLayer
from cigt.routing_manager_algorithms.information_gain_routing_manager import InformationGainRoutingManager


class CigtClean(nn.Module):
    def __init__(self, run_id, configs, model_definition):
        super().__init__()
        self.runId = run_id
        self.modelBackbone = configs.backbone
        assert self.modelBackbone in {"ResNet", "LeNet"}
        self.configInterpreter = None
        if self.modelBackbone == "ResNet":
            self.configInterpreter = ResnetConfigInterpreter
        elif self.modelBackbone == "LeNet":
            self.configInterpreter = LenetConfigInterpreter
        else:
            raise NotImplementedError()

        self.modelDefinition = model_definition

    def forward(self, x, labels, temperature):
        pass


