import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CigtReinforceMultipath(nn.Module):
    def __init__(self, cigt_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cigtModel = cigt_model


