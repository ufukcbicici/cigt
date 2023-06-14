from cigt.cigt_model import BasicBlock
from cigt.custom_layers.convolutional_block_attention_module import CBAM
import torch.nn.functional as F


class BasicBlockWithCbam(BasicBlock):
    def __init__(self, in_planes, planes, cbam_reduction_ratio, stride=1):
        super().__init__(in_planes, planes, stride)
        self.cbamReductionRatio = cbam_reduction_ratio

        self.cbam = CBAM(gate_channels=planes, reduction_ratio=cbam_reduction_ratio)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out = self.shortcut(x) + out
        out = F.relu(out)
        return out



