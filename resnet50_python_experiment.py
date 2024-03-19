import torch
import torchvision
from torchvision.models import ResNet50_Weights

# model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torchvision.models.resnet101()

# weights = ResNet50_Weights.verify(weights)
#
# return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)

resnet50_thin = torchvision.models.resnet._resnet(block=torchvision.models.resnet.Bottleneck,
                                                  layers=[3, 4, 6, 3],
                                                  weights=None,
                                                  progress=False,
                                                  width_per_group=32)

print("X")