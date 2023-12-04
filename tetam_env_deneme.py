import torch
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device:{0}".format(device))
x = torch.rand(size=(10, 5)).to(device)
print(x.device)
y = np.random.uniform(size=(100, 500))
print(np.mean(y))
