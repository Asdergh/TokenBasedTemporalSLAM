import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use("dark_background")


a = th.rand(10, 3, 128, 128)
conv = nn.Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 3),
    padding=1
)

print(conv(conv(a)).size())