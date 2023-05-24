import torch
from torch import nn
from torch.nn import functional as F


class ImageNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: dimension of tags
        :param output_dim: bit number of the final binary code
        """
        super(ImageNet, self).__init__()
        self.module_name = "image_model"

        # full-conv layers
        mid_num1 = 4096
        mid_num2 = 2048
        mid_num3 = 1024
        self.fc1 = nn.Linear(input_dim, mid_num1)
        self.fc2 = nn.Linear(mid_num1, mid_num2)
        self.fc2_2 = nn.Linear(mid_num2, mid_num3)
        self.fc3 = nn.Linear(mid_num3, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_2(x))
        x = self.fc3(x)
        norm = torch.norm(x, dim=1, keepdim=True)
        x = x / norm
        return x

    def _init_weights(self):
        for m in self._modules:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=0, b=1)
