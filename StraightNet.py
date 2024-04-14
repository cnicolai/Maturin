import torch
from torch import nn, sigmoid
import torch.nn.functional as F


class StraightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_size = 64 * 64 * 16
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.linear_layer_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        out = torch.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv3(out)
        out = torch.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv4(out)
        out = torch.relu(out)
        out = F.max_pool2d(out, 2)

        out = out.view(-1, self.linear_layer_size)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        out = self.fc3(out)
        out = sigmoid(out)
        return out
