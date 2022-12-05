import torch.nn as nn
import torch
from torchvision import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.CBAM import cbam_block


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(3, 2601),
            nn.BatchNorm1d(2601),
            nn.ReLU(),
            nn.Linear(2601, 2601),
            nn.BatchNorm1d(2601),
            nn.ReLU(),
            nn.Linear(2601, 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1]

        out = self.linear(x2)

        return out


class Conv1d(nn.Module):
    def __init__(self):
        super(Conv1d, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 867, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(867),
            nn.ReLU(),
            nn.Conv1d(867, 867, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(867),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(867 * 3, 2)
        )

    def forward(self, x):
        x1, x2 = x[0][:, 0:-1, :, :], x[1].view(-1, 1, 3)

        x2 = self.conv1d(x2).view(-1, 867 * 3)

        out = self.linear(x2)

        return out
