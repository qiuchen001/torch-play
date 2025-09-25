import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BasicBlock(nn.Module):
    expansion = 1  # BasicBlock 的扩展因子为 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        # 主路径
        self.conv1: nn.Conv2d = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        self.conv2: nn.Conv2d = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(planes)

        # 残差连接（short shortcut）
        self.shortcut: nn.Sequential = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    # 主路径：由两个 3*3 的卷积层（self.conv1, self.conv2）和两个 BatchNorm2d 层（self.bn1, self.bn2）组成。这是数据变换的主要部分。
    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = F.relu(self.bn1(self.conv1(x)))
        out: Tensor = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 核心：残差连接
        out = F.relu(input=out)
        return out
#  输入 x 首先经过 卷积 -> 批标准化 -> ReLU激活函数。
#   这个 BasicBlock 模块实现了一个残差单元。通过让网络学习对输入的“残差”（即 out 部分），而不是直接学习一个完整的变换，ResNet
#   极大地缓解了深度神经网络中的梯度消失问题，使得构建非常深的网络成为可能，从而在图像分类等任务上取得了突破性的成果。
