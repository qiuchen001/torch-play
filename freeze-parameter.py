import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        return out


model = MyModel()

print(model.state_dict().keys())

# 冻结 conv1 层的参数
model.conv1.weight.requires_grad = False
model.conv1.bias.requires_grad = False

# 解冻 conv1 层的参数
model.conv1.weight.requires_grad = True
model.conv1.bias.requires_grad = True
