import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 定义简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # 展平为一维
        x = self.fc1(x)
        return x


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, num_samples, num_classes):
        self.num_samples = num_samples
        self.num_classes = num_classes
        # 生成随机图像和标签
        self.data = torch.randn(num_samples, 3, 32, 32)  # 假设图像大小为32x32，3个通道
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 参数设置
num_samples = 1000
num_classes = 3  # 狗、猫、鸟
num_epochs = 500
batch_size = 64
learning_rate = 0.001

# 创建自定义数据集
dataset = CustomDataset(num_samples, num_classes)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 对比学习损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # 计算特征的相似性矩阵
        similarity_matrix = torch.matmul(features, features.T)
        # 对角线上的元素是正样本对
        positive_pairs = torch.diag(similarity_matrix)
        # 负样本对是矩阵中非对角线上的元素
        negative_pairs = similarity_matrix - torch.diag(positive_pairs)
        # 计算损失
        loss = -torch.log(torch.exp(positive_pairs / self.temperature) / (torch.exp(positive_pairs / self.temperature) + torch.sum(torch.exp(negative_pairs / self.temperature))))
        return loss.mean()


# 对比学习训练过程
contrastive_loss = ContrastiveLoss()


# 检查批次中是否有重复的标签或图像
def check_duplicates(batch):
    images, labels = batch
    unique_images = torch.unique(images, dim=0)
    unique_labels = torch.unique(labels, dim=0)
    if len(unique_images) < len(images):
        print("Duplicate images found in batch.")
    if len(unique_labels) < len(labels):
        print("Duplicate labels found in batch.")


# 训练过程
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 检查批次中是否有重复的标签或图像
        # check_duplicates((images, labels))

        outputs = model(images)
        loss = criterion(outputs, labels)
        # loss = contrastive_loss(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")