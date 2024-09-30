import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel, BertTokenizer


# 图像编码器
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # 去掉全连接层

    def forward(self, x):
        return self.resnet(x)


# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 标记的嵌入


# 投影头
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# 示例数据
image_encoder = ImageEncoder()
text_encoder = TextEncoder()

# 图像数据
image_data = torch.randn(16, 3, 224, 224)  # 16 张 224x224 的图像
image_features = image_encoder(image_data)

# 文本数据
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
texts = ["This is a sample text."] * 16
input_ids = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')['input_ids']
attention_mask = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')['attention_mask']
text_features = text_encoder(input_ids, attention_mask)

# 投影头
image_projection_head = ProjectionHead(image_features.shape[1], 512)
text_projection_head = ProjectionHead(text_features.shape[1], 512)

# 投影并归一化
image_embeddings = nn.functional.normalize(image_projection_head(image_features), dim=1)
text_embeddings = nn.functional.normalize(text_projection_head(text_features), dim=1)

# 计算相似性矩阵
logits = torch.matmul(image_embeddings, text_embeddings.T)
logits = logits * torch.exp(torch.tensor(1.0))  # 温度参数 t

# 标签
labels = torch.arange(logits.shape[0])

# 对图像的损失
loss_i = F.cross_entropy(logits, labels)

# 对文本的损失
loss_t = F.cross_entropy(logits.T, labels)

# 最终损失
loss = (loss_i + loss_t) / 2

# 优化器
optimizer = torch.optim.Adam([
    {'params': image_encoder.parameters()},
    {'params': text_encoder.parameters()},
    {'params': image_projection_head.parameters()},
    {'params': text_projection_head.parameters()}
], lr=1e-4)

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Total Loss: {loss.item()}")
