import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from qwen_vl_feature_extractor import QwenVLFeatureExtractor

class QwenVLFeatureExtractorWithTraining:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", projection_dim=512):
        self.qwen_extractor = QwenVLFeatureExtractor(model_name)
        
        # 获取隐藏层维度
        hidden_size = self.qwen_extractor.model.config.hidden_size
        
        # 投影层（需要训练）
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # 冻结Qwen2.5-VL的参数（可选）
        self._freeze_backbone()
        
        print(f"特征提取器初始化完成")
        print(f"  隐藏层维度: {hidden_size}")
        print(f"  投影维度: {projection_dim}")
    
    def _freeze_backbone(self):
        """冻结Qwen2.5-VL的参数"""
        for param in self.qwen_extractor.model.parameters():
            param.requires_grad = False
        print("已冻结Qwen2.5-VL参数")
    
    def _unfreeze_backbone(self):
        """解冻Qwen2.5-VL的参数（微调）"""
        for param in self.qwen_extractor.model.parameters():
            param.requires_grad = True
        print("已解冻Qwen2.5-VL参数")

# 对比学习损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, text_features, image_features, labels=None):
        """
        计算对比学习损失
        Args:
            text_features: 文本特征 [batch_size, feature_dim]
            image_features: 图像特征 [batch_size, feature_dim]
            labels: 标签（可选）
        """
        # 归一化特征
        text_features = F.normalize(text_features, p=2, dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # 计算相似度矩阵
        logits = torch.mm(text_features, image_features.t()) / self.temperature
        
        # 如果没有标签，假设对角线为正样本
        if labels is None:
            labels = torch.arange(logits.size(0)).to(logits.device)
        
        # 计算对比损失
        loss_text = F.cross_entropy(logits, labels)
        loss_image = F.cross_entropy(logits.t(), labels)
        
        return (loss_text + loss_image) / 2

# 训练数据集
class MultimodalDataset(Dataset):
    def __init__(self, text_image_pairs):
        """
        Args:
            text_image_pairs: [(text, image_path), ...]
        """
        self.pairs = text_image_pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        text, image_path = self.pairs[idx]
        # 这里需要实现图像加载逻辑
        image = self.load_image(image_path)
        return text, image
    
    def load_image(self, image_path):
        # 实现图像加载
        pass

# 训练函数
def train_feature_extractor(model, train_loader, num_epochs=10, lr=1e-4):
    """
    训练特征提取器
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = ContrastiveLoss(temperature=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    print(f"开始训练，设备: {device}")
    print(f"学习率: {lr}")
    print(f"训练轮数: {num_epochs}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (texts, images) in enumerate(train_loader):
            # 将数据移到设备
            images = images.to(device)
            
            # 前向传播
            text_features = model.encode_text(texts)
            image_features = model.encode_image(texts, images)
            
            # 计算损失
            loss = criterion(text_features, image_features)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    print("训练完成！")

# 训练策略选择
def training_strategies():
    print("=== 训练策略选择 ===")
    
    print("\n1. 仅训练投影层（推荐开始）:")
    print("""
    # 冻结Qwen2.5-VL，只训练投影层
    extractor = QwenVLFeatureExtractorWithTraining()
    extractor._freeze_backbone()  # 冻结主干网络
    
    # 训练投影层
    train_feature_extractor(extractor, train_loader, num_epochs=5)
    """)
    
    print("\n2. 微调整个模型:")
    print("""
    # 解冻所有参数进行微调
    extractor._unfreeze_backbone()
    
    # 使用更小的学习率
    train_feature_extractor(extractor, train_loader, num_epochs=3, lr=1e-5)
    """)
    
    print("\n3. 渐进式训练:")
    print("""
    # 阶段1: 训练投影层
    extractor._freeze_backbone()
    train_feature_extractor(extractor, train_loader, num_epochs=3)
    
    # 阶段2: 微调最后几层
    extractor._unfreeze_last_layers()
    train_feature_extractor(extractor, train_loader, num_epochs=2, lr=1e-5)
    
    # 阶段3: 全模型微调
    extractor._unfreeze_backbone()
    train_feature_extractor(extractor, train_loader, num_epochs=1, lr=1e-6)
    """)

# 数据准备建议
def data_preparation_tips():
    print("\n=== 数据准备建议 ===")
    
    print("1. 训练数据格式:")
    print("   - 文本-图像对: (text, image_path)")
    print("   - 正样本: 匹配的文本和图像")
    print("   - 负样本: 不匹配的文本和图像")
    
    print("\n2. 数据增强:")
    print("   - 文本: 同义词替换、句式变换")
    print("   - 图像: 裁剪、旋转、颜色变换")
    
    print("\n3. 数据量建议:")
    print("   - 最小: 1K-5K 对")
    print("   - 推荐: 10K-50K 对")
    print("   - 理想: 100K+ 对")

# 评估方法
def evaluation_methods():
    print("\n=== 评估方法 ===")
    
    print("1. 相似度计算:")
    print("   - 计算文本-图像对的余弦相似度")
    print("   - 评估正样本的相似度是否高于负样本")
    
    print("\n2. 检索任务:")
    print("   - 给定文本，检索相关图像")
    print("   - 给定图像，检索相关文本")
    
    print("\n3. 分类任务:")
    print("   - 使用提取的特征进行下游分类")
    print("   - 评估特征的质量")

if __name__ == "__main__":
    training_strategies()
    data_preparation_tips()
    evaluation_methods()
    
    print("\n=== 总结 ===")
    print("是否需要额外训练取决于您的应用场景:")
    print("1. 简单应用: 直接使用，无需训练")
    print("2. 性能要求高: 需要训练投影层")
    print("3. 最佳效果: 需要微调整个模型")
    print("4. 推荐策略: 先训练投影层，再考虑微调") 