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
def train_feature_extractor(model, train_loader, num_epochs=10, lr=1e-4, save_path="./trained_model"):
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
    print(f"模型保存路径: {save_path}")
    
    best_loss = float('inf')
    
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
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, save_path, epoch, avg_loss, is_best=True)
            print(f"✅ 保存最佳模型 (Loss: {avg_loss:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            save_model(model, save_path, epoch, avg_loss, is_best=False)
            print(f"💾 保存检查点 (Epoch {epoch+1})")
    
    print("训练完成！")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"模型已保存到: {save_path}")

def save_model(model, save_path, epoch, loss, is_best=False):
    """
    保存模型
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 保存模型状态
    model_state = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'projection_state_dict': model.projection.state_dict(),
        'is_best': is_best
    }
    
    # 保存检查点
    checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save(model_state, checkpoint_path)
    
    # 如果是最佳模型，也保存一个best_model.pt
    if is_best:
        best_path = os.path.join(save_path, 'best_model.pt')
        torch.save(model_state, best_path)
    
    # 保存最终模型
    final_path = os.path.join(save_path, 'final_model.pt')
    torch.save(model_state, final_path)

def load_model(model, load_path, load_best=True):
    """
    加载训练好的模型
    """
    import os
    
    if load_best:
        model_path = os.path.join(load_path, 'best_model.pt')
    else:
        model_path = os.path.join(load_path, 'final_model.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型状态
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.projection.load_state_dict(checkpoint['projection_state_dict'])
    
    print(f"✅ 模型加载成功: {model_path}")
    print(f"   训练轮数: {checkpoint['epoch']+1}")
    print(f"   损失值: {checkpoint['loss']:.4f}")
    print(f"   是否最佳: {checkpoint['is_best']}")
    
    return model

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

# 完整的训练和保存示例
def complete_training_example():
    print("=== 完整训练和保存示例 ===\n")
    
    print("1. 训练模型:")
    print("""
    # 创建模型
    extractor = QwenVLFeatureExtractorWithTraining()
    
    # 准备数据
    train_loader = prepare_data_loader()
    
    # 训练并保存
    train_feature_extractor(
        model=extractor,
        train_loader=train_loader,
        num_epochs=10,
        lr=1e-4,
        save_path="./trained_qwen_vl_model"
    )
    """)
    
    print("\n2. 加载模型:")
    print("""
    # 创建新模型实例
    new_extractor = QwenVLFeatureExtractorWithTraining()
    
    # 加载训练好的参数
    load_model(new_extractor, "./trained_qwen_vl_model", load_best=True)
    
    # 现在可以使用训练好的模型
    features = new_extractor.encode_text("test text")
    """)
    
    print("\n3. 模型文件结构:")
    print("""
    trained_qwen_vl_model/
    ├── best_model.pt          # 最佳模型
    ├── final_model.pt         # 最终模型
    ├── checkpoint_epoch_5.pt  # 第5轮检查点
    ├── checkpoint_epoch_10.pt # 第10轮检查点
    └── training_log.txt       # 训练日志
    """)

# 模型保存的最佳实践
def model_saving_best_practices():
    print("\n=== 模型保存最佳实践 ===")
    
    print("1. 保存内容:")
    print("   - 模型参数 (state_dict)")
    print("   - 优化器状态 (可选)")
    print("   - 训练轮数和损失")
    print("   - 模型配置信息")
    print("   - 数据预处理参数")
    
    print("\n2. 保存策略:")
    print("   - 保存最佳模型 (基于验证损失)")
    print("   - 定期保存检查点 (防止训练中断)")
    print("   - 保存最终模型 (训练完成)")
    print("   - 保存多个版本 (便于比较)")
    
    print("\n3. 文件命名:")
    print("   - 包含模型名称和版本")
    print("   - 包含训练日期和时间")
    print("   - 包含性能指标")
    print("   - 使用有意义的后缀")
    
    print("\n4. 存储管理:")
    print("   - 定期清理旧检查点")
    print("   - 备份重要模型")
    print("   - 记录模型性能")
    print("   - 版本控制管理")

# 模型部署示例
def model_deployment_example():
    print("\n=== 模型部署示例 ===")
    
    print("1. 生产环境加载:")
    print("""
    class ProductionFeatureExtractor:
        def __init__(self, model_path):
            self.model = QwenVLFeatureExtractorWithTraining()
            load_model(self.model, model_path, load_best=True)
            self.model.eval()  # 设置为评估模式
        
        def extract_features(self, text, image):
            with torch.no_grad():
                return self.model.encode_multimodal(text, image)
    
    # 使用
    extractor = ProductionFeatureExtractor("./trained_qwen_vl_model")
    features = extractor.extract_features("描述图像", image)
    """)
    
    print("\n2. 模型压缩和优化:")
    print("""
    # 模型量化
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # 模型剪枝
    pruned_model = torch.nn.utils.prune.global_unstructured(
        model, pruning_method=torch.nn.utils.prune.L1Unstructured, amount=0.3
    )
    """)

if __name__ == "__main__":
    training_strategies()
    data_preparation_tips()
    evaluation_methods()
    complete_training_example()
    model_saving_best_practices()
    model_deployment_example()
    
    print("\n=== 总结 ===")
    print("模型保存的重要性:")
    print("1. 保存训练成果，避免重复训练")
    print("2. 便于模型部署和分享")
    print("3. 支持模型版本管理")
    print("4. 便于实验对比和调优")
    print("5. 防止训练中断导致的数据丢失")
    print("\n建议: 训练时一定要设置合理的保存策略！") 