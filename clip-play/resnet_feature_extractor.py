import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()
        
        # 加载预训练的ResNet
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # 关键步骤：移除全连接层
        self.resnet.fc = nn.Identity()  # 去掉全连接层
        
        print("ResNet结构:")
        print(self.resnet)
        print(f"\n移除全连接层后的输出维度: {self.get_feature_dim()}")
    
    def get_feature_dim(self):
        """获取特征维度"""
        with torch.no_grad():
            # 创建一个测试输入
            test_input = torch.randn(1, 3, 224, 224)
            features = self.resnet(test_input)
            return features.shape[1]
    
    def forward(self, x):
        # 现在只输出特征，不进行分类
        return self.resnet(x)

# 对比：原始ResNet vs 特征提取器
def compare_models():
    print("=== 原始ResNet ===")
    original_resnet = models.resnet50(pretrained=False)
    test_input = torch.randn(1, 3, 224, 224)
    original_output = original_resnet(test_input)
    print(f"原始ResNet输出形状: {original_output.shape}")  # [1, 1000]
    print(f"原始ResNet最后一层: {original_resnet.fc}")
    
    print("\n=== 特征提取器 ===")
    feature_extractor = ResNetFeatureExtractor(pretrained=False)
    feature_output = feature_extractor(test_input)
    print(f"特征提取器输出形状: {feature_output.shape}")  # [1, 2048]
    print(f"特征提取器最后一层: {feature_extractor.resnet.fc}")

# CLIP风格的图像编码器示例
class CLIPImageEncoder(nn.Module):
    def __init__(self, feature_dim=2048, projection_dim=512):
        super(CLIPImageEncoder, self).__init__()
        
        # 使用ResNet作为特征提取器
        self.backbone = ResNetFeatureExtractor(pretrained=True)
        
        # 添加投影层（类似CLIP的做法）
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        print(f"CLIP图像编码器:")
        print(f"  特征维度: {feature_dim}")
        print(f"  投影维度: {projection_dim}")
    
    def forward(self, x):
        # 1. 提取特征
        features = self.backbone(x)  # [batch_size, 2048]
        
        # 2. 投影到共享空间
        projected = self.projection(features)  # [batch_size, 512]
        
        # 3. 归一化（CLIP中的关键步骤）
        normalized = nn.functional.normalize(projected, p=2, dim=1)
        
        return normalized

# 使用示例
if __name__ == "__main__":
    print("=== 模型对比 ===")
    compare_models()
    
    print("\n" + "="*50 + "\n")
    
    print("=== CLIP图像编码器示例 ===")
    clip_encoder = CLIPImageEncoder()
    
    # 测试输入
    test_images = torch.randn(4, 3, 224, 224)  # 4张图片
    encoded_features = clip_encoder(test_images)
    
    print(f"输入形状: {test_images.shape}")
    print(f"输出形状: {encoded_features.shape}")
    print(f"特征范数: {torch.norm(encoded_features, dim=1)}")  # 应该接近1（归一化后）
    
    print("\n=== 总结 ===")
    print("1. nn.Identity() 的作用：")
    print("   - 移除ResNet的分类层")
    print("   - 只保留特征提取能力")
    print("   - 输出原始特征向量")
    
    print("\n2. 在CLIP中的应用：")
    print("   - 使用ResNet提取图像特征")
    print("   - 添加投影层映射到共享空间")
    print("   - 实现图像-文本对比学习") 