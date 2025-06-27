import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

class QwenVLFeatureExtractor:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"加载Qwen2.5-VL模型: {model_name}")
        print(f"模型配置: {self.model.config}")
    
    def extract_features(self, text, image=None, extract_layer=-1):
        """
        提取Qwen2.5-VL的特征
        Args:
            text: 输入文本
            image: 输入图像（可选）
            extract_layer: 提取哪一层的特征（-1表示最后一层）
        """
        # 1. 准备输入
        inputs = self.tokenizer(text, return_tensors="pt")
        if image is not None:
            inputs["images"] = image
        
        # 2. 获取中间层输出
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # 3. 提取指定层的特征
        hidden_states = outputs.hidden_states
        target_layer = hidden_states[extract_layer]  # 最后一层
        
        # 4. 类似BERT的[CLS]提取方式
        # 对于Qwen2.5-VL，我们可以提取最后一个token的特征
        last_token_features = target_layer[:, -1, :]  # 取最后一个token
        
        return last_token_features, outputs
    
    def extract_multimodal_features(self, text, image):
        """
        提取多模态特征（文本+图像）
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs["images"] = image
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # 提取最后一层的特征
        last_layer = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        
        # 方法1: 取最后一个token（类似[CLS]）
        last_token_features = last_layer[:, -1, :]
        
        # 方法2: 平均池化所有token
        mean_features = last_layer.mean(dim=1)
        
        # 方法3: 取图像token的特征（如果有的话）
        # 这需要知道图像token的位置
        image_features = self._extract_image_features(last_layer, inputs)
        
        return {
            'last_token': last_token_features,
            'mean_pooling': mean_features,
            'image_features': image_features
        }
    
    def _extract_image_features(self, hidden_states, inputs):
        """
        提取图像相关的特征
        """
        # 这里需要根据具体的tokenizer实现
        # 通常图像token会有特殊的标识
        return hidden_states[:, -1, :]  # 简化处理

# 类似CLIP的特征提取器
class QwenVLCLIPStyleExtractor:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", projection_dim=512):
        self.qwen_extractor = QwenVLFeatureExtractor(model_name)
        
        # 获取隐藏层维度
        hidden_size = self.qwen_extractor.model.config.hidden_size
        print(f"Qwen2.5-VL隐藏层维度: {hidden_size}")
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        print(f"投影层: {hidden_size} -> {projection_dim}")
    
    def encode_text(self, text):
        """编码文本"""
        features, _ = self.qwen_extractor.extract_features(text)
        projected = self.projection(features)
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized
    
    def encode_image(self, text, image):
        """编码图像（通过文本描述）"""
        multimodal_features = self.qwen_extractor.extract_multimodal_features(text, image)
        
        # 使用平均池化的特征
        features = multimodal_features['mean_pooling']
        projected = self.projection(features)
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized
    
    def encode_multimodal(self, text, image):
        """编码多模态输入"""
        multimodal_features = self.qwen_extractor.extract_multimodal_features(text, image)
        
        # 可以选择不同的特征提取方式
        features = multimodal_features['last_token']  # 或 'mean_pooling'
        projected = self.projection(features)
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized

# 使用示例
def demonstrate_feature_extraction():
    print("=== Qwen2.5-VL特征提取示例 ===\n")
    
    # 注意：这里需要实际的模型和图像
    # 由于模型较大，这里只展示代码结构
    
    # 1. 基本特征提取
    print("1. 基本文本特征提取:")
    print("""
    extractor = QwenVLFeatureExtractor()
    text = "Describe this image"
    features, outputs = extractor.extract_features(text)
    print(f"特征形状: {features.shape}")
    """)
    
    # 2. 多模态特征提取
    print("\n2. 多模态特征提取:")
    print("""
    text = "What is in this image?"
    image = load_image("example.jpg")
    multimodal_features = extractor.extract_multimodal_features(text, image)
    
    for key, value in multimodal_features.items():
        print(f"{key}: {value.shape}")
    """)
    
    # 3. CLIP风格编码器
    print("\n3. CLIP风格编码器:")
    print("""
    clip_extractor = QwenVLCLIPStyleExtractor()
    
    # 文本编码
    text_features = clip_extractor.encode_text("a photo of a cat")
    
    # 图像编码
    image_features = clip_extractor.encode_image("describe this image", image)
    
    # 计算相似度
    similarity = torch.cosine_similarity(text_features, image_features)
    print(f"相似度: {similarity.item()}")
    """)

# 实际实现建议
def practical_implementation_tips():
    print("\n=== 实际实现建议 ===")
    print("1. 模型加载:")
    print("   - 使用device_map='auto'自动处理设备分配")
    print("   - 考虑使用torch.float16节省内存")
    print("   - 可以使用量化版本减少内存占用")
    
    print("\n2. 特征提取位置:")
    print("   - 最后一层: 最丰富的语义信息")
    print("   - 倒数第二层: 可能更稳定")
    print("   - 多层融合: 结合不同层的特征")
    
    print("\n3. 特征聚合方式:")
    print("   - 最后一个token: 类似BERT的[CLS]")
    print("   - 平均池化: 所有token的平均")
    print("   - 加权平均: 根据注意力权重")
    print("   - 图像token: 专门提取图像相关特征")
    
    print("\n4. 优化建议:")
    print("   - 使用torch.no_grad()节省内存")
    print("   - 批量处理提高效率")
    print("   - 缓存中间特征避免重复计算")

if __name__ == "__main__":
    demonstrate_feature_extraction()
    practical_implementation_tips()
    
    print("\n=== 总结 ===")
    print("借鉴BERT [CLS]提取方式，您可以:")
    print("1. 提取Qwen2.5-VL最后一层的特征")
    print("2. 选择最后一个token作为全局表示")
    print("3. 添加投影层映射到共享空间")
    print("4. 实现类似CLIP的对比学习")
    print("\n这种方法可以充分利用Qwen2.5-VL的强大多模态理解能力！") 