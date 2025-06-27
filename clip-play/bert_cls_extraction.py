import torch
from transformers import BertTokenizer, BertModel

class BERTTextEncoder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # 设置为评估模式
        
        print(f"加载BERT模型: {model_name}")
        print(f"隐藏层维度: {self.model.config.hidden_size}")
    
    def encode_text(self, texts):
        """
        编码文本并提取[CLS]标记
        """
        # 1. 分词
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        print(f"输入文本: {texts}")
        print(f"分词结果: {self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}")
        print(f"输入形状: {encoded['input_ids'].shape}")
        
        # 2. 前向传播
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        # 3. 提取[CLS]标记
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # 关键代码
        
        print(f"BERT输出形状: {outputs.last_hidden_state.shape}")
        print(f"[CLS]嵌入形状: {cls_embeddings.shape}")
        
        return cls_embeddings
    
    def analyze_embeddings(self, texts):
        """
        分析不同位置的嵌入
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=10,  # 短序列便于分析
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        hidden_states = outputs.last_hidden_state
        
        print(f"\n=== 嵌入分析 ===")
        print(f"完整隐藏状态形状: {hidden_states.shape}")
        
        # 分析不同位置的嵌入
        for i in range(min(5, hidden_states.shape[1])):
            token_id = encoded['input_ids'][0][i].item()
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            embedding = hidden_states[0, i, :]
            
            print(f"位置 {i}: {token}")
            print(f"  嵌入形状: {embedding.shape}")
            print(f"  嵌入范数: {torch.norm(embedding):.4f}")
            print(f"  前5个值: {embedding[:5].tolist()}")
            print()

# CLIP风格的文本编码器
class CLIPTextEncoder:
    def __init__(self, model_name='bert-base-uncased', projection_dim=512):
        self.bert_encoder = BERTTextEncoder(model_name)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(768, projection_dim),  # BERT hidden_size = 768
            torch.nn.ReLU(),
            torch.nn.Linear(projection_dim, projection_dim)
        )
        
        print(f"CLIP文本编码器:")
        print(f"  BERT维度: 768")
        print(f"  投影维度: {projection_dim}")
    
    def encode(self, texts):
        # 1. 提取[CLS]嵌入
        cls_embeddings = self.bert_encoder.encode_text(texts)
        
        # 2. 投影到共享空间
        projected = self.projection(cls_embeddings)
        
        # 3. 归一化（CLIP中的关键步骤）
        normalized = torch.nn.functional.normalize(projected, p=2, dim=1)
        
        return normalized

# 使用示例
if __name__ == "__main__":
    print("=== BERT [CLS]标记提取示例 ===\n")
    
    # 1. 基本示例
    encoder = BERTTextEncoder()
    texts = ["Hello world", "This is a test"]
    cls_embeddings = encoder.encode_text(texts)
    
    print(f"\n[CLS]嵌入形状: {cls_embeddings.shape}")
    print(f"第一个样本[CLS]嵌入前5个值: {cls_embeddings[0, :5].tolist()}")
    
    # 2. 详细分析
    print("\n" + "="*50)
    encoder.analyze_embeddings(["Hello world"])
    
    # 3. CLIP风格编码器
    print("\n" + "="*50)
    clip_text_encoder = CLIPTextEncoder()
    clip_embeddings = clip_text_encoder.encode(texts)
    
    print(f"CLIP文本嵌入形状: {clip_embeddings.shape}")
    print(f"归一化后范数: {torch.norm(clip_embeddings, dim=1)}")
    
    print("\n=== 总结 ===")
    print("1. [CLS]标记的作用:")
    print("   - BERT序列的第一个特殊标记")
    print("   - 用于表示整个序列的语义信息")
    print("   - 在分类任务中作为序列表示")
    
    print("\n2. 在CLIP中的应用:")
    print("   - 提取文本的全局语义表示")
    print("   - 与图像特征进行对比学习")
    print("   - 实现跨模态相似度计算")
    
    print("\n3. 关键代码解析:")
    print("   outputs.last_hidden_state[:, 0, :]")
    print("   - outputs.last_hidden_state: BERT最后一层输出")
    print("   - [:, 0, :]: 选择所有样本的第0个位置([CLS])")
    print("   - 结果: [batch_size, hidden_size]") 