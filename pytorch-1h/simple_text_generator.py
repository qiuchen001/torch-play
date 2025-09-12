import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# 设置随机种子
torch.manual_seed(42)
random.seed(42)

class SimpleTextDataset(Dataset):
    """简化的文本数据集类"""
    
    def __init__(self, text, sequence_length=10):
        self.text = text
        self.sequence_length = sequence_length
        
        # 创建字符到索引的映射
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        print(f"词汇表: {self.chars}")
        print(f"词汇表大小: {self.vocab_size}")
        
        # 创建训练序列
        self.sequences = []
        self.targets = []
        
        for i in range(len(text) - sequence_length):
            seq = text[i:i + sequence_length]
            target = text[i + sequence_length]
            
            self.sequences.append([self.char_to_idx[ch] for ch in seq])
            self.targets.append(self.char_to_idx[target])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

class SimpleTextGenerator(nn.Module):
    """简化的文本生成模型"""
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super(SimpleTextGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 嵌入层：将字符索引转换为向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层：处理序列信息
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # 输出层：将隐藏状态转换为字符概率
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # 1. 嵌入：字符索引 -> 向量
        embedded = self.embedding(x)
        
        # 2. LSTM：处理序列
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # 3. 输出：隐藏状态 -> 字符概率
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化LSTM的隐藏状态"""
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

def prepare_simple_dataset():
    """准备简单的训练数据"""
    # 使用简单的文本
    text = "hello world hello python hello pytorch"
    return text

def train_simple_model(model, train_loader, num_epochs=50, device='cpu'):
    """训练简化模型"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # 初始化隐藏状态
            hidden = model.init_hidden(sequences.size(0), device)
            
            # 前向传播
            outputs, _ = model(sequences, hidden)
            
            # 计算损失（使用最后一个时间步的输出）
            loss = criterion(outputs[:, -1, :], targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def generate_simple_text(model, dataset, start_text, length=20, device='cpu'):
    """生成简单文本"""
    model.eval()
    model.to(device)
    
    # 将起始文本转换为索引
    start_indices = [dataset.char_to_idx[ch] for ch in start_text if ch in dataset.char_to_idx]
    if not start_indices:
        start_indices = [0]
    
    generated_text = start_text
    current_sequence = torch.tensor([start_indices], dtype=torch.long).to(device)
    
    # 确保序列长度正确
    if current_sequence.size(1) < dataset.sequence_length:
        padding = torch.zeros(1, dataset.sequence_length - current_sequence.size(1), dtype=torch.long).to(device)
        current_sequence = torch.cat([padding, current_sequence], dim=1)
    elif current_sequence.size(1) > dataset.sequence_length:
        current_sequence = current_sequence[:, -dataset.sequence_length:]
    
    with torch.no_grad():
        hidden = model.init_hidden(1, device)
        
        for _ in range(length):
            # 前向传播
            output, hidden = model(current_sequence, hidden)
            
            # 获取最后一个时间步的输出
            last_output = output[:, -1, :]
            
            # 应用softmax并采样
            probabilities = F.softmax(last_output, dim=-1)
            next_char_idx = torch.multinomial(probabilities, 1).item()
            
            # 添加新字符
            next_char = dataset.idx_to_char[next_char_idx]
            generated_text += next_char
            
            # 更新序列（滑动窗口）
            current_sequence = torch.cat([current_sequence[:, 1:], torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)
    
    return generated_text

def main():
    """主函数"""
    print("=== 简化文本生成任务 ===")
    
    # 准备数据
    text = prepare_simple_dataset()
    print(f"训练文本: '{text}'")
    print(f"文本长度: {len(text)} 字符")
    
    # 创建数据集
    sequence_length = 5
    dataset = SimpleTextDataset(text, sequence_length)
    print(f"训练序列数量: {len(dataset)}")
    
    # 创建数据加载器
    batch_size = 4
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = SimpleTextGenerator(
        vocab_size=dataset.vocab_size,
        embedding_dim=32,
        hidden_dim=64
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 训练模型
    print("\n开始训练...")
    train_simple_model(model, train_loader, num_epochs=100, device=device)
    
    # 生成文本
    print("\n=== 文本生成结果 ===")
    start_texts = ["hello", "world", "pyt"]
    
    for start_text in start_texts:
        generated = generate_simple_text(model, dataset, start_text, length=15, device=device)
        print(f"起始文本: '{start_text}'")
        print(f"生成文本: {generated}")
        print("-" * 40)

if __name__ == "__main__":
    main()
