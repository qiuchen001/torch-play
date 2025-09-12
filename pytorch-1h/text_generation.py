import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import string

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TextDataset(Dataset):
    """文本数据集类，用于字符级别的文本生成"""
    
    def __init__(self, text, sequence_length=50):
        self.text = text
        self.sequence_length = sequence_length
        
        # 创建字符到索引的映射
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
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

class TextGenerator(nn.Module):
    """基于LSTM的文本生成模型"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(TextGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        # 嵌入
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM前向传播
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # 输出层
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

def prepare_dataset():
    """准备训练数据"""
    # 示例文本数据（您可以用自己的文本替换）
    sample_text = """
    从前有一个小村庄，村庄里住着许多善良的人们。
    他们每天辛勤工作，过着简单而快乐的生活。
    春天来了，花儿开了，鸟儿在枝头歌唱。
    夏天到了，阳光明媚，孩子们在河边玩耍。
    秋天来了，果实累累，农民们收获着丰收的喜悦。
    冬天到了，雪花纷飞，家家户户围炉取暖。
    这就是小村庄的故事，一个充满爱与温暖的地方。
    """
    
    # 清理文本
    text = ''.join([char for char in sample_text if char not in '\n\t'])
    return text

def train_model(model, train_loader, num_epochs=50, learning_rate=0.001, device='cpu'):
    """训练模型"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # 初始化隐藏状态
            hidden = model.init_hidden(sequences.size(0), device)
            
            # 前向传播
            outputs, _ = model(sequences, hidden)
            
            # 计算损失（只使用最后一个时间步的输出）
            loss = criterion(outputs[:, -1, :], targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def generate_text(model, dataset, start_text, length=100, temperature=0.8, device='cpu'):
    """生成文本"""
    model.eval()
    model.to(device)
    
    # 将起始文本转换为索引
    start_indices = [dataset.char_to_idx[ch] for ch in start_text if ch in dataset.char_to_idx]
    if not start_indices:
        start_indices = [0]  # 如果起始文本为空，使用第一个字符
    
    generated_text = start_text
    current_sequence = torch.tensor([start_indices], dtype=torch.long).to(device)
    
    # 确保序列长度正确
    if current_sequence.size(1) < dataset.sequence_length:
        # 如果起始序列太短，用零填充
        padding = torch.zeros(1, dataset.sequence_length - current_sequence.size(1), dtype=torch.long).to(device)
        current_sequence = torch.cat([padding, current_sequence], dim=1)
    elif current_sequence.size(1) > dataset.sequence_length:
        # 如果起始序列太长，截取最后的部分
        current_sequence = current_sequence[:, -dataset.sequence_length:]
    
    with torch.no_grad():
        hidden = model.init_hidden(1, device)
        
        for _ in range(length):
            # 前向传播
            output, hidden = model(current_sequence, hidden)
            
            # 获取最后一个时间步的输出
            last_output = output[:, -1, :] / temperature
            
            # 应用softmax并采样
            probabilities = F.softmax(last_output, dim=-1)
            next_char_idx = torch.multinomial(probabilities, 1).item()
            
            # 将新字符添加到生成文本
            next_char = dataset.idx_to_char[next_char_idx]
            generated_text += next_char
            
            # 更新序列（滑动窗口）
            current_sequence = torch.cat([current_sequence[:, 1:], torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)
    
    return generated_text

def main():
    """主函数"""
    print("=== 文本生成任务 ===")
    
    # 准备数据
    text = prepare_dataset()
    print(f"训练文本长度: {len(text)} 字符")
    print(f"前100个字符: {text[:100]}...")
    
    # 创建数据集
    sequence_length = 50
    dataset = TextDataset(text, sequence_length)
    print(f"词汇表大小: {dataset.vocab_size}")
    print(f"训练序列数量: {len(dataset)}")
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = TextGenerator(
        vocab_size=dataset.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 训练模型
    print("\n开始训练...")
    train_model(model, train_loader, num_epochs=100, learning_rate=0.001, device=device)
    
    # 生成文本
    print("\n=== 文本生成结果 ===")
    start_texts = ["从前", "春天", "夏天", "秋天", "冬天"]
    
    for start_text in start_texts:
        generated = generate_text(model, dataset, start_text, length=50, temperature=0.8, device=device)
        print(f"起始文本: '{start_text}'")
        print(f"生成文本: {generated}")
        print("-" * 50)
    
    # 保存模型
    torch.save(model.state_dict(), 'text_generator_model.pth')
    print("模型已保存为 'text_generator_model.pth'")

if __name__ == "__main__":
    main()
