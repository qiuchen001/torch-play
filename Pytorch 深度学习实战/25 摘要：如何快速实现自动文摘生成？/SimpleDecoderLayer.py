import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _generate_causal_mask(self, seq_len):
        """生成因果掩码（下三角矩阵为False，上三角为True）"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def forward(self, x):
        # x 是已经过右移处理的解码器输入 [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        
        # 1. 生成因果掩码
        causal_mask = self._generate_causal_mask(seq_len).to(x.device) # [seq_len, seq_len]
        
        # 2. 带掩码的自注意力
        attn_output, _ = self.self_attention(
            query=x, key=x, value=x, 
            attn_mask=causal_mask, # 关键：传入因果掩码
            is_causal=False # 使用自定义掩码时通常设为False
        )
        
        # 3. 残差连接与层归一化
        x = self.norm1(x + attn_output)
        
        # 4. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

# 假设使用方式
decoder_layer = SimpleDecoderLayer()
batch_size = 2
seq_len = 5
d_model = 512

# 假设 dec_input 是已经过右移和嵌入处理的输入
# 例如，原始目标序列为 [3, 4, 2, 0, 0]（0为填充符）
# 右移后输入为 [1, 3, 4, 2, 0]（1为起始符）
dec_input = torch.randn(batch_size, seq_len, d_model) 

# 前向传播
output = decoder_layer(dec_input) 
print(output.shape) # torch.Size([2, 5, 512])