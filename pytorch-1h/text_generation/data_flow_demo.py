# 数据流转过程演示
print("=== 数据流转过程演示 ===\n")

# 1. 原始数据
text = "hello world"
print(f"1. 原始文本: '{text}'")

# 2. 字符映射
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"\n2. 字符映射:")
for char, idx in char_to_idx.items():
    print(f"   '{char}' → {idx}")

# 3. 创建训练序列
sequence_length = 3
print(f"\n3. 创建训练序列 (长度={sequence_length}):")
for i in range(len(text) - sequence_length):
    seq = text[i:i + sequence_length]
    target = text[i + sequence_length]
    seq_indices = [char_to_idx[ch] for ch in seq]
    target_idx = char_to_idx[target]
    print(f"   输入: '{seq}' {seq_indices} → 目标: '{target}' {target_idx}")

# 4. 模型处理过程
print(f"\n4. 模型处理过程 (以 'hel' 为例):")
seq = "hel"
seq_indices = [char_to_idx[ch] for ch in seq]
print(f"   输入序列: '{seq}' → 索引: {seq_indices}")
print(f"   嵌入层: 索引 → 向量 (每个字符变成64维向量)")
print(f"   LSTM层: 处理序列，记住 'h' 和 'e' 的信息")
print(f"   输出层: 预测下一个字符的概率")
print(f"   真实目标: 'l' (索引 {char_to_idx['l']})")

# 5. 生成过程
print(f"\n5. 文本生成过程:")
print(f"   起始: 'hel'")
print(f"   预测: 'l' (概率最高)")
print(f"   更新: 'ell'")
print(f"   预测: 'o' (概率最高)")
print(f"   更新: 'llo'")
print(f"   继续...")

print(f"\n=== 关键理解点 ===")
print(f"• 模型学习的是字符序列的模式")
print(f"• 每个字符都有其向量表示")
print(f"• LSTM记住前面的字符信息")
print(f"• 输出是下一个字符的概率分布")
print(f"• 生成时通过采样选择下一个字符")
