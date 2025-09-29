#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的自动文摘生成代码
解决了 shift_tokens_right 函数缺少 decoder_start_token_id 参数的问题
"""

import torch
import pandas as pd
import datasets
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right

# 加载模型和分词器
print("Loading BART model and tokenizer...")
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# 创建示例数据集用于演示
print("Creating sample dataset...")
sample_data = {
    'text': [
        "This is a long article about artificial intelligence and machine learning. It discusses various aspects of AI development and its applications in different industries. The field has grown rapidly in recent years with advances in deep learning, neural networks, and computational power.",
        "Climate change is one of the most pressing issues of our time. Scientists around the world are working to understand its impacts and develop solutions. Rising temperatures, melting ice caps, and extreme weather events are becoming more frequent.",
        "The stock market experienced significant volatility today as investors reacted to new economic data and policy announcements from central banks. Trading volumes were higher than usual as uncertainty dominated market sentiment."
    ],
    'summary': [
        "Article discusses AI and machine learning applications in various industries.",
        "Climate change is a pressing global issue requiring scientific solutions.",
        "Stock market volatility due to economic data and policy announcements."
    ]
}

# 创建DataFrame并转换为datasets格式
df = pd.DataFrame(sample_data)
dataset = datasets.Dataset.from_pandas(df)
print(f"Dataset created with {len(dataset)} examples")
print(f"Features: {dataset.features}")

def convert_to_features(example_batch):
    """
    将文本和摘要转换为模型输入格式
    修复了 shift_tokens_right 函数调用中缺少 decoder_start_token_id 参数的问题
    """
    # 编码输入文本
    input_encodings = tokenizer.batch_encode_plus(
        example_batch['text'], 
        padding=True, 
        max_length=1024, 
        truncation=True
    )
    
    # 编码目标摘要
    target_encodings = tokenizer.batch_encode_plus(
        example_batch['summary'], 
        padding=True, 
        max_length=1024, 
        truncation=True
    )
    
    labels = target_encodings['input_ids']
    
    # 添加 decoder_start_token_id 参数
    decoder_input_ids = shift_tokens_right(
        torch.tensor(labels), 
        model.config.pad_token_id,  # 使用确保不是 None 的 pad_token_id
        model.config.decoder_start_token_id  # 这是之前缺少的参数
    )
    
    # 将 pad_token_id 替换为 -100 (忽略标签)
    labels = torch.tensor(labels)
    labels[labels == model.config.pad_token_id] = -100
    
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': decoder_input_ids.tolist(),
        'labels': labels.tolist(),
    }
    
    return encodings

# 应用数据转换
print("Processing dataset...")
dataset = dataset.map(convert_to_features, batched=True)
columns = ['input_ids', 'labels', 'decoder_input_ids', 'attention_mask']
dataset.set_format(type='torch', columns=columns)

print("Dataset preprocessing completed successfully!")
print(f"Dataset columns: {dataset.column_names}")

# 验证数据处理结果
print("\nValidating processed data:")
sample = dataset[0]
print("Sample processed data:")
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: tensor shape {value.shape}")
    else:
        print(f"{key}: {type(value)}")
        
# 可选：测试生成功能
print("\n测试文摘生成功能:")
test_text = sample_data['text'][0]
inputs = tokenizer([test_text], max_length=1024, return_tensors='pt', truncation=True)
summary_ids = model.generate(inputs['input_ids'], max_length=130, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"原文: {test_text[:100]}...")
print(f"生成摘要: {summary}")