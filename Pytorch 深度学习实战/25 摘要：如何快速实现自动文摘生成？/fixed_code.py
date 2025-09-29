#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构后的自动文摘生成代码
将各个功能模块用函数封装，使职责更清晰
"""

import torch
import pandas as pd
import datasets
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right


def load_model_and_tokenizer():
    """加载 BART 模型和分词器"""
    print("Loading BART model and tokenizer...")
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    return model, tokenizer


def create_sample_dataset():
    """创建示例数据集用于演示"""
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
    return dataset, sample_data


def convert_to_features(example_batch, model, tokenizer):
    """
    将文本和摘要转换为模型输入格式
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
        model.config.pad_token_id,
        model.config.decoder_start_token_id
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


def preprocess_dataset(dataset, model, tokenizer):
    """预处理数据集"""
    print("Processing dataset...")
    
    def convert_fn(example_batch):
        return convert_to_features(example_batch, model, tokenizer)
    
    dataset = dataset.map(convert_fn, batched=True)
    columns = ['input_ids', 'labels', 'decoder_input_ids', 'attention_mask']
    dataset.set_format(type='torch', columns=columns)
    
    print("Dataset preprocessing completed successfully!")
    print(f"Dataset columns: {dataset.column_names}")
    return dataset


def validate_processed_data(dataset):
    """验证处理后的数据"""
    print("\nValidating processed data:")
    sample = dataset[0]
    print("Sample processed data:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: tensor shape {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    return sample


def generate_summary(text, model, tokenizer):
    """生成文摘"""
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=130, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def main():
    """主函数：执行自动文摘生成流程"""
    # 1. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. 创建示例数据集
    dataset, sample_data = create_sample_dataset()
    
    # 3. 预处理数据集
    processed_dataset = preprocess_dataset(dataset, model, tokenizer)
    
    # 4. 验证处理后的数据
    validate_processed_data(processed_dataset)
    
    # 5. 测试文摘生成功能
    print("\n测试文摘生成功能:")
    test_text = sample_data['text'][0]
    summary = generate_summary(test_text, model, tokenizer)
    print(f"原文: {test_text[:100]}...")
    print(f"生成摘要: {summary}")
    
    print("\n自动文摘生成流程完成！")


if __name__ == "__main__":
    main()