#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构后的自动文摘生成代码
将各个功能模块用函数封装，使职责更清晰
包含训练集和验证集，使用标准的 Hugging Face 数据集格式
"""

import torch
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
    """创建示例数据集，使用标准的 Hugging Face 数据集格式"""
    print("Creating sample dataset with standard Hugging Face format...")
    
    # 训练集数据
    train_data = {
        'text': [
            "This is a long article about artificial intelligence and machine learning. It discusses various aspects of AI development and its applications in different industries. The field has grown rapidly in recent years with advances in deep learning, neural networks, and computational power.",
            "Climate change is one of the most pressing issues of our time. Scientists around the world are working to understand its impacts and develop solutions. Rising temperatures, melting ice caps, and extreme weather events are becoming more frequent.",
            "The stock market experienced significant volatility today as investors reacted to new economic data and policy announcements from central banks. Trading volumes were higher than usual as uncertainty dominated market sentiment.",
            "Renewable energy sources like solar and wind power are becoming increasingly important in the global energy mix. Countries around the world are investing heavily in clean energy infrastructure to reduce carbon emissions.",
            "The healthcare industry is undergoing a digital transformation with the adoption of telemedicine, electronic health records, and AI-powered diagnostic tools. These technologies are improving patient care and operational efficiency.",
            "Education systems worldwide are adapting to online learning platforms and digital tools. The pandemic accelerated this shift, forcing institutions to rethink traditional teaching methods.",
            "E-commerce has revolutionized the retail industry, with online shopping becoming the preferred method for many consumers. This trend is expected to continue as technology improves.",
            "Autonomous vehicles are advancing rapidly, with major tech and automotive companies investing billions in self-driving technology. Safety and regulatory challenges remain the biggest hurdles.",
            "Blockchain technology is finding applications beyond cryptocurrencies, including supply chain management, voting systems, and digital identity verification.",
            "Space exploration is entering a new era with private companies like SpaceX and Blue Origin leading the charge. Mars colonization and lunar bases are becoming realistic goals."
        ],
        'summary': [
            "Article discusses AI and machine learning applications in various industries.",
            "Climate change is a pressing global issue requiring scientific solutions.",
            "Stock market volatility due to economic data and policy announcements.",
            "Renewable energy investments growing to reduce carbon emissions worldwide.",
            "Healthcare digital transformation improving patient care with technology.",
            "Education systems adapting to online learning and digital tools.",
            "E-commerce revolutionizing retail with online shopping growth.",
            "Autonomous vehicles advancing despite safety and regulatory challenges.",
            "Blockchain applications expanding beyond cryptocurrencies.",
            "Private companies leading new era of space exploration."
        ]
    }
    
    # 验证集数据
    validation_data = {
        'text': [
            "Quantum computing represents a paradigm shift in computational power. Unlike classical computers that use bits, quantum computers use qubits that can exist in multiple states simultaneously.",
            "The Internet of Things (IoT) connects everyday devices to the internet, enabling data collection and remote control. Smart homes, cities, and industrial applications are driving IoT adoption.",
            "5G technology promises faster speeds, lower latency, and greater connectivity. This will enable new applications in areas like autonomous vehicles and augmented reality.",
            "Cybersecurity threats are evolving rapidly as more aspects of life move online. Organizations must invest in robust security measures to protect sensitive data.",
            "Virtual and augmented reality technologies are transforming entertainment, education, and training. These immersive experiences offer new ways to interact with digital content."
        ],
        'summary': [
            "Quantum computing uses qubits for unprecedented computational power.",
            "IoT connects devices for smart homes, cities, and industrial applications.",
            "5G enables faster connectivity for autonomous vehicles and AR applications.",
            "Cybersecurity threats evolving as more life moves online.",
            "VR and AR transforming entertainment, education, and training experiences."
        ]
    }
    
    # 创建训练集和验证集的 Dataset 对象
    train_dataset = datasets.Dataset.from_dict(train_data)
    validation_dataset = datasets.Dataset.from_dict(validation_data)
    
    # 创建包含 train 和 validation 分区的完整数据集
    dataset_dict = datasets.DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })
    
    print(f"Dataset created with train set: {len(dataset_dict['train'])} examples")
    print(f"Dataset created with validation set: {len(dataset_dict['validation'])} examples")
    print(f"Dataset features: {dataset_dict['train'].features}")
    
    return dataset_dict


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


def preprocess_dataset(dataset_dict, model, tokenizer):
    """预处理数据集字典，分别处理训练集和验证集"""
    print("Processing dataset dictionary...")
    
    def convert_fn(example_batch):
        return convert_to_features(example_batch, model, tokenizer)
    
    # 分别处理训练集和验证集
    train_dataset = dataset_dict['train'].map(convert_fn, batched=True)
    validation_dataset = dataset_dict['validation'].map(convert_fn, batched=True)
    
    # 设置格式
    columns = ['input_ids', 'labels', 'decoder_input_ids', 'attention_mask']
    train_dataset.set_format(type='torch', columns=columns)
    validation_dataset.set_format(type='torch', columns=columns)
    
    # 创建新的数据集字典
    processed_dataset_dict = datasets.DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })
    
    print("Dataset preprocessing completed successfully!")
    print(f"Train set columns: {train_dataset.column_names}")
    print(f"Validation set columns: {validation_dataset.column_names}")
    
    return processed_dataset_dict


def validate_processed_data(dataset_dict):
    """验证处理后的训练集和验证集数据"""
    print("\nValidating processed data:")
    
    print("Training set sample:")
    train_sample = dataset_dict['train'][0]
    for key, value in train_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: tensor shape {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    print("Validation set sample:")
    val_sample = dataset_dict['validation'][0]
    for key, value in val_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: tensor shape {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    return train_sample, val_sample


def generate_summary(text, model, tokenizer):
    """生成文摘"""
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=130, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def train(model, dataset):
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
    training_args = Seq2SeqTrainingArguments(
        output_dir='./models/bart-summarizer',# 模型输出目录
        num_train_epochs=1, # 训练轮数
        per_device_train_batch_size=1, # 训练过程bach_size
        per_device_eval_batch_size=1, # 评估过程bach_size
        warmup_steps=500, # 学习率相关参数
        weight_decay=0.01, # 学习率相关参数
        logging_dir='./logs', # 日志目录
    )

    trainer = Seq2SeqTrainer(
        model=model,                       
        args=training_args,                  
        train_dataset=dataset['train'],        
        eval_dataset=dataset['validation']   
    )

    trainer.train()


def main():
    """主函数：执行自动文摘生成流程"""
    # 1. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. 创建示例数据集（包含训练集和验证集）
    dataset_dict = create_sample_dataset()
    
    # 3. 预处理数据集
    processed_dataset_dict = preprocess_dataset(dataset_dict, model, tokenizer)
    
    # 4. 验证处理后的数据
    validate_processed_data(processed_dataset_dict)
    
    # 5. 测试文摘生成功能（使用训练集和验证集的样本）
    print("\n测试文摘生成功能:")
    
    # 测试训练集样本
    print("训练集样本测试:")
    train_text = dataset_dict['train']['text'][0]  # 第一个训练样本
    train_summary = generate_summary(train_text, model, tokenizer)
    print(f"原文: {train_text[:100]}...")
    print(f"生成摘要: {train_summary}")
    
    # 测试验证集样本
    print("\n验证集样本测试:")
    val_text = dataset_dict['validation']['text'][0]  # 第一个验证样本
    val_summary = generate_summary(val_text, model, tokenizer)
    print(f"原文: {val_text[:100]}...")
    print(f"生成摘要: {val_summary}")
    
    print("\n自动文摘生成流程完成！包含训练集和验证集。")

    # 6. 训练模型
    train(model, processed_dataset_dict)


if __name__ == "__main__":
    main()