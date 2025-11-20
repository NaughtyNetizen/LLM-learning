"""
完整示例：训练一个小型GPT模型

这个脚本展示了从零开始训练一个语言模型的完整流程
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_model.model import GPT
from gpt_model.config import get_config
from training.dataset import create_shakespeare_dataset, prepare_data
from training.trainer import Trainer, get_cosine_schedule_with_warmup
from tokenizer import CharTokenizer


def train_small_gpt():
    """
    训练一个小型GPT模型
    
    步骤:
    1. 准备数据
    2. 创建模型
    3. 设置优化器和学习率调度
    4. 训练模型
    5. 测试生成
    """
    
    print("=" * 70)
    print("训练小型GPT模型")
    print("=" * 70)
    
    # ==================== 1. 设置 ====================
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    
    # 模型配置
    model_type = 'gpt-mini'  # 可选: gpt-micro, gpt-mini, gpt-small
    config = get_config(model_type)
    print(f"模型类型: {model_type}")
    print(f"参数量: ~{sum(p.numel() for p in GPT(config).parameters())/1e6:.1f}M")
    
    # 训练配置
    batch_size = 8
    num_epochs = 5
    learning_rate = 3e-4
    gradient_accumulation_steps = 4
    save_dir = 'checkpoints/shakespeare'
    
    # ==================== 2. 准备数据 ====================
    
    print("\n" + "=" * 70)
    print("准备数据...")
    
    # 下载数据集
    data_path = create_shakespeare_dataset()
    
    # 读取文本以构建词汇表
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 使用新的Tokenizer模块
    print("构建Tokenizer...")
    # 获取所有字符
    chars = sorted(list(set(text)))
    tokenizer = CharTokenizer(chars)
    
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"数据长度: {len(text):,} 字符")
    
    # 保存Tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(save_dir)
    print(f"Tokenizer已保存到: {save_dir}")
    
    # 更新配置的词汇表大小
    config.vocab_size = tokenizer.vocab_size
    
    # 准备数据集
    train_dataset, val_dataset = prepare_data(
        data_path,
        tokenizer,
        seq_len=config.max_seq_len,
        train_split=0.9
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # ==================== 3. 创建模型 ====================
    
    print("\n" + "=" * 70)
    print("创建模型...")
    
    model = GPT(config)
    model = model.to(device)
    
    # ==================== 4. 设置优化器 ====================
    
    print("\n" + "=" * 70)
    print("设置优化器...")
    
    # AdamW优化器（GPT使用的优化器）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),  # GPT-3使用的beta值
        eps=1e-8,
        weight_decay=0.1
    )
    
    # 学习率调度（cosine with warmup）
    num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    num_warmup_steps = num_training_steps // 10  # 10% warmup
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=0.1
    )
    
    print(f"学习率: {learning_rate}")
    print(f"Warmup步数: {num_warmup_steps}")
    print(f"总训练步数: {num_training_steps}")
    
    # ==================== 5. 训练 ====================
    
    print("\n" + "=" * 70)
    print("开始训练...")
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
        use_amp=(device == 'cuda'),  # 只在GPU上使用混合精度
        log_interval=10,
        eval_interval=100,
        save_interval=500,
        save_dir=save_dir
    )
    
    # 训练
    trainer.train(num_epochs=num_epochs, lr_scheduler=lr_scheduler)
    
    # ==================== 6. 测试生成 ====================
    
    print("\n" + "=" * 70)
    print("测试文本生成...")
    
    model.eval()
    
    # 准备prompt
    prompt_text = "To be or not to be"
    prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long).to(device)
    
    print(f"\nPrompt: \"{prompt_text}\"")
    print("\n生成的文本:")
    print("-" * 70)
    
    # 生成文本
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=200,
            temperature=0.8,
            top_k=40
        )
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)
    print("-" * 70)
    
    print("\n✅ 训练完成！")
    print(f"模型和Tokenizer保存在: {save_dir}")


if __name__ == "__main__":
    train_small_gpt()

