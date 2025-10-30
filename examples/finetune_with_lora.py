"""
LoRA微调示例

展示如何使用LoRA高效微调预训练模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_model.model import GPT
from gpt_model.config import get_config
from finetuning.lora import apply_lora_to_model, get_lora_parameters, save_lora_weights
from training.dataset import InMemoryTextDataset
from training.trainer import Trainer, get_cosine_schedule_with_warmup


def finetune_with_lora():
    """
    使用LoRA微调模型的完整示例
    
    步骤:
    1. 加载预训练模型（或创建新模型）
    2. 应用LoRA
    3. 准备微调数据
    4. 微调
    5. 测试
    """
    
    print("=" * 70)
    print("LoRA微调示例")
    print("=" * 70)
    
    # ==================== 1. 设置 ====================
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    
    # ==================== 2. 加载或创建模型 ====================
    
    print("\n" + "=" * 70)
    print("创建模型...")
    
    config = get_config('gpt-mini')
    model = GPT(config)
    
    # 如果有预训练模型，可以这样加载：
    # checkpoint = torch.load('checkpoints/pretrained_model.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"原始模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== 3. 应用LoRA ====================
    
    print("\n" + "=" * 70)
    print("应用LoRA...")
    
    # 通常对attention层应用LoRA效果最好
    model = apply_lora_to_model(
        model,
        target_modules=['W_q', 'W_k', 'W_v', 'W_o'],  # Query, Key, Value, Output projection
        r=8,           # rank（越大参数越多，表达能力越强）
        lora_alpha=16, # 缩放因子
        lora_dropout=0.1
    )
    
    model = model.to(device)
    
    # ==================== 4. 准备微调数据 ====================
    
    print("\n" + "=" * 70)
    print("准备微调数据...")
    
    # 示例：在特定风格的文本上微调
    # 这里使用一些示例文本（实际应用中应该使用真实数据）
    finetune_texts = [
        "Once upon a time, in a land far away, there lived a wise old wizard.",
        "The brave knight rode through the dark forest, seeking the dragon's lair.",
        "In the beginning, there was nothing but chaos and darkness.",
        "The princess gazed out from her tower, dreaming of adventure.",
        "Magic filled the air as the ancient spell was cast.",
    ] * 50  # 重复以创建更多数据
    
    # 简单的字符tokenizer
    all_text = ' '.join(finetune_texts)
    chars = sorted(list(set(all_text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    class CharTokenizer:
        def encode(self, text):
            return [char_to_idx.get(ch, 0) for ch in text]
        
        def decode(self, ids):
            return ''.join([idx_to_char.get(i, '?') for i in ids])
    
    tokenizer = CharTokenizer()
    
    # 创建数据集
    train_dataset = InMemoryTextDataset(
        finetune_texts,
        tokenizer,
        seq_len=128,
        overlap=32
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True
    )
    
    # 小的验证集
    val_texts = finetune_texts[:10]
    val_dataset = InMemoryTextDataset(
        val_texts,
        tokenizer,
        seq_len=128,
        overlap=32
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False
    )
    
    # ==================== 5. 设置优化器 ====================
    
    print("\n" + "=" * 70)
    print("设置优化器（只优化LoRA参数）...")
    
    # 只优化LoRA参数！
    lora_params = get_lora_parameters(model)
    
    optimizer = torch.optim.AdamW(
        lora_params,  # 注意：只传入LoRA参数
        lr=1e-3,      # LoRA通常可以使用更高的学习率
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    
    num_epochs = 10
    num_training_steps = len(train_dataloader) * num_epochs
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=num_training_steps
    )
    
    # ==================== 6. 微调 ====================
    
    print("\n" + "=" * 70)
    print("开始微调...")
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=2,
        use_amp=(device == 'cuda'),
        log_interval=5,
        eval_interval=50,
        save_interval=200,
        save_dir='checkpoints/lora_finetune'
    )
    
    trainer.train(num_epochs=num_epochs, lr_scheduler=lr_scheduler)
    
    # ==================== 7. 保存LoRA权重 ====================
    
    print("\n" + "=" * 70)
    print("保存LoRA权重...")
    
    os.makedirs('checkpoints/lora_finetune', exist_ok=True)
    save_lora_weights(model, 'checkpoints/lora_finetune/lora_weights.pt')
    
    # 对比文件大小
    # 完整模型: ~10MB+
    # 只保存LoRA: ~几百KB
    
    # ==================== 8. 测试生成 ====================
    
    print("\n" + "=" * 70)
    print("测试文本生成...")
    
    model.eval()
    
    prompts = [
        "Once upon",
        "The brave",
        "Magic"
    ]
    
    for prompt in prompts:
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=100,
                temperature=0.8,
                top_k=40
            )
        
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        print(f"\nPrompt: \"{prompt}\"")
        print("-" * 70)
        print(generated_text)
        print("-" * 70)
    
    print("\n✅ LoRA微调完成！")
    
    # 参数效率总结
    print("\n" + "=" * 70)
    print("参数效率总结:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数: {total_params:,}")
    print(f"可训练参数（LoRA）: {trainable_params:,}")
    print(f"训练参数比例: {100 * trainable_params / total_params:.2f}%")
    print(f"节省: {100 * (1 - trainable_params / total_params):.2f}%")


if __name__ == "__main__":
    finetune_with_lora()
