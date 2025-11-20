"""
训练器

实现完整的训练循环，包括：
- 梯度累积
- 学习率调度
- 混合精度训练
- 梯度裁剪
- 训练监控
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import math
from typing import Optional, Dict
from tqdm import tqdm


class Trainer:
    """
    GPT模型训练器
    """
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        device='cuda',
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        use_amp=False,
        log_interval=10,
        eval_interval=100,
        save_interval=1000,
        save_dir='checkpoints'
    ):
        """
        初始化训练器
        
        参数:
            model: GPT模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            optimizer: 优化器
            device: 训练设备
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 最大梯度范数（用于梯度裁剪）
            use_amp: 是否使用混合精度训练
            log_interval: 日志记录间隔
            eval_interval: 验证间隔
            save_interval: 保存间隔
            save_dir: checkpoint保存目录
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.save_dir = save_dir
        
        # 混合精度训练
        self.scaler = GradScaler() if use_amp else None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 创建保存目录
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def train_epoch(self, lr_scheduler=None):
        """
        训练一个epoch
        
        参数:
            lr_scheduler: 学习率调度器（可选）
        
        返回:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            leave=False
        )
        
        for batch_idx, (input_ids, targets) in enumerate(progress_bar):
            # 移到设备
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播（使用混合精度）
            if self.use_amp:
                with autocast():
                    loss, _, _ = self.model(input_ids, targets)
                    loss = loss / self.gradient_accumulation_steps
            else:
                loss, _, _ = self.model(input_ids, targets)
                loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 累积梯度后更新
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # 优化器步骤
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # 学习率调度
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # 验证
                if self.global_step % self.eval_interval == 0:
                    val_loss = self.evaluate()
                    self.model.train()
                    
                    # 保存最佳模型
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best_model.pt')
                
                # 定期保存
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
            
            # 记录损失
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # 更新进度条
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'lr': f'{current_lr:.6f}'
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self):
        """
        在验证集上评估
        
        返回:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for input_ids, targets in self.val_dataloader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            loss, _, _ = self.model(input_ids, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        print(f"\n[Step {self.global_step}] Val Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return avg_loss
    
    def train(self, num_epochs, lr_scheduler=None):
        """
        训练多个epochs
        
        参数:
            num_epochs: 训练轮数
            lr_scheduler: 学习率调度器
        """
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"训练批次数: {len(self.train_dataloader)}")
        print(f"验证批次数: {len(self.val_dataloader)}")
        print(f"梯度累积步数: {self.gradient_accumulation_steps}")
        print(f"混合精度训练: {self.use_amp}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 训练一个epoch
            train_loss = self.train_epoch(lr_scheduler)
            
            # 评估
            val_loss = self.evaluate()
            
            # 记录
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print(f"  New Best Val Loss! Saved best_model.pt")
            
            print(f"  Best Val Loss: {self.best_val_loss:.4f}")
            
            elapsed_time = time.time() - start_time
            print(f"  Elapsed Time: {elapsed_time/60:.2f} min")
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"总时间: {(time.time() - start_time)/60:.2f} min")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print("=" * 60)
    
    def save_checkpoint(self, filename):
        """
        保存checkpoint
        
        参数:
            filename: 文件名
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.model.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        import os
        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"保存checkpoint到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载checkpoint
        
        参数:
            checkpoint_path: checkpoint路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"从 {checkpoint_path} 加载checkpoint")
        print(f"  Global Step: {self.global_step}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr_ratio=0.1
):
    """
    余弦学习率调度（带warmup）
    
    参数:
        optimizer: 优化器
        num_warmup_steps: warmup步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率比例
    
    返回:
        学习率调度器
    """
    def lr_lambda(current_step):
        # Warmup阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ====================== 演示代码 ======================

def demo_training():
    """演示训练流程"""
    print("=" * 60)
    print("训练流程演示")
    print("=" * 60)
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from llm_learning.model.model import GPT
    from llm_learning.model.config import get_config
    from llm_learning.data.dataset import TextDataset
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 创建模型
    print("\n创建模型...")
    config = get_config('gpt-micro')
    model = GPT(config)
    
    # 创建数据集
    print("\n创建模拟数据...")
    import numpy as np
    np.random.seed(42)
    train_data = np.random.randint(0, config.vocab_size, size=10000, dtype=np.int64)
    val_data = np.random.randint(0, config.vocab_size, size=2000, dtype=np.int64)
    
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/train_demo.npy', train_data)
    np.save('data/processed/val_demo.npy', val_data)
    
    train_dataset = TextDataset('data/processed/train_demo.npy', seq_len=64)
    val_dataset = TextDataset('data/processed/val_demo.npy', seq_len=64)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # 学习率调度
    num_epochs = 2
    num_training_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=2,
        use_amp=False,  # CPU不支持AMP
        log_interval=10,
        eval_interval=50,
        save_interval=100
    )
    
    # 训练
    trainer.train(num_epochs=num_epochs, lr_scheduler=lr_scheduler)
    
    print("\n✅ 演示完成！")


if __name__ == "__main__":
    demo_training()
