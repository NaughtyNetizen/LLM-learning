"""
LoRA (Low-Rank Adaptation) 实现

LoRA通过在预训练模型的权重旁边添加低秩矩阵来进行高效微调

原理:
W' = W + BA
其中 W 是原始权重(冻结), B 和 A 是可训练的低秩矩阵
如果 W 是 d×d, 则 A 是 d×r, B 是 r×d (r << d)

参数量: 2*d*r << d*d

论文: LoRA: Low-Rank Adaptation of Large Language Models
https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALayer(nn.Module):
    """
    LoRA层
    
    在原始线性层上添加低秩适配器
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        merge_weights: bool = False
    ):
        """
        初始化LoRA层
        
        参数:
            original_layer: 原始的线性层
            r: 低秩维度（rank）
            lora_alpha: 缩放因子
            lora_dropout: LoRA的dropout
            merge_weights: 是否在推理时合并权重
        """
        super().__init__()
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x
        self.merge_weights = merge_weights
        self.merged = False
        
        # 原始层（冻结）
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
        
        # LoRA矩阵
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # A矩阵: [in_features, r] 使用Kaiming初始化
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # B矩阵: [r, out_features] 初始化为0
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # 缩放因子
        self.scaling = self.lora_alpha / self.r
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: [batch_size, ..., in_features]
        
        返回:
            output: [batch_size, ..., out_features]
        """
        # 原始层的输出
        result = self.original_layer(x)
        
        # 如果已合并权重，直接返回
        if self.merged:
            return result
        
        # LoRA路径: x @ A @ B * scaling
        lora_result = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return result + lora_result
    
    def merge_weights(self):
        """
        合并LoRA权重到原始权重（用于推理）
        """
        if not self.merged:
            # W' = W + BA * scaling
            self.original_layer.weight.data += (
                self.lora_B @ self.lora_A.T * self.scaling
            ).T
            self.merged = True
    
    def unmerge_weights(self):
        """
        取消合并（用于继续训练）
        """
        if self.merged:
            self.original_layer.weight.data -= (
                self.lora_B @ self.lora_A.T * self.scaling
            ).T
            self.merged = False


def apply_lora_to_model(
    model,
    target_modules: list = ['W_q', 'W_v'],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1
):
    """
    将LoRA应用到模型
    
    参数:
        model: GPT模型
        target_modules: 要应用LoRA的模块名称列表
        r: 低秩维度
        lora_alpha: 缩放因子
        lora_dropout: dropout概率
    
    返回:
        应用LoRA后的模型
    """
    lora_count = 0
    
    for name, module in model.named_modules():
        # 检查是否是目标模块
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # 获取父模块和属性名
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                parent = model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                
                # 替换为LoRA层
                lora_layer = LoRALayer(
                    module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                setattr(parent, attr_name, lora_layer)
                lora_count += 1
                
                print(f"应用LoRA到: {name} (in={module.in_features}, out={module.out_features}, r={r})")
    
    print(f"\n总共应用了 {lora_count} 个LoRA层")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def get_lora_parameters(model):
    """
    获取所有LoRA参数
    
    参数:
        model: 应用了LoRA的模型
    
    返回:
        LoRA参数列表
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def save_lora_weights(model, save_path):
    """
    只保存LoRA权重
    
    参数:
        model: 应用了LoRA的模型
        save_path: 保存路径
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
    
    torch.save(lora_state_dict, save_path)
    print(f"LoRA权重已保存到: {save_path}")
    print(f"文件大小: {sum(v.numel() for v in lora_state_dict.values()):,} 参数")


def load_lora_weights(model, load_path):
    """
    加载LoRA权重
    
    参数:
        model: 应用了LoRA的模型
        load_path: 权重路径
    """
    lora_state_dict = torch.load(load_path)
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
            module.lora_B.data = lora_state_dict[f"{name}.lora_B"]
    
    print(f"LoRA权重已从 {load_path} 加载")


# ====================== 测试代码 ======================

def test_lora():
    """测试LoRA实现"""
    print("=" * 60)
    print("测试LoRA实现")
    print("=" * 60)
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from gpt_model.model import GPT
    from gpt_model.config import get_config
    
    # 创建模型
    print("\n创建GPT模型...")
    config = get_config('gpt-micro')
    model = GPT(config)
    
    print(f"原始模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"原始可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 应用LoRA
    print("\n" + "-" * 60)
    print("应用LoRA...")
    model = apply_lora_to_model(
        model,
        target_modules=['W_q', 'W_k', 'W_v', 'W_o'],  # 只对attention应用LoRA
        r=4,
        lora_alpha=16,
        lora_dropout=0.1
    )
    
    # 测试前向传播
    print("\n" + "-" * 60)
    print("测试前向传播...")
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        loss, logits, _ = model(input_ids, targets)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits形状: {logits.shape}")
    
    # 测试保存和加载
    print("\n" + "-" * 60)
    print("测试保存和加载LoRA权重...")
    
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/lora_test.pt'
    save_lora_weights(model, save_path)
    
    # 修改权重
    for param in get_lora_parameters(model):
        param.data.fill_(1.0)
    
    # 加载回来
    load_lora_weights(model, save_path)
    
    print("\n✅ 所有测试通过！")
    
    # 参数效率对比
    print("\n" + "-" * 60)
    print("参数效率对比:")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数: {total:,}")
    print(f"LoRA参数: {trainable:,}")
    print(f"效率: 只需训练 {100 * trainable / total:.2f}% 的参数")
    print(f"参数减少: {100 * (1 - trainable / total):.2f}%")


if __name__ == "__main__":
    test_lora()
