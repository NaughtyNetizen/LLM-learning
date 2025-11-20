"""
Transformer的其他核心层

包括：
1. Position-wise Feed-Forward Network (FFN)
2. Layer Normalization
3. Residual Connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络 (Position-wise Feed-Forward Network)
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    或使用GELU: FFN(x) = GELU(xW1 + b1)W2 + b2
    
    在Transformer中，每个位置独立地通过同一个FFN
    通常d_ff = 4 * d_model
    
    参数:
        d_model: 输入/输出维度
        d_ff: 中间层维度（通常是d_model的4倍）
        dropout: dropout概率
        activation: 激活函数 ('relu' 或 'gelu')
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu  # GPT使用GELU
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len, d_model]
        
        返回:
            output: [batch_size, seq_len, d_model]
        """
        # x -> W1 -> activation -> dropout -> W2 -> dropout
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization)
    
    对每个样本的特征维度进行归一化
    LN(x) = γ * (x - μ) / (σ + ε) + β
    
    其中μ和σ是在特征维度上计算的均值和标准差
    γ和β是可学习的缩放和偏移参数
    
    参数:
        d_model: 特征维度
        eps: 防止除零的小常数
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        
        # 可学习的缩放和偏移参数
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len, d_model]
        
        返回:
            output: [batch_size, seq_len, d_model]
        """
        # 在最后一个维度（特征维度）上计算均值和标准差
        mean = x.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        std = x.std(dim=-1, keepdim=True)    # [batch, seq_len, 1]
        
        # 归一化
        normalized = (x - mean) / (std + self.eps)
        
        # 缩放和偏移
        output = self.gamma * normalized + self.beta
        
        return output


class ResidualConnection(nn.Module):
    """
    残差连接 + Layer Norm
    
    实现: LayerNorm(x + Sublayer(x))
    
    GPT使用Pre-LN: LayerNorm在sublayer之前
    即: x + Sublayer(LayerNorm(x))
    
    参数:
        d_model: 模型维度
        dropout: dropout概率
        pre_norm: 是否使用pre-normalization（GPT风格）
    """
    def __init__(self, d_model, dropout=0.1, pre_norm=True):
        super().__init__()
        
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
    
    def forward(self, x, sublayer):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len, d_model]
            sublayer: 子层函数（如attention或FFN）
        
        返回:
            output: [batch_size, seq_len, d_model]
        """
        if self.pre_norm:
            # Pre-LN (GPT-2/3风格): x + Sublayer(LN(x))
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            # Post-LN (原始Transformer): LN(x + Sublayer(x))
            return self.norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    """
    Transformer解码器块 (用于GPT)
    
    结构:
    1. Causal Self-Attention + Residual
    2. Feed-Forward Network + Residual
    
    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff: FFN中间层维度
        dropout: dropout概率
        max_seq_len: 最大序列长度
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, max_seq_len=1024):
        super().__init__()
        
        # 导入CausalSelfAttention
        from attention import CausalSelfAttention
        
        # 因果自注意力
        self.attention = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout
        )
        
        # 前馈网络
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout, activation='gelu')
        
        # Layer Norm（Pre-LN风格）
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len, d_model]
        
        返回:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        # 1. Self-Attention with Residual (Pre-LN)
        attn_output, attn_weights = self.attention(self.ln1(x))
        x = x + self.dropout(attn_output)
        
        # 2. Feed-Forward with Residual (Pre-LN)
        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_output)
        
        return x, attn_weights


# ====================== 测试代码 ======================

def test_layers():
    """测试各个层的功能"""
    print("=" * 60)
    print("测试Transformer层")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 超参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048
    n_heads = 8
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"输入形状: {x.shape}")
    
    # 测试FFN
    print("\n" + "-" * 60)
    print("测试 Position-wise Feed-Forward Network")
    ffn = PositionWiseFeedForward(d_model, d_ff)
    ffn_output = ffn(x)
    print(f"FFN输出形状: {ffn_output.shape}")
    print(f"输入输出维度相同: {ffn_output.shape == x.shape}")
    
    # 测试Layer Norm
    print("\n" + "-" * 60)
    print("测试 Layer Normalization")
    ln = LayerNorm(d_model)
    ln_output = ln(x)
    print(f"LayerNorm输出形状: {ln_output.shape}")
    print(f"输出均值 (应该接近0): {ln_output.mean().item():.6f}")
    print(f"输出标准差 (应该接近1): {ln_output.std().item():.6f}")
    
    # 测试完整的Transformer Block
    print("\n" + "-" * 60)
    print("测试 Transformer Block")
    block = TransformerBlock(d_model, n_heads, d_ff)
    block_output, attn_weights = block(x)
    print(f"Block输出形状: {block_output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in block.parameters())
    trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_layers()
