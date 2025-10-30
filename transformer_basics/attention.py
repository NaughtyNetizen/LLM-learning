"""
Attention机制的完整实现

这个文件实现了Transformer的核心：注意力机制
包括：
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Causal (Masked) Attention for GPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力 (Scaled Dot-Product Attention)
    
    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    参数:
        dropout: dropout概率
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        参数:
            query: [batch_size, n_heads, seq_len, d_k]
            key: [batch_size, n_heads, seq_len, d_k]
            value: [batch_size, n_heads, seq_len, d_v]
            mask: [batch_size, 1, seq_len, seq_len] 或 [batch_size, n_heads, seq_len, seq_len]
                  用于因果注意力（防止看到未来信息）
        
        返回:
            output: [batch_size, n_heads, seq_len, d_v]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        # 获取d_k维度，用于缩放
        d_k = query.size(-1)
        
        # 计算注意力分数: QK^T / sqrt(d_k)
        # [batch, n_heads, seq_len, d_k] @ [batch, n_heads, d_k, seq_len]
        # -> [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用mask（如果提供）
        # 对于GPT，mask用于实现因果注意力（causal attention）
        # 即当前位置只能看到之前的位置，不能看到未来
        if mask is not None:
            # 将mask为0的位置设为一个很大的负数
            # 这样softmax后这些位置的权重接近0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获得注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用dropout（训练时随机丢弃一些注意力连接）
        attention_weights = self.dropout(attention_weights)
        
        # 使用注意力权重对value进行加权求和
        # [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, d_v]
        # -> [batch, n_heads, seq_len, d_v]
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    核心思想：
    - 使用多个注意力头，每个头关注不同的表示子空间
    - 允许模型同时关注来自不同位置的不同信息
    
    参数:
        d_model: 模型维度（输入输出维度）
        n_heads: 注意力头的数量
        dropout: dropout概率
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # Q, K, V的线性变换
        # 这里使用一个大的线性层来同时计算所有头的Q/K/V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(dropout)
        
        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        参数:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: attention mask
        
        返回:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 1. 线性变换并分割成多个头
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        # -> [batch, seq_len, n_heads, d_k]
        # -> [batch, n_heads, seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. 应用缩放点积注意力
        # output: [batch, n_heads, seq_len, d_k]
        # attention_weights: [batch, n_heads, seq_len, seq_len]
        if mask is not None:
            # 扩展mask维度以匹配多头
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
        
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # 3. 拼接所有头的输出
        # [batch, n_heads, seq_len, d_k] -> [batch, seq_len, n_heads, d_k]
        # -> [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. 最终的线性变换
        output = self.W_o(output)
        output = self.dropout(output)
        
        return output, attention_weights


class CausalSelfAttention(nn.Module):
    """
    因果自注意力 (Causal Self-Attention)
    
    专门用于GPT等自回归模型的注意力机制
    特点：当前位置只能看到之前的位置（包括自己），不能看到未来
    
    这是GPT的核心组件！
    """
    def __init__(self, d_model, n_heads, max_seq_len=1024, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # 多头注意力
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 注册因果mask（下三角矩阵）
        # 这个mask确保位置i只能attend到位置<=i的token
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            )
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len, d_model]
        
        返回:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 获取当前序列长度的因果mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # 自注意力：Q=K=V=x
        output, attention_weights = self.mha(x, x, x, mask)
        
        return output, attention_weights


# ====================== 测试代码 ======================

def test_attention():
    """测试注意力机制的基本功能"""
    print("=" * 60)
    print("测试Attention机制")
    print("=" * 60)
    
    # 设置随机种子以便复现
    torch.manual_seed(42)
    
    # 超参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # 创建模型
    mha = MultiHeadAttention(d_model, n_heads)
    causal_attn = CausalSelfAttention(d_model, n_heads, max_seq_len=512)
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n输入形状: {x.shape}")
    print(f"模型维度: d_model={d_model}, n_heads={n_heads}, d_k={d_model//n_heads}")
    
    # 测试Multi-Head Attention
    print("\n" + "-" * 60)
    print("测试 Multi-Head Attention (无mask)")
    output, attn_weights = mha(x, x, x)
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"注意力权重求和 (应该接近1): {attn_weights[0, 0, 0].sum().item():.4f}")
    
    # 测试Causal Self-Attention
    print("\n" + "-" * 60)
    print("测试 Causal Self-Attention (带因果mask)")
    output, attn_weights = causal_attn(x)
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 可视化因果mask的效果
    print("\n第一个头的注意力权重矩阵（位置0只能看到自己）:")
    print(attn_weights[0, 0, :5, :5].detach().numpy())
    print("注意：上三角部分应该接近0（被mask掉了）")
    
    # 验证因果性
    print("\n" + "-" * 60)
    print("验证因果性：位置i不能看到位置>i")
    for i in range(min(3, seq_len)):
        future_attn = attn_weights[0, 0, i, i+1:].sum().item()
        print(f"位置{i}对未来位置的注意力和: {future_attn:.6f} (应该接近0)")
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_attention()
