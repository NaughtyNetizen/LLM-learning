"""
嵌入层实现

包括：
1. Token Embedding
2. Positional Embedding
3. 组合的输入嵌入
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token嵌入层
    
    将token ID转换为dense vector
    
    参数:
        vocab_size: 词汇表大小
        d_model: 嵌入维度
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len] token IDs
        
        返回:
            embeddings: [batch_size, seq_len, d_model]
        """
        # 嵌入并缩放（某些实现会乘以sqrt(d_model)）
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """
    位置嵌入层
    
    有两种常见实现：
    1. 固定的正弦位置编码（原始Transformer）
    2. 可学习的位置嵌入（GPT-2/3）
    
    这里实现可学习的位置嵌入（GPT风格）
    
    参数:
        max_seq_len: 最大序列长度
        d_model: 嵌入维度
    """
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        
        self.embedding = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len, d_model] 或 [batch_size, seq_len]
        
        返回:
            pos_embeddings: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 创建位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        return self.embedding(positions)


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码（原始Transformer论文）
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    优点：
    - 不需要学习参数
    - 可以推广到训练时未见过的序列长度
    
    参数:
        d_model: 嵌入维度
        max_seq_len: 最大序列长度
        dropout: dropout概率
    """
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵 [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # 计算除数项
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        # 应用sin和cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度并注册为buffer（不会被优化器更新）
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len, d_model]
        
        返回:
            output: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 添加位置编码
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class InputEmbedding(nn.Module):
    """
    组合的输入嵌入
    
    Token Embedding + Positional Embedding
    
    参数:
        vocab_size: 词汇表大小
        d_model: 嵌入维度
        max_seq_len: 最大序列长度
        dropout: dropout概率
        pos_encoding_type: 位置编码类型 ('learned' 或 'sinusoidal')
    """
    def __init__(
        self, 
        vocab_size, 
        d_model, 
        max_seq_len=1024, 
        dropout=0.1,
        pos_encoding_type='learned'
    ):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        
        if pos_encoding_type == 'learned':
            # GPT风格：可学习的位置嵌入
            self.pos_embedding = PositionalEmbedding(max_seq_len, d_model)
        elif pos_encoding_type == 'sinusoidal':
            # 原始Transformer：固定的正弦位置编码
            self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            raise ValueError(f"不支持的位置编码类型: {pos_encoding_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding_type = pos_encoding_type
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len] token IDs
        
        返回:
            embeddings: [batch_size, seq_len, d_model]
        """
        # Token嵌入
        token_emb = self.token_embedding(x)
        
        # 添加位置信息
        if self.pos_encoding_type == 'learned':
            pos_emb = self.pos_embedding(x)
            embeddings = token_emb + pos_emb
            embeddings = self.dropout(embeddings)
        else:  # sinusoidal
            embeddings = self.pos_embedding(token_emb)
        
        return embeddings


# ====================== 测试代码 ======================

def test_embeddings():
    """测试嵌入层"""
    print("=" * 60)
    print("测试嵌入层")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 超参数
    batch_size = 2
    seq_len = 10
    vocab_size = 50000
    d_model = 512
    max_seq_len = 1024
    
    # 创建随机token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Token IDs形状: {token_ids.shape}")
    print(f"Token IDs范围: [{token_ids.min()}, {token_ids.max()}]")
    
    # 测试Token Embedding
    print("\n" + "-" * 60)
    print("测试 Token Embedding")
    token_emb = TokenEmbedding(vocab_size, d_model)
    token_output = token_emb(token_ids)
    print(f"Token嵌入输出形状: {token_output.shape}")
    
    # 测试Learned Positional Embedding
    print("\n" + "-" * 60)
    print("测试 Learned Positional Embedding (GPT风格)")
    pos_emb = PositionalEmbedding(max_seq_len, d_model)
    pos_output = pos_emb(token_ids)
    print(f"位置嵌入输出形状: {pos_output.shape}")
    
    # 测试Sinusoidal Positional Encoding
    print("\n" + "-" * 60)
    print("测试 Sinusoidal Positional Encoding (原始Transformer)")
    sin_pos = SinusoidalPositionalEncoding(d_model, max_seq_len)
    sin_output = sin_pos(token_output)
    print(f"正弦位置编码输出形状: {sin_output.shape}")
    
    # 测试完整的Input Embedding (Learned)
    print("\n" + "-" * 60)
    print("测试 完整Input Embedding (Learned)")
    input_emb_learned = InputEmbedding(
        vocab_size, d_model, max_seq_len, pos_encoding_type='learned'
    )
    learned_output = input_emb_learned(token_ids)
    print(f"输出形状: {learned_output.shape}")
    
    # 测试完整的Input Embedding (Sinusoidal)
    print("\n" + "-" * 60)
    print("测试 完整Input Embedding (Sinusoidal)")
    input_emb_sin = InputEmbedding(
        vocab_size, d_model, max_seq_len, pos_encoding_type='sinusoidal'
    )
    sin_output = input_emb_sin(token_ids)
    print(f"输出形状: {sin_output.shape}")
    
    # 参数量对比
    print("\n" + "-" * 60)
    print("参数量对比:")
    learned_params = sum(p.numel() for p in input_emb_learned.parameters())
    sin_params = sum(p.numel() for p in input_emb_sin.parameters())
    print(f"  Learned位置编码参数量: {learned_params:,}")
    print(f"  Sinusoidal位置编码参数量: {sin_params:,}")
    print(f"  差异: {learned_params - sin_params:,} (位置嵌入表的大小)")
    
    # 可视化正弦位置编码的模式
    print("\n" + "-" * 60)
    print("正弦位置编码的前5个位置、前10个维度:")
    sin_pe = SinusoidalPositionalEncoding(d_model, max_seq_len)
    print(sin_pe.pe[0, :5, :10])
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_embeddings()
