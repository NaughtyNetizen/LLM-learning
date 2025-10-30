"""
完整的GPT模型实现

基于GPT-2/GPT-3架构的Decoder-only Transformer
"""

import torch
import torch.nn as nn
import sys
import os

# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_basics.attention import CausalSelfAttention
from transformer_basics.layers import PositionWiseFeedForward, LayerNorm
from transformer_basics.embeddings import InputEmbedding
from gpt_model.config import GPTConfig


class GPTBlock(nn.Module):
    """
    GPT Transformer块
    
    结构（Pre-LN风格）:
    1. x = x + Attention(LayerNorm(x))
    2. x = x + FFN(LayerNorm(x))
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # Layer Normalization
        self.ln1 = LayerNorm(config.d_model)
        self.ln2 = LayerNorm(config.d_model)
        
        # Causal Self-Attention
        self.attention = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        
        # Feed-Forward Network
        self.ffn = PositionWiseFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation='gelu'
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len, d_model]
        
        返回:
            output: [batch_size, seq_len, d_model]
            attn_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        # 1. Self-Attention with Residual
        attn_output, attn_weights = self.attention(self.ln1(x))
        x = x + attn_output
        
        # 2. FFN with Residual
        ffn_output = self.ffn(self.ln2(x))
        x = x + ffn_output
        
        return x, attn_weights


class GPT(nn.Module):
    """
    完整的GPT模型
    
    架构:
    1. Input Embedding (Token + Position)
    2. N x Transformer Blocks
    3. Final Layer Norm
    4. LM Head (线性层输出logits)
    
    参数:
        config: GPTConfig配置对象
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config
        
        # 1. Input Embedding
        self.input_embedding = InputEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            pos_encoding_type='learned'  # GPT使用可学习的位置嵌入
        )
        
        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config.n_layers)
        ])
        
        # 3. Final Layer Norm
        self.ln_f = LayerNorm(config.d_model)
        
        # 4. LM Head（输出层）
        # 注意：在实际的GPT-2中，这个权重与token embedding共享
        # 这里为了简单起见，使用独立的线性层
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 特殊初始化：缩放残差连接的权重
        for name, param in self.named_parameters():
            if name.endswith('W_o.weight') or name.endswith('linear2.weight'):
                # 缩放residual projection的权重
                torch.nn.init.normal_(param, mean=0.0, std=0.02/torch.sqrt(torch.tensor(2.0 * config.n_layers)))
        
        print(f"GPT模型初始化完成")
        print(f"参数量: {self.get_num_params()/1e6:.2f}M")
    
    def _init_weights(self, module):
        """
        初始化权重
        
        使用标准正态分布初始化，std=0.02（GPT-2的做法）
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.beta)
            torch.nn.init.ones_(module.gamma)
    
    def forward(self, input_ids, targets=None):
        """
        前向传播
        
        参数:
            input_ids: [batch_size, seq_len] token IDs
            targets: [batch_size, seq_len] 目标token IDs（训练时使用）
        
        返回:
            如果targets为None（推理）:
                logits: [batch_size, seq_len, vocab_size]
            如果targets不为None（训练）:
                loss: 交叉熵损失
                logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # 检查序列长度
        assert seq_len <= self.config.max_seq_len, \
            f"序列长度 {seq_len} 超过最大长度 {self.config.max_seq_len}"
        
        # 1. Input Embedding
        x = self.input_embedding(input_ids)  # [batch, seq_len, d_model]
        
        # 2. 通过所有Transformer块
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attention_weights.append(attn)
        
        # 3. Final Layer Norm
        x = self.ln_f(x)  # [batch, seq_len, d_model]
        
        # 4. LM Head - 输出logits
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        # 计算损失（如果提供了targets）
        loss = None
        if targets is not None:
            # 重塑以计算交叉熵
            # [batch * seq_len, vocab_size] vs [batch * seq_len]
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # 忽略padding token
            )
        
        return loss, logits, attention_weights
    
    def get_num_params(self, non_embedding=False):
        """
        获取参数总数
        
        参数:
            non_embedding: 如果为True，不计入embedding参数
        
        返回:
            参数总数
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            # 减去embedding参数
            n_params -= self.input_embedding.token_embedding.embedding.weight.numel()
            n_params -= self.input_embedding.pos_embedding.embedding.weight.numel()
        
        return n_params
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成文本
        
        参数:
            idx: [batch_size, seq_len] 初始token序列
            max_new_tokens: 要生成的最大token数
            temperature: 温度参数（越高越随机）
            top_k: Top-K采样参数
        
        返回:
            generated: [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # 如果序列太长，裁剪到max_seq_len
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # 前向传播
            _, logits, _ = self.forward(idx_cond)
            
            # 只取最后一个位置的logits
            logits = logits[:, -1, :] / temperature  # [batch, vocab_size]
            
            # 可选：Top-K采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 应用softmax获得概率分布
            probs = nn.functional.softmax(logits, dim=-1)
            
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # 拼接到序列
            idx = torch.cat([idx, idx_next], dim=1)  # [batch, seq_len + 1]
        
        return idx


# ====================== 测试代码 ======================

def test_gpt_model():
    """测试GPT模型"""
    print("=" * 60)
    print("测试GPT模型")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 使用微型配置进行测试
    from config import get_config
    config = get_config('gpt-micro')
    
    print("\n模型配置:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  vocab_size: {config.vocab_size}")
    
    # 创建模型
    print("\n创建模型...")
    model = GPT(config)
    
    # 创建随机输入
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n输入形状: {input_ids.shape}")
    
    # 测试前向传播（训练模式）
    print("\n" + "-" * 60)
    print("测试训练模式（计算损失）")
    loss, logits, attn_weights = model(input_ids, targets)
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits形状: {logits.shape}")
    print(f"注意力权重数量: {len(attn_weights)} (每层一个)")
    
    # 测试前向传播（推理模式）
    print("\n" + "-" * 60)
    print("测试推理模式")
    model.eval()
    with torch.no_grad():
        _, logits, _ = model(input_ids)
        print(f"Logits形状: {logits.shape}")
        
        # 预测下一个token
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        print(f"预测的下一个token: {next_token}")
    
    # 测试文本生成
    print("\n" + "-" * 60)
    print("测试文本生成")
    prompt = input_ids[:1, :10]  # 使用前10个token作为prompt
    print(f"Prompt长度: {prompt.shape[1]}")
    
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=50)
    print(f"生成序列长度: {generated.shape[1]}")
    print(f"生成的token IDs (前30个): {generated[0, :30].tolist()}")
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_gpt_model()
