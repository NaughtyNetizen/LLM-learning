"""
GPT模型配置

定义各种规模的GPT模型配置
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    GPT模型配置类
    
    参数说明：
        vocab_size: 词汇表大小
        max_seq_len: 最大序列长度（上下文窗口）
        d_model: 模型维度（隐藏层大小）
        n_layers: Transformer层数
        n_heads: 注意力头数
        d_ff: FFN中间层维度（通常是d_model的4倍）
        dropout: Dropout概率
        bias: 是否在Linear和LayerNorm中使用bias
    """
    # 模型架构
    vocab_size: int = 50257  # GPT-2的词汇表大小
    max_seq_len: int = 1024  # 最大序列长度
    d_model: int = 768       # 模型维度
    n_layers: int = 12       # 层数
    n_heads: int = 12        # 注意力头数
    d_ff: int = 3072         # FFN维度 (4 * d_model)
    
    # 正则化
    dropout: float = 0.1
    bias: bool = True        # GPT-2使用bias，GPT-3不使用
    
    def __post_init__(self):
        """验证配置的合法性"""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) 必须能被 n_heads ({self.n_heads}) 整除"
        assert self.d_ff % self.d_model == 0 or self.d_model % self.d_ff == 0, \
            f"建议 d_ff 是 d_model 的整数倍"


# ====================== 预定义配置 ======================

def get_config(model_type: str) -> GPTConfig:
    """
    获取预定义的模型配置
    
    参数:
        model_type: 模型类型
            - 'gpt-micro': 微型模型，用于快速实验 (~1M参数)
            - 'gpt-mini': 迷你模型，用于学习 (~10M参数)
            - 'gpt-small': 小型模型 (~50M参数)
            - 'gpt2': GPT-2 117M
            - 'gpt2-medium': GPT-2 345M
            - 'gpt2-large': GPT-2 774M
            - 'gpt2-xl': GPT-2 1.5B
    
    返回:
        GPTConfig对象
    """
    configs = {
        # 学习用的超小模型
        'gpt-micro': GPTConfig(
            vocab_size=50257,
            max_seq_len=256,
            d_model=128,
            n_layers=2,
            n_heads=2,
            d_ff=512,
            dropout=0.1,
        ),
        
        # 学习用的小模型
        'gpt-mini': GPTConfig(
            vocab_size=50257,
            max_seq_len=512,
            d_model=384,
            n_layers=6,
            n_heads=6,
            d_ff=1536,
            dropout=0.1,
        ),
        
        # 小型实用模型
        'gpt-small': GPTConfig(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=512,
            n_layers=8,
            n_heads=8,
            d_ff=2048,
            dropout=0.1,
        ),
        
        # GPT-2系列（OpenAI官方配置）
        'gpt2': GPTConfig(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=3072,
            dropout=0.1,
            bias=True,
        ),
        
        'gpt2-medium': GPTConfig(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            d_ff=4096,
            dropout=0.1,
            bias=True,
        ),
        
        'gpt2-large': GPTConfig(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=1280,
            n_layers=36,
            n_heads=20,
            d_ff=5120,
            dropout=0.1,
            bias=True,
        ),
        
        'gpt2-xl': GPTConfig(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=1600,
            n_layers=48,
            n_heads=25,
            d_ff=6400,
            dropout=0.1,
            bias=True,
        ),
    }
    
    if model_type not in configs:
        raise ValueError(
            f"未知的模型类型: {model_type}\n"
            f"支持的类型: {list(configs.keys())}"
        )
    
    return configs[model_type]


def print_config_summary(config: GPTConfig):
    """
    打印模型配置摘要
    
    参数:
        config: GPTConfig对象
    """
    print("=" * 60)
    print("GPT模型配置")
    print("=" * 60)
    
    print("\n架构参数:")
    print(f"  词汇表大小:      {config.vocab_size:,}")
    print(f"  最大序列长度:    {config.max_seq_len:,}")
    print(f"  模型维度:        {config.d_model:,}")
    print(f"  层数:            {config.n_layers:,}")
    print(f"  注意力头数:      {config.n_heads:,}")
    print(f"  每个头的维度:    {config.d_model // config.n_heads:,}")
    print(f"  FFN维度:         {config.d_ff:,}")
    
    print("\n正则化:")
    print(f"  Dropout:         {config.dropout}")
    print(f"  使用Bias:        {config.bias}")
    
    # 估算参数量
    params = estimate_parameters(config)
    print(f"\n估算参数量:       {params:,} ({params/1e6:.2f}M)")


def estimate_parameters(config: GPTConfig) -> int:
    """
    估算模型参数量
    
    参数:
        config: GPTConfig对象
    
    返回:
        参数总数
    """
    # Token嵌入
    token_emb = config.vocab_size * config.d_model
    
    # 位置嵌入
    pos_emb = config.max_seq_len * config.d_model
    
    # 每个Transformer层的参数
    # Attention: Q, K, V, O
    attn_params = 4 * config.d_model * config.d_model
    # LayerNorm (x2)
    ln_params = 2 * 2 * config.d_model
    # FFN
    ffn_params = 2 * config.d_model * config.d_ff
    
    layer_params = attn_params + ln_params + ffn_params
    
    # 所有层
    all_layers = config.n_layers * layer_params
    
    # 最终LayerNorm
    final_ln = 2 * config.d_model
    
    # LM head（通常与token embedding共享权重）
    # 这里不计入，因为共享
    
    total = token_emb + pos_emb + all_layers + final_ln
    
    return total


# ====================== 测试代码 ======================

if __name__ == "__main__":
    print("测试所有预定义配置:\n")
    
    model_types = [
        'gpt-micro', 'gpt-mini', 'gpt-small',
        'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
    ]
    
    for model_type in model_types:
        config = get_config(model_type)
        print(f"\n{'='*60}")
        print(f"配置: {model_type}")
        print(f"{'='*60}")
        print(f"参数量: {estimate_parameters(config)/1e6:.1f}M")
        print(f"d_model={config.d_model}, n_layers={config.n_layers}, "
              f"n_heads={config.n_heads}")
