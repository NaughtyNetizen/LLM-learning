"""
文本生成器

整合各种采样策略，提供统一的生成接口
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from sampling import (
    greedy_decode, 
    temperature_sampling,
    top_k_sampling,
    top_p_sampling
)


class TextGenerator:
    """
    文本生成器
    
    支持多种生成策略和参数配置
    """
    def __init__(self, model, tokenizer=None):
        """
        初始化
        
        参数:
            model: GPT模型
            tokenizer: tokenizer（可选）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        strategy: str = 'top_p',
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop_token_id: Optional[int] = None,
        num_return_sequences: int = 1
    ) -> torch.Tensor:
        """
        生成文本
        
        参数:
            prompt_ids: [batch_size, seq_len] 输入prompt的token IDs
            max_new_tokens: 最多生成多少个新token
            strategy: 采样策略
                - 'greedy': 贪心解码
                - 'temperature': 温度采样
                - 'top_k': Top-K采样
                - 'top_p': Top-P (Nucleus)采样
            temperature: 温度参数
            top_k: Top-K参数
            top_p: Top-P参数
            repetition_penalty: 重复惩罚系数
            stop_token_id: 停止token的ID
            num_return_sequences: 返回多少个生成结果
        
        返回:
            generated_ids: [batch_size * num_return_sequences, seq_len] 生成的token IDs
        """
        device = prompt_ids.device
        batch_size = prompt_ids.size(0)
        
        # 如果需要多个序列，复制prompt
        if num_return_sequences > 1:
            prompt_ids = prompt_ids.repeat_interleave(num_return_sequences, dim=0)
        
        # 初始化生成序列
        generated_ids = prompt_ids.clone()
        
        # 用于跟踪哪些序列已经结束
        unfinished_sequences = torch.ones(
            generated_ids.shape[0], dtype=torch.long, device=device
        )
        
        for step in range(max_new_tokens):
            # 如果所有序列都已结束，停止生成
            if unfinished_sequences.max() == 0:
                break
            
            # 获取当前序列（可能需要截断）
            input_ids = generated_ids
            if input_ids.size(1) > self.model.config.max_seq_len:
                input_ids = input_ids[:, -self.model.config.max_seq_len:]
            
            # 前向传播
            _, logits, _ = self.model(input_ids)
            
            # 只使用最后一个位置的logits
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, repetition_penalty
                )
            
            # 根据策略采样下一个token
            if strategy == 'greedy':
                next_tokens = greedy_decode(next_token_logits)
            elif strategy == 'temperature':
                next_tokens = temperature_sampling(next_token_logits, temperature)
            elif strategy == 'top_k':
                if top_k is None:
                    top_k = 50
                next_tokens = top_k_sampling(next_token_logits, top_k, temperature)
            elif strategy == 'top_p':
                next_tokens = top_p_sampling(next_token_logits, top_p, temperature)
            else:
                raise ValueError(f"未知的采样策略: {strategy}")
            
            # 对于已结束的序列，使用padding token（或重复最后一个token）
            next_tokens = next_tokens * unfinished_sequences + \
                          generated_ids[:, -1] * (1 - unfinished_sequences)
            
            # 将新token添加到序列
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 更新未结束序列的标记
            if stop_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != stop_token_id).long()
                )
        
        return generated_ids
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        generated_ids: torch.Tensor, 
        penalty: float
    ) -> torch.Tensor:
        """
        应用重复惩罚
        
        对已生成过的token降低其logits
        
        参数:
            logits: [batch_size, vocab_size]
            generated_ids: [batch_size, seq_len]
            penalty: 惩罚系数（>1.0降低重复）
        
        返回:
            penalized_logits: [batch_size, vocab_size]
        """
        if penalty == 1.0:
            return logits
        
        # 对于已经出现过的token，除以penalty
        for i in range(logits.shape[0]):
            unique_tokens = generated_ids[i].unique()
            logits[i, unique_tokens] = logits[i, unique_tokens] / penalty
        
        return logits
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> List[str]:
        """
        从文本prompt生成文本（需要tokenizer）
        
        参数:
            prompt: 输入的文本prompt
            max_new_tokens: 最多生成多少个新token
            **kwargs: 其他生成参数
        
        返回:
            generated_texts: 生成的文本列表
        """
        if self.tokenizer is None:
            raise ValueError("需要提供tokenizer才能使用generate_text方法")
        
        # 编码prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # 移到正确的设备
        device = next(self.model.parameters()).device
        prompt_ids = prompt_ids.to(device)
        
        # 生成
        generated_ids = self.generate(
            prompt_ids, 
            max_new_tokens=max_new_tokens, 
            **kwargs
        )
        
        # 解码
        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]
        
        return generated_texts


# ====================== 演示代码 ======================

def demo_text_generation():
    """演示文本生成"""
    print("=" * 60)
    print("文本生成演示")
    print("=" * 60)
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from gpt_model.model import GPT
    from gpt_model.config import get_config
    
    # 创建小型模型
    print("\n创建GPT模型...")
    config = get_config('gpt-micro')
    model = GPT(config)
    model.eval()
    
    # 创建生成器
    generator = TextGenerator(model)
    
    # 创建随机prompt
    prompt_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    print(f"\nPrompt token IDs: {prompt_ids[0].tolist()}")
    print(f"Prompt长度: {prompt_ids.size(1)}")
    
    # 测试不同的采样策略
    strategies = {
        'greedy': {},
        'temperature': {'temperature': 0.8},
        'top_k': {'top_k': 20},
        'top_p': {'top_p': 0.9},
    }
    
    for strategy_name, params in strategies.items():
        print("\n" + "-" * 60)
        print(f"策略: {strategy_name}")
        print(f"参数: {params}")
        
        generated = generator.generate(
            prompt_ids,
            max_new_tokens=20,
            strategy=strategy_name,
            **params
        )
        
        print(f"生成长度: {generated.size(1)}")
        print(f"生成的token IDs: {generated[0, :30].tolist()}")
    
    # 测试生成多个序列
    print("\n" + "-" * 60)
    print("生成多个序列 (num_return_sequences=3)")
    
    generated = generator.generate(
        prompt_ids,
        max_new_tokens=15,
        strategy='top_p',
        num_return_sequences=3
    )
    
    print(f"生成形状: {generated.shape}")
    for i, seq in enumerate(generated):
        print(f"序列 {i+1}: {seq[:25].tolist()}")
    
    print("\n✅ 演示完成！")


if __name__ == "__main__":
    demo_text_generation()
