"""
文本生成的采样策略

实现多种采样方法：
1. Greedy Decoding (贪心解码)
2. Temperature Sampling (温度采样)
3. Top-K Sampling
4. Top-P (Nucleus) Sampling
5. Beam Search
"""

import torch
import torch.nn.functional as F


def greedy_decode(logits):
    """
    贪心解码：选择概率最高的token
    
    参数:
        logits: [batch_size, vocab_size] 或 [vocab_size]
    
    返回:
        next_token: [batch_size] 或 标量
    """
    return torch.argmax(logits, dim=-1)


def temperature_sampling(logits, temperature=1.0):
    """
    温度采样
    
    temperature < 1.0: 输出更确定（更保守）
    temperature = 1.0: 原始分布
    temperature > 1.0: 输出更随机（更有创造性）
    
    参数:
        logits: [batch_size, vocab_size]
        temperature: 温度参数
    
    返回:
        next_token: [batch_size]
    """
    # 应用温度
    logits = logits / temperature
    
    # 转换为概率分布
    probs = F.softmax(logits, dim=-1)
    
    # 从分布中采样
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return next_token


def top_k_sampling(logits, k, temperature=1.0):
    """
    Top-K采样：只从概率最高的k个token中采样
    
    参数:
        logits: [batch_size, vocab_size]
        k: 保留的top token数量
        temperature: 温度参数
    
    返回:
        next_token: [batch_size]
    """
    # 应用温度
    logits = logits / temperature
    
    # 获取top-k的值和索引
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # 将不在top-k中的token的logits设为-inf
    logits_with_mask = torch.full_like(logits, float('-inf'))
    logits_with_mask.scatter_(-1, top_k_indices, top_k_logits)
    
    # 转换为概率并采样
    probs = F.softmax(logits_with_mask, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return next_token


def top_p_sampling(logits, p, temperature=1.0):
    """
    Top-P (Nucleus) 采样：从累积概率达到p的最小token集合中采样
    
    优点：动态调整候选集大小
    
    参数:
        logits: [batch_size, vocab_size]
        p: 累积概率阈值 (0 < p <= 1)
        temperature: 温度参数
    
    返回:
        next_token: [batch_size]
    """
    # 应用温度
    logits = logits / temperature
    
    # 转换为概率并排序
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 移除累积概率超过p的token
    sorted_indices_to_remove = cumulative_probs > p
    
    # 保留至少一个token（第一个token）
    sorted_indices_to_remove[..., 0] = False
    
    # 创建mask
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    
    # 将被移除的token的logits设为-inf
    logits_with_mask = logits.clone()
    logits_with_mask[indices_to_remove] = float('-inf')
    
    # 重新计算概率并采样
    probs = F.softmax(logits_with_mask, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return next_token


def beam_search(model, input_ids, beam_width, max_length, temperature=1.0):
    """
    Beam Search解码
    
    维护多个候选序列，每次扩展最有希望的序列
    
    参数:
        model: GPT模型
        input_ids: [batch_size, seq_len] 初始序列
        beam_width: beam的宽度
        max_length: 最大生成长度
        temperature: 温度参数
    
    返回:
        best_sequence: [batch_size, max_length] 最佳序列
        best_score: [batch_size] 最佳得分
    """
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # 初始化beams: [batch_size, beam_width, seq_len]
    beams = input_ids.unsqueeze(1).repeat(1, beam_width, 1)
    beam_scores = torch.zeros(batch_size, beam_width, device=device)
    
    for step in range(max_length):
        # 获取当前所有beam的logits
        all_candidates = []
        
        for b in range(beam_width):
            current_beam = beams[:, b, :]  # [batch_size, seq_len]
            
            # 前向传播
            with torch.no_grad():
                _, logits, _ = model(current_beam)
            
            # 只取最后一个位置的logits
            logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 获取top-k候选
            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
            
            # 计算候选得分
            for k in range(beam_width):
                candidate_score = beam_scores[:, b] + top_log_probs[:, k]
                candidate_seq = torch.cat([current_beam, top_indices[:, k:k+1]], dim=1)
                all_candidates.append((candidate_score, candidate_seq))
        
        # 从所有候选中选择top beam_width个
        all_scores = torch.stack([c[0] for c in all_candidates], dim=1)  # [batch, beam_width^2]
        top_scores, top_indices = torch.topk(all_scores, beam_width, dim=1)
        
        # 更新beams
        new_beams = []
        for b in range(beam_width):
            idx = top_indices[:, b]
            selected_seq = torch.stack([all_candidates[i.item()][1][j] for j, i in enumerate(idx)])
            new_beams.append(selected_seq)
        
        beams = torch.stack(new_beams, dim=1)
        beam_scores = top_scores
    
    # 返回得分最高的序列
    best_beam_idx = torch.argmax(beam_scores, dim=1)
    best_sequence = beams[torch.arange(batch_size), best_beam_idx]
    best_score = beam_scores[torch.arange(batch_size), best_beam_idx]
    
    return best_sequence, best_score


# ====================== 测试代码 ======================

def test_sampling_strategies():
    """测试各种采样策略"""
    print("=" * 60)
    print("测试采样策略")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 创建模拟的logits
    vocab_size = 100
    batch_size = 2
    logits = torch.randn(batch_size, vocab_size)
    
    print(f"Logits形状: {logits.shape}")
    print(f"词汇表大小: {vocab_size}")
    
    # 测试贪心解码
    print("\n" + "-" * 60)
    print("1. 贪心解码")
    greedy_tokens = greedy_decode(logits)
    print(f"选择的tokens: {greedy_tokens}")
    print(f"对应的概率: {F.softmax(logits, dim=-1)[range(batch_size), greedy_tokens]}")
    
    # 测试温度采样
    print("\n" + "-" * 60)
    print("2. 温度采样")
    for temp in [0.5, 1.0, 2.0]:
        tokens = temperature_sampling(logits, temperature=temp)
        print(f"Temperature={temp}: tokens={tokens}")
    
    # 测试Top-K采样
    print("\n" + "-" * 60)
    print("3. Top-K采样")
    for k in [5, 10, 20]:
        tokens = top_k_sampling(logits, k=k)
        print(f"K={k}: tokens={tokens}")
    
    # 测试Top-P采样
    print("\n" + "-" * 60)
    print("4. Top-P (Nucleus)采样")
    for p in [0.5, 0.9, 0.95]:
        tokens = top_p_sampling(logits, p=p)
        print(f"P={p}: tokens={tokens}")
    
    # 可视化不同温度的分布
    print("\n" + "-" * 60)
    print("5. 温度对分布的影响")
    print("Top-5概率分布:")
    for temp in [0.5, 1.0, 2.0]:
        probs = F.softmax(logits[0] / temp, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        print(f"\nTemperature={temp}:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            print(f"  Token {idx.item()}: {prob.item():.4f}")
    
    print("\n✅ 所有测试完成！")


if __name__ == "__main__":
    test_sampling_strategies()
