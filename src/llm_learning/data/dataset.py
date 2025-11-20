"""
数据集处理

实现语言模型预训练的数据处理pipeline
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Optional
import os


class TextDataset(Dataset):
    """
    简单的文本数据集
    
    用于语言模型预训练（Next Token Prediction）
    """
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        tokenizer=None,
        stride: Optional[int] = None
    ):
        """
        初始化
        
        参数:
            data_path: 数据文件路径（包含token IDs的.npy或.pt文件）
            seq_len: 序列长度
            tokenizer: tokenizer（可选，用于从文本文件加载）
            stride: 滑动窗口步长（如果为None，则等于seq_len）
        """
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        
        # 加载数据
        if data_path.endswith('.npy'):
            self.data = np.load(data_path).astype(np.int64)
        elif data_path.endswith('.pt'):
            self.data = torch.load(data_path).numpy()
        elif data_path.endswith('.txt'):
            if tokenizer is None:
                raise ValueError("加载文本文件需要提供tokenizer")
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.data = np.array(tokenizer.encode(text), dtype=np.int64)
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")
        
        print(f"加载数据: {len(self.data):,} tokens")
        
        # 计算样本数量
        self.n_samples = max(1, (len(self.data) - seq_len) // self.stride)
        print(f"数据集大小: {self.n_samples:,} 样本")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        返回:
            input_ids: [seq_len] 输入token IDs
            targets: [seq_len] 目标token IDs（input_ids向右移动1位）
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len + 1  # +1因为需要target
        
        # 获取序列
        sequence = self.data[start_idx:end_idx]
        
        # 如果序列不够长，padding
        if len(sequence) < self.seq_len + 1:
            sequence = np.pad(
                sequence,
                (0, self.seq_len + 1 - len(sequence)),
                mode='constant',
                constant_values=0
            )
        
        # 分离input和target
        input_ids = torch.from_numpy(sequence[:-1].astype(np.int64))
        targets = torch.from_numpy(sequence[1:].astype(np.int64))
        
        return input_ids, targets


class InMemoryTextDataset(Dataset):
    """
    内存中的文本数据集（用于小数据集）
    
    直接从文本字符串列表创建数据集
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        seq_len: int,
        overlap: int = 0
    ):
        """
        初始化
        
        参数:
            texts: 文本列表
            tokenizer: tokenizer
            seq_len: 序列长度
            overlap: 序列间的重叠长度
        """
        self.seq_len = seq_len
        self.overlap = overlap
        self.tokenizer = tokenizer
        
        # Tokenize所有文本
        print("Tokenizing texts...")
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        self.data = np.array(all_tokens, dtype=np.int64)
        print(f"总token数: {len(self.data):,}")
        
        # 创建样本
        self.samples = []
        stride = seq_len - overlap
        for i in range(0, len(self.data) - seq_len, stride):
            input_ids = self.data[i:i+seq_len]
            targets = self.data[i+1:i+seq_len+1]
            self.samples.append((input_ids, targets))
        
        print(f"创建了 {len(self.samples):,} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_ids, targets = self.samples[idx]
        return torch.from_numpy(input_ids), torch.from_numpy(targets)


def create_shakespeare_dataset():
    """
    创建一个简单的Shakespeare数据集作为示例
    
    这个函数会下载tiny shakespeare数据集
    """
    import urllib.request
    
    data_dir = 'data/raw'
    os.makedirs(data_dir, exist_ok=True)
    
    # Update to point to WikiText if available, otherwise fallback or error
    data_path = os.path.join(data_dir, 'wikitext', 'train.txt')
    
    if not os.path.exists(data_path):
        # Fallback to tiny shakespeare if wikitext not found
        fallback_path = os.path.join(data_dir, 'tiny_shakespeare.txt')
        if os.path.exists(fallback_path):
            print(f"WikiText not found, using fallback: {fallback_path}")
            return fallback_path
            
        print(f"数据集不存在: {data_path}")
        print("请运行: python scripts/download_data.py --dataset wikitext")
        raise FileNotFoundError(f"无法找到数据集. 请先运行下载脚本.")
    else:
        print(f"使用 WikiText 数据集: {data_path}")
    
    return data_path


def prepare_data(
    data_path: str,
    tokenizer,
    seq_len: int = 256,
    train_split: float = 0.9
):
    """
    准备训练和验证数据集
    
    参数:
        data_path: 文本文件路径
        tokenizer: tokenizer
        seq_len: 序列长度
        train_split: 训练集比例
    
    返回:
        train_dataset, val_dataset
    """
    # Check if separate validation file exists
    data_dir = os.path.dirname(data_path)
    val_path = os.path.join(data_dir, 'validation.txt')
    
    if os.path.exists(val_path) and os.path.basename(data_path) == 'train.txt':
        print(f"发现独立验证集: {val_path}")
        
        # Load train
        with open(data_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
            
        # Load val
        with open(val_path, 'r', encoding='utf-8') as f:
            val_text = f.read()
            
        print(f"训练文本长度: {len(train_text):,} 字符")
        print(f"验证文本长度: {len(val_text):,} 字符")
        
    else:
        # Fallback to splitting
        print("未发现独立验证集，从训练集切分...")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 分割训练集和验证集
        split_idx = int(len(text) * train_split)
        train_text = text[:split_idx]
        val_text = text[split_idx:]
        
        print(f"训练文本长度: {len(train_text):,} 字符")
        print(f"验证文本长度: {len(val_text):,} 字符")
    
    # 创建数据集
    # 注意：对于大文件，这里可能会占用较多内存
    # 建议后续优化为Lazy Loading或Memory Mapping
    train_dataset = InMemoryTextDataset([train_text], tokenizer, seq_len)
    val_dataset = InMemoryTextDataset([val_text], tokenizer, seq_len)
    
    return train_dataset, val_dataset


# ====================== 测试代码 ======================

def test_dataset():
    """测试数据集"""
    print("=" * 60)
    print("测试数据集")
    print("=" * 60)
    
    # 创建模拟数据
    print("\n创建模拟数据...")
    np.random.seed(42)
    data = np.random.randint(0, 1000, size=10000, dtype=np.int64)
    
    # 保存到文件
    os.makedirs('data/processed', exist_ok=True)
    data_path = 'data/processed/test_data.npy'
    np.save(data_path, data)
    print(f"保存到: {data_path}")
    
    # 创建数据集
    print("\n" + "-" * 60)
    print("创建TextDataset")
    seq_len = 128
    dataset = TextDataset(data_path, seq_len=seq_len)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    input_ids, targets = dataset[0]
    print(f"\n样本形状:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  targets: {targets.shape}")
    print(f"\n前10个tokens:")
    print(f"  input: {input_ids[:10].tolist()}")
    print(f"  target: {targets[:10].tolist()}")
    print(f"\n验证target是input右移1位: {torch.all(input_ids[1:] == targets[:-1])}")
    
    # 测试DataLoader
    print("\n" + "-" * 60)
    print("测试DataLoader")
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    batch_inputs, batch_targets = next(iter(dataloader))
    print(f"Batch形状:")
    print(f"  inputs: {batch_inputs.shape}")
    print(f"  targets: {batch_targets.shape}")
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    test_dataset()
