"""
字符级Tokenizer

最简单的Tokenizer，将每个字符映射为一个ID
"""

from typing import List, Dict, Optional
import json
import os
from .base import BaseTokenizer

class CharTokenizer(BaseTokenizer):
    """
    字符级Tokenizer
    """
    def __init__(self, chars: Optional[List[str]] = None):
        super().__init__()
        if chars:
            self.setup(chars)
        else:
            self.chars = []
            self.char_to_idx = {}
            self.idx_to_char = {}
            self.vocab_size = 0
            
    def setup(self, chars: List[str]):
        """
        根据字符列表建立词汇表
        """
        # 确保包含特殊字符（如果有）并排序
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars)
        
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, text: str) -> List[int]:
        """
        将文本转换为token IDs
        """
        ids = []
        for ch in text:
            if ch in self.char_to_idx:
                ids.append(self.char_to_idx[ch])
            else:
                # 处理未知字符，这里简单忽略或报错
                # 实际应用中应该有一个<UNK> token
                print(f"Warning: Unknown character '{ch}'")
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        将token IDs转换为文本
        """
        return ''.join([self.idx_to_char.get(i, '') for i in ids])
        
    def save(self, path: str):
        """
        保存tokenizer到JSON文件
        """
        data = {
            'type': 'char',
            'chars': self.chars
        }
        
        # 如果path是目录，保存为tokenizer.json
        if os.path.isdir(path):
            file_path = os.path.join(path, 'tokenizer.json')
        else:
            file_path = path
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Tokenizer saved to {file_path}")
        
    @classmethod
    def load(cls, path: str):
        """
        从JSON文件加载tokenizer
        """
        # 如果path是目录，寻找tokenizer.json
        if os.path.isdir(path):
            file_path = os.path.join(path, 'tokenizer.json')
        else:
            file_path = path
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenizer file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if data.get('type') != 'char':
            raise ValueError(f"Invalid tokenizer type: {data.get('type')}")
            
        tokenizer = cls(data['chars'])
        return tokenizer
