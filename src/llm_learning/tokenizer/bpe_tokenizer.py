"""
BPE Tokenizer

基于tiktoken的BPE tokenizer封装
"""

from typing import List, Optional
import os
import json
from .base import BaseTokenizer

class BPETokenizer(BaseTokenizer):
    """
    BPE Tokenizer (Wrapper around tiktoken)
    """
    def __init__(self, model_name: str = "gpt2", encoding_name: Optional[str] = None):
        super().__init__()
        try:
            import tiktoken
        except ImportError:
            raise ImportError("tiktoken not installed. Please run: pip install tiktoken")
            
        if encoding_name:
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.model_name = encoding_name
        else:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                self.encoding = tiktoken.get_encoding("gpt2")
            self.model_name = model_name
            
        self.vocab_size = self.encoding.n_vocab
        
    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)
    
    def decode(self, ids: List[int]) -> str:
        return self.encoding.decode(ids)
        
    def save(self, path: str):
        """
        保存配置
        """
        data = {
            'type': 'bpe',
            'model_name': self.model_name
        }
        
        if os.path.isdir(path):
            file_path = os.path.join(path, 'tokenizer.json')
        else:
            file_path = path
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Tokenizer config saved to {file_path}")
        
    @classmethod
    def load(cls, path: str):
        """
        加载tokenizer
        """
        if os.path.isdir(path):
            file_path = os.path.join(path, 'tokenizer.json')
        else:
            file_path = path
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenizer file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if data.get('type') != 'bpe':
            raise ValueError(f"Invalid tokenizer type: {data.get('type')}")
            
        return cls(data['model_name'])
