"""
Tokenizer基类

定义所有Tokenizer必须实现的接口
"""

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
import json
import os

class BaseTokenizer(ABC):
    """
    Tokenizer抽象基类
    """
    def __init__(self):
        self.vocab_size = 0
        
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        将文本转换为token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """
        将token IDs转换为文本
        """
        pass
        
    @abstractmethod
    def save(self, path: str):
        """
        保存tokenizer配置和词汇表
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """
        加载tokenizer
        """
        pass
