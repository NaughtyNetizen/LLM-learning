"""推理和文本生成包"""

from .sampling import (
    greedy_decode,
    temperature_sampling,
    top_k_sampling,
    top_p_sampling,
    beam_search
)

from .generator import TextGenerator

__all__ = [
    'greedy_decode',
    'temperature_sampling',
    'top_k_sampling',
    'top_p_sampling',
    'beam_search',
    'TextGenerator',
]
