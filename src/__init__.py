"""
Pattern-Driven Cognitive Language Model (PD-CLM)

Bu modül, geleneksel tokenizer yaklaşımının ötesinde,
Pattern Stream Encoder (PSE) tabanlı yeni bir LLM mimarisi sunar.

Author: inkbytefo
Modified: 2025-11-16
"""

from .pse import PatternStreamEncoder
from .model import PDCLMBase
from .utils import *

__version__ = "0.1.0"
__all__ = ["PatternStreamEncoder", "PDCLMBase"]
