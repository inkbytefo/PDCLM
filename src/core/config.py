"""
PDCLM Configuration Module.
Centralizes all hyperparameters and architectural settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class PDCLMConfig:
    """Configuration for the PD-CLM model family."""
    
    # Model Architecture
    embed_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    
    # Pattern Stream Encoder (PSE)
    window_size: int = 64
    stride: int = 16  # Overlap factor
    use_entropy_modulation: bool = True
    
    # Hierarchical Memory Router (HMR)
    memory_slots: int = 128
    memory_dim: int = 512
    
    # High-Level Cognitive Loop (HCL)
    hcl_steps: int = 4
    reflection_threshold: float = 0.8
    
    # Training
    max_sequence_length: int = 2048
    vocab_size: int = 260  # 256 bytes + 4 special tokens (PAD, BOS, EOS, MASK)
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def __post_init__(self):
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
