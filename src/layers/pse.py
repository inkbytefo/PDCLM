## Developer: inkbytefo
## Modified: 2025-11-21

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from ..utils.entropy import compute_entropy_profile_vectorized

# Define TorchTensor type alias
TorchTensor = Union[torch.Tensor]

class PatternStreamEncoder(nn.Module):
    def __init__(self, vocab_size: int = 256, embed_dim: int = 512, window_size: int = 128, overlap_ratio: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.overlap = int(window_size * overlap_ratio)
        self.stride = window_size - self.overlap
        
        # CPU threading optimization for multi-core systems
        torch.set_num_threads(4)
        
        # CPU acceleration optimizations
        torch.backends.mkldnn.enabled = True
        torch.set_flush_denormal(True)
        
        # Optimized embedding: Manual weight instead of nn.Embedding for better CPU performance
        self.embed_weight = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)
        self.proj = nn.Linear(embed_dim * window_size, embed_dim)
        self.scale = nn.Parameter(torch.tensor(2.0))  # Entropy modulation scale

    def _compute_entropy_profile(self, bytes_seq: TorchTensor) -> TorchTensor:
        """
        Optimized entropy profile computation using vectorized operations.
        
        Uses torch.unfold for sliding windows and batched operations for entropy calculation.
        Significantly faster than loop-based approach.
        """
        return compute_entropy_profile_vectorized(
            bytes_seq, self.window_size, self.stride, self.vocab_size
        )

    def _embed_bytes(self, bytes_seq: TorchTensor) -> TorchTensor:
        """
        Optimized embedding computation using manual embedding lookup.
        Faster than nn.Embedding on CPU due to reduced overhead.
        """
        return F.embedding(bytes_seq, self.embed_weight)

    def forward(self, raw_text: str) -> TorchTensor:
        device = next(self.parameters()).device
        
        # Unicode karakterleri vocab_size'a göre mod alarak sınırlandır
        bytes_seq = torch.tensor([ord(c) % self.vocab_size for c in raw_text], dtype=torch.long, device=device)
        
        # Fast embedding computation
        embedded = self._embed_bytes(bytes_seq)  # [seq_len, embed_dim]
        
        # Entropy profile computation
        entropy_profile = self._compute_entropy_profile(bytes_seq)  # [num_windows]
        entropy_profile = F.sigmoid(self.scale * entropy_profile)  # [0,1] normalize
        
        # Optimized entropy upsampling using repeat instead of interpolate
        if entropy_profile.size(0) > 1:
            # Calculate how many times to repeat each entropy value
            repeat_factor = (bytes_seq.size(0) + entropy_profile.size(0) - 1) // entropy_profile.size(0)
            upsampled = entropy_profile.repeat_interleave(repeat_factor)[:bytes_seq.size(0)]
        else:
            upsampled = entropy_profile.expand(bytes_seq.size(0))
        
        modulated = embedded * upsampled.unsqueeze(-1)
        
        # Pad and reshape to windows
        pad_len = (self.stride - (modulated.size(0) % self.stride)) % self.stride
        if pad_len > 0:
            modulated = F.pad(modulated, (0, 0, 0, pad_len))
        
        windows = modulated.unfold(0, self.window_size, self.stride)  # [num_windows, window_size, embed_dim]
        flattened = windows.contiguous().view(windows.size(0), -1)  # [num_windows, window_size * embed_dim]
        
        return self.proj(flattened)  # [num_windows, embed_dim]

    def get_performance_stats(self, text: str) -> dict:
        """Get detailed performance statistics for analysis."""
        device = next(self.parameters()).device
        bytes_seq = torch.tensor([ord(c) for c in text], dtype=torch.long, device=device)
        
        # Time components separately
        import time
        
        # Embedding timing
        start = time.time()
        embedded = self._embed_bytes(bytes_seq)
        embedding_time = time.time() - start
        
        # Entropy timing
        start = time.time()
        entropy = self._compute_entropy_profile(bytes_seq)
        entropy_time = time.time() - start
        
        # Full pipeline timing
        start = time.time()
        output = self(text)
        full_time = time.time() - start
        
        num_windows = output.shape[0]
        compression_ratio = num_windows / len(text)
        
        return {
            'total_time': full_time,
            'embedding_time': embedding_time,
            'entropy_time': entropy_time,
            'other_time': full_time - embedding_time - entropy_time,
            'num_windows': num_windows,
            'compression_ratio': compression_ratio,
            'input_length': len(text)
        }
