## Developer: inkbytefo
## Modified: 2025-11-16

"""
Entropy calculation utilities optimized for batched processing.
This module provides vectorized entropy computation for PatternStreamEncoder.
"""

import torch
import torch.nn.functional as F
from typing import Union

# Define TorchTensor type alias
TorchTensor = Union[torch.Tensor]


def compute_entropy_profile_vectorized(
    bytes_seq: TorchTensor, 
    window_size: int, 
    stride: int, 
    vocab_size: int = 256
) -> TorchTensor:
    """
    Optimized entropy profile computation using torch operations.
    
    Args:
        bytes_seq: Input byte sequence [seq_len]
        window_size: Size of sliding windows
        stride: Step size for sliding windows
        vocab_size: Vocabulary size (default 256 for byte values)
        
    Returns:
        Entropy profile tensor [num_windows]
    """
    device = bytes_seq.device
    seq_len = bytes_seq.size(0)
    
    if seq_len < window_size:
        return torch.zeros(1, device=device)
    
    # Unfold windows: [seq_len] -> [num_windows, window_size]
    unfolded = bytes_seq.unfold(0, window_size, stride)  # [num_windows, window_size]
    
    # Batched bincount: count occurrences of each vocab item in each window
    num_windows = unfolded.size(0)
    hist = torch.zeros(num_windows, vocab_size, device=device, dtype=torch.float32)
    
    # Scatter-add for batched histogram computation
    # This efficiently computes histograms for all windows simultaneously
    hist.scatter_add_(1, unfolded, torch.ones_like(unfolded, dtype=torch.float32))
    
    # Normalize to probabilities
    probabilities = hist / window_size
    probabilities = probabilities + 1e-10  # Numerical stability
    
    # Compute entropy for each window: H = -sum(p * log(p))
    entropy = -(probabilities * torch.log(probabilities)).sum(dim=1)  # [num_windows]
    
    return entropy


def compute_entropy_profile_unoptimized(
    bytes_seq: TorchTensor, 
    window_size: int, 
    stride: int, 
    vocab_size: int = 256
) -> TorchTensor:
    """
    Original unoptimized entropy computation for comparison.
    
    Args:
        bytes_seq: Input byte sequence [seq_len]
        window_size: Size of sliding windows
        stride: Step size for sliding windows
        vocab_size: Vocabulary size (default 256 for byte values)
        
    Returns:
        Entropy profile tensor [num_windows]
    """
    device = bytes_seq.device
    seq_len = bytes_seq.size(0)
    profiles = []
    
    for i in range(0, seq_len - window_size + 1, stride):
        window = bytes_seq[i:i + window_size]
        hist = torch.bincount(window, minlength=vocab_size).float()
        hist = hist / hist.sum()
        hist = hist + 1e-10
        ent = -(hist * hist.log()).sum()
        profiles.append(ent)
    
    if not profiles:
        return torch.zeros(1, device=device)
    return torch.stack(profiles)


def benchmark_entropy_computation(
    bytes_seq: TorchTensor,
    window_size: int,
    stride: int,
    vocab_size: int = 256,
    num_runs: int = 5
) -> dict:
    """
    Benchmark vectorized vs unoptimized entropy computation.
    
    Args:
        bytes_seq: Input byte sequence
        window_size: Window size for sliding windows
        stride: Stride for sliding windows
        vocab_size: Vocabulary size
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Vectorized version
    vectorized_times = []
    for _ in range(num_runs):
        start = time.time()
        vectorized_result = compute_entropy_profile_vectorized(
            bytes_seq, window_size, stride, vocab_size
        )
        vectorized_times.append(time.time() - start)
    
    # Unoptimized version
    unoptimized_times = []
    for _ in range(num_runs):
        start = time.time()
        unoptimized_result = compute_entropy_profile_unoptimized(
            bytes_seq, window_size, stride, vocab_size
        )
        unoptimized_times.append(time.time() - start)
    
    # Verify results are equivalent
    vectorized_mean = torch.mean(vectorized_result)
    unoptimized_mean = torch.mean(unoptimized_result)
    result_diff = abs(vectorized_mean - unoptimized_mean)
    
    return {
        'vectorized_time': sum(vectorized_times) / len(vectorized_times),
        'unoptimized_time': sum(unoptimized_times) / len(unoptimized_times),
        'speedup': sum(unoptimized_times) / sum(vectorized_times),
        'result_equivalence': result_diff.item() < 1e-6,
        'vectorized_mean_entropy': vectorized_mean.item(),
        'unoptimized_mean_entropy': unoptimized_mean.item(),
        'result_difference': result_diff.item()
    }


if __name__ == "__main__":
    # Test the entropy functions
    print("Testing entropy computation functions...")
    
    # Create test data
    test_seq = torch.randint(0, 256, (10000,))
    window_size = 128
    stride = 64
    vocab_size = 256
    
    # Benchmark
    results = benchmark_entropy_computation(test_seq, window_size, stride, vocab_size)
    
    print(f"\nBenchmark Results:")
    print(f"Vectorized time: {results['vectorized_time']:.4f}s")
    print(f"Unoptimized time: {results['unoptimized_time']:.4f}s")
    print(f"Speedup: {results['speedup']:.2f}x")
    print(f"Results equivalent: {results['result_equivalence']}")
    print(f"Entropy difference: {results['result_difference']:.2e}")
    
    if results['speedup'] >= 3:
        print("✅ Vectorization successful - achieving target speedup!")
    else:
        print("❌ Vectorization speedup below target")
