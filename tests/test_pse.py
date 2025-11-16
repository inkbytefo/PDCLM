## Developer: inkbytefo
## Modified: 2025-11-16

import pytest
import torch
import time
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.pse import PatternStreamEncoder

def test_pse_output_shape():
    pse = PatternStreamEncoder(window_size=64)
    text = "a" * 1000
    out = pse(text)
    assert out.dim() == 2
    assert out.size(1) == 512

def test_entropy_positive():
    pse = PatternStreamEncoder()
    bytes_seq = torch.randint(0, 256, (128,))
    ent = pse._compute_entropy_profile(bytes_seq)
    assert (ent > 0).all()

def test_entropy_batched_speed():
    """Test that optimized entropy computation is fast enough for real-time processing."""
    pse = PatternStreamEncoder()
    bytes_seq = torch.randint(0, 256, (10000,))
    
    start = time.time()
    entropy_result = pse._compute_entropy_profile(bytes_seq)
    duration = time.time() - start
    
    print(f"Entropy computation time for 10k seq: {duration:.4f}s")
    
    # Target: < 0.05s for 10k sequence (200k chars/sec throughput)
    assert duration < 0.05, f"Entropy computation too slow: {duration:.4f}s (target: < 0.05s)"
    
    # Ensure results are reasonable
    assert entropy_result.dim() == 1
    assert entropy_result.size(0) > 0
    assert not torch.isnan(entropy_result).any()
    assert (entropy_result >= 0).all()

def test_entropy_optimization_vs_baseline():
    """Test that optimized version is significantly faster than baseline."""
    from src.entropy_utils import compute_entropy_profile_vectorized, compute_entropy_profile_unoptimized
    
    # Use larger test sequence for meaningful performance comparison
    test_seq = torch.randint(0, 256, (20000,))
    window_size = 128
    stride = 64
    vocab_size = 256
    
    # Time optimized version
    start = time.time()
    optimized_result = compute_entropy_profile_vectorized(test_seq, window_size, stride, vocab_size)
    optimized_time = time.time() - start
    
    # Time baseline version  
    start = time.time()
    baseline_result = compute_entropy_profile_unoptimized(test_seq, window_size, stride, vocab_size)
    baseline_time = time.time() - start
    
    speedup = baseline_time / optimized_time
    result_equivalent = torch.allclose(optimized_result, baseline_result, atol=1e-6)
    
    print(f"Optimized: {optimized_time:.4f}s")
    print(f"Baseline: {baseline_time:.4f}s") 
    print(f"Speedup: {speedup:.2f}x")
    print(f"Results equivalent: {result_equivalent}")
    
    # Must be at least 3x faster
    assert speedup >= 3.0, f"Speedup too low: {speedup:.2f}x (target: >= 3x)"
    
    # Results must be equivalent
    assert result_equivalent, "Optimized results don't match baseline"

def test_entropy_accuracy():
    """Test that entropy values are in expected range and consistent."""
    pse = PatternStreamEncoder()
    
    # Test with uniform distribution (high entropy)
    uniform_seq = torch.randint(0, 256, (1000,))
    uniform_entropy = pse._compute_entropy_profile(uniform_seq)
    
    # Test with repetitive sequence (low entropy)  
    repetitive_seq = torch.tensor([65] * 1000)  # All 'A'
    repetitive_entropy = pse._compute_entropy_profile(repetitive_seq)
    
    print(f"Uniform entropy mean: {uniform_entropy.mean():.4f}")
    print(f"Repetitive entropy mean: {repetitive_entropy.mean():.4f}")
    
    # Uniform should have higher entropy than repetitive
    assert uniform_entropy.mean() > repetitive_entropy.mean(), "Entropy not distinguishing patterns correctly"
    
    # All entropy values should be in valid range [0, log2(256)] ≈ [0, 8]
    max_entropy = 8.0  # Maximum possible entropy for byte values
    assert (uniform_entropy >= 0).all() and (uniform_entropy <= max_entropy).all()
    assert (repetitive_entropy >= 0).all() and (repetitive_entropy <= max_entropy).all()
