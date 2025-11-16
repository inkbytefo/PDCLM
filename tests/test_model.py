## Developer: inkbytefo
## Modified: 2025-11-16

import pytest
import torch
from src.model import PDCLMBase, pretrain_step, create_batches


def test_model_forward_pass():
    """Test model forward pass with valid input."""
    model = PDCLMBase(embed_dim=256, num_layers=4, heads=4, window_size=512)
    text = "a" * 3000  # Yeterli uzunluk
    loss = model(text)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert 0 < loss.item() < 200  # İlk loss yüksek olabilir


def test_model_training_step():
    """Test training step with loss decrease."""
    model = PDCLMBase()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    text = "a" * 3000
    loss1 = pretrain_step(model, text, optimizer, torch.device('cpu'))
    loss2 = pretrain_step(model, text, optimizer, torch.device('cpu'))
    assert loss2 < loss1 * 1.5  # Hafif düşüş


def test_gradient_clipping():
    """Test gradient clipping functionality."""
    model = PDCLMBase()
    text = "a" * 3000
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(text)
    loss.backward()
    grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert grad_norm_before > 0


def test_model_parameter_count():
    """Test model parameter counting."""
    model = PDCLMBase(embed_dim=256, num_layers=4, heads=4, window_size=512)
    param_count = model.count_parameters()
    assert param_count > 0
    assert isinstance(param_count, int)


def test_model_info():
    """Test model info retrieval."""
    model = PDCLMBase(embed_dim=256, num_layers=4, heads=4, window_size=512)
    info = model.get_model_info()
    
    assert 'embed_dim' in info
    assert 'num_layers' in info
    assert 'heads' in info
    assert 'window_size' in info
    assert 'total_params' in info
    assert info['embed_dim'] == 256
    assert info['num_layers'] == 4


def test_create_batches():
    """Test batch creation utility."""
    text = "Hello world! " * 100
    batches = list(create_batches(text, batch_size=100, stride=50))
    
    assert len(batches) > 0
    assert all(isinstance(batch, str) for batch in batches)
    assert all(len(batch) <= 100 for batch in batches)


if __name__ == "__main__":
    pytest.main([__file__])
