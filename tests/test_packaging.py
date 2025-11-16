## Developer: inkbytefo
## Modified: 2025-11-17

import os
import sys
sys.path.append('..')

import torch
from src.pdclm import PDCLM
from src.utils import save_checkpoint, load_checkpoint


def test_checkpoint_save_and_load():
    os.makedirs('checkpoints', exist_ok=True)
    model = PDCLM(embed_dim=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = 'checkpoints/test_ckpt.pt'
    save_checkpoint(model, optimizer, 0, 0.0, path)
    epoch, loss = load_checkpoint(model, optimizer, path)
    assert epoch == 0
    assert isinstance(loss, float)