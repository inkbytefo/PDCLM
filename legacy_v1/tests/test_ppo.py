## Developer: inkbytefo
## Modified: 2025-11-18

import pytest
from src.pdclm import PDCLM, ppo_train


def test_ppo_train_runs():
    model = PDCLM(embed_dim=256)
    ppo_train(model, epochs=1, tasks_per_epoch=2)
    assert True


if __name__ == "__main__":
    pytest.main([__file__])