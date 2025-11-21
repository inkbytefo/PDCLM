## Developer: inkbytefo
## Modified: 2025-11-17

import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pdclm import PDCLM
from src.utils import load_checkpoint


class EvalWrapper(torch.nn.Module):
    def __init__(self, model: PDCLM):
        super().__init__()
        self.model = model

    def forward(self, task: str):
        _, reward, _ = self.model.evaluate_with_answer(task)
        return torch.tensor([reward], dtype=torch.float32)


def main():
    model = PDCLM(embed_dim=512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ckpt = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'pdclm_final.pt')
    if os.path.exists(ckpt):
        load_checkpoint(model, optimizer, ckpt)
    wrapper = EvalWrapper(model)
    try:
        scripted = torch.jit.script(wrapper)
        out_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'pdclm_eval_script.pt')
        scripted.save(out_path)
        print(f"Saved TorchScript to: {out_path}")
    except Exception as e:
        out_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'pdclm_eval_fallback.pt')
        torch.save(wrapper.state_dict(), out_path)
        print(f"TorchScript failed, saved fallback state_dict to: {out_path}. Error: {e}")


if __name__ == '__main__':
    main()