## Developer: inkbytefo
## Modified: 2025-11-17

import sys
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pdclm import PDCLM
from src.utils import load_checkpoint


class EvalRequest(BaseModel):
    task: str


app = FastAPI()

model = PDCLM(embed_dim=512)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
ckpt = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'pdclm_final.pt')
if os.path.exists(ckpt):
    load_checkpoint(model, optimizer, ckpt)
if torch.cuda.is_available():
    model = model.cuda()


@app.post('/evaluate')
def evaluate(req: EvalRequest):
    _, reward, answer = model.evaluate_with_answer(req.task)
    return {"reward": float(reward), "answer": answer}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)