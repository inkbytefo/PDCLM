## Developer: inkbytefo
## Modified: 2025-11-18

import re
import math
import os
import json
import numpy as np
from typing import List
import torch
import faiss
from .pse import PatternStreamEncoder


class BaseTool:
    def execute(self, query: str) -> str:
        raise NotImplementedError


class CalculatorTool(BaseTool):
    def execute(self, query: str) -> str:
        expr = re.sub(r"[^0-9\+\-\*\/\(\)\s]", "", query)
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return "ERROR"


class SearchTool(BaseTool):
    def __init__(self, kb_text: str | None = None, index_dir: str | None = None):
        self.kb_text = kb_text or ""
        self.pse = PatternStreamEncoder()
        self.index = None
        self.chunks = []
        if index_dir and os.path.exists(os.path.join(index_dir, "faiss.index")):
            self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
            with open(os.path.join(index_dir, "chunks.jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    t = o.get("text", "")
                    if t:
                        self.chunks.append(t)

    def _split(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"[\n\.!?]", text) if s.strip()]

    def execute(self, query: str) -> str:
        min_len = getattr(self.pse, "window_size", 32)
        qtxt = query
        if len(qtxt) < min_len:
            repeat = (min_len // max(len(qtxt), 1)) + 1
            qtxt = (qtxt + " ") * repeat
        qv = self.pse(qtxt)
        q = qv.mean(dim=0).detach().float().cpu().numpy()
        q /= (np.linalg.norm(q) + 1e-9)
        if self.index is not None and self.chunks:
            D, I = self.index.search(q.reshape(1, -1).astype(np.float32), 3)
            sel = [self.chunks[i] for i in I[0] if i < len(self.chunks)]
            return " \n".join(sel)
        if not self.kb_text:
            return ""
        sentences = self._split(self.kb_text)
        if not sentences:
            return ""
        scores = []
        for s in sentences:
            stxt = s
            if len(stxt) < min_len:
                repeat = (min_len // max(len(stxt), 1)) + 1
                stxt = (stxt + " ") * repeat
            sv = self.pse(stxt)
            t = sv.mean(dim=0)
            sim = torch.nn.functional.cosine_similarity(torch.tensor(q), t.detach().float().cpu(), dim=0)
            scores.append(float(sim.item()))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        return " \n".join([sentences[i] for i in idx])


class ToolExecutor:
    def __init__(self):
        self.tools = {}

    def register(self, name: str, tool: BaseTool):
        self.tools[name.lower()] = tool

    def run_call(self, call_text: str) -> str:
        m = re.search(r"\[TOOL_CALL:\s*(\w+)\(\"(.*?)\"\)\s*\]", call_text)
        if not m:
            return ""
        name = m.group(1).lower()
        arg = m.group(2)
        tool = self.tools.get(name)
        if not tool:
            return ""
        return tool.execute(arg)