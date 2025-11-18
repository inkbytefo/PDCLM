## Developer: inkbytefo
## Modified: 2025-11-18

import os
import json
import argparse
from datasets import load_dataset
import numpy as np
import faiss
import torch
from src.pse import PatternStreamEncoder

def write_text(path, dataset, text_key: str = "text"):
    with open(path, "w", encoding="utf-8") as f:
        for ex in dataset:
            t = ex.get(text_key, "") if isinstance(ex, dict) else ex[text_key]
            if t and str(t).strip():
                f.write(str(t).strip() + "\n")

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        c = text[i:j].strip()
        if c:
            chunks.append(c)
        i = j
    return chunks

def build_faiss_index(chunks: list[str], embed_dim: int = 512, device: str = "cpu"):
    pse = PatternStreamEncoder(embed_dim=embed_dim)
    if device == "cuda" and torch.cuda.is_available():
        pse = pse.cuda()
    vecs = []
    for c in chunks:
        v = pse(c)
        m = v.mean(dim=0).detach().float().cpu().numpy()
        vecs.append(m)
    X = np.stack(vecs, axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Xn = X / norms
    index = faiss.IndexFlatIP(Xn.shape[1])
    index.add(Xn.astype(np.float32))
    return index, Xn

def save_index(index, out_dir: str, chunks: list[str], embed_dim: int):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"embed_dim": embed_dim}, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset_pct", type=float, default=1.0)
    parser.add_argument("--full", type=str, default="false")
    parser.add_argument("--build_index", type=str, default="true")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--index_dir", type=str, default="data/index")
    parser.add_argument("--embed_dim", type=int, default=512)
    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)

    ds_name = args.dataset.lower()
    full = args.full.lower() == "true"
    pct = args.subset_pct if not full else 100.0
    if ds_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"{args.split}[:{pct}%]")
        out_txt = f"data/raw/wikitext_{args.split}.txt" if full else "data/raw/wikitext_sample.txt"
        write_text(out_txt, ds, "text")
        src_path = out_txt
    elif ds_name == "c4":
        ds = load_dataset("c4", "en", split=f"{args.split}[:{pct}%]")
        out_txt = f"data/raw/c4_{args.split}.txt" if full else "data/raw/c4_sample.txt"
        write_text(out_txt, ds, "text")
        src_path = out_txt
    elif ds_name == "pile":
        ds = load_dataset("EleutherAI/pile", split=f"{args.split}[:{pct}%]")
        out_txt = f"data/raw/pile_{args.split}.txt" if full else "data/raw/pile_sample.txt"
        write_text(out_txt, ds, "text")
        src_path = out_txt
    elif ds_name == "wikipedia":
        ds = load_dataset("wikipedia", "20220301.en", split=f"{args.split}[:{pct}%]")
        out_txt = f"data/raw/wikipedia_{args.split}.txt" if full else "data/raw/wikipedia_sample.txt"
        write_text(out_txt, ds, "text")
        src_path = out_txt
    else:
        raise SystemExit(1)

    if args.build_index.lower() == "true":
        txt = read_text(src_path)
        chunks = chunk_text(txt, args.chunk_size)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        index, Xn = build_faiss_index(chunks, embed_dim=args.embed_dim, device=device)
        save_index(index, args.index_dir, chunks, embed_dim=args.embed_dim)

if __name__ == "__main__":
    main()
