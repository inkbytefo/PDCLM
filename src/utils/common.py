## Developer: inkbytefo
## Modified: 2025-11-21

"""
Utility functions for PD-CLM project.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import json
from scipy.special import softmax
import time
import os
import re

def load_data(file_path: str) -> Dict[str, Any]:
    """
    Load raw text data and basic stats.
    """
    if not os.path.exists(file_path):
        return {"text": "", "length": 0}
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return {"text": text, "length": len(text)}


def preprocess_text(text: str) -> str:
    """
    Preprocess text data.
    
    Args:
        text: Raw text input
        
    Returns:
        Preprocessed text
    """
    # Basic text preprocessing
    text = text.strip()
    text = text.lower()
    return text


def build_reasoning_tasks(n: int = 10) -> List[str]:
    import random
    tasks = []
    for _ in range(n):
        choice = random.choice(["add", "mul", "even", "sort"])
        if choice == "add":
            a, b = random.randint(1, 100), random.randint(1, 100)
            tasks.append(f"What is {a} + {b}?")
        elif choice == "mul":
            a, b = random.randint(1, 50), random.randint(1, 50)
            tasks.append(f"What is {a} * {b}?")
        elif choice == "even":
            a = random.randint(1, 100)
            tasks.append(f"Is {a} even? Answer True/False.")
        else:
            arr = ', '.join(map(str, random.sample(range(1, 20), 5)))
            tasks.append(f"Sort these numbers: {arr}")
    return tasks


def measure_forward_latency(model: torch.nn.Module, text: str) -> float:
    """
    Measure forward pass latency in milliseconds for a given text.
    """
    start = time.time()
    # Handle potential missing attributes safely
    min_len = 32
    if hasattr(model, 'pse') and hasattr(model.pse, 'window_size'):
        min_len = model.pse.window_size
        
    if len(text) < min_len:
        repeat = (min_len // max(len(text), 1)) + 1
        text = (text + " ") * repeat
    with torch.no_grad():
        _ = model(text)
    return (time.time() - start) * 1000.0


def parse_expected_answer(task: str) -> Tuple[Optional[str], Any]:
    t = task.strip()
    if "What is" in t and "+" in t:
        try:
            expr = t.split("is")[1].split("?")[0]
            a, b = [int(x.strip()) for x in expr.split("+")]
            return ("add", a + b)
        except: pass
    if "What is" in t and "*" in t:
        try:
            expr = t.split("is")[1].split("?")[0]
            a, b = [int(x.strip()) for x in expr.split("*")]
            return ("mul", a * b)
        except: pass
    if "even" in t:
        try:
            num = int(t.split("even?")[0].split()[-1])
            return ("even", (num % 2 == 0))
        except: pass
    if "Sort these numbers" in t:
        try:
            part = t.split(":")[1]
            arr = [int(x.strip()) for x in part.split(",")]
            return ("sort", sorted(arr))
        except: pass
    return (None, None)


def compute_hallucination_penalty(task: str, answer: Optional[str]) -> float:
    try:
        kind, expected = parse_expected_answer(task)
        if kind is None or answer is None:
            return 0.1
        if kind in ("add", "mul"):
            return 0.0 if str(expected) == str(int(answer)) else 0.1
        if kind == "even":
            return 0.0 if (answer.lower() in ("true", "false") and ((answer.lower() == "true") == expected)) else 0.1
        if kind == "sort":
            try:
                parsed = [int(x.strip()) for x in answer.strip('[]').split(',') if x.strip()]
            except Exception:
                parsed = None
            return 0.0 if parsed == expected else 0.1
        return 0.05
    except Exception:
        return 0.1


def calculate_entropy(logits: torch.Tensor) -> float:
    """
    Calculate entropy of model predictions.
    
    Args:
        logits: Model output logits
        
    Returns:
        Entropy value
    """
    probabilities = softmax(logits.detach().cpu().numpy(), axis=-1)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=-1)
    return float(np.mean(entropy))


def visualize_training_curve(train_losses: List[float], 
                           val_losses: List[float] = None,
                           save_path: str = None):
    """
    Visualize training and validation losses.
    
    Args:
        train_losses: Training loss values
        val_losses: Validation loss values (optional)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Training Loss', color='blue')
    
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='red')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   save_path: str):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Checkpoint save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   checkpoint_path: str):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file
        
    Returns:
        epoch: Loaded epoch number
        loss: Loaded loss value
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def setup_wandb(project_name: str, experiment_name: str):
    """
    Setup Weights & Biases for experiment tracking.
    
    Args:
        project_name: W&B project name
        experiment_name: Experiment name
    """
    import wandb
    
    wandb.init(
        project=project_name,
        name=experiment_name,
        config={
            "model": "PD-CLM",
            "timestamp": "2025-11-21"
        }
    )
    
    return wandb


def build_sft_dataset(num_samples: int, agent, max_steps: int = 8) -> List[Dict[str, Any]]:
    # from src.hcl import task_generator # TODO: Fix import when HCL is ready
    # For now, we'll assume task_generator is passed or available
    # This might need refactoring
    samples = []
    # Placeholder for task_generator usage
    return samples


def save_sft_jsonl(path: str, samples: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def build_tool_augmented_sft_dataset(num_samples: int, agent, tools) -> List[Dict[str, Any]]:
    import random
    samples = []
    for _ in range(num_samples):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        task = f"What is the result of {a} * {b} and what is the capital of Turkey?"
        calc_q = f"{a} * {b}"
        # Assuming tools has run_call method
        calc_res = tools.run_call(f'[TOOL_CALL: calculator("{calc_q}")]')
        srch_q = "capital of Turkey"
        srch_res = tools.run_call(f'[TOOL_CALL: search("{srch_q}")]')
        target = (
            f"Step 1: I need to solve the multiplication. [TOOL_CALL: calculator(\"{calc_q}\")]\n"
            f"[TOOL_RESULT: \"{calc_res}\"]\n"
            f"Step 2: Now I need to find the capital of Turkey. [TOOL_CALL: search(\"{srch_q}\")]\n"
            f"[TOOL_RESULT: \"{srch_res}\"]\n"
            f"Final answer: The result is {calc_res} and the capital of Turkey is Ankara."
        )
        samples.append({"input": task, "target": target})
    return samples


def _extract_number(text: str) -> int | None:
    nums = re.findall(r"[-+]?\d+", text)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except Exception:
        return None


def _parse_gsm8k_answer(ans: str) -> int | None:
    m = re.findall(r"####\s*([-+]?\d+)", ans)
    if m:
        try:
            return int(m[-1])
        except Exception:
            return None
    return _extract_number(ans)


def evaluate_gsm8k(model, split: str = "test", sample_size: int = 100, max_new_bytes: int = 128) -> Dict[str, Any]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    import random
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[:min(sample_size, len(ds))]
    correct = 0
    results = []
    for i in idxs:
        q = ds[i]["question"].strip()
        gt = _parse_gsm8k_answer(ds[i]["answer"])  # type: ignore
        prompt = q + "\nStep 1: "
        gen = model.generate_text(prompt, max_new_bytes=max_new_bytes)
        pred = None
        if "Final answer:" in gen:
            pred = _extract_number(gen.split("Final answer:")[-1])
        if pred is None:
            pred = _extract_number(gen)
        ok = (pred is not None and gt is not None and int(pred) == int(gt))
        correct += int(bool(ok))
        results.append({"question": q, "pred": pred, "gt": gt, "ok": bool(ok)})
    acc = correct / max(len(idxs), 1)
    return {"split": split, "sample_size": len(idxs), "accuracy": acc, "results": results}
