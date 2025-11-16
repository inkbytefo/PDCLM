## Developer: inkbytefo
## Modified: 2025-11-17

"""
Utility functions for PD-CLM project.
"""

import torch
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from scipy.special import softmax
import time
import os


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
    min_len = getattr(getattr(model, 'pse', None), 'window_size', 32)
    if len(text) < min_len:
        repeat = (min_len // max(len(text), 1)) + 1
        text = (text + " ") * repeat
    with torch.no_grad():
        _ = model(text)
    return (time.time() - start) * 1000.0


def parse_expected_answer(task: str):
    t = task.strip()
    if "What is" in t and "+" in t:
        expr = t.split("is")[1].split("?")[0]
        a, b = [int(x.strip()) for x in expr.split("+")]
        return ("add", a + b)
    if "What is" in t and "*" in t:
        expr = t.split("is")[1].split("?")[0]
        a, b = [int(x.strip()) for x in expr.split("*")]
        return ("mul", a * b)
    if "even" in t:
        num = int(t.split("even?")[0].split()[-1])
        return ("even", (num % 2 == 0))
    if "Sort these numbers" in t:
        part = t.split(":")[1]
        arr = [int(x.strip()) for x in part.split(",")]
        return ("sort", sorted(arr))
    return (None, None)


def compute_hallucination_penalty(task: str, answer: str | None) -> float:
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
            "timestamp": "2025-11-16"
        }
    )
    
    return wandb
