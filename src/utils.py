## Developer: inkbytefo
## Modified: 2025-11-16

"""
Utility functions for PD-CLM project.
"""

import torch
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from scipy.special import softmax


def load_data(file_path: str) -> Dict[str, Any]:
    """
    Load data from file.
    
    Args:
        file_path: Path to data file
        
    Returns:
        Loaded data dictionary
    """
    # TODO: Implement specific data loading logic
    pass


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
