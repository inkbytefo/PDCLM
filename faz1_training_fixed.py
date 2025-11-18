#!/usr/bin/env python3
## Developer: inkbytefo
## Modified: 2025-11-18
"""
PDCLM Faz-1 Training Script (500 iterations) - FIXED VERSION
T4 GPU ile optimized training script
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# Import our model and utilities
from src.model import PDCLMBase, pretrain_step, create_batches
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def load_text_file(file_path):
    """Simple text file loader"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return None

def main():
    print("🚀 PDCLM Faz-1 Training Script (500 iterations) - FIXED")
    print("="*60)
    
    # Setup
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    ddp = world > 1
    if ddp and not dist.is_initialized():
        dist.init_process_group(backend='gloo')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    print(f"🔥 GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Load data
    data_path = "data/raw/wikitext_sample.txt"
    print(f"\n📖 Loading data from: {data_path}")
    
    raw_text = load_text_file(data_path)
    if raw_text:
        print(f"✅ Data loaded successfully: {len(raw_text):,} characters")
    else:
        print(f"❌ Data file not found: {data_path}")
        # Create sample data
        raw_text = "This is a sample text for testing the PDCLM model. " * 1000
        print(f"📝 Using synthetic data: {len(raw_text):,} characters")
    
    # Split data for validation (last 10k chars)
    val_split = max(10000, len(raw_text) // 10)  # At least 10k or 10% of data
    train_text = raw_text[:-val_split]
    val_text = raw_text[-val_split:]
    
    print(f"📊 Data split:")
    print(f"  - Training: {len(train_text):,} characters")
    print(f"  - Validation: {len(val_text):,} characters")
    
    # Initialize model
    print(f"\n🤖 Initializing PDCLMBase model...")
    model = PDCLMBase(embed_dim=256, num_layers=4, heads=4, window_size=512)
    model = model.to(device)
    model_wrapped = DDP(model) if ddp else model
    
    print(f"✅ Model created")
    print(f"📊 Parameters: {model.count_parameters():,}")
    
    # Initialize optimizer
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training parameters
    batch_size = 10000  # characters per batch
    num_iterations = 500
    log_interval = 50
    val_interval = 50
    
    print(f"\n🎯 Training configuration:")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Batch size: {batch_size:,} characters")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Log interval: {log_interval}")
    print(f"  - Validation interval: {val_interval}")
    
    # Create training batches
    print(f"\n📦 Creating training batches...")
    train_batches = list(create_batches(train_text, batch_size=batch_size))
    print(f"✅ Created {len(train_batches)} training batches")
    
    # Training loop
    print(f"\n🚀 Starting 500-iteration training...")
    print("="*60)
    
    train_losses = []
    val_losses = []
    start_time = datetime.now()
    
    model.train()
    for iteration in range(num_iterations):
        # Select batch (cycle through available batches)
        batch_text = train_batches[iteration % len(train_batches)]
        
        # Training step
        try:
            loss = pretrain_step(model_wrapped, batch_text, optimizer, device)
            train_losses.append(loss)
            
            # Logging
            if iteration % log_interval == 0 and rank == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"Iteration {iteration:3d}/{num_iterations} | Train Loss: {loss:.6f} | Time: {elapsed:.1f}s")
            
            # Validation
            if iteration % val_interval == 0 and rank == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = model(val_text)
                    val_losses.append(val_loss.item())
                    print(f"           | Val Loss:   {val_loss.item():.6f}")
                model.train()
            
            # Check for NaN
            if np.isnan(loss):
                print(f"❌ NaN loss detected at iteration {iteration}")
                break
                
        except Exception as e:
            print(f"❌ Error at iteration {iteration}: {str(e)}")
            break
    
    # Training completed
    total_time = (datetime.now() - start_time).total_seconds()
    if rank == 0:
        print("="*60)
        print(f"✅ Training completed!")
        print(f"📊 Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"📈 Training iterations: {len(train_losses)}")
        print(f"🔍 Validation checks: {len(val_losses)}")
    
    if train_losses and rank == 0:
        final_train_loss = train_losses[-1]
        min_train_loss = min(train_losses)
        print(f"📈 Final train loss: {final_train_loss:.6f}")
        print(f"📉 Best train loss: {min_train_loss:.6f}")
    
    if val_losses and rank == 0:
        final_val_loss = val_losses[-1]
        min_val_loss = min(val_losses)
        print(f"🔍 Final val loss: {final_val_loss:.6f}")
        print(f"🔍 Best val loss: {min_val_loss:.6f}")
        
        # Convergence check
        if len(val_losses) >= 10:
            recent_vals = val_losses[-10:]
            convergence = np.std(recent_vals) < 0.001  # Low variance in last 10 checks
            print(f"🎯 Convergence: {'YES' if convergence else 'NO'}")
    
    # Loss visualization
    print(f"\n📊 Creating loss plot...")
    
    if rank == 0:
        plt.figure(figsize=(14, 8))
        
        # Plot training losses
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Faz-1 Next-Pattern Prediction Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add loss statistics
        if train_losses:
            final_loss = train_losses[-1]
            min_loss = min(train_losses)
            plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Target Loss (0.5)')
            plt.text(0.02, 0.98, f'Final: {final_loss:.4f}\nMin: {min_loss:.4f}', 
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.legend()
    
    # Plot validation losses
        plt.subplot(2, 1, 2)
        val_iters = list(range(0, len(train_losses), val_interval))[:len(val_losses)]
        plt.plot(val_iters, val_losses, 'r-', linewidth=2, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Validation Loss')
        plt.title('Faz-1 Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    if val_losses:
        final_val = val_losses[-1]
        min_val = min(val_losses)
        plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Target (0.5)')
        plt.text(0.02, 0.98, f'Final: {final_val:.4f}\nMin: {min_val:.4f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.legend()
    
        plt.tight_layout()
    
    # Save plot
    if rank == 0:
        plot_path = "experiments/pretrain_loss.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 Loss plot saved: {plot_path}")
    
    # plt.show()  # Comment out for headless operation
    
    # Final assessment
    if rank == 0:
        print(f"\n" + "="*60)
        print(f"🏁 FAZ-1 TRAINING ASSESSMENT")
        print(f"="*60)
    
    final_train_loss = train_losses[-1] if train_losses else float('inf')
    final_val_loss = val_losses[-1] if val_losses else float('inf')
    
    if rank == 0:
        print(f"📊 Results:")
        print(f"  - Final training loss: {final_train_loss:.6f}")
        print(f"  - Final validation loss: {final_val_loss:.6f}")
        print(f"  - Training time: {total_time:.1f}s")
        print(f"  - Iterations completed: {len(train_losses)}/500")
    
    # Target evaluation
    if rank == 0:
        print(f"\n🎯 Target Evaluation:")
    
    if final_train_loss < 0.7:
        print(f"  ✅ TRAIN LOSS < 0.7: SUCCESS")
        decision = "Faz-1 TAMAM, Cognitive Loop'a geç"
    elif final_train_loss > 1.0:
        print(f"  ❌ TRAIN LOSS > 1.0: NEEDS OPTIMIZATION")
        decision = "PSE output variance artır (scale=5.0) veya data çeşitlendir"
    else:
        print(f"  ⚠️ TRAIN LOSS between 0.7-1.0: MODERATE")
        decision = "Faz-1 kabul edilebilir, optimize edilebilir"
    
    if rank == 0:
        print(f"\n🏆 DECISION: {decision}")
    
    # Save results
    results = {
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'min_train_loss': min(train_losses) if train_losses else None,
        'min_val_loss': min(val_losses) if val_losses else None,
        'training_time': total_time,
        'iterations': len(train_losses),
        'decision': decision
    }
    
    import json
    if rank == 0:
        os.makedirs('experiments', exist_ok=True)
        with open('experiments/faz1_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to experiments/faz1_results.json")
        print(f"🎉 PDCLM Faz-1 training completed!")
    
    return final_train_loss

if __name__ == "__main__":
    main()
