#!/usr/bin/env python3
## Developer: inkbytefo
## Modified: 2025-11-18
"""
PDCLM Faz-1 Training Script (500 iterations) - FIXED VERSION
T4 GPU ile optimized training script
"""

import sys
import os
import argparse
sys.path.append('.')

import torch
import torch.nn as nn
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import random

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/raw/wikitext_sample.txt')
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--val-interval', type=int, default=50)
    parser.add_argument('--val-chars', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='pdclm')
    parser.add_argument('--wandb-run', type=str, default='faz1')
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--window-size', type=int, default=256)
    parser.add_argument('--max-windows', type=int, default=16)
    parser.add_argument('--pse-scale', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--early-stop-delta', type=float, default=1e-3)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    args = parser.parse_args()

    print("🚀 PDCLM Faz-1 Training Script (500 iterations) - FIXED")
    print("="*60)
    
    # Setup
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    ddp = world > 1
    if ddp and not dist.is_initialized():
        dist.init_process_group(backend=('nccl' if torch.cuda.is_available() else 'gloo'))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    print(f"🔧 Device: {device}")
    print(f"🔥 GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    if not args.wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    else:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run, config={
                'iterations': args.iterations,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'log_interval': args.log_interval,
                'val_interval': args.val_interval,
                'data_path': args.data_path
            })
        except Exception:
            os.environ['WANDB_DISABLED'] = 'true'
    
    data_path = args.data_path
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
    # Seeding for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = PDCLMBase(embed_dim=args.embed_dim, num_layers=args.num_layers, heads=args.heads, window_size=args.window_size, max_windows=args.max_windows)
    model = model.to(device)
    if ddp:
        model_wrapped = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None, output_device=local_rank if device.type == 'cuda' else None, find_unused_parameters=False)
    else:
        model_wrapped = model
    
    if args.pse_scale is not None:
        try:
            model.pse.scale.data = torch.tensor(float(args.pse_scale), device=device)
            print(f"🔧 PSE scale set to {args.pse_scale}")
        except Exception:
            print(f"⚠️ Could not set PSE scale")
    print(f"✅ Model created")
    print(f"📊 Parameters: {model.count_parameters():,}")
    
    learning_rate = args.lr
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    torch.cuda.empty_cache()
    os.environ['PYTORCH_ALLOC_CONF'] = os.environ.get('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    
    batch_size = args.batch_size
    num_iterations = args.iterations
    log_interval = args.log_interval
    val_interval = args.val_interval
    
    print(f"\n🎯 Training configuration:")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Batch size: {batch_size:,} characters")
    print(f"  - Accumulation steps: {args.accumulation_steps}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Log interval: {log_interval}")
    print(f"  - Validation interval: {val_interval}")
    
    # Create training batches
    print(f"\n📦 Creating training batches...")
    stride = max(batch_size // 2, args.window_size)
    train_batches = list(create_batches(train_text, batch_size=batch_size, stride=stride))
    print(f"✅ Created {len(train_batches)} training batches")
    
    # Training loop
    print(f"\n🚀 Starting {num_iterations}-iteration training...")
    print("="*60)
    
    train_losses = []
    val_losses = []
    start_time = datetime.now()
    
    model.train()
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_iter = 0
    if args.resume:
        ckpt_path = ckpt_dir / 'faz1_last.pt'
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location=device)
            (model_wrapped.module if isinstance(model_wrapped, DDP) else model).load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_iter = int(state.get('iteration', 0))
    best_val = float('inf')
    no_improve = 0
    accumulation_steps = max(1, int(args.accumulation_steps))
    optimizer.zero_grad()
    for iteration in range(start_iter, num_iterations):
        # Select batch (cycle through available batches)
        batch_text = train_batches[iteration % len(train_batches)]
        
        # Training step
        try:
            mini_chars = min(1024, max(batch_size // 8, 512))
            try:
                loss = pretrain_step(
                    model_wrapped,
                    batch_text[:batch_size],
                    optimizer,
                    device,
                    mini_batch_chars=mini_chars,
                    do_step=False,
                    scale_loss=1.0/accumulation_steps
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                mini_chars = max(256, mini_chars // 2)
                loss = pretrain_step(
                    model_wrapped,
                    batch_text[:batch_size],
                    optimizer,
                    device,
                    mini_batch_chars=mini_chars,
                    do_step=False,
                    scale_loss=1.0/accumulation_steps
                )
            train_losses.append(loss)

            if (iteration + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_((model_wrapped.module if isinstance(model_wrapped, DDP) else model).parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Logging
            if iteration % log_interval == 0 and rank == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                current_loss = loss
                print(f"Iteration {iteration:3d}/{num_iterations} | Train Loss: {current_loss:.6f} | Time: {elapsed:.1f}s")
                if args.wandb and os.environ.get('WANDB_DISABLED') != 'true':
                    try:
                        import wandb
                        wandb.log({'train_loss': current_loss, 'iter': iteration, 'time_s': elapsed})
                    except Exception:
                        pass
            
            if iteration % val_interval == 0 and rank == 0:
                model.eval()
                with torch.no_grad():
                    target_model = (model_wrapped.module if isinstance(model_wrapped, DDP) else model_wrapped)
                    vlen = max(args.window_size, min(args.val_chars, len(val_text)))
                    val_chunk = val_text[-vlen:]
                    val_loss = target_model(val_chunk)
                    val_losses.append(val_loss.item())
                    print(f"           | Val Loss:   {val_loss.item():.6f}")
                    if args.wandb and os.environ.get('WANDB_DISABLED') != 'true':
                        try:
                            import wandb
                            wandb.log({'val_loss': val_loss.item(), 'iter': iteration})
                        except Exception:
                            pass
                model.train()

                if val_losses:
                    current = val_losses[-1]
                    if current + args.early_stop_delta < best_val:
                        best_val = current
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= args.early_stop_patience:
                            print(f"🛑 Early stopping at iteration {iteration} (patience={args.early_stop_patience})")
                            break
            # Check for NaN
            if np.isnan(loss):
                print(f"❌ NaN loss detected at iteration {iteration}")
                break
                
            if args.save_every and rank == 0 and iteration % args.save_every == 0:
                state = {
                    'model': (model_wrapped.module if isinstance(model_wrapped, DDP) else model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iteration': iteration
                }
                torch.save(state, str(ckpt_dir / 'faz1_last.pt'))
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
    
    results = {
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'min_train_loss': min(train_losses) if train_losses else None,
        'min_val_loss': min(val_losses) if val_losses else None,
        'training_time': total_time,
        'iterations': len(train_losses),
        'decision': decision
    }
    
    if rank == 0:
        os.makedirs('experiments', exist_ok=True)
        with open('experiments/faz1_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        state = {
            'model': (model_wrapped.module if isinstance(model_wrapped, DDP) else model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': len(train_losses)
        }
        torch.save(state, str(ckpt_dir / 'faz1_last.pt'))
        
        print(f"\n💾 Results saved to experiments/faz1_results.json")
        print(f"🎉 PDCLM Faz-1 training completed!")
    
    return final_train_loss

if __name__ == "__main__":
    main()
