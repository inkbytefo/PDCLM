## Developer: inkbytefo
## Modified: 2025-11-17

"""
Quick test script for PDCLM Faz-1 validation
Kısa training testi - model çalışıyor mu kontrol et
"""

import sys
import os
sys.path.append('..')

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt

from src.model import PDCLMBase, pretrain_step, create_batches

print("✅ Quick Faz-1 Test Starting...")

# Load smaller dataset
data_path = "../data/raw/wikitext_sample.txt"
with open(data_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

# Use smaller subset for quick test
test_text = raw_text[:50000]  # 50k characters only
print(f"📏 Test data: {len(test_text):,} characters")

# Initialize smaller model for quick test
model = PDCLMBase(embed_dim=128, num_layers=2, heads=2, window_size=256)
device = torch.device('cpu')
model = model.to(device)

print(f"🤖 Model parameters: {model.count_parameters():,}")

# Quick training test (50 iterations only)
optimizer = AdamW(model.parameters(), lr=1e-3)  # Higher LR for faster convergence
losses = []

print("🚀 Quick training test (50 iterations)...")
model.train()

for i in range(50):
    # Small batch
    batch_text = test_text[i*1000:(i+1)*1000] if i < 50 else test_text[:1000]
    
    try:
        loss = pretrain_step(model, batch_text, optimizer, device)
        losses.append(loss)
        
        if i % 10 == 0:
            print(f"  Iteration {i:2d} | Loss: {loss:.6f}")
            
        if np.isnan(loss):
            print(f"❌ NaN loss at iteration {i}")
            break
            
    except Exception as e:
        print(f"❌ Error at iteration {i}: {str(e)}")
        break

# Results
final_loss = losses[-1] if losses else float('inf')
min_loss = min(losses) if losses else float('inf')

print(f"\n📊 Quick Test Results:")
print(f"  - Completed iterations: {len(losses)}/50")
print(f"  - Final loss: {final_loss:.6f}")
print(f"  - Best loss: {min_loss:.6f}")
print(f"  - Loss reduction: {((losses[0] - final_loss) / losses[0] * 100):.1f}%")

# Model validation
model.eval()
with torch.no_grad():
    val_loss = model(test_text[:2000])  # 2k chars validation
    print(f"  - Validation loss: {val_loss.item():.6f}")

# Assessment
if final_loss < 2.0 and not np.isnan(final_loss):
    print(f"\n✅ FAZ-1 MODEL WORKS! Pattern prediction functional.")
    convergence = "Evet" if final_loss < 1.0 else "Kısmi"
else:
    print(f"\n❌ Model needs optimization")

# Save loss plot
plt.figure(figsize=(10, 6))
plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Faz-1 Quick Test - Next-Pattern Prediction Loss')
plt.grid(True, alpha=0.3)
plt.legend()

final_loss_str = f"{final_loss:.4f}"
plt.text(0.02, 0.98, f'Final: {final_loss_str}\nMin: {min_loss:.4f}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("quick_test_loss.png", dpi=300, bbox_inches='tight')
print(f"💾 Loss plot saved: quick_test_loss.png")

print(f"\n🏁 Quick Test Complete!")
