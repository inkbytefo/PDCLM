#!/usr/bin/env python
# coding: utf-8

# # PSE (PatternStreamEncoder) Test Notebook
# 
# Bu notebook PSE'nin performansını test eder:
# - Veri yükleme ve hazırlık
# - PSE output hesaplama
# - Entropy profili görselleştirme
# - Hız karşılaştırması (PSE vs BPE)

# In[ ]:


# 1. Import'lar ve Setup
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from transformers import GPT2TokenizerFast
from src.pse import PatternStreamEncoder

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# In[ ]:


# 2. PSE Instance Oluşturma
pse = PatternStreamEncoder(window_size=128)

# CUDA kullanılabilirse GPU'ya taşı
if torch.cuda.is_available():
    pse = pse.cuda()
    print("PSE GPU'ya taşındı")
else:
    print("PSE CPU'da çalışıyor")

print(f"PSE parameters: {sum(p.numel() for p in pse.parameters())} parameters")


# In[ ]:


# 3. Veri Yükleme
print("Veri yükleme...")
with open("data/raw/wikitext_sample.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# İlk 50k karakteri al
text = full_text[:50000]
print(f"Toplam dosya boyutu: {len(full_text)} karakter")
print(f"Test için kullanılan: {len(text)} karakter")
print(f"İlk 100 karakter: {text[:100]}...")


# In[ ]:


# 4. PSE Output Hesaplama
print("\n=== PSE OUTPUT HESAPLAMA ===")

start_time = time.time()
stream = pse(text)
pse_time = time.time() - start_time

print(f"Input length: {len(text)} chars")
print(f"Stream shape: {stream.shape}")
print(f"Effective compression: {stream.shape[0] / len(text):.4f} windows/char")
print(f"PSE processing time: {pse_time:.4f}s")


# In[ ]:


# 5. Entropy Profili Görselleştirme
print("\n=== ENTROPY PROFİLİ HESAPLAMA ===")

device = next(pse.parameters()).device
entropy_profile = pse._compute_entropy_profile(torch.tensor([ord(c) for c in text], device=device))

print(f"Entropy profile shape: {entropy_profile.shape}")
print(f"Mean entropy: {entropy_profile.mean().item():.4f}")
print(f"Std entropy: {entropy_profile.std().item():.4f}")

# Görselleştirme
plt.figure(figsize=(12, 4))
plt.plot(entropy_profile.cpu().numpy())
plt.title("Entropy Profile - Pattern Detection")
plt.xlabel("Window Position")
plt.ylabel("Entropy Value")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Experiments klasörüne kaydet
plt.savefig("experiments/entropy_profile.png", dpi=150, bbox_inches='tight')
print("Entropy profili kaydedildi: experiments/entropy_profile.png")
plt.show()


# In[ ]:


# 6. Hız Karşılaştırması: BPE vs PSE
print("\n=== HIZ KARŞILAŞTIRMASI ===")

# BPE Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# BPE test
start = time.time()
bpe_tokens = tokenizer.encode(text)
bpe_time = time.time() - start

# PSE test (tekrar ama daha hassas timing)
start = time.time()
pse_output = pse(text)
pse_time_new = time.time() - start

print(f"BPE time: {bpe_time:.4f}s")
print(f"PSE time: {pse_time_new:.4f}s")
print(f"Speedup: {bpe_time/pse_time_new:.2f}x")
print(f"BPE tokens: {len(bpe_tokens)}")
print(f"PSE windows: {pse_output.shape[0]}")

# Hız analizi
if bpe_time / pse_time_new >= 1.2:
    print("✅ PSE BPE'den hızlı!")
else:
    print("❌ PSE yavaş, window_size optimizasyonu gerekli")
    print("Window size 256 ile tekrar denenmeli.")


# In[ ]:


# 7. Window Size 256 ile Optimizasyon Testi
print("\n=== WINDOW SIZE 256 OPTİMİZASYONU ===")

# PSE256 oluştur
pse_256 = PatternStreamEncoder(window_size=256)
if torch.cuda.is_available():
    pse_256 = pse_256.cuda()

# Test
start = time.time()
stream_256 = pse_256(text)
pse_256_time = time.time() - start

print(f"PSE-128 time: {pse_time_new:.4f}s")
print(f"PSE-256 time: {pse_256_time:.4f}s")
print(f"PSE-128 compression: {stream.shape[0] / len(text):.4f} windows/char")
print(f"PSE-256 compression: {stream_256.shape[0] / len(text):.4f} windows/char")
print(f"PSE-256 vs BPE speedup: {bpe_time / pse_256_time:.2f}x")

if bpe_time / pse_256_time >= 1.2:
    print("✅ PSE-256 BPE'den hızlı!")
else:
    print("❌ PSE-256 de yavaş.")

print(f"\n=== SONUÇ ÖZETİ ===")
print(f"Veri boyutu: {len(text)} karakter")
print(f"BPE tokenization: {bpe_time:.4f}s")
print(f"PSE-128: {pse_time_new:.4f}s (speedup: {bpe_time/pse_time_new:.2f}x)")
print(f"PSE-256: {pse_256_time:.4f}s (speedup: {bpe_time/pse_256_time:.2f}x)")
print(f"Entropy profile kaydedildi: experiments/entropy_profile.png")

