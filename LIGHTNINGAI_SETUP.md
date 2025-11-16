# PDCLM Faz-1 Deneyleri - LightningAI T4 GPU Rehberi

## 🚀 Hızlı Başlangıç

LightningAI T4 GPU üzerinde PDCLM Faz-1 deneylerini şu sırayla yap:

### 1. Proje Setup
```bash
# Repository'yi clone et
!git clone https://github.com/inkbytefo/PDCLM.git
%cd PDCLM

# Dependencies kur
!pip install -r requirements.txt

# Test et
!pytest tests/test_model.py -v
```

### 2. GPU Kontrolü
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 3. Faz-1 Deneyleri

#### Deney A: Hızlı Model Validation (5 dk)
```bash
python experiments/quick_test.py
```
**Hedef:** Modelin çalışıp çalışmadığını kontrol et

#### Deney B: 100 Iterasyon Training (15 dk)
```python
# Notebook: experiments/train_test_updated.ipynb
# num_iterations = 100 (GPU ile test)
# Hedef: Loss düşüşü gözlemle
```

#### Deney C: 500 Iterasyon Full Training (30-45 dk)
```python
# Notebook: experiments/train_test_updated.ipynb
# num_iterations = 500
# Final loss < 0.5 hedefi
```

## 🎯 Başarı Kriterleri

| Deney | Hedef | Süre | Sonuç |
|-------|-------|------|-------|
| Quick Test | Model çalışıyor | 5 dk | ✅/❌ |
| 100 Iter | Loss düşüyor | 15 dk | ✅/❌ |
| 500 Iter | Final < 0.5 | 45 dk | ✅/❌ |

## 📊 Monitoring

### Loss Tracking
```python
# Her 50 iterasyonda log
iteration: 50/500 | Loss: 1.234 | Val Loss: 1.456
iteration: 100/500 | Loss: 0.987 | Val Loss: 1.123
```

### WandB (Opsiyonel)
```python
import wandb
wandb.init(project="pdclm-lightning")
wandb.log({"loss": loss, "iteration": i})
```

## ⚡ Optimizasyon İpuçları

### T4 GPU için Optimized Settings
```python
# Model boyutu
embed_dim = 256
num_layers = 4
heads = 4
window_size = 512

# Training
learning_rate = 1e-4
batch_size = 10000
num_iterations = 500

# Memory optimization
torch.cuda.empty_cache()  # Her epoch sonra
```

### Troubleshooting
**GPU Memory Error:**
- Batch size küçült: 5000
- Embed dim düşür: 128

**Slow Training:**
- Data loading optimize et
- Mixed precision kullan: `torch.cuda.amp`

## 🔧 Komutlar

```bash
# Temiz test
!python -c "from src.model import PDCLMBase; print('✅ Import OK')"

# Quick validation
!python experiments/quick_test.py

# Full test suite
!pytest tests/ -v

# GPU memory check
!python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

## 📋 Experiment Log Template

```markdown
## Faz-1 Deney Raporu
**Tarih:** [DATE]
**Hardware:** T4 GPU
**Deney A (Quick Test):** [PASS/FAIL]
**Deney B (100 iter):** [PASS/FAIL] 
**Deney C (500 iter):** Final Loss = [X.XXX]
**Sonuç:** [Faz-1 TAMAM/Faz-1 optimize et]
```

## 🚨 Kritik Komutlar

### Acil Durum - Memory Temizleme
```python
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
```

### Stop Training (Ctrl+C Alternative)
```python
# İçeride çalışan process'i durdur
!pkill -f python
```

---

**🎯 Ana Hedef:** Final Loss < 0.5  
**⚡ Beklenen Süre:** 30-45 dakika  
**🏆 Başarı:** Faz-2 Cognitive Loop'a geçiş
