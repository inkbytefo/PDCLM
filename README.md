# PDCLM - Pattern-Driven Cognitive Language Model

## Proje Hakkında

PDCLM, geleneksel tokenizer yaklaşımının ötesinde, **Pattern Stream Encoder (PSE)** tabanlı yeni bir LLM mimarisi sunar.

## Faz-1: Continuous Stream Pretraining ✅

- **PSE Performance**: 0.28s / 50k char (optimized)
- **Model**: PDCLMBase (4 layer, 256 dim, 4 head)
- **Parameter Count**: 37,899,777
- **Test Suite**: 6/6 pytest PASS ✅
- **Convergence**: Functional (GPU'da test edilmeli)

### Model Architecture
```
PDCLMBase(
  embed_dim=256,
  num_layers=4, 
  heads=4,
  window_size=512
)
```

### Test Results
```bash
pytest tests/test_model.py -v
# 6 passed in 25.84s ✅
```

## Google Colab'da Training

1. Repository'yi clone edin:
```bash
!git clone https://github.com/inkbytefo/PDCLM.git
%cd PDCLM
```

2. Notebook'u çalıştırın:
```bash
# experiments/train_test_updated.ipynb
# GPU runtime ile 500 iterasyon training
```

3. Quick Test (küçük model):
```bash
python experiments/quick_test.py
```

## Dosya Yapısı
```
pdclm_project/
├── src/
│   ├── model.py          # PDCLMBase model
│   ├── pse.py           # Pattern Stream Encoder
│   ├── utils.py         # Training utilities
│   └── entropy_utils.py # Entropy analysis
├── tests/
│   ├── test_model.py    # Model tests (6 PASS)
│   └── test_pse.py      # PSE tests
├── experiments/
│   ├── train_test_updated.ipynb    # 500 iter training
│   ├── quick_test.py               # Quick validation
│   └── pse_optimized_test.ipynb    # PSE optimization
├── data/
│   └── raw/
│       └── wikitext_sample.txt     # Training data (5.3M chars)
├── LICENSE                     # Özel kullanım lisansı
└── requirements.txt                # Dependencies
```

## Faz-2: Cognitive Loop (Next)
- Dynamic pattern adjustment
- Context-aware memory
- Self-optimization loop

## Performans Hedefleri

| Metric | Target | Current |
|--------|---------|---------|
| Final Loss | < 0.5 | TBD (GPU) |
| Training Speed | 100 iter/hr | CPU: 25s/iter |
| Memory Usage | < 8GB | ~2GB |

## Setup
```bash
pip install -r requirements.txt
pytest tests/                    # Run tests
python experiments/quick_test.py # Quick validation
```

## Lisans

Bu proje **Özel Kullanım Lisansı** altında yayınlanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## Author

**Tevfik İşkın**  
Türkiye Cumhuriyeti  
Pattern-Driven Language Model Research

---

> **Not:** Bu proje Tevfik İşkın'ın kişisel araştırmasıdır ve özel kullanım lisansı ile korunmaktadır.
