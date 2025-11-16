## Dosya ve Yapı
- `src/hcl.py`: HCL modülü (CCM, HCLAgent, AgentSim, task_generator, compute_reward, hcl_train_step)
- `tests/test_hcl.py`: CCM tutarlılık testi
- `experiments/hcl_experiment.py`: 50 iterasyon self-play ve ödül grafiği (notebook yerine script)

## Uygulama Detayları
- **CCM**: CoT adımlarını ortalama havuzla → `[dim]`; `sigmoid(linear)` ile [0,1]; derinlik ortalaması.
- **HCLAgent**: CoT üretimi; PSE+Transformer ile akış; `embed_cot` ile her adımı `[dim]`e indirgeme.
- **AgentSim**: proposer-critic diyalog; critic CCM ile skorlar; cevap çıkarımı basit toplama.
- **TaskGenerator**: `"What is a + b?"` üretir.
- **Reward**: `0.7*coherence + 0.3*correctness`.
- **RL**: `hcl_train_step` içinde sadece CCM parametrelerine basit gradyan yükselişi (model sabit).

## Script Akışı (`experiments/hcl_experiment.py`)
1. Nesneler: `PDCLMBase`, `CognitiveControlModule`, iki `HCLAgent`.
2. 50 iterasyon: Her iterasyonda `hcl_train_step(..., num_tasks=10)`; `rewards.append(...)`.
3. Grafik ve kayıt: `experiments/hcl_reward.png`.
4. Ortalama ödül kontrolü ve mesaj yazdırma.

## Çalıştırma
- `pytest -q`
- `python experiments/hcl_experiment.py`

## Kriterler
- Ortalama ödül ≥ 0.7 ise: "Faz-2 çalışıyor, Reflection'a geç."; aksi: "Task çeşitlendir, CoT decode ekle."