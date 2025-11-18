## Amaç

* Mevcut muhakeme değerlendiricisini, byte düzeyinde metin (CoT) üretebilen otonom bir ajana dönüştürmek.

* Kurallı `parse_and_reason` yerine modelin kendisinin CoT üretmesi.

## Mevcut Durum Özeti

* Model gövdesi: `PDCLMBase` sürekli pattern tahmini ve `MSELoss` kullanıyor (`src/model.py:34`, `src/model.py:36`).

* Yansıma: `ReflectivePDCLM` ile BCE tabanlı ek kayıp (`src/reflection.py:32`, `src/reflection.py:37`).

* HCL: CoT kural tabanlı üretiliyor (`HCLAgent`, `parse_and_reason` → `src/hcl.py:33`, `src/hcl.py:47`), veri üretimi `task_generator` (`src/hcl.py:102`).

* Tam eğitim döngüsü: `full_train` (`src/pdclm.py:85`), ana sınıf: `PDCLM` (`src/pdclm.py:18`).

* ByteLMHead kavramsal olarak dokümanda var, kodda yok.

## Teknik Hedefler

* Byte‑LM Head: `nn.Linear(embed_dim, 256)` ile sonraki byte logits üretimi.

* Kayıp dönüşümü: `CrossEntropyLoss` (teacher forcing, label smoothing opsiyonlu).

* Üretim: Byte‑düzeyinde greedy/top‑k/nucleus örnekleme; UTF‑8 decode.

* CoT üretimi: Model girdiden CoT zinciri üretir; HCL coherence/critic değerlendirmesi korunur.

* SFT: Kural tabanlı CoT’lerden bootstrap dataset; Instruction SFT döngüsü.

## Mimari Değişiklikler

* `src/model.py`

  * `PDCLMBase` içine `ByteLMHead` ekle; `MSELoss` → `CrossEntropyLoss`.

  * `forward` çıktısı: `logits` (B × T × 256), `loss_ce` ve yardımcı metrikler.

  * `generate` güncelle: byte sampling + durdurma kriterleri (EOS, max\_len, newline).

* `src/hcl.py`

  * `parse_and_reason` kademeli devre dışı; yeni `generate_cot` modeli çağırarak üretir.

  * `hcl_train_step` güncelle: üretken cevabı CCM/critic ile değerlendir.

* `src/pdclm.py`

  * `full_train` için SFT varyantı: `(task, target_cot)` ile CE eğitimi.

  * Değerlendirme: doğruluk, coherence, yansıma skoru, perplexity.

* `src/reflection.py`

  * `reflective_forward` logits ile birlikte çalışabilir; toplam kayıp kompozisyonunu güncelle (opsiyonel).

* `experiments/` ve testler

  * Yeni deney: Instruction SFT akışı.

  * PyTest: generate ve CE kaybı doğrulama.

## Veri ve Eğitim Planı

* Dataset üretimi (bootstrap):

  * `task_generator` (`src/hcl.py:102`) ile N\~50k görev üret.

  * `HCLAgent` ile kural tabanlı tam CoT üret (
    `src/hcl.py:33`, `src/hcl.py:47`).

  * Kaydet: JSONL `(input, target_cot)`; byte dizisine encode.

* Instruction SFT döngüsü:

  * Input: görev metni (prompt) → model.

  * Target: CoT metni (bytes) → CE teacher forcing.

  * Hiperparametreler: AdamW, LR schedule, grad clip, label smoothing (ε=0.1), warmup.

## Üretim (Inference) Tasarımı

* Sampling: greedy/top‑k(20)/nucleus(p=0.9), temperature ayarı.

* Durdurma: `EOS` (özel byte), `max_len`, `\n\nFinal answer:` görülünce durdur.

* Post‑hoc değerlendirme: CCM coherence, expected answer parse (`src/utils.py`).

## Aşamalar ve Zamanlama

* Faz‑1: ByteLMHead ekle + CE LM pretraining (kısa kurulum)

  * Kod noktaları: `src/model.py:9`, `src/model.py:34`, `src/model.py:36`.

  * Amaç: logits üretimi ve CE loss stabil çalışsın; basit corpus (mevcut görev metinleri).

* Faz‑2: Bootstrap Instruction SFT dataset

  * Kod noktaları: `src/hcl.py:33`, `src/hcl.py:47`, `src/hcl.py:102`.

  * Amaç: `(task, cot)` çiftleri üretimi ve temizleme.

* Faz‑3: Instruction SFT eğitimi

  * Kod noktaları: `src/pdclm.py:85` yeni SFT döngüsü.

  * Amaç: CoT üretimi öğrenilsin; perplexity ve doğruluk iyileşsin.

* Faz‑4: HCL entegrasyonu (üretken mod)

  * Kod noktaları: `src/hcl.py` tüm akış; `compute_reward` korunur.

  * Amaç: Model üretimi + CCM/critic ile uçtan uca çözüm.

* Paralel: HMR parametre optimizasyonu (sim\_max, ssm\_age\_max izleme).

## Değerlendirme ve Metrikler

* Dil modelleme: CE loss, byte‑perplexity.

* Görev metrikleri: doğruluk, coherence (CCM), reflection skoru, halüsinasyon cezası.

* HMR: `hmr/sim_max`, `hmr/ssm_age_max`.

* Verimlilik: ilerleme süresi (`measure_forward_latency`).

## Riskler ve Önlemler

* Mode collapse/tekrarlı CoT → nucleus sampling + temperature > 0.7.

* Öğrenme instabilitesi → label smoothing, grad clip, cosine LR schedule.

* UTF‑8 uyumsuz byte dizileri → strict decode ve fallback.

* HMR etkileşimi → başta HMR etkisini sınırlamak için bayrak (ablasyon), sonra kademeli aç.

## Teslim Kriterleri

* ByteLMHead ile CE training stabil çalışıyor; testler geçiyor.

* Model `generate_cot(task)` ile kuralsız CoT üretiyor.

* SFT ile doğruluk/coherence artışı ölçülüyor.

* Perplexity düşüş trendi net.

## Ortam ve Komutlar

* Bağımlılıklar: `torch`, `numpy`, `wandb`, `pytest`.

* Komutlar:

  * `make lint`

  * `make test`

  * `python -m experiments.final_alignment --epochs 1 --steps 30`

  * `python -m experiments.infer_pdclm --prompt "Task: ..."`

