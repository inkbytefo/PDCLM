## Developer: inkbytefo
## Modified: 2025-11-17

# Tokenizersız NLP Mimarisi ve Uygulama Planı

## Amaç
- Doğal dil anlama ve üretimi için tokenizer olmadan byte düzeyinde çalışmak.
- PD‑CLM çekirdeğiyle uyumlu: PSE, HCL, Reflection, HMR, RAG/Tool‑use ve çoklu ödül.
- Açık uçlu sorularda üretken yanıt, verifiable görevlerde yüksek doğruluk ve güvenli davranış.

## Yaklaşım
- ByteLMHead: `embed_dim → 256 sınıf (0–255)` çıkışı ile byte‑seviyesinde üretim, `CrossEntropyLoss` ile SFT.
- Byte‑Level Semantik Encoder: PSE çıktısı → semantik projeksiyon (normalize). Intent sınıflandırma, varlık/slot çıkarımı ve RAG eşleme.
- Intent Router: `math|logic|sort|qa|dynamic|code|web` kararını verir ve modül seçiminde tetikleyici olur.
- RAG/Tool‑Use: küçük KB ve FAISS/pgvector ile bilgi çekimi; Toolformer tarzı API çağrıları (hesap makinesi, arama, takvim).
- Çoklu Ödül: `coherence + correctness + reflection − efficiency(latency) − hallucination` ve kaynak zorunluluğu.

## Bileşenler
- PSE: bytes → pencere/stride → `[num_windows, dim]`, entropi modülasyonu.
- ByteLMHead: `Linear(dim, 256)` + `CE`; üretken yanıtı UTF‑8 decode eder.
- Semantik Encoder: `Linear → L2 normalize`; intent ve retrieval benzerliği için kullanılır.
- Intent Router: hafif sınıflandırıcı + kural; REST yanıtına `supported`, `intent`, `source` ekler.
- RAG: cümle vektörüyle nearest neighbor; chunked cross‑attention veya hafif bağlamsal birleştirme.
- Tool‑Use: API çağrıları için kendini öğretme; minimal demolarla self‑supervised filtre.

## Eğitim Protokolü
- Faz‑1 SFT (tokenizersız): `L_text = CE(bytes)`, `L_pattern = MSE(PSE)`, `L_reflect = BCEWithLogits`; `total = α*L_text + β*L_pattern + γ*L_reflect`.
- Faz‑2 HCL Self‑Play: coherence odaklı; Reflection ile hata sinyali.
- Faz‑3 RAG/Toolformer: few‑shot demolarla API kullanımı; doğru kaynak entegrasyonu.
- Faz‑4 Instruction SFT: açık uçlu görevler; CoT üretimini ByteLMHead ile metinsel hale getir.

## Değerlendirme
- Metrikler: `reward`, `reward_base`, `latency_ms`, `hmr/sim_max`, `intent_dist`, `rag_hit_rate`, `hallucination_rate`.
- Setler: GSM8K mini (math), basit QA (capitals/units), tool‑use (time/date), mantık örnekleri.
- Kriterler: final alignment avg reward ≥ 0.85; reflection target error < 0.2; hallucination düşük ve kaynaklı yanıt.

## REST Entegrasyonu
- Endpoint: `/evaluate` → `{"reward":..., "answer":..., "supported":..., "intent":..., "source":...}`.
- Desteklenmeyen intentlerde düşük ceza ve yönlendirme mesajı.

## Roadmap
- Sprint 1: ByteLMHead PoC, küçük SFT; intent router ve REST alanları.
- Sprint 2: Semantik Encoder + mini KB/RAG; source citation ve hallucination cezası iyileştirmesi.
- Sprint 3: Toolformer demoları; CoT üretimi metinsel; uzun CoT stres testleri.
- Sprint 4: Ölçekli SFT ve retrieval; güvenlik/guardrails; çoklu ödül ayarı.

## Referanslar
- ByT5 (token‑free byte‑level): Xue et al., 2022, arXiv:2105.13626; ACL Anthology TACL 2022.
- Reflexion (verbal RL): Shinn et al., 2023, arXiv:2303.11366.
- Toolformer (self‑supervised tool use): Schick et al., 2023, arXiv:2302.04761.
- RETRO/InstructRetro (retrieval augmentation): Borgeaud et al., 2021, arXiv:2112.04426; Ping et al., 2023.