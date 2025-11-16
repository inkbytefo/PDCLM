## Developer: inkbytefo
## Modified: 2025-11-17

# PD‑CLM — Pattern‑Driven Cognitive Language Model

PD‑CLM is a multi‑phase system that replaces classic tokenization with a Pattern Stream Encoder (PSE), builds reasoning with a High‑Level Cognitive Loop (HCL), optimizes consistency via a Reflection Network (R‑Layer), and routes long‑context memory using a Hierarchical Memory Router (HMR). It logs multi‑reward metrics to W&B and exposes a REST API for deployment.

## Highlights
- PSE: Continuous stream encoding with entropy modulation (tokenizer‑free)
- HCL: Proposer/critic agents, CoT generation, CCM coherence
- Reflection: BCEWithLogits with cosine target error
- HMR: SSM/MSM slots (EMA), per‑position routing, LSM KV‑store retrieval
- Multi‑Reward: coherence + correctness + reflection − efficiency/hallucination
- REST server and TorchScript export

## Install & Test
- `pip install -r requirements.txt`
- `pytest -q` (all tests passing)

## Final Alignment (short)
- `python experiments/final_alignment.py --embed_dim 512 --epochs 1 --steps 30`
- Inference check: `python experiments/infer_pdclm.py --embed_dim 512 --ckpt checkpoints/pdclm_final.pt --samples 50`

## Experiments
- Reflection (long CoT): `python experiments/reflection_experiment.py` → `experiments/reflection_longcot.png`
- REST server: `python experiments/rest_server.py`
  - `curl -X POST http://localhost:8000/evaluate -H "Content-Type: application/json" -d "{\"task\":\"What is 7 + 9?\"}"`
- TorchScript export: `python experiments/export_model.py`

## Code Map
- `src/pse.py` — PatternStreamEncoder
- `src/model.py` — PDCLMBase (transformer decoder)
- `src/hcl.py` — HCLAgent, CognitiveControlModule (CoT, coherence)
- `src/reflection.py` — ReflectionNetwork, ReflectivePDCLM
- `src/hmr.py` — HierarchicalMemoryRouter (EMA, routing, LSM)
- `src/pdclm.py` — Unified model, training loop, W&B
- `src/utils.py` — Data, latency, expected answers, hallucination penalty

## W&B Logging
- `reward`, `reward_base`, `loss`, `latency_ms`, `hmr/sim_max`, `hmr/ssm_age_max`
- Note: project views may require login

## Roadmap
See `ROADMAP.md` for a production training plan (data pipeline, distributed training, evaluation, deployment).
