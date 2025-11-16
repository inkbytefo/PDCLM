## Developer: inkbytefo
## Modified: 2025-11-17

# PD‑CLM Production Training Roadmap

## Phase 0 — Preparation
- Data governance: sources, licensing, PII filtering
- Infra: GPU cluster (≥8 for proto, 64+ for scale), storage/I/O
- Observability: W&B projects, system metrics

## Phase 1 — Continuous Stream Pretraining (PSE)
- Corpus ingestion → PSE preprocessing (window, stride)
- Mixed precision, gradient accumulation, checkpointing
- Targets: stable loss, throughput, memory footprint

## Phase 2 — HCL Self‑Play
- Proposer/critic agents, CCM coherence
- Task generator: math, logic, sorting; expand gradually
- Rewards: coherence/accuracy balance, curriculum schedule

## Phase 3 — Reflection (R‑Layer)
- BCEWithLogits targets via cosine error
- Long CoT runs (8–12 steps), LR scheduler
- Metrics: reflective loss, target/pred errors

## Phase 4 — HMR Routing & Retrieval
- EMA slot updates, per‑position routing
- LSM KV‑store retrieval, thresholds and aging
- Metrik: `hmr/sim_max`, `hmr/ssm_age_max`, latency

## Phase 5 — Multi‑Reward Expansion
- Hallucination penalty (expected answer comparison)
- Efficiency penalty (latency normalized)
- Weighted fusion tuned via validation

## Phase 6 — Evaluation & Alignment
- Held‑out task suites (GSM8K samples, logic checks)
- Inference Avg Reward ≥ 0.85 and stability across tasks
- W&B dashboards and alerts

## Phase 7 — Deployment
- TorchScript export
- REST service (FastAPI), autoscaling & logging
- Rollout plan and A/B checks

## Milestones
- M1: Pretraining stable, tests green
- M2: HCL rewards ≥ 0.75, CoT consistent
- M3: Reflection target error < 0.2
- M4: HMR metrics plateau (sim_max ~0.9)
- M5: Final alignment reward ≥ 0.85
- M6: REST+TorchScript operational