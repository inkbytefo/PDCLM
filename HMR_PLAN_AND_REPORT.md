## Developer: inkbytefo
## Modified: 2025-11-17

# PD-CLM Comparison Report and HMR Roadmap

## Summary
- Core cognitive stack implemented: PSE, HCL, Reflection.
- Missing: Hierarchical Memory Router (HMR), broader data pipeline, multi-modality, extended multi-reward.
- Decision: Implement HMR prototype first to address long-context memory routing.

## Current vs Target
- PSE: continuous stream with entropy modulation — aligned.
- HCL: proposer/critic, CCM coherence — aligned.
- Reflection: BCEWithLogitsLoss with cosine targets — aligned.
- Multi-Reward: partial (coherence/correctness/reflection) — extend later.
- HMR: absent → now prototyped (SSM/MSM slots) and integrated.

## HMR Prototype Scope
- File: `src/hmr.py` — learnable SSM/MSM slots, simple concatenation as memory.
- Integration: `src/model.py` — replace zero memory with HMR output.
- Tests: `tests/test_hmr.py` — module forward and integration checks.

## Roadmap
### Phase A: HMR v1 (Prototyping)
- Add slot update: EMA-based update from recent `input_stream`.
- Attention mask support: prioritize SSM over MSM depending on recency.
- Config hooks: slot counts, decay rates, enable/disable LSM.

### Phase B: HMR v2 (Routing Logic)
- Router softmax weights to blend slots per position.
- Memory compression: project input stream to slot space via `nn.Linear`.
- Slot aging and eviction policy.

### Phase C: HMR v3 (Long-Term Memory)
- LSM key-value store with retrieval by cosine similarity.
- Async refresh of slots from LSM at boundaries.

### Phase D: Training & Evaluation
- Synthetic long-CoT tasks for routing stress-test.
- Metrics: coherence over depth, retrieval accuracy, latency overhead.

### Phase E: Pipeline & Rewards Expansion
- Data pipeline in `src/utils.py`: PSE preprocessing and task split.
- Reward channels: hallucination penalty, energy/latency regularization.

## Acceptance Criteria
- Tests pass (`pytest -q`).
- HMR improves stability on long CoT tasks.
- Reward remains ≥ 0.85 with HMR enabled.