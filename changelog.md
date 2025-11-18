## Developer: inkbytefo
## Modified: 2025-11-17

# Changelog

All notable changes to this project are documented here.

## 2025-11-18

- Added Value head (`value_head = nn.Linear(embed_dim, 1)`) parallel to ByteLMHead for RL critic/value estimation.
- Implemented PPO training loop (`ppo_train`) optimizing policy and value with clipped objective; reward computed on model-generated CoT.
- Added rollout tracing in `generate_text` (actions, logprobs, values) and sequence logprob evaluation helper.
- Introduced generated-text reward evaluation helper to combine coherence, correctness, reflection and hallucination penalty.
- Added `tests/test_ppo.py` validating PPO loop runs.
- Introduced tool-use and RAG scaffolding: `src/tools.py` with `CalculatorTool`, `SearchTool`, and `ToolExecutor`.
- Added tool-aware CoT generation `generate_cot_with_tools` in `src/hcl.py` and default tool setup in `src/pdclm.py`.
- Added tests for tools and tool-aware CoT (`tests/test_tools_exec.py`, `tests/test_hcl_tools.py`).

## 2025-11-17

- Added Byte-level LM head (`nn.Linear(embed_dim, 256)`) to `PDCLMBase` and switched training loss from `MSELoss` to `CrossEntropyLoss` for next-byte prediction.
- Implemented byte-level text generation (`generate_text`) with temperature, top-k, and nucleus sampling in `src/model.py`.
- Updated `HCLAgent.generate_cot` to use model-driven generative CoT; preserved the previous rule-based path as `generate_rule_based_cot` in `src/hcl.py`.
- Introduced Instruction SFT training loop (`instruction_sft_train`) in `src/pdclm.py` with checkpointing and W&B logging.
- Added SFT dataset helpers in `src/utils.py` (`build_sft_dataset`, `save_sft_jsonl`) to bootstrap `(task, CoT)` pairs from the rule-based generator.
- Fixed short-prompt handling by adding automatic padding before PSE to avoid window/stride overflow in `src/model.py` forward and generation paths.
- Stabilized evaluation tests by using rule-based CoT inside `PDCLM._evaluate_task` and `evaluate_with_answer` for deterministic reward checks.
- Added tests:
  - `tests/test_generate_text.py` for byte-level generation
  - `tests/test_cot_generate.py` for generative CoT
  - `tests/test_sft_dataset.py` for SFT dataset creation
- Test suite result: 30 passed, 2 warnings (PyTest run on Windows, Python 3.11).

### Files Affected
- `src/model.py`: ByteLMHead, CE loss, generative text API, padding fix
- `src/hcl.py`: generative CoT, preserved rule-based CoT
- `src/pdclm.py`: instruction SFT training loop, evaluation path adjustment
- `src/utils.py`: SFT dataset builders
- `tests/*`: new tests for generation and SFT

## 2025-11-16

- Optimized `PatternStreamEncoder` (`src/pse.py`) with manual embedding, vectorized entropy profiling, MKLDNN enablement, and denormal flush for CPU performance.
- Implemented Phase-1 pretraining utilities in `src/model.py` (`pretrain_step` with autocast and gradient clipping; `create_batches` for sliding-window batch creation).
- Added initial unit tests across modules (`tests/`), including model, HMR, reflection, and utilities.