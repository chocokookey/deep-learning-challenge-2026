# 아주소중한딥러닝챌린지 2026 — Qwen2.5-3B Math Solver

Fine-tuned **Qwen2.5-3B-Instruct** for integer-answer math problems.
Pipeline: **QLoRA DPO** generation with **self-consistency (N=64)** + a **verifier** reranker.
Practice leaderboard score: **0.734**.

## Method
1. **Generation** — a DPO-tuned adapter samples N candidate solutions per problem (temp 0.8, top-p 0.95).
2. **Self-consistency** — vote over the extracted integer answers.
3. **Verifier reranking** — a separately-tuned verifier scores P(correct) per candidate.
   Final score per answer: `SC_count + λ · P(Yes) · N_total`  (λ = 0.7).

(Optional ensemble: add a Gemini-distilled generation adapter, N=32 each. **DPO-only is the confirmed setting.**)

## Environment
- Kaggle, GPU **T4 × 2**, Internet **on**
- `trl==0.12.2` — **required** (other versions broke DPO training)
- vLLM (inference, `pip install -U vllm transformers`) + transformers / peft / bitsandbytes / datasets / accelerate
- See `requirements.txt`

## Repo structure
```
train/train_dpo.py      # DPO training  -> qlora_dpo adapter
inference/inference.py  # confirmed pipeline; set TEST_CSV and run
requirements.txt
```

## Adapters (Kaggle Datasets)
| adapter | path |
|---|---|
| DPO (generation) | `h70jun/qlora-dpo/qlora_dpo` |
| Gemini distill (optional) | `h70jun/deeplearning2/qlora_gemini2_backup` |
| verifier | `h70jun/deeplearning2/qlora_verifier_backup` |

## Run
**Train** (reproduces the DPO adapter):
```bash
python train/train_dpo.py
```
**Infer** (produces the answer CSV):
```bash
# edit TEST_CSV at the top of the file, then:
python inference/inference.py     # -> /kaggle/working/submission.csv
```

## Notes for reproducibility
- Attach all three adapter datasets as Notebook Inputs **before** inference — a missing input surfaces as `EngineDeadError`, not a clear file error.
- `MAX_INPUT_TOK=2900` truncation prevents verifier overflow at `max_model_len=3072`.
- Generation seed is fixed (42) for deterministic re-runs (not in the original run — an intentional reproducibility upgrade).
- Solving/verifier prompts are embedded verbatim; submission columns are `id, answer` (answer cast to int64).
- `train_dpo.py` pins training to a single GPU via `CUDA_VISIBLE_DEVICES="0"`; inference uses both cards (`tensor_parallel_size=2`).

## Real test (Aug 31, ~2000 problems, 24h window)
Individual entry. Finishing beats a fractional score gain — a run that misses the window is 0.
- **Dry-run first:** set `SAMPLE_N=30`, run once, read the printed ETA (×~66 for 2000), then pick N.
- **N knob:** `N=32` roughly halves wall-clock for ~-0.005 practice score. Use `N=64` only if the dry-run leaves comfortable margin.
- **Ensemble off** for the real run unless practice ensemble is a clear blowout (it's ~2× slower).
- Submission CSV is rewritten every block, so an interrupted run still leaves a partial answer file.
- Conserve Kaggle GPU quota (30h/week) before Aug 31; commit the full run immediately at 00:00.
