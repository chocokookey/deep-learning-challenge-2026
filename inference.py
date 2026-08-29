"""
Inference — 아주소중한딥러닝챌린지 2026 (Qwen2.5-3B-Instruct + QLoRA)

Confirmed pipeline: DPO generation (self-consistency) + verifier rerank, lambda=0.7.
Practice score 0.734 at N=64. Prompts / model / sampling / scoring are VERBATIM from the
working DPO cells; wrapped in a block loop that saves incrementally and prints an ETA
(for the ~2000-problem, 24h real run on Aug 31).

REAL RUN (Aug 31):
  1. Set SAMPLE_N=30, run once, read the printed ETA, THEN pick N.
  2. N=32 ~halves wall-clock vs N=64 for ~-0.005 practice score -> safe when time is tight.
  3. submission.csv is rewritten every block, so an interrupted run still leaves a partial file.
  Change TEST_CSV only; run top to bottom.
Install:  pip install -U vllm transformers
"""
import re, math, time
from collections import Counter, defaultdict
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ============================= CONFIG (edit here) =============================
PIPELINE   = "dpo"        # "dpo" (confirmed 0.734) | "ensemble" (dpo+gemini, ~2x slower)
TEST_CSV   = "/kaggle/input/competitions/deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv"
OUT_CSV    = "/kaggle/working/submission.csv"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DPO_LORA   = "/kaggle/input/datasets/h70jun/qlora-dpo/qlora_dpo"
GEM_LORA   = "/kaggle/input/datasets/h70jun/deeplearning2/qlora_gemini2_backup"
VER_LORA   = "/kaggle/input/datasets/h70jun/deeplearning2/qlora_verifier_backup"

N_TOTAL       = 64        # candidates/question.  32 = ~half the time, ~-0.005 practice
LAMBDA        = 0.7       # Score = SC_count + LAMBDA * P(Yes) * N_TOTAL   (0.6~0.7 -> 0.734)
CHUNK         = 50        # questions per block (concurrency + incremental-save granularity)
MAX_INPUT_TOK = 2900
SEED          = 42        # NOT in the original run; added for deterministic re-runs
SAMPLE_N      = 0         # >0 -> only first N questions (timing dry-run). 0 -> all.

SYSTEM_PROMPT = ("You are a careful math problem solver. Solve the problem step by step. "
    "The final answer is always an integer. End your response with the final answer in the format \\boxed{answer}.")
VERIFY_SYSTEM = ("You are a math solution verifier. Given a problem and a proposed solution, "
    "judge whether the solution's final answer is correct. Respond with exactly 'Yes' or 'No'.")
YES_STRINGS = {"Yes", " Yes", "yes", " yes", "YES", " YES"}
ID_COL, ANSWER_COL = "id", "answer"
# =============================================================================

N_PER, MAX_LORAS = (N_TOTAL // 2, 3) if PIPELINE == "ensemble" else (N_TOTAL, 2)

lb = pd.read_csv(TEST_CSV)
lb.columns = [c.strip() for c in lb.columns]
print("columns:", list(lb.columns))
id_col = ID_COL       if ID_COL       in lb.columns else lb.columns[0]
q_col  = "question"   if "question"   in lb.columns else lb.columns[1]
if SAMPLE_N:
    lb = lb.head(SAMPLE_N)
    print(f"*** DRY RUN: first {SAMPLE_N} questions — extrapolate the ETA, do NOT submit ***")
ids       = lb[id_col].tolist()
questions = lb[q_col].tolist()
print(f"questions: {len(questions)}  pipeline: {PIPELINE}  N_TOTAL: {N_TOTAL}")

def extract_answer(text):
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    cand = boxed[-1] if boxed else None
    if cand is None:
        nums = re.findall(r"-?\d[\d,]*", text)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    try:
        v = int(cand.replace(",", ""))
        return None if abs(v) > 10**18 else v
    except ValueError:
        return None

def p_yes(logprob_dict):
    s = 0.0
    for lp in logprob_dict.values():
        if lp.decoded_token in YES_STRINGS:
            s += math.exp(lp.logprob)
    return min(s, 1.0)

tok = AutoTokenizer.from_pretrained(BASE_MODEL)

llm = LLM(model=BASE_MODEL, dtype="float16", gpu_memory_utilization=0.85,
          max_model_len=3072, enforce_eager=True, enable_lora=True,
          max_lora_rank=16, max_loras=MAX_LORAS, tensor_parallel_size=2)
DPO_REQ = LoRARequest("dpo",    1, DPO_LORA)
GEM_REQ = LoRARequest("gemini", 2, GEM_LORA)
VER_REQ = LoRARequest("ver",    3, VER_LORA)

gen_params = SamplingParams(temperature=0.8, top_p=0.95, n=N_PER, max_tokens=512, seed=SEED)
ver_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

GEN_LORAS = [DPO_REQ] if PIPELINE == "dpo" else [DPO_REQ, GEM_REQ]

def solve_block(qs):
    prompts = [tok.apply_chat_template(
                  [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user",   "content": q}],
                  tokenize=False, add_generation_prompt=True) for q in qs]

    counter = [Counter() for _ in qs]
    reps    = [defaultdict(list) for _ in qs]
    for lora in GEN_LORAS:                                # dpo (+ gemini if ensemble)
        for j, out in enumerate(llm.generate(prompts, gen_params, lora_request=lora)):
            for o in out.outputs:
                a = extract_answer(o.text)
                if a is None:
                    continue
                counter[j][a] += 1
                if len(reps[j][a]) < 2:
                    reps[j][a].append(o.text.strip())

    vprompts, vindex = [], []
    for j, q in enumerate(qs):
        for ans, sols in reps[j].items():
            for sol in sols:
                user = f"Problem:\n{q}\n\nProposed solution:\n{sol}\n\nIs the final answer correct?"
                p = tok.apply_chat_template(
                        [{"role": "system", "content": VERIFY_SYSTEM},
                         {"role": "user",   "content": user}],
                        tokenize=False, add_generation_prompt=True)
                pid = tok(p)["input_ids"]                 # truncate whole prompt (verifier overflow guard)
                if len(pid) > MAX_INPUT_TOK:
                    p = tok.decode(pid[:MAX_INPUT_TOK])
                vprompts.append(p); vindex.append((j, ans))

    pyes = defaultdict(float)
    if vprompts:
        for (j, ans), o in zip(vindex, llm.generate(vprompts, ver_params, lora_request=VER_REQ)):
            pyes[(j, ans)] = max(pyes[(j, ans)], p_yes(o.outputs[0].logprobs[0]))

    out_ans = []
    for j in range(len(qs)):
        cnt = counter[j]
        if not cnt:
            out_ans.append(0); continue
        best_a, best_s = None, -1
        for a in cnt:
            s = cnt[a] + LAMBDA * pyes.get((j, a), 0.0) * N_TOTAL
            if s > best_s:
                best_s, best_a = s, a
        out_ans.append(int(best_a))
    return out_ans

# --- main loop: incremental save + live ETA ---
final_ids, final_ans = [], []
t0 = time.time()
n_blocks = (len(questions) + CHUNK - 1) // CHUNK
for b, s in enumerate(range(0, len(questions), CHUNK), 1):
    final_ans += solve_block(questions[s:s + CHUNK])
    final_ids += ids[s:s + CHUNK]
    df = pd.DataFrame({id_col: final_ids, ANSWER_COL: final_ans})
    df[ANSWER_COL] = df[ANSWER_COL].astype("int64")
    df.to_csv(OUT_CSV, index=False)
    el = time.time() - t0
    print(f"block {b}/{n_blocks}  done {len(final_ids)}/{len(questions)}  "
          f"elapsed {el/60:.1f}m  ETA {el/b*(n_blocks-b)/60:.1f}m")
print("wrote", OUT_CSV, len(final_ids), f"| total {(time.time()-t0)/60:.1f}m")
