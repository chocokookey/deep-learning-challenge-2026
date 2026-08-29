"""
Inference — 아주소중한딥러닝챌린지 2026 (Qwen2.5-3B-Instruct + QLoRA)

확정된 파이프라인: DPO 생성 (Self-Consistency) + Verifier 리랭킹, lambda=0.7.
N=64 기준 연습 리더보드 0.734 달성. 프롬프트/모델/샘플링/채점 로직은 검증된 셀과 100% 동일함.
실전 24시간 런(약 2000문제)을 대비해 블록 단위 중간 저장 및 ETA 출력 기능 포함.

실전 적용 (8/31):
  1. SAMPLE_N=30으로 맞추고 한 번 실행 -> 출력되는 전체 ETA 확인 후 최종 N 결정.
  2. N=32로 낮추면 시간은 절반으로 줄고 점수 하락폭은 -0.005 내외이므로, 시간이 빡빡할 때 안전한 선택지.
  3. submission.csv는 블록마다 덮어쓰기로 저장되므로 중간에 터져도 부분 답안이 안전하게 남음.
  TEST_CSV 경로만 실전용으로 수정 후 전체 실행.
설치:  pip install -U vllm transformers
"""
import re, math, time
from collections import Counter, defaultdict
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ============================= 설정 (이 부분만 수정) =============================
PIPELINE   = "dpo"        # "dpo" (0.734 확정) | "ensemble" (dpo+gemini, 속도 2배 소요)
TEST_CSV   = "/kaggle/input/competitions/deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv"
OUT_CSV    = "/kaggle/working/submission.csv"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DPO_LORA   = "/kaggle/input/datasets/h70jun/qlora-dpo/qlora_dpo"
GEM_LORA   = "/kaggle/input/datasets/h70jun/deeplearning2/qlora_gemini2_backup"
VER_LORA   = "/kaggle/input/datasets/h70jun/deeplearning2/qlora_verifier_backup"

N_TOTAL       = 64        # 문제당 생성할 후보 수. 32로 낮추면 시간 절반, 점수 하락폭 미미함
LAMBDA        = 0.7       # 최종 점수 = 빈도수 + LAMBDA * P(Yes) * N_TOTAL (0.6~0.7 부근 최적)
CHUNK         = 50        # 한 번에 처리 및 저장할 문제 수 (메모리 확보 및 중간 저장 단위)
MAX_INPUT_TOK = 2900
SEED          = 42        # 재현성(Reproducibility)을 위한 난수 시드 고정
SAMPLE_N      = 0         # >0: 앞부분 N문제만 테스트 (ETA 확인용) / 0: 전체 문제 실행

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
    print(f"* DRY RUN: 앞부분 {SAMPLE_N}문제만 실행 — ETA 확인 후 SAMPLE_N=0으로 되돌려 본run 진행 *")
    
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
        return None if abs(v) > 1018 else v
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
    for lora in GEN_LORAS:                                # dpo 단독 (ensemble 모드일 경우 gemini 추가)
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
                pid = tok(p)["input_ids"]                 # verifier 최대 컨텍스트 길이 방어를 위한 프롬프트 자르기
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
