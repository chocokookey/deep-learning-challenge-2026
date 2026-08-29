"""
DPO 학습 코드 — 아주소중한딥러닝챌린지 2026
실전 0.734 달성에 사용된 `qlora_dpo` 어댑터 재현 코드.
Kaggle T4x2 환경에서 작성되었으며, 메모리 충돌 방지를 위해 단일 GPU에 할당함.
설치:  pip install -U transformers peft bitsandbytes datasets accelerate  &&  pip install trl==0.12.2
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                        # GPU 1장으로 제한 (무조건 파일 최상단에 위치)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer                            # trl==0.12.2 필수

MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"
VERIFIER_CSV = "/kaggle/input/datasets/h70jun/deeplearning2/verifier_data.csv"
OUT_DIR      = "/kaggle/working/qlora_dpo"

SYSTEM_PROMPT = ("You are a careful math problem solver. Solve the problem step by step. "
    "The final answer is always an integer. End your response with the final answer in the format \\boxed{answer}.")

# --- verifier_data.csv에서 DPO 데이터 쌍 추출 (문제당 정답 최대 2개 x 오답 최대 2개) ---
vdf = pd.read_csv(VERIFIER_CSV)
print(f"verifier data: {len(vdf)} rows (pos {(vdf['label']==1).sum()}, neg {(vdf['label']==0).sum()})")
pos_dict = vdf[vdf['label'] == 1].groupby('question')['solution'].apply(list).to_dict()
neg_dict = vdf[vdf['label'] == 0].groupby('question')['solution'].apply(list).to_dict()

dpo_data = []
for q in pos_dict:
    if q in neg_dict:
        for ps in pos_dict[q][:2]:
            for ns in neg_dict[q][:2]:
                dpo_data.append({"question": q, "chosen": ps, "rejected": ns})
dpo_df = pd.DataFrame(dpo_data)
print(f"DPO pairs: {len(dpo_df)}")                              # 약 1346 쌍

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.padding_side = "right"

def format_dpo(row):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": row["question"]}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt":   prompt,
            "chosen":   row["chosen"]   + "<|im_end|>",
            "rejected": row["rejected"] + "<|im_end|>"}

raw_ds = Dataset.from_pandas(dpo_df)
dpo_ds = raw_ds.map(format_dpo, remove_columns=raw_ds.column_names).shuffle(seed=42)

# --- 모델 로드 (4-bit QLoRA) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map={"": 0})
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# gradient_checkpointing + use_reentrant=False  -> CUBLAS / 메모리 접근 에러(illegal-memory) 해결 핵심 옵션
# processing_class=tok (tokenizer 아님)         -> trl 0.12.2 버전 문법 적용
dpo_config = DPOConfig(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False, bf16=False,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    beta=0.1,
    max_prompt_length=256,
    max_length=768,
)
trainer = DPOTrainer(model=model, ref_model=None, args=dpo_config,
                     train_dataset=dpo_ds, processing_class=tok)
model.config.use_cache = False
trainer.train()                                                 # 배치 크기 1, 누적 16 기준 약 84스텝 (checkpoint-84)
trainer.save_model(OUT_DIR)

# Kaggle Dataset으로 백업: Output 탭 -> New Dataset 생성 후 실전 노트북에 Add Input 필수
shutil.make_archive("/kaggle/working/qlora_dpo_backup", "zip", OUT_DIR)
print("saved:", OUT_DIR)
