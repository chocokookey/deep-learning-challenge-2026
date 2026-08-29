# Deep Learning Challenge 2026

본 저장소는 '제5회 대학 연합 아주 소중한 딥러닝 챌린지 2026' 제출을 위한 재현 코드 및 설명서입니다. 
베이스 모델인 `Qwen/Qwen2.5-3B-Instruct`의 수학 추론 능력을 극대화하기 위해 **DPO(Direct Preference Optimization)**와 **Verifier Best-of-N** 기법을 결합하여 구현했습니다.

## 🚀 Pipeline Overview

1. **Model Fine-Tuning (DPO)**
   - 모델이 자체 생성한 정답 풀이(Chosen)와 오답 풀이(Rejected) 쌍을 기반으로 DPO 학습을 진행하여, 수학 문제 해결에 대한 내재적 체급을 향상시켰습니다.
   - `trl` 라이브러리의 `DPOTrainer`를 활용했으며, QLoRA(4bit)를 적용해 제한된 자원 환경에서 메모리 효율을 극대화했습니다.

2. **Inference & Reranking (Verifier Best-of-N)**
   - **Generation:** DPO 학습이 완료된 어댑터를 사용하여 각 문제당 32개(N=32)의 후보 풀이를 생성합니다. (Temperature=0.8)
   - **Verification:** 별도로 학습된 Verifier 어댑터를 통해 각 후보 풀이가 정답일 확률 $P(Yes)$를 측정합니다.
   - **Hybrid Scoring:** 단순 다수결(Self-Consistency)의 맹점을 보완하기 위해 빈도수와 Verifier 확률을 결합한 하이브리드 스코어를 산출하여 최종 정답을 도출합니다. ($\lambda=0.6$)

## 🛠️ Environments
- **Platform:** Kaggle Notebook (GPU: T4 x2)
- **Library:** `vllm`, `transformers`, `peft`, `bitsandbytes`, `trl`

## 📂 File Description
- `train_dpo.py`: DPO 데이터셋 전처리 및 QLoRA 기반 학습 코드
- `inference.py` / `inference_8_31.ipynb`: vLLM 엔진을 활용한 병렬 추론 및 Verifier 채점 코드

## ⚙️ How to Reproduce
1. Kaggle 환경에서 T4 x2 GPU를 활성화합니다.
2. 실전 테스트 데이터셋 및 사전 학습된 어댑터 경로를 `TEST_PATH`, `GEN_LORA`, `VER_LORA` 변수에 맞게 설정합니다.
3. `inference_8_31.ipynb` 또는 `inference.py`를 실행하여 최종 `submission.csv`를 획득합니다.
