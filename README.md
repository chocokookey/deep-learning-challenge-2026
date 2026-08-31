# Deep Learning Challenge 2026

본 저장소는 '제5회 대학 연합 아주 소중한 딥러닝 챌린지 2026' 제출을 위한 재현 코드 및 설명서입니다. 베이스 모델인 `Qwen/Qwen2.5-3B-Instruct`의 수학 추론 능력을 극대화하기 위해 DPO(Direct Preference Optimization)와 Verifier Best-of-N 기법을 결합하여 구현했습니다.

## 🚀 Pipeline Overview

### 1. Model Fine-Tuning (DPO)
- 모델이 자체 생성한 정답 풀이(Chosen)와 오답 풀이(Rejected) 쌍을 기반으로 DPO 학습을 진행하여, 수학 문제 해결에 대한 내재적 체급을 향상시켰습니다.
- `trl` 라이브러리의 `DPOTrainer`를 활용했으며, QLoRA(4bit)를 적용해 제한된 자원 환경에서 메모리 효율을 극대화했습니다.

### 2. Inference & Reranking (Verifier Best-of-N)
- **Generation**: DPO 학습이 완료된 어댑터를 사용하여 각 문제당 32개(N=32)의 후보 풀이를 생성합니다. (Temperature=0.8)
- **Verification**: 별도로 학습된 Verifier 어댑터를 통해 각 후보 풀이가 정답일 확률 P(Yes)를 측정합니다.
- **Hybrid Scoring**: 단순 다수결(Self-Consistency)의 맹점을 보완하기 위해 빈도수와 Verifier 확률을 결합한 하이브리드 스코어를 산출하여 최종 정답을 도출합니다. `Score = SC_count + λ·P(Yes)·N` (λ=0.7)

## 📦 Model Weights (Adapters)

추론에 필요한 fine-tuning된 LoRA 어댑터는 아래 Kaggle 데이터셋에 공개되어 있습니다 (Public).

| 어댑터 | 용도 | Kaggle 데이터셋 | 노트북 내 경로 |
|---|---|---|---|
| DPO | 풀이 생성 | https://www.kaggle.com/datasets/h70jun/qlora-dpo | `/kaggle/input/datasets/h70jun/qlora-dpo/qlora_dpo` |
| Verifier | 정답 채점 | https://www.kaggle.com/datasets/h70jun/deeplearning2 | `/kaggle/input/datasets/h70jun/deeplearning2/qlora_verifier_backup` |

## 🛠️ Environments

- **Platform**: Kaggle Notebook (GPU: T4 x2)
- **Base Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Library**: `vllm`, `transformers`, `peft`, `bitsandbytes`, `trl==0.12.2`

## 📂 File Description

- `train_dpo.py`: DPO 데이터셋 전처리 및 QLoRA 기반 학습 코드
- `inference_8_31.ipynb` / `inference.py`: vLLM 엔진을 활용한 병렬 추론 및 Verifier 채점 코드 (최종 제출에 사용)

## ⚙️ How to Reproduce

1. Kaggle 노트북을 생성하고 **GPU T4 x2**를 활성화합니다.
2. 위 **Model Weights** 표의 두 Kaggle 데이터셋을 노트북 **Input**으로 연결합니다.
3. `inference_8_31.ipynb`를 엽니다.
4. 최상단 **CONFIG 셀**에서 `TEST_CSV` 변수를 채점용 테스트 데이터 경로로 설정합니다. (어댑터 경로는 위 표 기준으로 이미 설정되어 있습니다.)
5. 전체 셀을 실행하면 `/kaggle/working/submission.csv`가 생성됩니다.
6. 출력 형식: 원본 test 파일의 `answer` 컬럼에 예측 정수값을 채운 CSV.

## 📊 Pipeline Summary

DPO fine-tuned 어댑터로 각 문제를 N=32회 생성 → Verifier 어댑터로 각 후보의 정답 확률 P(Yes) 측정 → `Score = SC_count + λ·P(Yes)·N` (λ=0.7)으로 최종 답 선택 (Best-of-N Reranking).
