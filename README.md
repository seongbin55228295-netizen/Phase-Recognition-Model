# Phase-Recognition-Model

자기회귀 멀티모달 학습 기반 요리 영상 단계 인식 모델 (Real 팀, 빅데이터프로그래밍 2026-1)

CNN 기반 이미지 인코더(ResNet-50)와 Transformer 기반 텍스트 인코더(DistilBERT)를 직접 설계한 Co-attention 융합 모듈로 결합하여, 요리 영상의 현재 프레임과 모델이 스스로 예측한 단계 히스토리를 함께 입력받아 현재 조리 단계(Prep / Cut / Mix / Cook-Heat / Bake / Season / Plate / Idle)를 분류하는 End-to-End 파이프라인.

## 모델 아키텍처

### 전체 파이프라인

시점 `t` 의 예측은 **현재 프레임**과 **직전 k개 시점의 자기 예측 히스토리**를 함께 입력받아 수행됩니다.

```
                       ┌──────────────────────────────────────────────┐
                       │            예측 히스토리 버퍼 ŷ_{t-k:t-1}       │
                       │      예: "[t-3: Cut] [t-2: Mix] [t-1: Cook]"   │
                       └────────────────────┬─────────────────────────┘
                                            │ tokenize
                                            ▼
   ┌─────────────────┐               ┌──────────────────┐
   │  Frame x_t      │               │   DistilBERT     │
   │  (RGB 224×224)  │               │   Text Encoder   │
   └────────┬────────┘               │   (6L, 768d)     │
            │                        └────────┬─────────┘
            ▼                                 │
   ┌─────────────────┐                        │ H_text ∈ ℝ^{L×768}
   │   ResNet-50     │                        │
   │   (ImageNet pre)│                        │
   │   conv5 출력    │                        │
   └────────┬────────┘                        │
            │ F_img ∈ ℝ^{2048×7×7} → 49 토큰   │
            │ (1×1 conv → 512d projection)    │ (linear → 512d projection)
            ▼                                 ▼
   ┌───────────────────────────────────────────────────────┐
   │           Co-attention Fusion Block  ×2층              │
   │   img ←→ text 양방향 cross-attention + FFN + residual  │
   └────────────────────────┬──────────────────────────────┘
                            │ z_t ∈ ℝ^{512} (pooled)
                            ▼
                  ┌──────────────────────┐
                  │  MLP Head            │
                  │  512 → 256 → 8       │
                  └──────────┬───────────┘
                             ▼
                    ŷ_t ∈ {Prep, Cut, Mix,
                           Cook-Heat, Bake, Season,
                           Plate, Idle}
                             │
                             └──► 히스토리 버퍼에 추가 → t+1 시점 입력
```

### 구성 요소

| 모듈 | 구현 | 출력 |
| --- | --- | --- |
| 이미지 인코더 | ResNet-50 (ImageNet 사전학습) | `2048×7×7` feature map → 49 spatial 토큰, 512d로 projection |
| 텍스트 인코더 | DistilBERT (`distilbert-base-uncased`) | `L×768` 시퀀스 임베딩 → 512d projection |
| 융합 모듈 | 2층 Co-attention 블록 (직접 구현) | 양방향 cross-attention + FFN + residual, 512d 융합 벡터 |
| 분류 헤드 | 3층 MLP `512 → 256 → 8` | 8개 조리 단계 클래스 logits |

이미지·텍스트 두 모달의 손실 신호가 융합 모듈을 통과해 두 인코더까지 역전파되는 **End-to-End** 구조이며, 인코더는 낮은 학습률로 fine-tuning 됩니다.

### 자기회귀 추론

각 시점 `t` 의 출력 ŷ_t 는 히스토리 버퍼에 추가되어 `t+1` 입력으로 재사용됩니다. 히스토리 길이 `k` 는 하이퍼파라미터이며 E3 실험에서 `k ∈ {0, 1, 3, 5}` 로 비교합니다 (`k=0` 은 이미지 단독 베이스라인).

학습-추론 간 분포 차이(Exposure Bias)를 완화하기 위해 **Scheduled Sampling** 을 적용합니다. 학습 epoch 진행에 따라 정답 라벨 대신 모델 자신의 예측을 히스토리에 주입할 확률 `p` 를 `0 → 0.5` 로 선형 증가시킵니다.

| 시나리오 | 학습 입력 | 추론 입력 | 목적 |
| --- | --- | --- | --- |
| TF + TF (Oracle) | 정답 히스토리 | 정답 히스토리 | 상한 측정 |
| TF + FR | 정답 히스토리 | 자기 예측 히스토리 | Exposure Bias 노출 |
| SS + FR | 확률적 혼합 | 자기 예측 히스토리 | 완화 효과 검증 |

### 융합 기법 변형 (E2 비교)

동일 인코더·동일 분류 헤드 조건에서 세 가지 융합 구조를 직접 구현하여 정량 비교합니다.

| 변형 | 핵심 연산 | 특징 |
| --- | --- | --- |
| **Co-attention** (주 제안) | 양방향 cross-attention 2층 | 두 모달 간 토큰 단위 상호참조, attention map 시각화로 해석 가능 |
| **GMU** (Gated Multimodal Unit) | `z = g ⊙ h_img + (1−g) ⊙ h_text`, `g = σ(W·[h_img; h_text])` | 게이트 기반 적응적 가중합, 경량 |
| **Concat** | `[pool(h_img); pool(h_text)]` 후 MLP | 단순 결합 베이스라인 |

### 손실 함수

* **분류 손실**: Cross-Entropy
* **경계 프레임 가중치**: Soft Boundary — 구간 경계 ±2 프레임의 손실 가중치를 0.5로 감쇠하여 라벨 노이즈 영향 완화

## 저장소 구조

```
Phase-Recognition-Model/
├── README.md
├── requirements.txt              # Python 의존성 (작성 예정)
├── .gitignore
│
├── YouCookII/                    # YouCook2 원본 데이터셋 (외부 다운로드, 수정 금지)
│   ├── annotations/              #   youcookii_annotations_trainval.json
│   ├── features/                 #   사전추출 특징 (대용량, 별도 다운로드)
│   ├── scripts/                  #   download_youcookii_videos.py (공식 다운로더)
│   ├── splits/                   #   train/val/test 리스트, duration 정보
│   ├── label_foodtype.csv        #   89개 레시피 카테고리 매핑
│   └── youcookii_readme.pdf
│
├── configs/                      # 실험·모델 설정, 프로토타입 정의
│   └── action_class_prototypes.json   # 8개 단계 클래스의 프로토타입 문장 정의
│
├── data/                         # 원본·중간 데이터 (.gitignore, 별도 생성)
│   ├── videos/                   #   다운로드된 YouCook2 영상 (mp4)
│   └── frames/                   #   2 FPS로 추출한 프레임 (jpg)
│
├── processed/                    # 전처리 산출물 (CSV/JSON)
│   ├── action_annotations.csv    #   전체 annotation에 대한 프로토타입 분류 결과
│   └── required_annotations.csv  #   수동 검토가 필요한 모호 케이스 목록
│
├── src/                          # 학습·평가용 코어 코드
│   ├── data/                     #   Dataset, DataLoader, 샘플링
│   ├── models/
│   │   ├── encoders/             #   image_encoder.py (ResNet-50), text_encoder.py (DistilBERT)
│   │   ├── fusion/               #   co_attention.py, gmu.py, concat.py
│   │   └── classifier.py         #   MLP 분류 헤드 (512→256→8)
│   ├── training/                 #   Trainer, Scheduled Sampling 스케줄러
│   ├── evaluation/               #   Frame Acc, Macro F1, Segment IoU, Edit Distance
│   └── utils/                    #   로깅, 시각화 (Grad-CAM, attention map)
│
├── scripts/                      # 실행 가능한 파이프라인 스크립트
│   ├── embed_and_classify_with_prototypes.py   # 프로토타입 기반 자동 라벨링
│   ├── download_videos.py        # (예정) YouCook2 영상 다운로드 래퍼
│   ├── extract_frames.py         # (예정) 2 FPS 프레임 추출
│   ├── preprocess_annotations.py # (예정) annotation → 8단계 라벨 변환
│   ├── train.py                  # (예정) 학습 엔트리포인트
│   └── evaluate.py               # (예정) 평가 엔트리포인트
│
├── experiments/                  # E1~E8 실험 설정 파일 (YAML 등)
├── checkpoints/                  # 학습된 모델 가중치 (.gitignore)
├── reports/                      # 실험 결과 및 시각화 산출물
│   ├── figures/                  #   학습 곡선, attention map, Grad-CAM 등
│   ├── tables/                   #   실험 지표 표 (CSV/MD)
│   └── logs/                     #   학습 로그
└── notebooks/                    # EDA·결과 분석용 주피터 노트북
```

## 환경 설정

### Python 환경

```bash
# Python 3.10+ 권장
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 주요 의존성 (예정)

| 영역 | 라이브러리 |
| --- | --- |
| 딥러닝 | `torch`, `torchvision`, `transformers` |
| 임베딩 | `sentence-transformers` |
| 영상/프레임 | `opencv-python`, `ffmpeg-python`, `yt-dlp` |
| 데이터 | `numpy`, `pandas`, `scikit-learn` |
| 시각화 | `matplotlib`, `seaborn` |
| 학습 보조 | `tqdm`, `pyyaml`, `tensorboard` |

## 재현 가이드

### 1. 저장소 클론

```bash
git clone <repo-url> Phase-Recognition-Model
cd Phase-Recognition-Model
```

### 2. YouCook2 원본 데이터 확보

본 저장소에는 [YouCookII/](YouCookII/) 구조의 메타데이터(annotations, splits, label_foodtype 등)만 포함되어 있습니다. 영상과 사전추출 feature는 용량 제약으로 제외되어 있으니 아래 절차로 별도 확보하세요.

**(a) 영상 다운로드**

```bash
# YouCook2 공식 다운로드 스크립트는 YouCookII/scripts/ 에 포함
python YouCookII/scripts/download_youcookii_videos.py
# 다운로드된 영상은 data/videos/ 로 이동
```

> 일부 영상은 YouTube에서 비공개·삭제되어 다운로드 불가할 수 있습니다. 이 경우 학습/검증/테스트 셋의 실제 사용 가능 개수를 [processed/](processed/) 의 매니페스트로 별도 기록하세요.

**(b) (선택) 사전추출 특징 다운로드**

자체 인코더로 학습할 경우 불필요합니다. 공식 ResNet feature를 비교 베이스라인으로 쓰려면 YouCook2 공식 배포 페이지에서 다운로드하여 [YouCookII/features/](YouCookII/features/) 아래에 배치합니다.

### 3. 데이터 전처리

#### 3.1 2 FPS 프레임 추출

```bash
python scripts/extract_frames.py --input data/videos --output data/frames --fps 2
```

#### 3.2 annotation → 8단계 라벨 매핑

YouCook2 원본 주석은 자연어 문장입니다. 본 프로젝트는 8개 클래스 (Prep, Cut, Mix, Cook-Heat, Bake, Season, Plate, Idle) 로 재매핑하며, 이를 위해 사전 정의한 프로토타입 문장과의 임베딩 유사도로 자동 라벨링한 뒤 모호 케이스만 수동 검토합니다.

```bash
# (1) annotation 문장을 문장 단위 CSV 로 변환 → processed/annotation_sentences.csv
python scripts/preprocess_annotations.py

# (2) 프로토타입 분류 → processed/action_annotations.csv (전체 결과),
#                       processed/required_annotations.csv (검토 필요 케이스)
python scripts/embed_and_classify_with_prototypes.py --sample-size all
```

프로토타입 정의는 [configs/action_class_prototypes.json](configs/action_class_prototypes.json) 에 있으며, 임베딩 모델은 `all-MiniLM-L6-v2` 를 기본값으로 합니다.

### 4. 학습

```bash
# 기본 학습 (Co-attention 융합, Teacher Forcing)
python scripts/train.py --config experiments/E1_baseline.yaml

# E2 융합 기법 비교 (Co-attention / GMU / Concat)
python scripts/train.py --config experiments/E2_fusion_coattn.yaml
python scripts/train.py --config experiments/E2_fusion_gmu.yaml
python scripts/train.py --config experiments/E2_fusion_concat.yaml

# E4 Exposure Bias (Teacher Forcing 학습 + Free-running 평가)
python scripts/train.py --config experiments/E4_freerunning.yaml

# E5 Scheduled Sampling
python scripts/train.py --config experiments/E5_scheduled_sampling.yaml
```

체크포인트는 [checkpoints/](checkpoints/) 에 저장됩니다.

### 5. 평가

```bash
python scripts/evaluate.py --checkpoint checkpoints/<run_id>/best.pt --split test
```

평가 지표(Frame-level Accuracy, Macro F1, Segment IoU, Edit Distance, TF-FR Gap)와 시각화(Grad-CAM, Co-attention map)는 [reports/](reports/) 아래에 저장됩니다.

## 실험 체계

| ID | 주제 | 비교 축 |
| --- | --- | --- |
| E1 | 모달 Ablation | 이미지 단독 / 히스토리 단독 / 융합 |
| E2 | 융합 기법 비교 | Co-attention / GMU / Concat |
| E3 | 히스토리 길이 | k ∈ {0, 1, 3, 5} |
| E4 | Exposure Bias | TF 학습+TF 평가 / TF 학습+FR 평가 / SS 학습+FR 평가 |
| E5 | Scheduled Sampling 강도 | 샘플링 확률 스케줄 비교 |
| E6 | 레시피 카테고리별 난이도 | 10개 카테고리별 성능 |
| E7 | 도메인 일반화 | YouCook2 → EPIC-Kitchens 전이 |
| E8 | 최종 통합 평가 | 전체 베스트 설정 검증 |

## 데이터셋 출처

- **YouCook2**: Zhou et al., *Towards Automatic Learning of Procedures from Web Instructional Videos*, AAAI 2018. <http://youcook2.eecs.umich.edu/>
- **EPIC-Kitchens**: Damen et al., *Scaling Egocentric Vision: The EPIC-KITCHENS Dataset*, ECCV 2018. (E7 도메인 일반화 평가용)