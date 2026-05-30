# Phase-Recognition-Model

자기회귀 멀티모달 학습 기반 요리 영상 단계 인식 모델

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
├── configs/                      # 실험·모델 설정, 프로토타입, 선별 정의
│   ├── action_class_prototypes.json   # 8개 단계 클래스의 프로토타입 문장 정의
│   └── selected_video_ids.json        # 학습/검증에 사용하는 300개 video_id 고정 목록
│
├── data/                         # 원본·중간 데이터 (.gitignore, 별도 생성)
│   ├── videos/                   #   다운로드된 YouCook2 영상 (mp4)
│   └── frames/                   #   2 FPS로 추출한 프레임 (jpg)
│
├── processed/                       # 전처리 산출물 및 매니페스트 (CSV/JSON/TXT)
│   ├── failed_downloads.json         #   다운로드 실패 video_id 목록 (다운로더 출력, 후속 단계의 가용 영상 필터로 사용)
│   ├── action_annotations.csv        #   프로토타입 분류 결과 (subset·recipe_type·segment 포함)
│   ├── review_queue.csv              #   수동 검토가 필요한 모호 케이스 큐
│   ├── reviewed_annotations.csv      #   수동 검수 완료 결과 (predicted_label = 확정 라벨)
│   └── frame_labels/                 #   프레임 단위 라벨 (영상별 1 CSV, 학습 직접 입력) + _manifest.json (video 메타)
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
│   ├── download_videos.py        # YouTube 영상 다운로드 (yt-dlp 래퍼, URL 은 YouCookII JSON 에서 직접 추출)
│   ├── extract_frames.py         # 2 FPS 프레임 추출 (ffmpeg, 선별 300개 영상 대상)
│   ├── generate_annotation_labels.py   # 프로토타입 기반 자동 분류 + 검수 분기 (action_annotations.csv + review_queue.csv 동시 생성)
│   ├── generate_frame_labels.py  # auto + reviewed → segment 통합 → 프레임 단위 라벨 + Soft Boundary 가중치 (frame_labels/ + _manifest.json)
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

[scripts/download_videos.py](scripts/download_videos.py) 는 [YouCookII/annotations/youcookii_annotations_trainval.json](YouCookII/annotations/youcookii_annotations_trainval.json) 의 각 항목 `video_url` 필드에서 다운로드 URL 을 직접 추출해 yt-dlp 로 받아 [data/videos/](data/videos/) 에 저장합니다.

```bash
# 사전 설치: pip install yt-dlp; ffmpeg는 시스템에 설치되어 있어야 함
python scripts/download_videos.py
```

**YouTube 봇 차단 우회 — cookies.txt 준비 (선택, 대량 다운로드 시 권장)**

YouTube 는 동일 IP에서 다수 영상을 연속 요청할 경우 봇으로 판단해 차단합니다. yt-dlp 의 `--cookies` 옵션으로 본인 브라우저의 로그인 세션 쿠키를 전달하면 차단을 우회할 수 있습니다.

1. 쿠키 파일 생성 — 둘 중 하나
   - **yt-dlp 직접 추출 (권장):** `yt-dlp --cookies-from-browser chrome --print-to-file - secrets/cookies.txt "https://www.youtube.com"`
   - **브라우저 확장:** "Get cookies.txt LOCALLY" (Chrome/Firefox) 로 youtube.com 쿠키를 Netscape 형식으로 저장
2. 파일을 `secrets/cookies.txt` 에 배치 (디렉토리 자체가 [.gitignore](.gitignore) 에 등록되어 있어 절대 커밋되지 않음)
3. 다운로드 스크립트가 자동 감지 — 다른 경로를 쓰려면 `--cookies <path>` 또는 환경변수 `YT_COOKIES_FILE` 사용

> ⚠️ cookies.txt 는 본인의 YouTube 로그인 자격증명과 동등합니다. 절대 공유하거나 커밋하지 마세요. 의심되면 YouTube 비밀번호 변경으로 세션 무효화.

> 실행 결과는 [reports/logs/download_log.txt](reports/logs/) 와 [processed/failed_downloads.json](processed/failed_downloads.json) 에 기록되며, YouCook2 공식 다운로더([YouCookII/scripts/download_youcookii_videos.py](YouCookII/scripts/download_youcookii_videos.py)) 도 함께 제공됩니다. 일부 영상은 YouTube 정책 변경으로 비공개·삭제되어 받지 못할 수 있으며, 후속 전처리 스크립트(`generate_annotation_labels.py` 등)는 [processed/failed_downloads.json](processed/failed_downloads.json) 의 `failed_video_ids` 를 참조해 실패 영상을 자동으로 제외합니다.

**(b) (선택) 사전추출 특징 다운로드**

자체 인코더로 학습할 경우 불필요합니다. 공식 ResNet feature를 비교 베이스라인으로 쓰려면 YouCook2 공식 배포 페이지에서 다운로드하여 [YouCookII/features/](YouCookII/features/) 아래에 배치합니다.

### 3. 데이터 전처리

#### 3.1 학습 대상 300 영상 선별 및 프레임 추출

본 프로젝트는 YouCook2 train+val (1,790개) 중 다운로드 가능한 1,561개에서 **300개 영상을 선별**하여 사용합니다. 선별된 video_id 는 [configs/selected_video_ids.json](configs/selected_video_ids.json) 에 동결되어 있으며, 89개 YouCook2 레시피 카테고리에 걸친 다양 분포입니다 (training 216 / validation 84).

> **계획서 변경 사항** — 수행계획서 초안의 "10 카테고리 × 30 영상" 명세는 4개 카테고리(steak/omelet/smoothie/cookie)가 YouCook2 89개 레시피에 부재하여 그대로 구현 불가능했습니다. 동일 영상 수(300개)를 유지하되 89개 레시피 전반에 분포시키는 것으로 사양을 조정했습니다.

```bash
# 프레임 추출: configs/selected_video_ids.json 의 300개 영상에 대해
#   data/videos/ 의 영상을 2 FPS, 짧은 변 256px, q=2 JPEG 로 추출
#   data/frames/<video_id>/frame_NNNNNN.jpg 및 timestamps.json 으로 저장
#   카테고리(레시피명) 메타는 YouCookII JSON + label_foodtype.csv 에서 즉석 lookup
python scripts/extract_frames.py
```

추출 통계는 [reports/logs/frame_extraction_stats.json](reports/logs/) 에 기록됩니다. timestamps.json 은 시점 t의 프레임 ↔ segment 매핑 및 자기회귀 학습 시점 정합에 필수입니다.

#### 3.2 annotation → 8단계 라벨 매핑

YouCook2 원본 주석은 자연어 문장입니다. 본 프로젝트는 8개 클래스 (Prep, Cut, Mix, Cook-Heat, Bake, Season, Plate, Idle) 로 재매핑하며, 이를 위해 사전 정의한 프로토타입 문장과의 임베딩 유사도로 자동 라벨링한 뒤 모호 케이스만 수동 검토합니다.

```bash
# 프로토타입 임베딩과의 코사인 유사도로 자동 분류 + 검수 분기를 한 단계에 수행
#   입력: YouCookII/annotations/youcookii_annotations_trainval.json
#         − processed/failed_downloads.json 로 다운로드 실패 영상 자동 제외
#         + configs/action_class_prototypes.json (클래스 프로토타입)
#   출력:
#     processed/action_annotations.csv  (자동 학습용, good/weak 가중치 부여)
#     processed/review_queue.csv        (수동 검수 큐, 사유 + 제안 이슈 포함)
python scripts/generate_annotation_labels.py
```

프로토타입 정의는 [configs/action_class_prototypes.json](configs/action_class_prototypes.json) 에 있으며, 임베딩 모델은 `all-MiniLM-L6-v2` 를 기본값으로 합니다. 검수 분기 규칙·임계값(top1·margin 컷오프, 민감 라벨, 다중 액션 키워드 등)과 good/weak 경계는 [scripts/generate_annotation_labels.py](scripts/generate_annotation_labels.py) 상단의 `REVIEW_RULES`·`AUTO_QUALITY_RULES` 상수로 관리되며, 값이 안정화된 후 외부 JSON 으로부터 흡수되었습니다. 재조정 시 해당 상수를 직접 편집하세요.

#### 3.3 수동 검수 결과 변환

[processed/review_queue.csv](processed/review_queue.csv) 의 모호 케이스를 사람이 확정한 결과는 [processed/reviewed_annotations.csv](processed/reviewed_annotations.csv) 로 보관합니다. 1,839행은 동결 산출물이며 재검수는 일반적으로 필요하지 않습니다.

재검수가 필요한 경우 (예: split 임계값을 바꿔 review_queue 가 달라진 경우) 다음 컬럼 규약으로 직접 CSV 를 작성하세요:
`video_id, subset, recipe_type, segment_id, segment_start, segment_end, sentence, predicted_label` — 여기서 `predicted_label` 은 사람 확정 라벨입니다. Excel 사용 시 `-` 로 시작하는 video_id 가 `#NAME?` 로 손상되지 않도록 해당 셀을 텍스트 서식으로 강제하거나 앞에 작은따옴표(`'-...`)를 붙여 보호하세요.

#### 3.4 프레임 단위 학습 라벨 생성 (segment 통합 + Soft Boundary)

자동 분류(`action_annotations.csv`)와 수동 검수(`reviewed_annotations.csv`)를 메모리상에서 segment 단위로 병합한 뒤, 각 프레임에 해당 segment 라벨을 매핑하고 segment 양 끝 **±2 프레임의 가중치를 0.5배로 감쇠**해 경계 노이즈를 완화합니다 (Soft Boundary). 가중치 정책:
- `auto_good` → 1.0
- `auto_weak` → 0.5
- `reviewed` (사람 확정) → 1.0

```bash
python scripts/generate_frame_labels.py
# → processed/frame_labels/<video_id>.csv (300개)
# → processed/frame_labels/_manifest.json    (video 메타 — subset, recipe_type 등)
```

frame CSV 컬럼: `frame_index, frame_name, timestamp_sec, label, source, segment_id, sample_weight, is_boundary`

`_manifest.json` 에는 영상별 `{subset, recipe_type, n_segments, n_frames_total, n_frames_emitted, n_boundary_frames}` 와 전체 요약 통계가 담깁니다. 학습 코드는 이 manifest 로 train(216) / validation(84) 영상을 분리합니다.

설계 결정:
- **갭 프레임 제외**: 두 segment 사이의 빈 시간(주석되지 않은 구간)에 속하는 프레임은 출력하지 않음 — 학습 대상 아님
- **경계 폭**: 양 끝 ±2 프레임 (계획서 사양)
- **감쇠 인자**: 0.5 (`--boundary-factor` 로 조정 가능)
- **출력 분할**: 영상별 분리 (1 video = 1 csv)로 DataLoader 의 영상 단위 셔플·자기회귀 시퀀스 구성을 단순화
- **segment 머지 무중간 산출물**: action_annotations 와 reviewed_annotations 의 통합은 메모리상에서 수행되며, (video_id, segment_id) 키 파티션은 실행 중 `assert_partition` 으로 검증.

> ⚠️ **Excel 라운드트립 주의** — YouCook2의 일부 video_id 는 `-` 로 시작합니다 (예: `--bv0V6ZjWI`). Excel 은 이를 수식으로 오인해 `#NAME?` 로 치환합니다. processed/ 의 CSV 들은 Python 파이프라인이 생성·소비하므로 손상되지 않지만, **수작업으로 Excel 에 열어 다시 저장하면 손상이 발생**합니다. 부득이하게 Excel 로 열어야 한다면 video_id 컬럼 전체를 "텍스트" 서식으로 강제하거나 `-` 시작 셀에 작은따옴표(`'`)를 붙여 보호하세요. xlsx 임시 파일은 [.gitignore](.gitignore) 로 차단됩니다.

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