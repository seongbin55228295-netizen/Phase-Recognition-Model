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

각 시점 `t` 의 출력 ŷ_t 는 히스토리 버퍼에 추가되어 `t+1` 입력으로 재사용됩니다. 히스토리 길이 `k` 는 하이퍼파라미터입니다 (`k=0` 은 이미지 단독 베이스라인과 동치).

학습-추론 간 분포 차이(Exposure Bias)를 완화하기 위해 **Scheduled Sampling** 을 적용합니다. 학습 epoch 진행에 따라 정답 라벨 대신 모델 자신의 예측을 히스토리에 주입할 확률 `p` 를 `0 → 0.5` 로 선형 증가시킵니다.

| 시나리오 | 학습 입력 | 추론 입력 | 목적 |
| --- | --- | --- | --- |
| TF + TF (Oracle) | 정답 히스토리 | 정답 히스토리 | 상한 측정 |
| TF + FR | 정답 히스토리 | 자기 예측 히스토리 | Exposure Bias 노출 |
| SS + FR | 확률적 혼합 | 자기 예측 히스토리 | 완화 효과 검증 |

### 융합 기법 변형

동일 인코더·동일 분류 헤드 조건에서 두 가지 융합 구조를 직접 구현해 정량 비교합니다.

| 변형 | 핵심 연산 | 특징 |
| --- | --- | --- |
| **Co-attention** (주 제안) | 양방향 cross-attention 2층 | 두 모달 간 토큰 단위 상호참조, attention map 시각화로 해석 가능 |
| **Concat** | `[pool(h_img); pool(h_text)]` 후 MLP | 단순 결합 베이스라인 |

### 손실 함수

* **분류 손실**: Cross-Entropy
* **경계 프레임 가중치**: Soft Boundary — 구간 경계 ±2 프레임의 손실 가중치를 0.5로 감쇠하여 라벨 노이즈 영향 완화

## 저장소 구조

```
Phase-Recognition-Model/
├── README.md
├── requirements.txt              # Python 의존성
├── .gitignore
│
├── configs/                      # 실험·모델 설정, 프로토타입, 선별 정의
│   ├── action_class_prototypes.json   # 8개 단계 클래스의 프로토타입 문장 정의
│   └── selected_video_ids.json        # 학습/검증에 사용하는 300개 video_id 고정 목록
│
├── data/                         # 모든 데이터 (유형별 하위 폴더)
│   ├── raw/                      #   다운로드된 YouCook2 영상 mp4 (.gitignore, 별도 생성)
│   ├── frames/                   #   2 FPS로 추출한 프레임 jpg (.gitignore, 별도 생성)
│   ├── processed/                #   전처리 산출물·매니페스트 (CSV/JSON, git 추적)
│   │   ├── failed_downloads.json     #     다운로드 실패 video_id (후속 단계 가용 영상 필터)
│   │   ├── action_annotations.csv    #     프로토타입 분류 결과 (subset·recipe_type·segment 포함)
│   │   ├── review_queue.csv          #     수동 검토 필요 모호 케이스 큐
│   │   ├── reviewed_annotations.csv  #     수동 검수 완료 결과 (predicted_label = 확정 라벨)
│   │   └── frame_labels/             #     프레임 단위 라벨 (영상별 1 CSV, 학습 직접 입력) + _manifest.json (video 메타)
│   └── external/                 #   외부 데이터셋 (수정 금지)
│       └── YouCookII/            #     YouCook2 원본: annotations, splits, features, label_foodtype, 공식 다운로더
│
├── src/                          # 학습·평가·추론 코어 코드
│   ├── data/                     #   Dataset, DataLoader, 샘플링, labels
│   ├── preprocessing/            #   라벨링·프레임 추출 코어 로직 (torchvision 불필요, 단위 테스트 대상)
│   │                             #     frame_extraction.py   (ffmpeg 2fps/256 프레임 추출)
│   │                             #     annotation_labeling.py (프로토타입 분류 + 검수 분기 규칙)
│   │                             #     frame_labeling.py      (segment 병합 + Soft Boundary)
│   ├── models/
│   │   ├── encoders/             #   image_encoder.py (ResNet-50), text_encoder.py (DistilBERT)
│   │   ├── fusion/               #   co_attention.py, concat.py  (gmu.py 는 향후 작업, 미구현)
│   │   └── classifier.py         #   MLP 분류 헤드 (512→256→8)
│   ├── training/                 #   Trainer, Scheduled Sampling 스케줄러
│   ├── evaluation/               #   Frame Acc, Macro F1, Segment IoU, Edit Distance
│   ├── inference/                #   Predictor(전체영상 Free-Running), 영상 다운로드/프레임 로딩, YouCook2 의사 GT
│   └── utils/                    #   (예정) 로깅, 시각화 (Grad-CAM, attention map) — 현재 placeholder
│
├── scripts/                      # 실행 가능한 파이프라인 스크립트
│   ├── download_videos.py        # YouTube 영상 다운로드 (yt-dlp 래퍼, URL 은 YouCookII JSON 에서 직접 추출)
│   ├── extract_frames.py         # 2 FPS 프레임 추출 (선별 300개 대상, src/preprocessing 공용 함수 사용)
│   ├── generate_annotation_labels.py   # 프로토타입 기반 자동 분류 + 검수 분기 (얇은 CLI; 로직은 src/preprocessing/annotation_labeling.py)
│   ├── generate_frame_labels.py  # auto + reviewed → segment 통합 → 프레임 라벨 + Soft Boundary (얇은 CLI; 로직은 src/preprocessing/frame_labeling.py)
│   ├── train.py                  # 학습 엔트리포인트
│   ├── evaluate.py               # 전체영상 결정적 평가 (FR+TF, 변형당 reports/metrics/<variant>.json)
│   ├── compare_variants.py       # 변형별 metrics → 4개 비교 축 표 생성 (reports/tables/)
│   └── infer_video.py            # test 영상 추론 (--source youtube|youcook2 — 정성/정량)
│
├── experiments/                  # 학습 변형별 YAML 설정
├── checkpoints/                  # 학습된 모델 가중치 (.gitignore)
├── reports/                      # 실험 결과 및 시각화 산출물
│   ├── figures/                  #   학습 곡선, attention map, Grad-CAM 등
│   ├── metrics/                  #   evaluate.py 변형별 평가 결과 (<variant>.json)
│   ├── tables/                   #   compare_variants.py 비교 축 표 (CSV/MD)
│   ├── inference/                #   infer_video.py 추론 산출물 (영상별 predictions/segments/timeline/metrics)
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

### 주요 의존성

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

본 저장소에는 [data/external/YouCookII/](data/external/YouCookII/) 구조의 메타데이터(annotations, splits, label_foodtype 등)만 포함되어 있습니다. 영상과 사전추출 feature는 용량 제약으로 제외되어 있으니 아래 절차로 별도 확보하세요.

**(a) 영상 다운로드**

[scripts/download_videos.py](scripts/download_videos.py) 는 [data/external/YouCookII/annotations/youcookii_annotations_trainval.json](data/external/YouCookII/annotations/youcookii_annotations_trainval.json) 의 각 항목 `video_url` 필드에서 다운로드 URL 을 직접 추출해 yt-dlp 로 받아 [data/raw/](data/raw/) 에 저장합니다.

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

> 실행 결과는 [reports/logs/download_log.txt](reports/logs/) 와 [data/processed/failed_downloads.json](data/processed/failed_downloads.json) 에 기록되며, YouCook2 공식 다운로더([data/external/YouCookII/scripts/download_youcookii_videos.py](data/external/YouCookII/scripts/download_youcookii_videos.py)) 도 함께 제공됩니다. 일부 영상은 YouTube 정책 변경으로 비공개·삭제되어 받지 못할 수 있으며, 후속 전처리 스크립트(`generate_annotation_labels.py` 등)는 [data/processed/failed_downloads.json](data/processed/failed_downloads.json) 의 `failed_video_ids` 를 참조해 실패 영상을 자동으로 제외합니다.

**(b) (선택) 사전추출 특징 다운로드**

자체 인코더로 학습할 경우 불필요합니다. 공식 ResNet feature를 비교 베이스라인으로 쓰려면 YouCook2 공식 배포 페이지에서 다운로드하여 [data/external/YouCookII/features/](data/external/YouCookII/features/) 아래에 배치합니다.

### 3. 데이터 전처리

#### 3.1 학습 대상 300 영상 선별 및 프레임 추출

본 프로젝트는 YouCook2 train+val (1,790개) 중 다운로드 가능한 1,561개에서 **300개 영상을 선별**하여 사용합니다. 선별된 video_id 는 [configs/selected_video_ids.json](configs/selected_video_ids.json) 에 동결되어 있으며, 89개 YouCook2 레시피 카테고리에 걸친 다양 분포입니다 (training 216 / validation 84).

> **계획서 변경 사항** — 수행계획서 초안의 "10 카테고리 × 30 영상" 명세는 4개 카테고리(steak/omelet/smoothie/cookie)가 YouCook2 89개 레시피에 부재하여 그대로 구현 불가능했습니다. 동일 영상 수(300개)를 유지하되 89개 레시피 전반에 분포시키는 것으로 사양을 조정했습니다.

```bash
# 프레임 추출: configs/selected_video_ids.json 의 300개 영상에 대해
#   data/raw/ 의 영상을 2 FPS, 짧은 변 256px, q=2 JPEG 로 추출
#   data/frames/<video_id>/frame_NNNNNN.jpg 및 timestamps.json 으로 저장
#   카테고리(레시피명) 메타는 YouCookII JSON + label_foodtype.csv 에서 즉석 lookup
python scripts/extract_frames.py
```

추출 통계는 [reports/logs/frame_extraction_stats.json](reports/logs/) 에 기록됩니다. timestamps.json 은 시점 t의 프레임 ↔ segment 매핑 및 자기회귀 학습 시점 정합에 필수입니다.

#### 3.2 annotation → 8단계 라벨 매핑

YouCook2 원본 주석은 자연어 문장입니다. 본 프로젝트는 8개 클래스 (Prep, Cut, Mix, Cook-Heat, Bake, Season, Plate, Idle) 로 재매핑하며, 이를 위해 사전 정의한 프로토타입 문장과의 임베딩 유사도로 자동 라벨링한 뒤 모호 케이스만 수동 검토합니다.

```bash
# 프로토타입 임베딩과의 코사인 유사도로 자동 분류 + 검수 분기를 한 단계에 수행
#   입력: data/external/YouCookII/annotations/youcookii_annotations_trainval.json
#         − data/processed/failed_downloads.json 로 다운로드 실패 영상 자동 제외
#         + configs/action_class_prototypes.json (클래스 프로토타입)
#   출력:
#     data/processed/action_annotations.csv  (자동 학습용, good/weak 가중치 부여)
#     data/processed/review_queue.csv        (수동 검수 큐, 사유 + 제안 이슈 포함)
python scripts/generate_annotation_labels.py
```

프로토타입 정의는 [configs/action_class_prototypes.json](configs/action_class_prototypes.json) 에 있으며, 임베딩 모델은 `all-MiniLM-L6-v2` 를 기본값으로 합니다. 검수 분기 규칙·임계값(top1·margin 컷오프, 민감 라벨, 다중 액션 키워드 등)과 good/weak 경계는 [scripts/generate_annotation_labels.py](scripts/generate_annotation_labels.py) 상단의 `REVIEW_RULES`·`AUTO_QUALITY_RULES` 상수로 관리되며, 값이 안정화된 후 외부 JSON 으로부터 흡수되었습니다. 재조정 시 해당 상수를 직접 편집하세요.

#### 3.3 수동 검수 결과 변환

[data/processed/review_queue.csv](data/processed/review_queue.csv) 의 모호 케이스를 사람이 확정한 결과는 [data/processed/reviewed_annotations.csv](data/processed/reviewed_annotations.csv) 로 보관합니다. 1,839행은 동결 산출물이며 재검수는 일반적으로 필요하지 않습니다.

재검수가 필요한 경우 (예: split 임계값을 바꿔 review_queue 가 달라진 경우) 다음 컬럼 규약으로 직접 CSV 를 작성하세요:
`video_id, subset, recipe_type, segment_id, segment_start, segment_end, sentence, predicted_label` — 여기서 `predicted_label` 은 사람 확정 라벨입니다. Excel 사용 시 `-` 로 시작하는 video_id 가 `#NAME?` 로 손상되지 않도록 해당 셀을 텍스트 서식으로 강제하거나 앞에 작은따옴표(`'-...`)를 붙여 보호하세요.

#### 3.4 프레임 단위 학습 라벨 생성 (segment 통합 + Soft Boundary)

자동 분류(`action_annotations.csv`)와 수동 검수(`reviewed_annotations.csv`)를 메모리상에서 segment 단위로 병합한 뒤, 각 프레임에 해당 segment 라벨을 매핑하고 segment 양 끝 **±2 프레임의 가중치를 0.5배로 감쇠**해 경계 노이즈를 완화합니다 (Soft Boundary). 가중치 정책:
- `auto_good` → 1.0
- `auto_weak` → 0.5
- `reviewed` (사람 확정) → 1.0

```bash
python scripts/generate_frame_labels.py
# → data/processed/frame_labels/<video_id>.csv (300개)
# → data/processed/frame_labels/_manifest.json    (video 메타 — subset, recipe_type 등)
```

frame CSV 컬럼: `frame_index, frame_name, timestamp_sec, label, source, segment_id, sample_weight, is_boundary`

`_manifest.json` 에는 영상별 `{subset, recipe_type, n_segments, n_frames_total, n_frames_emitted, n_boundary_frames}` 와 전체 요약 통계가 담깁니다. 학습 코드는 이 manifest 로 train(216) / validation(84) 영상을 분리합니다.

설계 결정:
- **갭 프레임 제외**: 두 segment 사이의 빈 시간(주석되지 않은 구간)에 속하는 프레임은 출력하지 않음 — 학습 대상 아님
- **경계 폭**: 양 끝 ±2 프레임 (계획서 사양)
- **감쇠 인자**: 0.5 (`--boundary-factor` 로 조정 가능)
- **출력 분할**: 영상별 분리 (1 video = 1 csv)로 DataLoader 의 영상 단위 셔플·자기회귀 시퀀스 구성을 단순화
- **segment 머지 무중간 산출물**: action_annotations 와 reviewed_annotations 의 통합은 메모리상에서 수행되며, (video_id, segment_id) 키 파티션은 실행 중 `assert_partition` 으로 검증.

> ⚠️ **Excel 라운드트립 주의** — YouCook2의 일부 video_id 는 `-` 로 시작합니다 (예: `--bv0V6ZjWI`). Excel 은 이를 수식으로 오인해 `#NAME?` 로 치환합니다. data/processed/ 의 CSV 들은 Python 파이프라인이 생성·소비하므로 손상되지 않지만, **수작업으로 Excel 에 열어 다시 저장하면 손상이 발생**합니다. 부득이하게 Excel 로 열어야 한다면 video_id 컬럼 전체를 "텍스트" 서식으로 강제하거나 `-` 시작 셀에 작은따옴표(`'`)를 붙여 보호하세요. xlsx 임시 파일은 [.gitignore](.gitignore) 로 차단됩니다.

### 4. 학습

#### 4.1 학습 사양 (베이스라인 기본값)

런타임 단일 소스는 [experiments/baseline.yaml](experiments/baseline.yaml) 입니다. 아래 표는 그 값의 근거이며, 변형 YAML 들이 일부 키만 덮어씁니다.

| 영역 | 결정 | 위치 |
| --- | --- | --- |
| **샘플 단위** | 영상당 1 윈도우 = 연속 64 프레임 (CSV 행 기준). 64프레임 미만(3/300 영상)은 tail-padding + `valid_mask=False` | [src/data/dataset.py](src/data/dataset.py) `window_size=64` |
| **에폭당 윈도우 수** | 영상당 4 윈도우 (216 × 4 = 864 샘플/에폭 @ batch 4 → ~216 step) | YAML `data.num_windows_per_video=4` |
| **배치 크기** | 4 (AMP on). SS 경로는 윈도우당 W=64 시점의 그래프가 누적되어 backward 메모리가 batch에 비례 곱셈으로 늘어남 — RTX 5090 32GB에서 batch=8 OOM 확인되어 4로 통일 (7 variants 비교 조건 동일화) | YAML `data.batch_size=4` |
| **DataLoader 병렬화** | `num_workers=8`, `persistent_workers=True`, `prefetch_factor=4` — JPEG 디코드를 워커 풀로 분산해 GPU starvation 방지 | YAML `data.num_workers=8`, [src/data/dataset.py](src/data/dataset.py) `build_dataloaders` |
| **히스토리 포맷** | `"[t-3: Cut] [t-2: Mix] [t-1: Cook-Heat]"` (oldest first). 첫 시점은 `"[START]"`, 부분 prefix는 가용 토큰만 | [src/training/history.py](src/training/history.py) |
| **기본 히스토리 길이 k** | 3 (configurable, `k=0`은 이미지 단독과 동치) | YAML `training.history_length=3` |
| **BPTT** | 없음. 자기회귀 히스토리는 *문자열* 로만 전달되므로 prior prediction의 그래프가 끊김 (자동 detach) | [src/training/trainer.py](src/training/trainer.py) `_step_scheduled_sampling` |
| **윈도우 처리** | TF (p=0): `(B×64, …)` 단일 forward — 빠름. SS (p>0): 이미지 인코더는 윈도우당 **단일 호출로 캐싱** (`encode_image` on `B*W` 이미지) 후, 텍스트 인코더+융합+분류기만 시점별 sequential forward (history가 prior pred에 의존하므로 직렬 불가피). 캐싱 전 7.7s/it → 캐싱 후 3.2s/it 측정 | [src/training/trainer.py](src/training/trainer.py) `_step_scheduled_sampling`, [src/models/phase_recognition_model.py](src/models/phase_recognition_model.py) `forward_with_cached_image` |
| **Scheduled Sampling** | `p: 0.0 → 0.5` 선형 ramp 10 epoch, 이후 고정. 각 prior 위치는 p 확률로 모델 예측, (1-p) 확률로 정답 라벨 (위치별 독립 샘플링) | YAML `training.scheduled_sampling`, [src/training/scheduled_sampling.py](src/training/scheduled_sampling.py) |
| **옵티마이저** | AdamW, weight_decay=1e-4 | [scripts/train.py](scripts/train.py) `build_optimizer` |
| **Differential LR** | 인코더(ResNet-50 backbone + DistilBERT body) 1e-5, 헤드(projection + fusion + classifier) 1e-4 | YAML `training.optimizer` |
| **Grad clip** | global L2-norm 1.0 | YAML `training.grad_clip` |
| **AMP** | CUDA 시 ON, CPU/MPS 시 자동 OFF | YAML `training.use_amp` |
| **에폭 수** | 20 | YAML `training.epochs` |
| **융합 모듈** | Co-attention 2층, 8 heads, FFN dim 2048, dropout 0.1 | YAML `model.fusion_kwargs`, [src/models/fusion/co_attention.py](src/models/fusion/co_attention.py) |
| **분류 헤드** | 512 → 256 → 8, ReLU, dropout 0.1 | [src/models/classifier.py](src/models/classifier.py) |
| **클래스 인덱스** | `Prep=0, Cut=1, Mix=2, Cook-Heat=3, Bake=4, Season=5, Plate=6, Idle=7` ([configs/action_class_prototypes.json](configs/action_class_prototypes.json) `labels` 순) | [src/data/labels.py](src/data/labels.py) |
| **이미지 정규화** | ImageNet `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)` | [src/data/dataset.py](src/data/dataset.py) |
| **학습 augmentation** | Resize 256 → RandomCrop 224 → HorizontalFlip(0.5) | [src/data/dataset.py](src/data/dataset.py) `build_train_transform` |
| **검증 augmentation** | Resize 256 → CenterCrop 224 (random 요소 제거) | [src/data/dataset.py](src/data/dataset.py) `build_eval_transform` |
| **텍스트 max len** | 64 토큰 (k=3 + `[START]` 여유 포함) | YAML `training.max_text_len` |
| **손실** | CE × `sample_weight × valid_mask` — Soft Boundary 가중치는 전처리에서 이미 부여됨 | [src/training/trainer.py](src/training/trainer.py) `_weighted_ce` |
| **검증 모드** | Free-Running (모델 자기 예측만으로 히스토리 구성, 정답 라벨 사용 금지) | [src/training/trainer.py](src/training/trainer.py) `evaluate` |
| **체크포인트** | val frame_accuracy 최고치 갱신 시 `best.pt` (런마다 1개) + 학습 종료 시 `history.json` (epoch별 메트릭). `save_every=999`로 epoch별 weight 덤프는 비활성화 (변형당 ~8GB 디스크 사용 회피) | [scripts/train.py](scripts/train.py) |
| **타겟 하드웨어** | 단일 GPU, VRAM ≥ 16 GB (batch=4 + AMP). 측정 환경: Runpod RTX 5090 (32GB), torch 2.8+cu128 / 2.10+cu128 — 양쪽 모두 Blackwell sm_120 정상 동작 확인 | — |

#### 4.2 실행

본 프로젝트가 실제로 학습·비교하는 변형은 **4개 비교 축 × 총 7개 변형**입니다. 모든 변형은 [§4.1 학습 사양](#41-학습-사양-베이스라인-기본값)을 공유하고, 아래 표의 "핵심 변경" 키만 덮어씁니다.

| 비교 축 | 의도 | 변형 | YAML | 핵심 변경 |
| --- | --- | --- | --- | --- |
| **모달 ablation** | 자기회귀·텍스트 도움 여부 | image-only | [image_only.yaml](experiments/image_only.yaml) | `model.type: image_only`, SS off |
| | | 융합 (기준) | [baseline.yaml](experiments/baseline.yaml) | — |
| **융합 기법** | Co-attention 채택 정당화 | Co-attention (기준) | [baseline.yaml](experiments/baseline.yaml) | — |
| | | Concat | [fusion_concat.yaml](experiments/fusion_concat.yaml) | `model.fusion: concat` |
| **Exposure Bias** | TF↔FR 분포 차이 입증·완화 | TF 학습 + TF 평가 (Oracle 상한) | [tf_oracle.yaml](experiments/tf_oracle.yaml) | SS p_end=0, `eval_mode: tf` |
| | | TF 학습 + FR 평가 | [tf_freerun.yaml](experiments/tf_freerun.yaml) | SS p_end=0, `eval_mode: fr` |
| | | SS 학습 + FR 평가 (기준) | [baseline.yaml](experiments/baseline.yaml) | — |
| **SS 강도** | 완화 곡선 탐색 | p_end = 0.25 | [ss_low.yaml](experiments/ss_low.yaml) | `scheduled_sampling.p_end: 0.25` |
| | | p_end = 0.5 (기준) | [baseline.yaml](experiments/baseline.yaml) | — |
| | | p_end = 0.75 | [ss_high.yaml](experiments/ss_high.yaml) | `scheduled_sampling.p_end: 0.75` |

기준(baseline) 한 번이 4개 비교 축의 기준점을 모두 채우므로 총 학습 회수는 **7회**입니다.

```bash
# 기준 — 4개 비교 축 공통 기준점
python scripts/train.py --config experiments/baseline.yaml

# 모달 ablation
python scripts/train.py --config experiments/image_only.yaml

# 융합 기법 비교
python scripts/train.py --config experiments/fusion_concat.yaml

# Exposure Bias
python scripts/train.py --config experiments/tf_oracle.yaml
python scripts/train.py --config experiments/tf_freerun.yaml

# Scheduled Sampling 강도
python scripts/train.py --config experiments/ss_low.yaml
python scripts/train.py --config experiments/ss_high.yaml
```

각 변형은 [checkpoints/&lt;variant&gt;/](checkpoints/) 아래에 `best.pt` (val_acc 최고치, ~420MB) + `history.json` (epoch별 메트릭, 수 KB)을 남깁니다. `history.json`은 git에 commit되어 시각화·재현성에 활용되며, `best.pt`는 용량 사유로 git에서 제외(`checkpoints/**/*.pt`)되어 별도 배포(GitHub Releases 등)됩니다.

#### 4.3 사전 점검 — 오버피팅 스모크 테스트

본 학습 시작 전, 2개 영상만 사용해 의도적 과적합으로 train loss가 ~0에 수렴하는지 확인하는 [overfit_smoke.yaml](experiments/overfit_smoke.yaml)을 두었습니다. 데이터 로딩 / loss masking / history 토큰화 / fusion / classifier / gradient flow 전체 경로의 무결성을 ~3분 안에 검증.

```bash
python scripts/train.py --config experiments/overfit_smoke.yaml
```

합격 기준: 첫 5 epoch에서 train loss 50%↓, 40 epoch 안에 < 0.05. 안 떨어지면 본 학습 진입 금지. 사용 manifest는 [data/processed/frame_labels/_manifest_overfit.json](data/processed/frame_labels/_manifest_overfit.json) (2 training + 1 validation video, 후자는 DataLoader 충족용 — 메트릭 무시).

### 5. 평가 및 추론

학습이 끝난 변형들은 두 축으로 검증한다 — (5.1) **선별 300개 내 검증셋** 정량 평가, (5.2) **선별 외 test 영상** 추론(정성/정량).

#### 5.1 정량 평가 — 검증셋 전체영상 ([evaluate.py](scripts/evaluate.py))

학습 중 검증(영상당 무작위 64프레임 1윈도우)과 달리, 각 영상을 **처음부터 끝까지** 결정적으로 자기회귀 추론해 segment 단위 지표가 의미를 갖도록 한다. 동일 가중치로 **FR(Free-Running)·TF(Teacher-Forcing) 두 체제**를 모두 평가하며, 노출편향 지표 `tf_fr_gap = frame_acc_TF − frame_acc_FR` 를 함께 산출한다.

```bash
# 변형 1개 평가 → reports/metrics/<variant>.json
python scripts/evaluate.py --config experiments/baseline.yaml
#   옵션: --split {validation|training}  (기본 validation, 84영상)
#         --limit N (앞 N개만; 스모크)   --chunk 128 (forward 청크 크기)

# 나머지 변형(image_only / fusion_concat / tf_oracle / tf_freerun / ss_low / ss_high)도 각각 평가한 뒤,
# 7개 결과를 §4.2 의 4개 비교 축 표로 집계 → reports/tables/
python scripts/compare_variants.py
```

- 체크포인트는 `--checkpoint` 가 아니라 YAML 의 `checkpoint.dir` 에서 `best.pt` 를 자동 탐색한다.
- 지표: Frame Accuracy(비가중 대표값 — `tf_fr_gap` 산출 기준, sample_weight 가중값 `frame_accuracy_weighted` 도 병기), Macro F1, Segment IoU, Edit Score ([src/evaluation/metrics.py](src/evaluation/metrics.py)).
- [compare_variants.py](scripts/compare_variants.py) 는 4개 비교 축 표(modal ablation / fusion / exposure bias / SS 강도) + `all_metrics.csv` 를 [reports/tables/](reports/) 에 쓴다.
- 핵심 로직은 torchvision 없이 단위 테스트된다: `python tests/test_metrics.py`, `python tests/test_evaluate_logic.py` (메트릭·롤아웃), `python tests/test_preprocessing_labeling.py` (라벨링 분기·Soft Boundary 가중치).

#### 5.2 test 영상 추론 ([infer_video.py](scripts/infer_video.py))

선별 300개에 포함되지 않은 영상으로 일반화를 확인한다. 입력 종류는 `--source` 로 분기하지만 전처리·추론 경로는 동일하다(ffmpeg 2fps/짧은변 256 추출 → eval transform → **영상 전체 1시퀀스 Free-Running**).

```bash
# 임의 YouTube 영상 — 정성 (정답 없음): 예측 + 단계 타임라인
python scripts/infer_video.py --source youtube --url "https://youtu.be/XXXX" \
    --config experiments/baseline.yaml --checkpoint checkpoints/baseline/best.pt
# 로컬 mp4 도 가능: --video clip.mp4

# 선별 외 YouCook2 영상 — 정량: annotation 으로 의사 GT 복원 후 지표 산출
python scripts/infer_video.py --source youcook2 --video-id <id> \
    --config experiments/baseline.yaml --checkpoint checkpoints/baseline/best.pt
# 여러 개 일괄: --video-list ids.txt
```

- 산출물은 [reports/inference/](reports/)`<name>/` 아래: `predictions.csv`, `segments.json`(예측 단계 타임라인), `timeline.png`, youcook2 는 `metrics.json` 추가.
- **youcook2 정량평가의 GT 는 프로토타입 임베딩 자동 의사라벨**이다(§3.2 와 동일 매핑, 단 수동검수 없음). 즉 "사람 정답"이 아니라 "자동 라벨링과의 일치도"를 재는 것이므로 해석 시 유의한다.
- 사전 요구: `ffmpeg`, `yt-dlp`(다운로드 시). YouCook2 영상은 annotation 의 `video_url` 로 자동 다운로드되며, 봇 차단 시 `--cookies` 로 우회(2(a) 절 참고).

> 시각화(Grad-CAM, Co-attention attention map)는 [src/utils/](src/utils/) 에 예정되어 있으나 아직 미구현이다.

## 데이터셋 출처

- **YouCook2**: Zhou et al., *Towards Automatic Learning of Procedures from Web Instructional Videos*, AAAI 2018. <http://youcook2.eecs.umich.edu/>
- **EPIC-Kitchens**: Damen et al., *Scaling Egocentric Vision: The EPIC-KITCHENS Dataset*, ECCV 2018. (향후 작업의 도메인 일반화 평가용 후보 데이터셋)

## 향후 작업

본 학기 일정 안에서는 다루지 않지만, 자연스러운 후속 비교들입니다. 가성비 순서로 정리.

| 주제 | 비교 축 | 추가 필요 작업 |
| --- | --- | --- |
| 히스토리 길이 ablation | k ∈ {0, 1, 3, 5} | YAML 1개씩만 추가 — 코드 변경 없음 |
| 레시피 카테고리별 난이도 | recipe_type 그룹별 성능 | Evaluator 에 그룹 집계 코드 ~30 줄 (manifest 에 정보 이미 있음) |
| 융합 기법 추가 변형 | GMU (게이트 기반 가중합) | `src/models/fusion/gmu.py` 작성 (~50 줄) + YAML 1 개 |
| 최종 통합 평가 | 위 비교들의 최선 조합 재학습 | 비교 결과 확정 후 YAML 1 개 |
| 도메인 일반화 | YouCook2 → EPIC-Kitchens 전이 평가 | 새 데이터셋 확보 + 라벨 매핑 + 전처리 파이프라인 일체 |