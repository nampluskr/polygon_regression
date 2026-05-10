# quad_regression 프로젝트 코드 작성 규칙

## 1. 프로젝트 개요

```
목적:  CNN backbone 을 이용한 볼록 사각형 4개 꼭짓점 좌표 회귀
입력:  이미지 → resize → ImageNet 정규화
출력:  (B, 8) — [x1,y1, x2,y2, x3,y3, x4,y4], 정규화 좌표 [0,1]
단계:  Pretrain (공개 데이터) → Finetune (실측 데이터)
```

---

## 2. 폴더 구조

### 2-1. 루트 폴더 구조

```
quad_regression/
  ├── annotations/
  ├── configs/
  ├── docs/
  ├── experiments/
  ├── notebooks/
  ├── outputs/
  ├── scripts/
  ├── src/
  └── tests/
```

### 2-2. 상세 폴더 구조 + 파일명

```
quad_regression/
│
├── annotations/
│   ├── midv2020/
│   │   ├── annotations-midv2020.csv
│   │   ├── annotations-midv2020-train.csv
│   │   └── annotations-midv2020-valid.csv
│   ├── oxford_pets/
│   │   ├── annotations-oxford_pets.csv
│   │   ├── annotations-oxford_pets-train.csv
│   │   └── annotations-oxford_pets-valid.csv
│   ├── pmd/
│   │   ├── annotations-pmd.csv
│   │   ├── annotations-pmd-train.csv
│   │   └── annotations-pmd-valid.csv
│   └── smartdoc/
│       ├── annotations-smartdoc.csv
│       ├── annotations-smartdoc-train.csv
│       └── annotations-smartdoc-valid.csv
│
├── configs/
│   ├── finetune.yaml
│   ├── paths.yaml
│   └── pretrain.yaml
│
├── docs/
│   └── instructions.md
│
├── experiments/
│   ├── analysis/
│   │   ├── compare_backbones.py
│   │   ├── compare_runs.py
│   │   ├── inspect_weights.py
│   │   ├── metric_finetune.py
│   │   ├── metric_pretrain.py
│   │   ├── visualize_coords.py
│   │   └── visualize_errors.py
│   ├── debug/
│   │   ├── debug_backbone_freeze.py
│   │   ├── debug_load_weights.py
│   │   ├── debug_loss.py
│   │   ├── debug_model_build.py
│   │   ├── debug_model_output.py
│   │   └── debug_single_batch.py
│   ├── explore/
│   │   ├── check_annotation.py
│   │   ├── check_dataloader.py
│   │   ├── check_dataset.py
│   │   └── check_transforms.py
│   └── prototype/
│       ├── proto_augmentation.py
│       ├── proto_loss_fn.py
│       └── proto_scheduler.py
│
├── notebooks/
│   └── annotations/
│       ├── make_annotations_midv2020.ipynb
│       ├── make_annotations_oxford_pets.ipynb
│       ├── make_annotations_pmd.ipynb
│       └── make_annotations_smartdoc.ipynb
│
├── outputs/
│   ├── finetune/
│   │   ├── archive/
│   │   │   ├── finetune-resnet50-img224-run1.pth
│   │   │   └── finetune-resnet50-img224-run2.pth
│   │   ├── finetune-resnet50-img224.log
│   │   ├── finetune-resnet50-img224.pth
│   │   └── finetune-resnet50-img224.yaml
│   └── pretrain/
│       ├── pretrain-resnet50-img224.log
│       ├── pretrain-resnet50-img224.pth
│       └── pretrain-resnet50-img224.yaml
│
├── scripts/
│   ├── run_finetune.py
│   └── run_pretrain.py
│
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── path_utils.py
│   │   └── utils.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── coords.py
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   └── transform.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── model_builder.py
│   │   ├── regressor.py
│   │   └── simple_cnn.py
│   ├── __init__.py
│   ├── engine.py
│   └── inference.py
│
└── tests/
    ├── data/
    │   ├── test_dataset.py
    │   └── test_transforms.py
    ├── models/
    │   ├── test_model_builder.py
    │   └── test_regressor.py
    └── conftest.py
```

### 의존성 방향 (단방향 엄수)

```
core/ ← data/ ← models/ ← engine.py ← scripts/
```

---

## 3. 파일명 규칙

| 항목 | 규칙 | 예시 |
|------|------|------|
| 출력 가중치 | `{stage}-{backbone}-img{image_size}.pth` | `pretrain-resnet50-img224.pth` |
| 출력 로그 | `{stage}-{backbone}-img{image_size}.log` | `pretrain-resnet50-img224.log` |
| 출력 설정 | `{stage}-{backbone}-img{image_size}.yaml` | `pretrain-resnet50-img224.yaml` |
| finetune archive | `finetune-{backbone}-img{image_size}-run{N}.pth` | `finetune-resnet50-img224-run1.pth` |
| annotation CSV | `annotations-{dataset}.csv` | `annotations-midv2020.csv` |
| annotation train CSV | `annotations-{dataset}-train.csv` | `annotations-midv2020-train.csv` |
| annotation valid CSV | `annotations-{dataset}-valid.csv` | `annotations-midv2020-valid.csv` |
| notebooks | `make_annotations_{dataset}.ipynb` | `make_annotations_midv2020.ipynb` |
| experiments 파일 | `{폴더접두어}_{대상}.py` | `debug_model_build.py` |
| tests 파일 | `test_{대상}.py` | `test_model_builder.py` |

### experiments 접두어 규칙

| 폴더 | 접두어 | 예시 |
|------|--------|------|
| `explore/` | `check_` | `check_dataset.py` |
| `debug/` | `debug_` | `debug_single_batch.py` |
| `prototype/` | `proto_` | `proto_loss_fn.py` |
| `analysis/` | 행위동사 | `visualize_coords.py` |

---

## 4. 네이밍 규칙

### 용어 통일

| 금지 | 사용 | 이유 |
|------|------|------|
| `poly` | `quad` | 4각형 특정, 프로젝트 전체 일관성 |
| `trained` | `pretrain` | 태스크 기준 사전학습 본질 반영 |
| `tuned` | `finetune` | 도메인 fine-tuning 본질 반영 |
| `model_name` | `backbone_name` | 선택 대상이 backbone 임을 명확히 |
| `{model}` (파일명) | `{backbone}` | 파일을 구분하는 실질적 키 |
| `val` | `valid` | 5글자 통일 |
| `quad_regressor.py` | `regressor.py` | 파일명 단순화 |
| `transforms.py` | `transform.py` | 단수형 통일 |
| `get_val_transform` | `get_valid_transform` | valid 용어 통일 |
| `anotation` | `annotation` | 오타 수정 |

### 변수명 규칙

| 변수 | 범위 | 의미 |
|------|------|------|
| `results` | batch step | `train_step` / `eval_step` 반환 dict |
| `train_metrics` | epoch | `train()` 반환 집계 dict |
| `valid_metrics` | epoch | `evaluate()` 반환 집계 dict |
| `config` | 전역 | `pretrain.yaml` / `finetune.yaml` 로드 결과 |
| `paths` | 전역 | `paths.yaml` 로드 결과 |

### 함수명 규칙

| 패턴 | 예시 | 의미 |
|------|------|------|
| `build_` | `build_model` | 객체 생성 |
| `load_` | `load_backbone_weights` | 파일에서 로드 |
| `save_` | `save_model_weights` | 파일로 저장 |
| `get_` | `get_logger` | 객체 반환 |
| `format_` | `format_metrics` | 문자열 변환 |
| `run()` | experiments 진입점 | 탐색 / 디버그 실행 |
| `main()` | scripts 진입점 | 공식 학습 실행 |

---

## 5. 코드 작성 규칙

### 공통

```
- 타입힌트 / docstring 제거
- 클래스 / 함수 설명 → """docstring""" (1회만)
- 코드 동작 설명    → # comment (영어)
- 파일 첫줄         → # {파일경로} 주석
- logger 이름       → "train" 으로 전체 통일
- 등호(=) / 콜론(:) 세로 정렬 금지
```

### import 규칙

```python
# 같은 패키지 내부 → 상대 import
from .metrics import QuadIOU, PointAccuracy

# 외부 패키지 → 절대 import
from src.core.config import load_config
```

### `__init__.py`

```
src/__init__.py          # 빈 파일
src/core/__init__.py     # 빈 파일
src/data/__init__.py     # 빈 파일
src/models/__init__.py   # 빈 파일
```

---

## 6. 학습 단계 규칙

### Pretrain

```
backbone weight 로드  → load_backbone_weights() strict=False
공개 데이터 학습      → MIDV2020 + SmartDoc + oxford_pets
저장                  → pretrain-{backbone}-img{image_size}.pth/log/yaml
```

### Finetune

```
pretrain weight 로드  → load_model_weights() strict=True
실측 데이터 반복학습  → PMD
저장                  → finetune-{backbone}-img{image_size}.pth/log/yaml
archive               → finetune-{backbone}-img{image_size}-run{N}.pth
```

### `image_size` 관리 원칙

```
평가 대상: 224 (default) / 384 / 512
가중치와 image_size 는 항상 쌍으로 관리
로딩 시 동일 경로 .yaml 에서 image_size 자동 참조
→ load_model_with_config(weight_path) 사용
```

---

## 7. 모델 구조 규칙

### `build_model(backbone_name, output_dim=8)`

```
output_dim = 8 고정 (프로젝트 불변)
model.backbone / model.head 명시적 분리 필수
→ QuadRegressor 에서 freeze 제어에 사용
```

### `QuadRegressor` (`src/models/regressor.py`)

```
model      → build_model() 또는 SimpleCNN() 결과 주입
backbone   → model.backbone 참조
head       → model.head 참조
device     → 생성자에서 결정, model.to(device)
sigmoid    → forward() 내부 고정 (use_sigmoid 플래그 없음)
optimizer  → backbone / head 개별 lr 설정
             backbone_lr: 1e-5 / head_lr: 1e-4
freeze     → freeze_backbone() 시 optimizer lr=0.0 동시 적용
```

### `engine.py` 인터페이스

```
trainer.train_step(batch) → dict (loss, iou, acc, batch_size)
trainer.eval_step(batch)  → dict (loss, iou, acc, batch_size)
trainer.model.state_dict()
trainer.optimizer.param_groups
trainer.scheduler (StepLR 기준)
```

---

## 8. Loss / Metrics 규칙

### Loss (`src/models/losses.py`) — 학습용

```
QuadIOU      PyTorch 기반, GPU 지원, backprop 가능
QuadIOULoss  1 - QuadIOU
WingLoss     랜드마크 회귀 특화
CombinedLoss WingLoss + QuadIOULoss (기본 loss_fn)
```

### Metrics (`src/models/metrics.py`) — 평가용

```
QuadIOU        shapely 기반, CPU 전용, 정확한 IoU
PointAccuracy  threshold 기반 정확도 (기본: p3=0.03)
NME            정규화 평균 오차
MDE            픽셀 기준 평균 거리
```

---

## 9. Dataset / Annotation 규칙

### Dataset 구조

```
BaseDataset (ABC)     공통 로직 — csv 로드 / samples 구성
      ↓
QuadDataset           좌표 특화 — split / sampling / __getitem__
```

### `split` 값

```
"train" / "valid"   (val 사용 금지)
```

### `image_size` 기본값

```
default: 224
평가 대상: 224 / 384 / 512
항상 config["image_size"] 에서 참조 — 코드 내 하드코딩 금지
```

### Annotation CSV 생성 원칙

```
notebooks/annotations/make_annotations_{dataset}.ipynb 에서 생성

노트북 구성:
  Section 1: 데이터셋 구조 탐색 (폴더 구조 / annotation 형식 / 샘플 출력)
  Section 2: CSV 생성 (파싱 / DataFrame / 유효성 검사 / 저장)
  Section 3: CSV 검증 (로드 / 샘플 수 / 좌표 분포)
  Section 4: 시각화 (이미지 + 사각형 좌표 오버레이 / train / valid 각각)

split_and_save(df, save_dir, dataset_name, sampling, test_size, seed)
  → annotations-{dataset}.csv
  → annotations-{dataset}-train.csv
  → annotations-{dataset}-valid.csv
```

### `paths.yaml` annotation 키

```yaml
annotations:          # 전체 CSV
train_annotations:    # 학습 CSV
valid_annotations:    # 검증 CSV
images:               # 이미지 폴더
```

---

## 10. Config 규칙

### `paths.yaml`

```
경로 정보만 담당
동적 파일명 (backbone/image_size 포함) → path_utils.py 에서 생성
${변수} 참조 지원 (최대 10회 반복 치환)
구분자: 하이픈(-) 통일
```

### `pretrain.yaml` / `finetune.yaml`

```
두 파일은 완전히 동일한 항목 구조
차이: datasets / freeze_backbone / batch_size / max_epoch
image_size → 출력 파일명에 반영되는 핵심 학습 변수
```

### Early Stopping 기본값

```
monitor:   iou
mode:      max
patience:  10
scheduler: StepLR (step_size=10, gamma=0.5)
```

---

## 11. `.gitignore` 규칙

```
outputs/                  # 학습 결과물 제외
experiments/explore/      # 탐색 스크립트 제외
experiments/debug/        # 디버그 스크립트 제외
experiments/prototype/    # 프로토타입 스크립트 제외
experiments/analysis/     # 분석 결과 추적 포함
notebooks/                # jupyter 선택적
annotations/pmd/          # PMD 보안상 제외
.pytest_cache/
__pycache__/
.env
```

---

## 12. 확정된 파일 목록

| 파일 | 상태 |
|------|------|
| `configs/paths.yaml` | 작성 완료 |
| `configs/pretrain.yaml` | 작성 완료 |
| `configs/finetune.yaml` | 작성 완료 |
| `src/core/config.py` | 작성 완료 |
| `src/core/logger.py` | 작성 완료 |
| `src/core/path_utils.py` | 작성 완료 |
| `src/core/utils.py` | 작성 완료 |
| `src/data/coords.py` | 작성 완료 |
| `src/data/dataloader.py` | 작성 완료 |
| `src/data/dataset.py` | 작성 완료 |
| `src/data/transform.py` | 작성 완료 |
| `src/models/losses.py` | 작성 완료 |
| `src/models/metrics.py` | 작성 완료 |
| `src/models/model_builder.py` | 작성 완료 |
| `src/models/regressor.py` | 작성 완료 |
| `src/models/simple_cnn.py` | 작성 완료 |
| `src/engine.py` | 작성 완료 |
| `src/inference.py` | 작성 완료 |
| `notebooks/annotations/make_annotations_oxford_pets.ipynb` | 작성 완료 |
| `notebooks/annotations/make_annotations_midv2020.ipynb` | 작성 완료 |
| `notebooks/annotations/make_annotations_smartdoc.ipynb` | 작성 완료 |
| `notebooks/annotations/make_annotations_pmd.ipynb` | 미작성 |
| `scripts/run_pretrain.py` | 미작성 |
| `scripts/run_finetune.py` | 미작성 |
| `tests/` | 미확정 |