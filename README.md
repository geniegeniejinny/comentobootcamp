# AI 재활용품 분류기 (PyTorch)

## 목표

이미지(쓰레기 사진)를 입력하면 **paper / plastic / metal / glass / other**로 분류합니다.

## 빠른 시작

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 데이터 구성

```bash
Kaggle Garbage Classification 데이터셋을 80:20으로 분할해 사용합니다.

data/
  train/
    cardboard/ glass/ metal/ paper/ plastic/ trash/
  val/
    cardboard/ glass/ metal/ paper/ plastic/ trash/


```

## 학습(Training)

```bash
# 기본
python3 src/train.py --data data --epochs 5 --model models/recycle_resnet18.pt

# 권장(성능 개선: 증강/가중치/LR 스케줄러 적용본)
python3 src/train.py --data data --epochs 15 --batch 32 --lr 5e-4 --model models/recycle_resnet18.pt

```

## 추론(Inference, CLI)

```bash
python3 src/infer.py --model models/recycle_resnet18.pt --image data/val/plastic/예시파일.jpg
# 출력: {"label": "...", "prob": 0.XXX}

```

## 평가 지표

- Validation Accuracy: **~0.93 (epoch≈15, ResNet18 전이학습)**
- 혼동행렬(Confusion Matrix) 및 오분류 분석 결과는 아래와 같습니다.

<p align="center">
  <img src="https://github.com/geniegeniejinny/comentobootcamp/blob/08b05eacdf4f4792ab8e15e9fad3b47bc789492c/outputs/confusion_matrix_20251101-195227.png" width="500" alt="Confusion Matrix">
</p>

## 폴더 구조

src/ # 학습/추론
app/ # FastAPI 엔드포인트
tests/ # pytest
models/ # 저장된 가중치(.pt)
.vscode/ # 런치/테스트 설정

## 라이선스/출처

- Dataset: Kaggle Garbage Classification (학습·교육 목적 사용)
