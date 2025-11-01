
# AI 재활용품 분류기 (PyTorch + FastAPI)

## 목표
이미지(쓰레기 사진)를 입력하면 **paper / plastic / metal / glass / other**로 분류합니다.

## 빠른 시작
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 1) 데이터 폴더 구성
```
data/
  train/
    paper/    ...이미지들
    plastic/  ...이미지들
    metal/
    glass/
    other/
  val/
    paper/ ...
    plastic/ ...
    metal/ ...
    glass/ ...
    other/ ...
```
자체 수집 또는 공개 데이터 일부만 사용해도 됩니다.

### 2) 학습
```bash
python src/train.py --data data --epochs 2 --model models/recycle_resnet18.pt
```

### 3) 예측 (CLI)
```bash
python src/infer.py --model models/recycle_resnet18.pt --image path/to/test.jpg
```

### 4) FastAPI 서버
```bash
uvicorn app.api:app --reload --port 8000
# POST /predict  (multipart/form-data, field: file)
```

## 테스트
```bash
pytest -q
```

## 폴더 구조
src/         # 학습/추론
app/         # FastAPI 엔드포인트
tests/       # pytest
models/      # 저장된 가중치(.pt)
.vscode/     # 런치/테스트 설정
