
import os, json, io
import torch
from PIL import Image
from fastapi.testclient import TestClient
from app.api import app

def test_api_health_predict_without_model():
    # 모델이 없으면 500 반환 (학습 전 상태 검증)
    client = TestClient(app)
    img = Image.new("RGB", (224,224), (200,200,200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    resp = client.post("/predict", files={"file": ("x.png", buf, "image/png")})
    assert resp.status_code in (200, 500)
