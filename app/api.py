
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image
import io, os

MODEL_PATH = os.environ.get("RECYCLE_MODEL", "models/recycle_resnet18.pt")

app = FastAPI(title="Recycle Classifier API")

def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu")
    backbone = ckpt.get("backbone", "resnet18")
    classes = ckpt["classes"]
    if backbone == "resnet18":
        model = models.resnet18()
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, len(classes))
    else:
        model = models.vit_b_16()
        in_f = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_f, len(classes))
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, classes

model, classes = None, None
if Path(MODEL_PATH).exists():
    model, classes = load_model(Path(MODEL_PATH))

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, classes
    if model is None:
        return JSONResponse({"error": "Model not loaded. Train first."}, status_code=500)
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = out.softmax(1)[0]
        idx = int(prob.argmax().item())
        return {"label": classes[idx], "prob": float(prob[idx].item())}
