
import argparse, json
from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image

def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu")
    backbone = ckpt.get("backbone", "resnet18")
    classes = ckpt["classes"]
    if backbone == "resnet18":
        model = models.resnet18()
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, len(classes))
    elif backbone == "vit_b_16":
        model = models.vit_b_16()
        in_f = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_f, len(classes))
    else:
        raise ValueError("Unsupported backbone")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, classes

def preprocess(img: Image.Image):
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tf(img).unsqueeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    args = ap.parse_args()

    model, classes = load_model(Path(args.model))
    img = Image.open(args.image).convert("RGB")
    x = preprocess(img)
    with torch.no_grad():
        out = model(x)
        prob = out.softmax(1)[0]
        idx = int(prob.argmax().item())
        print(json.dumps({"label": classes[idx], "prob": float(prob[idx].item())}, ensure_ascii=False))

if __name__ == "__main__":
    main()
