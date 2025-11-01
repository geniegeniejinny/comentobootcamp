import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torchvision import datasets, transforms, models
from pathlib import Path
import numpy as np
from datetime import datetime
import json

# -------------------------------
# 설정
# -------------------------------
val_dir = Path("data/val")
model_path = Path("models/recycle_resnet18.pt")
out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
cm_png = out_dir / f"confusion_matrix_{ts}.png"
report_txt = out_dir / f"classification_report_{ts}.txt"
meta_json = out_dir / f"metrics_{ts}.json"

# -------------------------------
# 데이터 로드
# -------------------------------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
dataset = datasets.ImageFolder(str(val_dir), transform=tf)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# -------------------------------
# 모델 로드
# -------------------------------
ckpt = torch.load(model_path, map_location="cpu")
classes = ckpt["classes"]
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

# -------------------------------
# 예측 수집
# -------------------------------
all_labels, all_preds = [], []
with torch.no_grad():
    for x, y in loader:
        out = model(x)
        preds = out.argmax(1)
        all_labels.extend(y.numpy())
        all_preds.extend(preds.numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)

# -------------------------------
# 지표 & 혼동행렬
# -------------------------------
cm = confusion_matrix(all_labels, all_preds)
acc = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=classes, digits=3)

print("📊 Validation accuracy:", f"{acc:.4f}")
print("📄 Classification Report:\n", report)

# 텍스트/메타 저장
report_txt.write_text(f"Accuracy: {acc:.4f}\n\n{report}", encoding="utf-8")
meta_json.write_text(json.dumps({
    "accuracy": float(acc),
    "classes": classes,
    "timestamp": ts,
}, ensure_ascii=False, indent=2), encoding="utf-8")

# 그림 저장
plt.figure(figsize=(7.2, 6.4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (acc={acc:.3f})")
plt.tight_layout()
plt.savefig(cm_png, dpi=200)
plt.show()

print(f"\n 저장 완료:\n - {cm_png}\n - {report_txt}\n - {meta_json}")
