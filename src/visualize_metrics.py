import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def pick_column(df, keyword):
    cols = [c for c in df.columns if keyword in c]
    if not cols:
        raise KeyError(f"'{keyword}' 포함 컬럼 없음. df.columns 확인 필요: {list(df.columns)[:10]} ...")
    return cols[0]

def main():
    csv_path = Path("runs/detect/train/results.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} 없음. 먼저 train을 실행하세요.")

    df = pd.read_csv(csv_path)

    epoch = df['epoch'] if 'epoch' in df.columns else df.index
    prec = df[pick_column(df, 'metrics/precision')]
    rec  = df[pick_column(df, 'metrics/recall')]
    map50 = df[pick_column(df, 'metrics/mAP50')]

    Path("runs/plots").mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(epoch, prec, label='Precision')
    plt.plot(epoch, rec, label='Recall')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Precision & Recall per Epoch')
    plt.legend(); plt.grid(True)
    plt.savefig("runs/plots/precision_recall.png", dpi=150)

    plt.figure()
    plt.plot(epoch, map50, label='mAP@50')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('mAP@50 per Epoch')
    plt.legend(); plt.grid(True)
    plt.savefig("runs/plots/map50.png", dpi=150)

    print("[저장] runs/plots/precision_recall.png, runs/plots/map50.png")

if __name__ == "__main__":
    main()
