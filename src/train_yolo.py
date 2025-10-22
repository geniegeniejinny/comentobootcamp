from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="data/data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,          
        patience=10,
        augment=True,
        device='cpu'       
    )
    metrics = model.val()
    print("\n[VAL METRICS]", metrics)

if __name__ == "__main__":
    main()
