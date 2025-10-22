import cv2
from ultralytics import YOLO
from utils_cv import crop_with_xyxy, analyze_patch
from pathlib import Path

def main():
    weights = Path("runs/detect/train/weights/best.pt")
    model = YOLO(str(weights)) if weights.exists() else YOLO("yolov8n.pt")

    image_path = "data/test_image.jpg"  
    image = cv2.imread(image_path)
    assert image is not None, f"이미지 없음: {image_path}"

    results = model(image, imgsz=640, conf=0.25, device='cpu')

    vis = image.copy()
    analyses = []

    for r in results:
        names = r.names
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(vis, f"{label} {conf:.2f}", (x1, max(0, y1-7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            patch = crop_with_xyxy(image, x1, y1, x2, y2)
            stats = analyze_patch(patch)
            analyses.append({"bbox": (x1,y1,x2,y2), "label": label, **stats})

            text = f"b:{stats['mean_brightness']:.0f} e:{stats['edge_density']:.2f}"
            cv2.putText(vis, text, (x1, min(y2+18, vis.shape[0]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


    try:
        cv2.imshow("YOLO Detection + Pattern", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass

    Path("runs/analysis").mkdir(parents=True, exist_ok=True)
    out_path = "runs/analysis/detect_result.jpg"
    cv2.imwrite(out_path, vis)
    print(f"[저장] {out_path}")
    print("[분석 요약] 예:", analyses[:3])

if __name__ == "__main__":
    main()
