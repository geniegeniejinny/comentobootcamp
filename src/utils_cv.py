import cv2
import numpy as np

def crop_with_xyxy(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return img[y1:y2, x1:x2]

def analyze_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness  = float(np.std(gray))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    hist_b = cv2.calcHist([patch],[0],None,[8],[0,256]).flatten()
    hist_g = cv2.calcHist([patch],[1],None,[8],[0,256]).flatten()
    hist_r = cv2.calcHist([patch],[2],None,[8],[0,256]).flatten()
    hist = np.concatenate([hist_b, hist_g, hist_r])
    hist = hist / (np.sum(hist) + 1e-6)
    top3_bins = hist.argsort()[-3:][::-1].tolist()

    return {
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "edge_density": edge_density,
        "top3_bins": top3_bins
    }
