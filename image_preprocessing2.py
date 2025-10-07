import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)

def to_grayscale_and_normalize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm = gray / 255.0
    return gray, norm

def apply_threshold(gray_image, threshold=127):
    _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def denoise_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def augment_image(image):
    augmented = {}
    augmented['flip'] = cv2.flip(image, 1)

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 90, 1)
    augmented['rotate'] = cv2.warpAffine(image, M, (w, h))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 50) % 180
    augmented['color'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return augmented

def detect_red(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def dark_removed(image, threshold=50):
    """어두운 이미지 여부 저장"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < threshold:
        return True
    return False

def small_removed(image, min_ratio=0.05):
    """객체 크기 작은 이미지 여부 저장"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True
    largest_area = max(cv2.contourArea(c) for c in contours)
    h, w = gray.shape
    image_area = h * w
    return (largest_area / image_area) < min_ratio

def preprocess_and_save(img_path, save_dir="preprocessed_samples2"):
    image = cv2.imread(img_path)
    if image is None:
        print("이미지를 찾을 수 없습니다:", img_path)
        return

    os.makedirs(save_dir, exist_ok=True)

    # 1. 크기 조정
    resized = resize_image(image)
    cv2.imwrite(os.path.join(save_dir, "resized.jpg"), resized)

    # 2. Grayscale & 정규화
    gray, gray_norm = to_grayscale_and_normalize(resized)
    cv2.imwrite(os.path.join(save_dir, "grayscale.jpg"), gray)
    cv2.imwrite(os.path.join(save_dir, "grayscale_norm.jpg"), (gray_norm*255).astype(np.uint8))

    # 3. Threshold 적용
    binary = apply_threshold(gray, threshold=127)
    cv2.imwrite(os.path.join(save_dir, "threshold.jpg"), binary)

    # 4. Blur 처리
    blurred = denoise_blur(resized)
    cv2.imwrite(os.path.join(save_dir, "blurred.jpg"), blurred)

    # 5. 데이터 증강
    aug = augment_image(resized)
    for k, v in aug.items():
        cv2.imwrite(os.path.join(save_dir, f"{k}.jpg"), v)

    # 6. 빨간색 검출
    red_detected = detect_red(resized)
    cv2.imwrite(os.path.join(save_dir, "red_detected.jpg"), red_detected)

    # 7. 어두운 이미지 여부 저장
    if dark_removed(resized):
        cv2.imwrite(os.path.join(save_dir, "dark_removed.jpg"), resized)

    # 8. 작은 객체 이미지 여부 저장
    if small_removed(resized):
        cv2.imwrite(os.path.join(save_dir, "small_removed.jpg"), resized)

    print(f"모든 전처리 및 검사 이미지가 '{save_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    preprocess_and_save("sample.jpg")
