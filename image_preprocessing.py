import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def resize_image(image, size=(224, 224)):
    """224x224 크기로 조정"""
    return cv2.resize(image, size)

def to_grayscale_and_normalize(image):
    """Grayscale 변환 후 Normalize (0~1 스케일)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm = gray / 255.0
    return gray, norm

def apply_threshold(gray_image, threshold=127):
    """그레이스케일 이미지에 임계값 적용 (이진화)"""
    _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def denoise_blur(image):
    """노이즈 제거 (Gaussian Blur)"""
    return cv2.GaussianBlur(image, (5, 5), 0)

def augment_image(image):
    """데이터 증강: 좌우 반전, 회전, 색상 변화"""
    augmented = {}
    # 좌우 반전
    augmented['flip'] = cv2.flip(image, 1)
    # 회전 (90도)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 90, 1)
    augmented['rotate'] = cv2.warpAffine(image, M, (w, h))
    # 색상 변화 (HSV에서 Hue shift)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 50) % 180
    augmented['color'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return augmented

def detect_red(image):
    """빨간색 영역 추출"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 빨간색 범위 두 구간 지정
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def preprocess_and_save(img_path, save_dir="preprocessed_samples"):
    """전체 전처리 파이프라인 실행"""
    image = cv2.imread(img_path)
    if image is None:
        print("이미지를 찾을 수 없습니다:", img_path)
        return

    os.makedirs(save_dir, exist_ok=True)

    # 1. 크기 조정
    resized = resize_image(image)
    cv2.imwrite(os.path.join(save_dir, "resized.jpg"), resized)

    # 2. 그레이스케일 & 정규화
    gray, gray_norm = to_grayscale_and_normalize(resized)
    cv2.imwrite(os.path.join(save_dir, "grayscale.jpg"), gray)
    cv2.imwrite(os.path.join(save_dir, "grayscale_norm.jpg"), (gray_norm*255).astype(np.uint8))

    # 3. Threshold 적용
    binary = apply_threshold(gray, threshold=127)
    cv2.imwrite(os.path.join(save_dir, "threshold.jpg"), binary)

    # 4. 블러 처리
    blurred = denoise_blur(resized)
    cv2.imwrite(os.path.join(save_dir, "blurred.jpg"), blurred)

    # 5. 데이터 증강
    aug = augment_image(resized)
    for k, v in aug.items():
        cv2.imwrite(os.path.join(save_dir, f"{k}.jpg"), v)

    # 6. 빨간색 검출
    red_detected = detect_red(resized)
    cv2.imwrite(os.path.join(save_dir, "red_detected.jpg"), red_detected)

    # 결과 출력 (비교용)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 5, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.title("Grayscale")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.title("Threshold")
    plt.imshow(binary, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.title("Blurred")
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.title("Red Detected")
    plt.imshow(cv2.cvtColor(red_detected, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()
    print(f"처리된 이미지들이 '{save_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    preprocess_and_save("sample.jpg")

