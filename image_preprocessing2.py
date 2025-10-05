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
    return norm

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
    gray_norm = to_grayscale_and_normalize(resized)
    cv2.imwrite(os.path.join(save_dir, "grayscale.jpg"), (gray_norm*255).astype(np.uint8))

    # 3. 블러 처리
    blurred = denoise_blur(resized)
    cv2.imwrite(os.path.join(save_dir, "blurred.jpg"), blurred)

    # 4. 데이터 증강
    aug = augment_image(resized)
    for k, v in aug.items():
        cv2.imwrite(os.path.join(save_dir, f"{k}.jpg"), v)

    # 결과 출력 (비교용)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Resized")
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Blurred")
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()
    print(f"처리된 이미지들이 '{save_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    preprocess_and_save("sample.jpg")
