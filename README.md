# Image Preprocessing Pipeline

이 프로젝트는 OpenCV를 활용하여 이미지 전처리 과정을 실습하는 코드입니다. 주요 목적은 GitHub를 통한 코드 관리와 픽셀 단위 이미지 처리에 대한 이해입니다.

---

## 전처리 과정 (image_preprocessing2.py)

1. **크기 조정 (Resize)**

   - 모든 이미지를 `224 x 224` 크기로 변환하여 모델 입력 크기에 맞춤

2. **그레이스케일 변환 & 정규화**

   - 이미지를 Grayscale로 변환하고,
   - 픽셀 값을 `0 ~ 1` 범위로 정규화하여 밝기 분포를 표준화

3. **Threshold 적용**

   - `cv2.threshold()`를 사용하여 픽셀 값을 기준으로 이진화
   - 배경과 객체를 명확히 구분하기 위한 단계
   - 결과물은 `threshold.jpg`로 저장

4. **노이즈 제거 (Gaussian Blur)**

   - `(5,5)` 커널 크기의 Gaussian Blur 필터 적용
   - 불필요한 노이즈를 제거하고 부드러운 이미지를 생성

5. **데이터 증강 (Augmentation)**
   - 좌우 반전(`flip.jpg`)
   - 90도 회전(`rotate.jpg`)
   - 색상 변화(Hue shift, `color.jpg`)
   - 다양한 데이터셋 상황을 가정하여 이미지 다양성을 확보

---

## 결과물 (preprocessed_samples/)

전처리 수행 후 결과물은 `preprocessed_samples/` 폴더에 저장됩니다.

- `resized.jpg` : 크기 조정 결과
- `grayscale.jpg` : Grayscale 변환 결과
- `grayscale_norm.jpg` : 정규화된 Grayscale 결과
- `threshold.jpg` : 이진화 결과
- `blurred.jpg` : Blur 필터 적용 결과
- `flip.jpg` : 좌우 반전 결과
- `rotate.jpg` : 90도 회전 결과
- `color.jpg` : 색상(Hue) 변화 결과
