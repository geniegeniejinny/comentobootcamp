# Image Preprocessing Pipeline

이 프로젝트는 OpenCV를 활용하여 이미지 전처리 과정을 실습하는 코드입니다. 주요 목적은 GitHub를 통한 코드 관리와 픽셀 단위 이미지 처리에 대한 이해입니다.

main 브랜치에는 image_preprocessing.py가 있고 image_preprocessing.py에는 resized, grayscale, threshold, blurred, flip, rotate, color, red_detected가 있습니다.

feature/image-processing 브랜치에는 image_preprocessing2.py가 추가되어있습니다.
image_preprocessing2.py에는 dark_removed, small_removed가 추가되어있습니다.

---

## 전처리 코드 구분

- **기본 전처리 (image_preprocessing.py)**  
  → 필수적인 이미지 처리 단계 구현

- **심화 전처리 (image_preprocessing2.py)**  
  → 기본 전처리 + 추가적인 필터링(어두운 이미지/작은 객체 제거)까지 확장

---

## 기본 전처리 과정 (image_preprocessing.py)

1. **크기 조정 (Resize)**

   - 이미지를 `224 x 224` 크기로 변환

2. **그레이스케일 변환 & 정규화**

   - 이미지를 Grayscale로 변환
   - 픽셀 값을 `0~1` 범위로 정규화

3. **Threshold 적용**

   - `cv2.threshold()`로 이진화 → 배경과 객체 구분

4. **노이즈 제거 (Gaussian Blur)**

   - `(5,5)` 커널 크기 Blur 필터로 노이즈 제거

5. **데이터 증강 (Augmentation)**

   - 좌우 반전 (`flip.jpg`)
   - 90도 회전 (`rotate.jpg`)
   - 색상 변화 (`color.jpg`)

6. **빨간색 검출 (Red Detection)**
   - HSV 색공간에서 빨간색 범위 지정
   - 빨간색 영역만 마스크 처리 (`red_detected.jpg`)

---

## 심화 전처리 과정 (image_preprocessing2.py)

위의 기본 전처리 과정에 더해, 다음 두 단계가 추가됩니다.

7. **너무 어두운 이미지 제거 (Dark Removal)**

   - 평균 밝기를 계산하여 임계값 이하일 경우 `dark_removed.jpg`로 저장

8. **작은 객체 이미지 제거 (Small Object Removal)**
   - Threshold 후 객체 윤곽선을 탐지 → 가장 큰 객체의 면적 계산
   - 전체 이미지 대비 작은 객체일 경우 `small_removed.jpg`로 저장

---

## 결과물 저장 경로

- **기본 전처리 결과** → `preprocessed_samples/`
- **심화 전처리 결과** → `preprocessed_samples2/`
