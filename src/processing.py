# src/processing.py
import cv2
import numpy as np

def ensure_color(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError("지원하지 않는 이미지 형식입니다. (HxW 또는 HxWx3)")

def to_gray(image: np.ndarray) -> np.ndarray:
    image = ensure_color(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def depth_from_gray(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    if gray is None or gray.ndim != 2:
        raise ValueError("단일 채널 그레이스케일 입력이 필요합니다.")
    gray_f = gray.astype(np.float32) / 255.0  # [0,1]
    if invert:
        gray_f = 1.0 - gray_f
    return gray_f

def generate_depth_map(image: np.ndarray, invert: bool = False) -> np.ndarray:
    gray = to_gray(image)
    depth_scalar = depth_from_gray(gray, invert=invert)  # [0,1]
    gray_u8 = (depth_scalar * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)
    return depth_map  # HxWx3, uint8 (BGR)

def point_cloud_from_depth(depth_scalar: np.ndarray) -> np.ndarray:
    if depth_scalar is None or depth_scalar.ndim != 2:
        raise ValueError("단일 채널 깊이 스칼라(HxW)가 필요합니다.")
    h, w = depth_scalar.shape
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    Z = (depth_scalar.astype(np.float32) * 255.0)  # 보기 편하게 0~255 스케일
    return np.dstack((X, Y, Z)).astype(np.float32)  # HxWx3
