import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import cv2
from src.processing import (
    ensure_color, to_gray, depth_from_gray,
    generate_depth_map, point_cloud_from_depth
)

def synthetic_gradient_image(h=120, w=160):
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

def test_ensure_color_3ch():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    out = ensure_color(img)
    assert out.shape == img.shape

def test_ensure_color_gray():
    gray = np.zeros((10, 10), dtype=np.uint8)
    out = ensure_color(gray)
    assert out.shape == (10, 10, 3)

def test_ensure_color_none():
    with pytest.raises(ValueError):
        ensure_color(None)

def test_to_gray_shape_and_type():
    img = synthetic_gradient_image()
    gray = to_gray(img)
    assert gray.ndim == 2
    assert gray.shape == img.shape[:2]
    assert gray.dtype == np.uint8

def test_depth_from_gray_range_and_invert():
    img = synthetic_gradient_image()
    gray = to_gray(img)
    d = depth_from_gray(gray, invert=False)
    assert d.min() >= 0.0 and d.max() <= 1.0
    d_inv = depth_from_gray(gray, invert=True)
    assert np.isclose((d + d_inv).mean(), 1.0, atol=0.05)

def test_generate_depth_map_basic():
    img = synthetic_gradient_image()
    dm = generate_depth_map(img)
    assert isinstance(dm, np.ndarray)
    assert dm.shape == img.shape
    assert dm.dtype == np.uint8

def test_generate_depth_map_error_on_none():
    with pytest.raises(ValueError):
        generate_depth_map(None)

def test_point_cloud_from_depth_shape_and_monotonic():
    img = synthetic_gradient_image(h=20, w=30)
    gray = to_gray(img)
    depth = depth_from_gray(gray, invert=False)
    pc = point_cloud_from_depth(depth)
    assert pc.shape == (20, 30, 3)
    z = pc[..., 2]
    col_means = z.mean(axis=0)
    assert np.all(np.diff(col_means) >= -1e-3)

def test_point_cloud_from_depth_invalid_input():
    with pytest.raises(ValueError):
        point_cloud_from_depth(None)
    with pytest.raises(ValueError):
        point_cloud_from_depth(np.zeros((10, 10, 3), dtype=np.float32))
