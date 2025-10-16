import os
import cv2
import numpy as np
from pathlib import Path
from src.processing import to_gray, generate_depth_map, depth_from_gray, point_cloud_from_depth

# ===== 시각 효과 파라미터 =====
DEPTH_SCALE = 10.0   # 깊이 과장 배율 (크면 클수록 산처럼 솟음)
GAMMA = 0.8          # 감마 보정 (0.6~1.2 사이로 조절 추천; 1.0=보정 없음)
GAUSS_KSIZE = 5      # 가우시안 블러 커널(홀수). 0이나 1이면 블러 없음
SURF_MAX_SIDE = 220  # 표면그래프 다운샘플 목표 한 변 크기
PC_MAX_SIDE = 260    # 포인트클라우드 다운샘플 목표 한 변 크기
ELEV, AZIM = 35, -55 # 3D 시점 각도

def load_or_synthesize(path="samples/gom.png", h=240, w=320):
    p = Path(path)
    if p.exists():
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"이미지 로드 실패: {p.resolve()}")
        print(f"[INFO] 입력 이미지 사용: {p.resolve()}")
        return img
    print("[INFO] samples/gom.png 없음 → 합성 그라디언트 이미지 사용")
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

def apply_gamma(depth_scalar, gamma=1.0):
    if gamma == 1.0:
        return depth_scalar
    ds = np.clip(depth_scalar, 0, 1).astype(np.float32)
    return np.power(ds, gamma)

def smooth(depth_scalar, ksize=0):
    if not ksize or ksize <= 1:
        return depth_scalar
    tmp = (np.clip(depth_scalar, 0, 1) * 255).astype(np.uint8)
    tmp = cv2.GaussianBlur(tmp, (ksize, ksize), 0)
    return tmp.astype(np.float32) / 255.0

def must_save(filename, img):
    ok = cv2.imwrite(filename, img)
    if not ok:
        raise RuntimeError(f"파일 저장 실패: {os.path.abspath(filename)}")
    print(f"[OK] 저장됨 → {os.path.abspath(filename)}")

if __name__ == "__main__":
    print(f"[INFO] 작업 폴더: {os.getcwd()}")

    # 1) 입력 로드 (없으면 합성)
    image = load_or_synthesize("samples/gom.png")
    gray = to_gray(image)

    # 2) 깊이맵(BGR 컬러) 생성 및 저장 (항상 저장)
    depth_map = generate_depth_map(image)
    must_save("out_original.jpg", image)
    must_save("out_depth_map.jpg", depth_map)

    # 3) 깊이 스칼라 → 감마/스무딩 → 포인트클라우드(Z 과장)
    depth_scalar = depth_from_gray(gray)             # [0,1]
    depth_scalar = apply_gamma(depth_scalar, GAMMA)  # 감마
    depth_scalar = smooth(depth_scalar, GAUSS_KSIZE) # 스무딩

    pc = point_cloud_from_depth(depth_scalar)        # HxWx3 (X,Y,Z[0~255])
    pc[..., 2] = pc[..., 2] * DEPTH_SCALE            # Z 과장

    # 4) Z(깊이) 평면 컬러맵 저장
    z = pc[..., 2].astype(np.float32)
    z_norm = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    z_color = cv2.applyColorMap(z_norm, cv2.COLORMAP_TURBO)
    must_save("out_depth_Z_colormap.jpg", z_color)

    # 5) (옵션) 3D 그래프 저장 — matplotlib 설치 시만 수행
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib.colors import LightSource

        h, w = z.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        # 다운샘플 비율 계산
        def downstep(size, target):
            return max(1, int(round(size / max(1, target))))

        step_surf = max(downstep(h, SURF_MAX_SIDE), downstep(w, SURF_MAX_SIDE))
        step_pc   = max(downstep(h, PC_MAX_SIDE),   downstep(w, PC_MAX_SIDE))

        Xs, Ys, Zs = X[::step_surf, ::step_surf], Y[::step_surf, ::step_surf], z[::step_surf, ::step_surf]
        Xp, Yp, Zp = X[::step_pc,   ::step_pc],   Y[::step_pc,   ::step_pc],   z[::step_pc,   ::step_pc]

        # --- 표면 그래프 (조명/셰이딩) ---
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        Zs_min, Zs_max = np.min(Zs), np.max(Zs)
        Zs_norm = (Zs - Zs_min) / max(1e-6, (Zs_max - Zs_min))

        ls = LightSource(azdeg=315, altdeg=45)
        shaded_rgb = ls.shade(Zs_norm, cmap=plt.get_cmap('jet'), vert_exag=1.0, blend_mode='soft')

        ax.plot_surface(Xs, Ys, Zs, rstride=1, cstride=1,
                        facecolors=shaded_rgb, linewidth=0, antialiased=False)
        ax.set_title(f"3D Depth Surface (scale={DEPTH_SCALE}, gamma={GAMMA}, blur={GAUSS_KSIZE})")
        ax.set_xlabel("X (width)"); ax.set_ylabel("Y (height)"); ax.set_zlabel("Depth (Z)")
        ax.view_init(elev=ELEV, azim=AZIM)
        plt.tight_layout()
        fig.savefig("out_depth_surface_exaggerated.png", dpi=220)
        print(f"[OK] 저장됨 → {os.path.abspath('out_depth_surface_exaggerated.png')}")

        # --- 포인트클라우드 산점도 ---
        fig2 = plt.figure(figsize=(7, 5))
        ax2 = fig2.add_subplot(111, projection='3d')
        sc = ax2.scatter(Xp.ravel(), Yp.ravel(), Zp.ravel(), s=1, c=Zp.ravel(), cmap='turbo')
        ax2.set_title("3D Point Cloud (downsampled)")
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
        ax2.view_init(elev=ELEV, azim=AZIM)
        plt.tight_layout()
        fig2.savefig("out_depth_pointcloud_exaggerated.png", dpi=220)
        print(f"[OK] 저장됨 → {os.path.abspath('out_depth_pointcloud_exaggerated.png')}")
    except Exception as e:
        print(f"[WARN] 3D 그래프 생략(옵션): {e}")

    print("[DONE] 모든 저장 완료. 이미지 저장 경로:", os.getcwd())
