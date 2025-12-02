from pathlib import Path
import time

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------------------------------------------------
# 0. 디바이스 / PyTorch 설정
# ---------------------------------------------------

# ✅ 여기서 디바이스 강제 선택
# - "cpu"  로 두면, GPU가 있어도 무조건 CPU로만 돌림
# - "cuda" 로 두면, GPU가 있어야 하고, 없으면 에러 날 수 있음
# - None 으로 두면, "cuda 있으면 cuda, 없으면 cpu" 자동 선택
FORCE_DEVICE = "cpu"   # <- 지금은 CPU 테스트용. GPU 쓰고 싶으면 "cuda" 또는 None 으로 변경.

if FORCE_DEVICE is not None:
    DEVICE = FORCE_DEVICE
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧 DEVICE = {DEVICE}")

torch.set_grad_enabled(False)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True  # 입력 크기 고정이면 성능 ↑

# ---------------------------------------------------
# 1. 기본 설정
# ---------------------------------------------------

# FPS를 측정할 입력 영상 경로
VIDEO_PATH = r"D:\dataset\AIhub\test_video3.avi"   # 네 테스트 영상

# 모델별 실험 폴더 / 가중치 / 이름 설정
MODELS = {
    "yolov8n": {
        "exp_dir": Path(r"D:\dataset\AIhub\runs\detect\yolov8n"),
        "weights": Path(r"D:\dataset\AIhub\runs\detect\yolov8n\weights\best.pt"),
    },
    "fine_tune(epochs=80)": {
        "exp_dir": Path(r"D:\dataset\AIhub\runs\detect\fine_tune_ft6"),
        "weights": Path(r"D:\dataset\AIhub\runs\detect\fine_tune_ft6\weights\best.pt"),
    },
    "fine_tune_hyp(epochs=80)": {
        "exp_dir": Path(r"D:\dataset\AIhub\runs\detect\fine_tune_ft7"),
        "weights": Path(r"D:\dataset\AIhub\runs\detect\fine_tune_ft7\weights\best.pt"),
    },
}

# FPS 측정 시 사용할 프레임 개수
# ⚠ CPU에서 너무 느리면 200 → 100 또는 50으로 줄여도 됨
NUM_FRAMES_FOR_FPS = 200

# 파이프라인 FPS를 몇 번 반복 측정해서 평균낼지
PIPELINE_FPS_REPEAT = 1   # CPU는 1 권장, GPU는 3 정도로 올려도 됨

# 워밍업용 더미 추론 횟수
WARMUP_ITERS = 20

SAVE_DIR = Path(r"D:\dataset\AIhub")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = SAVE_DIR / "score.png"

# ---------------------------------------------------
# 2. mAP50-95를 CSV에서 추출
# ---------------------------------------------------

def load_map_from_csv(exp_dir: Path) -> float:
    """
    Ultralytics가 저장한 results.csv에서 마지막 에폭의 mAP50-95(B)를 읽어온다.
    """
    csv_path = exp_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv를 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    col_name = "metrics/mAP50-95(B)"
    if col_name not in df.columns:
        raise KeyError(
            f"{col_name} 열이 없습니다. CSV 열 이름을 한 번 확인해 주세요.\n열들: {list(df.columns)}"
        )

    map5095 = float(df[col_name].iloc[-1])  # 마지막 에폭 값
    return map5095

# ---------------------------------------------------
# 3. ROI 정의 (정규좌표) - car_count 코드와 맞춤
# ---------------------------------------------------

LEFT_LANE_POINTS_NORM = [
    (0.01, 0.85),
    (0.43, 0.85),
    (0.48, 0.5),
    (0.33, 0.5),
]

RIGHT_LANE_POINTS_NORM = [
    (0.58, 0.85),
    (0.92, 0.85),
    (0.65, 0.5),
    (0.535, 0.5),
]


def norm_to_pixels(norm_points, width, height):
    return np.array(
        [[int(x * width), int(y * height)] for (x, y) in norm_points],
        dtype=np.int32
    )


def in_lane_roi(cx: int, cy: int, left_poly: np.ndarray, right_poly: np.ndarray) -> bool:
    """
    박스 중심점 (cx, cy)가 왼쪽/오른쪽 차선 폴리곤 내부에 있는지 여부.
    """
    pt = (float(cx), float(cy))
    in_left = cv2.pointPolygonTest(left_poly, pt, False) >= 0
    in_right = cv2.pointPolygonTest(right_poly, pt, False) >= 0
    return in_left or in_right

# ---------------------------------------------------
# 4. GFLOPs / Params 계산 (thop 사용)
# ---------------------------------------------------

def get_model_flops_params(weights_path: Path,
                           img_size: int = 640):
    """
    thop을 이용해서 GFLOPs / Params 계산.
    thop이 없으면 (None, None) 리턴.
    """
    try:
        from thop import profile
    except ImportError:
        print(f"\n[MODEL INFO] {weights_path}")
        print("  (thop 패키지가 없어 GFLOPs/Params 계산을 생략합니다. 'pip install thop' 으로 설치 가능)")
        return None, None

    print(f"\n[MODEL INFO] {weights_path}")
    model = YOLO(str(weights_path))
    model.to(DEVICE)

    dummy = torch.zeros(1, 3, img_size, img_size, device=DEVICE)

    net = model.model
    net.eval()

    flops, params = profile(net, inputs=(dummy,), verbose=False)
    gflops = flops / 1e9
    mparams = params / 1e6

    print(f"  GFLOPs: {gflops:.2f}")
    print(f"  Params: {mparams:.2f} M")

    return gflops, mparams

# ---------------------------------------------------
# 5. 순수 모델 FPS 측정 (더미 입력만 사용)
# ---------------------------------------------------

def measure_model_only_fps(weights_path: Path,
                           img_size: int = 640,
                           num_iters: int = 200) -> float:
    """
    비디오/ROI/루프 다 빼고, 순수하게 모델 forward + NMS 속도만 측정.
    더미 텐서(1, 3, img_size, img_size)만 계속 넣어서 FPS 계산.
    """
    print(f"\n[MODEL ONLY FPS] 모델: {weights_path}")
    model = YOLO(str(weights_path))
    model.to(DEVICE)

    dummy = torch.zeros(1, 3, img_size, img_size, device=DEVICE)

    # 워밍업
    for _ in range(WARMUP_ITERS):
        _ = model(dummy, conf=0.45, verbose=False)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(num_iters):
        _ = model(dummy, conf=0.45, verbose=False)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - t0

    fps = num_iters / total_time if total_time > 0 else 0.0
    print(f"  - 반복 횟수: {num_iters}")
    print(f"  - 총 시간: {total_time:.3f} s")
    print(f"  -> MODEL ONLY FPS: {fps:.2f}")
    return fps

# ---------------------------------------------------
# 6. 영상 + ROI 포함 파이프라인 FPS 측정
# ---------------------------------------------------

def measure_fps_single_run(model: YOLO,
                           names,
                           video_path: str,
                           num_frames: int = 200) -> float:
    """
    한 번만 돌려서 FPS 측정하는 함수 (내부용).
    비디오 읽기 + YOLO 추론 + ROI 필터링 포함.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")

    # 먼저 한 프레임 읽어서 해상도 얻고 ROI 폴리곤 픽셀 좌표 계산
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("영상에서 프레임을 읽을 수 없습니다 (빈 영상?).")

    h, w = first_frame.shape[:2]
    left_poly = norm_to_pixels(LEFT_LANE_POINTS_NORM, w, h)
    right_poly = norm_to_pixels(RIGHT_LANE_POINTS_NORM, w, h)

    # --- GPU/모델 워밍업: 실제 측정 전에 더미 프레임으로 몇 번 돌려주기 ---
    dummy = np.zeros_like(first_frame)
    for _ in range(WARMUP_ITERS):
        _ = model(dummy, conf=0.45, verbose=False)

    # 측정을 위해 다시 처음부터
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝

        # 실제 추론
        results = model(frame, conf=0.45, verbose=False)[0]

        # car_count와 동일하게: 클래스 'car' + ROI 내 중심점만 남기는 후처리
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))

                if cls_name.lower() != "car":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if not in_lane_roi(cx, cy, left_poly, right_poly):
                    continue

                # FPS 측정 목적이라 추가 작업은 안 함
                pass

        frame_count += 1

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - t0
    cap.release()

    if frame_count == 0 or total_time == 0:
        return 0.0

    fps = frame_count / total_time
    return fps


def measure_pipeline_fps(weights_path: Path,
                         video_path: str,
                         num_frames: int = 200,
                         repeat: int = 1) -> float:
    """
    주어진 가중치와 영상으로 YOLO를 여러 번 돌려서 평균 FPS를 측정한다.
    - repeat: 같은 세팅으로 몇 번 반복 측정할지
    """
    print(f"\n[PIPELINE FPS] 모델: {weights_path}")

    model = YOLO(str(weights_path))
    model.to(DEVICE)
    names = model.model.names  # 클래스 이름 dict

    fps_list = []
    for i in range(repeat):
        fps_i = measure_fps_single_run(model, names, video_path, num_frames)
        fps_list.append(fps_i)
        print(f"   - run {i+1}/{repeat}: {fps_i:.2f} FPS")

    fps_mean = float(np.mean(fps_list))
    fps_std = float(np.std(fps_list)) if len(fps_list) > 1 else 0.0
    print(f"  => 평균 PIPELINE FPS: {fps_mean:.2f} (± {fps_std:.2f})")

    return fps_mean

# ---------------------------------------------------
# 7. 전체 파이프라인: mAP + FPS + GFLOPs 수집 후 그래프 그리기
# ---------------------------------------------------

def main():
    results = {}

    for name, info in MODELS.items():
        exp_dir = info["exp_dir"]
        weights = info["weights"]

        print("\n====================================")
        print(f"=== 모델: {name} ===")

        # 1) mAP50-95 읽기
        map5095 = load_map_from_csv(exp_dir)
        print(f"  mAP50-95: {map5095:.4f}")

        # 2) GFLOPs / Params 계산
        gflops, mparams = get_model_flops_params(weights_path=weights, img_size=640)

        # 3) 순수 모델 FPS
        model_only_fps = measure_model_only_fps(
            weights_path=weights,
            img_size=640,
            num_iters=200 if DEVICE == "cuda" else 100  # CPU면 반복 수 살짝 줄이기
        )

        # 4) 실제 파이프라인 FPS (비디오 + ROI 포함)
        pipeline_fps = measure_pipeline_fps(
            weights_path=weights,
            video_path=VIDEO_PATH,
            num_frames=NUM_FRAMES_FOR_FPS,
            repeat=PIPELINE_FPS_REPEAT
        )

        results[name] = {
            "mAP50-95": map5095,
            "fps_model_only": model_only_fps,
            "fps_pipeline": pipeline_fps,
            "GFLOPs": gflops,
            "Params(M)": mparams,
        }

    # 요약 출력
    print("\n========== 요약 ==========")
    for name, stats in results.items():
        gflops_str = f"{stats['GFLOPs']:.2f}" if stats["GFLOPs"] is not None else "N/A"
        params_str = f"{stats['Params(M)']:.2f}" if stats["Params(M)"] is not None else "N/A"
        print(
            f"{name:24s} | mAP50-95 = {stats['mAP50-95']:.4f} | "
            f"GFLOPs = {gflops_str:>5s} | Params = {params_str:>5s} M | "
            f"MODEL FPS = {stats['fps_model_only']:.2f} | "
            f"PIPELINE FPS = {stats['fps_pipeline']:.2f}"
        )

    # 3) FPS vs mAP50-95 그래프 (파이프라인 기준)
    plt.figure(figsize=(7, 6))

    for name, stats in results.items():
        x = stats["fps_pipeline"]        # 전체 파이프라인 FPS
        y = stats["mAP50-95"]
        plt.scatter(x, y)
        plt.text(x + 0.1, y, name, fontsize=9)

    plt.xlabel("Pipeline FPS (video + ROI)")
    plt.ylabel("mAP50-95 (higher is better)")
    plt.title(f"Model Comparison on {DEVICE.upper()}: Pipeline FPS vs mAP50-95")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(SAVE_PATH, dpi=300)
    print(f"\n그래프 저장 완료: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
