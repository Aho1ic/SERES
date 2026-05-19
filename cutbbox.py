#!/usr/bin/env python3
"""
批量检测并裁剪检测框脚本（YOLOv8）
支持：
1) 图片文件夹推理裁剪
2) 视频流/视频文件推理裁剪
3) 仅裁剪指定标签
"""

from pathlib import Path
from typing import Dict, Optional, Set

import cv2
from ultralytics import YOLO

# ===================== 配置区（按需修改） =====================
MODEL_PATH = "/home/algorithm/yolo11s.pt"   # yolov8s 权重路径
INPUT_MODE = "image"                # "image" 或 "video"

# 图片模式配置
INPUT_DIR = "/home/algorithm/chongqing/aibox/jpg/"              # 输入图片文件夹
OUTPUT_DIR = "/home/algorithm/chongqing/aibox/personcrops/"       # 输出裁剪图片文件夹（也用于视频模式）

# 视频模式配置（可填本地视频路径、rtsp://、rtmp://、http:// 等）
VIDEO_SOURCE = ""
VIDEO_FRAME_STRIDE = 1              # 每隔N帧推理一次，1表示每帧都推理
VIDEO_MAX_FRAMES = 0                # 0表示不限制，>0表示最多处理多少帧

CONF_THRES = 0.2                   # 置信度阈值
DEVICE = "0"                        # "cpu" / "0"(第一张GPU)

# 选择要裁剪的标签：可填类别名(str)或类别ID(int)
# 例如：[] 表示全部类别，["person", 0] 表示只保留person
TARGET_LABELS = [0]
# ============================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_target_class_ids(model_names: Dict[int, str]) -> Optional[Set[int]]:
    if not TARGET_LABELS:
        return None

    name_to_id = {name: idx for idx, name in model_names.items()}
    target_ids: Set[int] = set()

    for item in TARGET_LABELS:
        if isinstance(item, int):
            target_ids.add(item)
            continue
        if isinstance(item, str):
            stripped = item.strip()
            if stripped.isdigit():
                target_ids.add(int(stripped))
                continue
            if stripped in name_to_id:
                target_ids.add(name_to_id[stripped])
                continue
            raise ValueError(f"TARGET_LABELS 中的类别名不存在: {item}")
        raise TypeError(f"TARGET_LABELS 仅支持 int 或 str，收到类型: {type(item)}")

    return target_ids


def save_crops_from_frame(
    img,
    boxes,
    output_dir: Path,
    file_stem: str,
    start_det_id: int,
    target_class_ids: Optional[Set[int]],
) -> int:
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return 0

    saved_count = 0
    for box in boxes:
        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        if target_class_ids is not None and cls_id not in target_class_ids:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
        det_id = start_det_id + saved_count
        save_name = (
            f"{file_stem}_det{det_id:03d}_cls{cls_id}_"
            f"conf{conf:.2f}_{x1}_{y1}_{x2}_{y2}.jpg"
        )
        cv2.imwrite(str(output_dir / save_name), crop)
        saved_count += 1

    return saved_count


def crop_from_images(model: YOLO, output_dir: Path, target_class_ids: Optional[Set[int]]) -> None:
    input_dir = Path(INPUT_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir.resolve()}")

    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    if not image_paths:
        print(f"输入目录中没有图片: {input_dir.resolve()}")
        return

    total_crops = 0
    total_images = len(image_paths)
    print(f"开始处理，共 {total_images} 张图片...")

    for idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{idx}/{total_images}] 读取失败，跳过: {img_path.name}")
            continue

        results = model.predict(
            source=img,
            conf=CONF_THRES,
            device=DEVICE,
            verbose=False,
        )

        image_crop_count = save_crops_from_frame(
            img=img,
            boxes=results[0].boxes,
            output_dir=output_dir,
            file_stem=img_path.stem,
            start_det_id=1,
            target_class_ids=target_class_ids,
        )
        if image_crop_count == 0:
            print(f"[{idx}/{total_images}] 无检测框: {img_path.name}")
            continue

        total_crops += image_crop_count

        print(f"[{idx}/{total_images}] {img_path.name} -> 裁剪 {image_crop_count} 个目标")

    print(f"处理完成，总共保存 {total_crops} 张裁剪图，输出目录: {output_dir.resolve()}")


def crop_from_video(model: YOLO, output_dir: Path, target_class_ids: Optional[Set[int]]) -> None:
    source = str(VIDEO_SOURCE)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    print(f"开始处理视频流: {source}")
    frame_idx = 0
    infer_count = 0
    total_crops = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if VIDEO_FRAME_STRIDE > 1 and (frame_idx - 1) % VIDEO_FRAME_STRIDE != 0:
            continue
        if VIDEO_MAX_FRAMES > 0 and infer_count >= VIDEO_MAX_FRAMES:
            break

        infer_count += 1
        results = model.predict(
            source=frame,
            conf=CONF_THRES,
            device=DEVICE,
            verbose=False,
        )

        frame_crop_count = save_crops_from_frame(
            img=frame,
            boxes=results[0].boxes,
            output_dir=output_dir,
            file_stem=f"frame_{frame_idx:06d}",
            start_det_id=1,
            target_class_ids=target_class_ids,
        )
        total_crops += frame_crop_count

        if infer_count % 20 == 0 or frame_crop_count > 0:
            print(
                f"帧 {frame_idx}（已推理 {infer_count} 帧）-> "
                f"裁剪 {frame_crop_count} 个目标，累计 {total_crops}"
            )

    cap.release()
    print(f"视频处理完成，总共保存 {total_crops} 张裁剪图，输出目录: {output_dir.resolve()}")


def crop_all_bboxes() -> None:
    model_path = Path(MODEL_PATH)
    output_dir = Path(OUTPUT_DIR)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    target_class_ids = build_target_class_ids(model.names)
    if target_class_ids is None:
        print("标签过滤: 未启用（保留全部类别）")
    else:
        print(f"标签过滤: 仅保留类别ID {sorted(target_class_ids)}")

    mode = INPUT_MODE.strip().lower()
    if mode == "image":
        crop_from_images(model=model, output_dir=output_dir, target_class_ids=target_class_ids)
    elif mode == "video":
        crop_from_video(model=model, output_dir=output_dir, target_class_ids=target_class_ids)
    else:
        raise ValueError(f"INPUT_MODE 仅支持 'image' 或 'video'，当前为: {INPUT_MODE}")


if __name__ == "__main__":
    crop_all_bboxes()
