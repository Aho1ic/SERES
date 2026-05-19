#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path


FILE = Path(__file__).resolve()
ALLROOT = FILE.parents[1]
cv2 = None
np = None
YOLO = None

PROD_CONF = 0.5
LOW_CONF = 0.01
HEAD_KPT_CONF = 0.3
BODY_KPT_CONF = 0.4
Y_DIFF_THRESHOLD = 1
MIN_SHOULDER_TO_ELBOW = 10

KPT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def load_runtime_dependencies():
    global cv2, np, YOLO
    try:
        import cv2 as cv2_module
        import numpy as np_module
        from ultralytics import YOLO as yolo_class
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少运行依赖，请在部署环境或包含 cv2、numpy、ultralytics 的 Python 环境中执行"
        ) from exc

    cv2 = cv2_module
    np = np_module
    YOLO = yolo_class


def valid_point(point):
    point = np.asarray(point)
    return point.shape == (2,) and not np.all(point == 0)


def format_point(point):
    if not valid_point(point):
        return "missing"
    return f"({point[0]:.1f}, {point[1]:.1f})"


def build_valid_keypoints(person_keypoints):
    valid_keypoints = []
    keypoint_scores = []

    for idx, keypoint in enumerate(person_keypoints):
        if len(keypoint) >= 3:
            x, y, score = float(keypoint[0]), float(keypoint[1]), float(keypoint[2])
        else:
            x, y, score = float(keypoint[0]), float(keypoint[1]), 1.0

        threshold = HEAD_KPT_CONF if idx in (0, 1, 2, 3, 4) else BODY_KPT_CONF
        keypoint_scores.append(score)
        if score > threshold:
            valid_keypoints.append([x, y])
        else:
            valid_keypoints.append([0, 0])

    return valid_keypoints, keypoint_scores


def diagnose_arm(wrist, elbow, shoulder, side_name):
    missing = []
    if not valid_point(wrist):
        missing.append("wrist")
    if not valid_point(elbow):
        missing.append("elbow")
    if not valid_point(shoulder):
        missing.append("shoulder")

    if missing:
        return {
            "pass": False,
            "reason": f"{side_name}手关键点缺失: {', '.join(missing)}",
            "wrist": format_point(wrist),
            "elbow": format_point(elbow),
            "shoulder": format_point(shoulder),
        }

    shoulder_to_elbow = float(np.linalg.norm(shoulder - elbow))
    if shoulder_to_elbow < MIN_SHOULDER_TO_ELBOW:
        return {
            "pass": False,
            "reason": (
                f"{side_name}肩膀到肘距离过小: {shoulder_to_elbow:.1f}px "
                f"< {MIN_SHOULDER_TO_ELBOW}px"
            ),
            "shoulder_to_elbow": shoulder_to_elbow,
            "wrist": format_point(wrist),
            "elbow": format_point(elbow),
            "shoulder": format_point(shoulder),
        }

    if wrist[1] >= elbow[1] - Y_DIFF_THRESHOLD:
        return {
            "pass": False,
            "reason": (
                f"{side_name}手腕没有明显高于手肘: wrist_y={wrist[1]:.1f}, "
                f"elbow_y={elbow[1]:.1f}, 需要 wrist_y < elbow_y - {Y_DIFF_THRESHOLD}"
            ),
            "shoulder_to_elbow": shoulder_to_elbow,
            "wrist": format_point(wrist),
            "elbow": format_point(elbow),
            "shoulder": format_point(shoulder),
        }

    return {
        "pass": True,
        "reason": (
            f"{side_name}手通过: 手腕高于手肘, wrist_y={wrist[1]:.1f}, "
            f"elbow_y={elbow[1]:.1f}"
        ),
        "shoulder_to_elbow": shoulder_to_elbow,
        "wrist": format_point(wrist),
        "elbow": format_point(elbow),
        "shoulder": format_point(shoulder),
    }


def diagnose_photo_pose(valid_keypoints):
    points = [np.asarray(point, dtype=np.float32) for point in valid_keypoints]

    left_shoulder = points[5].copy()
    right_shoulder = points[6].copy()
    left_elbow = points[7].copy()
    right_elbow = points[8].copy()
    left_wrist = points[9].copy()
    right_wrist = points[10].copy()

    notes = []
    estimated = {"left_shoulder": False, "right_shoulder": False}

    if not (valid_point(left_shoulder) and valid_point(right_shoulder)):
        notes.append("肩膀关键点不完整，按生产逻辑尝试用同侧手肘和手腕估算肩膀")

        if not valid_point(left_shoulder):
            if valid_point(left_elbow) and valid_point(left_wrist):
                left_shoulder = left_elbow - (left_wrist - left_elbow) * 0.8
                estimated["left_shoulder"] = True
                notes.append(f"估算左肩: {format_point(left_shoulder)}")
            else:
                notes.append("左肩无法估算: 左肘或左腕缺失")

        if not valid_point(right_shoulder):
            if valid_point(right_elbow) and valid_point(right_wrist):
                right_shoulder = right_elbow - (right_wrist - right_elbow) * 0.8
                estimated["right_shoulder"] = True
                notes.append(f"估算右肩: {format_point(right_shoulder)}")
            else:
                notes.append("右肩无法估算: 右肘或右腕缺失")

        if not (valid_point(left_shoulder) or valid_point(right_shoulder)):
            return {
                "pass": False,
                "reason": "左右肩膀都缺失且无法估算，后处理无法判断拍照姿势",
                "left_arm": None,
                "right_arm": None,
                "notes": notes,
                "estimated": estimated,
            }

    left_arm = diagnose_arm(left_wrist, left_elbow, left_shoulder, "左")
    right_arm = diagnose_arm(right_wrist, right_elbow, right_shoulder, "右")
    passed = left_arm["pass"] or right_arm["pass"]

    if passed:
        reason = "至少一只手满足生产后处理: 手腕高于手肘"
    else:
        reason = "左右手都不满足生产后处理: 手腕高于手肘"

    return {
        "pass": passed,
        "reason": reason,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "notes": notes,
        "estimated": estimated,
    }


def collect_candidates(results, conf_threshold):
    candidates = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        keypoints = None
        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()

        for idx, box in enumerate(boxes):
            conf = float(confs[idx])
            candidate = {
                "index": len(candidates) + 1,
                "bbox": [int(v) for v in box.tolist()],
                "conf": conf,
                "conf_ok": conf >= conf_threshold,
                "pose_pass": False,
                "pose": None,
                "keypoint_scores": {},
            }

            if keypoints is None or idx >= len(keypoints):
                candidate["pose"] = {
                    "pass": False,
                    "reason": "模型结果没有关键点，无法进入拍照姿势后处理",
                    "left_arm": None,
                    "right_arm": None,
                    "notes": [],
                    "estimated": {},
                }
            else:
                valid_keypoints, scores = build_valid_keypoints(keypoints[idx])
                candidate["keypoint_scores"] = {
                    KPT_NAMES[kpt_idx]: round(float(score), 4)
                    for kpt_idx, score in enumerate(scores[: len(KPT_NAMES)])
                }
                candidate["pose"] = diagnose_photo_pose(valid_keypoints)
                candidate["pose_pass"] = bool(candidate["pose"]["pass"])

            candidates.append(candidate)

    candidates.sort(key=lambda item: item["conf"], reverse=True)
    for new_idx, candidate in enumerate(candidates, start=1):
        candidate["rank"] = new_idx
    return candidates


def build_summary(candidates, conf_threshold):
    if not candidates:
        return (
            "低阈值下也没有任何检测框。原因更可能是 YOLO 模型本身未检出目标，"
            "不是当前 conf 阈值或后处理逻辑导致。"
        )

    detected = [item for item in candidates if item["conf_ok"] and item["pose_pass"]]
    if detected:
        return (
            "按当前 YOLO conf 和拍照后处理逻辑，这张图应该能通过。"
            "如果线上没有上报，建议继续排查取帧是否取到这张图、大模型二次验证、上传接口或模型版本。"
        )

    low_conf_pose_pass = [item for item in candidates if not item["conf_ok"] and item["pose_pass"]]
    high_conf_pose_fail = [item for item in candidates if item["conf_ok"] and not item["pose_pass"]]

    if low_conf_pose_pass and not high_conf_pose_fail:
        best = low_conf_pose_pass[0]
        return (
            f"主要原因是 YOLO 置信度不够。候选框姿态后处理通过，"
            f"但最高相关 conf={best['conf']:.4f} < 当前阈值 {conf_threshold:.2f}。"
        )

    if high_conf_pose_fail and not low_conf_pose_pass:
        return "主要原因是后处理逻辑未通过。存在 conf 达标的候选框，但拍照姿势规则没有通过。"

    if low_conf_pose_pass and high_conf_pose_fail:
        return (
            "同时存在两类问题：有候选框姿态通过但 conf 不够，也有 conf 达标但后处理未通过。"
            "需要结合候选框明细确认目标人对应哪一个框。"
        )

    max_conf = max(item["conf"] for item in candidates)
    return (
        f"置信度和后处理都未通过。最高 conf={max_conf:.4f}，"
        f"当前阈值 {conf_threshold:.2f}。"
    )


def draw_visualization(image, candidates, output_path):
    canvas = image.copy()
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["bbox"]
        if candidate["conf_ok"] and candidate["pose_pass"]:
            color = (0, 200, 0)
            status = "PASS"
        elif candidate["pose_pass"]:
            color = (0, 200, 255)
            status = "LOW_CONF"
        elif candidate["conf_ok"]:
            color = (0, 0, 255)
            status = "POST_FAIL"
        else:
            color = (160, 160, 160)
            status = "FAIL"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"#{candidate['rank']} {status} conf={candidate['conf']:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.imwrite(str(output_path), canvas)


def print_human_report(image_path, model_path, conf_threshold, low_conf, candidates, summary):
    print("=" * 80)
    print("SpyCam 图片识别诊断")
    print("=" * 80)
    print(f"图片: {image_path}")
    print(f"模型: {model_path}")
    print(f"生产 conf 阈值: {conf_threshold}")
    print(f"诊断 low_conf 阈值: {low_conf}")
    print(f"结论: {summary}")
    print("-" * 80)

    if not candidates:
        return

    for candidate in candidates:
        conf_status = "达标" if candidate["conf_ok"] else "不够"
        pose_status = "通过" if candidate["pose_pass"] else "不通过"
        print(
            f"[候选 #{candidate['rank']}] conf={candidate['conf']:.4f} ({conf_status}), "
            f"bbox={candidate['bbox']}, 后处理={pose_status}"
        )

        pose = candidate["pose"]
        print(f"  后处理原因: {pose['reason']}")

        for note in pose.get("notes", []):
            print(f"  备注: {note}")

        left_arm = pose.get("left_arm")
        right_arm = pose.get("right_arm")
        if left_arm:
            print(f"  左手: {'通过' if left_arm['pass'] else '失败'} - {left_arm['reason']}")
            print(
                f"    shoulder={left_arm['shoulder']}, elbow={left_arm['elbow']}, "
                f"wrist={left_arm['wrist']}"
            )
        if right_arm:
            print(f"  右手: {'通过' if right_arm['pass'] else '失败'} - {right_arm['reason']}")
            print(
                f"    shoulder={right_arm['shoulder']}, elbow={right_arm['elbow']}, "
                f"wrist={right_arm['wrist']}"
            )

        useful_scores = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        ]
        score_text = ", ".join(
            f"{name}={candidate['keypoint_scores'].get(name, 0):.3f}"
            for name in useful_scores
        )
        print(f"  关键点置信度: {score_text}")
        print("-" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="诊断 spycam 单张图片为什么没有识别为拍照姿势")
    parser.add_argument("image", type=str, help="待诊断图片路径")
    parser.add_argument(
        "--model",
        type=str,
        default=str(ALLROOT / "weights" / "spycam.pt"),
        help="spycam YOLO 模型路径",
    )
    parser.add_argument("--conf", type=float, default=PROD_CONF, help="生产 YOLO 置信度阈值")
    parser.add_argument(
        "--low-conf",
        type=float,
        default=LOW_CONF,
        help="诊断时使用的低置信度阈值，用于判断是否只是 conf 不够",
    )
    parser.add_argument("--device", type=str, default=None, help="推理设备，例如 0、cpu、cuda:0")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO 输入尺寸，不填则使用模型默认")
    parser.add_argument("--save-vis", type=str, default=None, help="保存可视化诊断图片")
    parser.add_argument("--json", action="store_true", help="额外输出 JSON 诊断结果")
    return parser.parse_args()


def main():
    args = parse_args()
    load_runtime_dependencies()

    image_path = Path(args.image).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if args.low_conf > args.conf:
        raise ValueError("--low-conf 应该小于或等于 --conf")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    model = YOLO(str(model_path))
    predict_kwargs = {"conf": args.low_conf, "verbose": False}
    if args.device is not None:
        predict_kwargs["device"] = args.device
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz

    results = model(str(image_path), **predict_kwargs)
    candidates = collect_candidates(results, args.conf)
    summary = build_summary(candidates, args.conf)

    if args.save_vis:
        vis_path = Path(args.save_vis).expanduser().resolve()
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        draw_visualization(image, candidates, vis_path)
        print(f"可视化图片已保存: {vis_path}")

    print_human_report(image_path, model_path, args.conf, args.low_conf, candidates, summary)

    if args.json:
        print(
            json.dumps(
                {
                    "image": str(image_path),
                    "model": str(model_path),
                    "conf": args.conf,
                    "low_conf": args.low_conf,
                    "summary": summary,
                    "candidates": candidates,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
