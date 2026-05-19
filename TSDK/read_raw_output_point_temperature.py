import os
import sys
import uuid
import json
import glob
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / 'weights' / 'solarpanel.pt'
TASK_DIR = PROJECT_ROOT / 'task'
UPLOAD_CEWEN_DIR = PROJECT_ROOT / 'upload' / 'cewen'


class DJIThermalProcessor:
    def __init__(self, sdk_path="/home/algorithm/chongqing/TSDK"):
        """
        初始化DJI热成像处理器
        :param sdk_path: DJI Thermal SDK的安装路径
        """
        self.sdk_path = Path(sdk_path)
        self.irp_path = self.sdk_path / "utility/bin/linux/release_x64/dji_irp"
        self.lib_path = self.sdk_path / "utility/bin/linux/release_x64"


        self._setup_environment()

    def _setup_environment(self):
        """设置必要的环境权限"""
        try:
            subprocess.run(["chmod", "+x", str(self.irp_path)], check=True)

            os.environ["LD_LIBRARY_PATH"] = f"/usr/local/lib:{self.lib_path}"
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"环境设置失败: {str(e)}")

    def generate_raw(self, input_image, output_dir="."):
        """
        从DJI热成像JPEG生成RAW文件
        :param input_image: 输入的热成像JPEG路径
        :param output_dir: 输出目录
        :return: 生成的RAW文件路径
        """
        input_file_name = Path(input_image).stem
        output_path = Path(output_dir) / f"{input_file_name}.raw"

        try:
            cmd = [
                str(self.irp_path),
                "-s", str(input_image),
                "-a", "measure",
                "-o", str(output_path)
            ]
            subprocess.run(cmd, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"RAW文件生成失败: {str(e)}")

    def process_directory(self, input_dir, output_dir=".", extensions=("jpeg", "jpg", "JPG")):
        """
        处理指定目录下所有热成像图片，生成RAW文件
        :param input_dir: 输入目录，包含热成像JPEG图片
        :param output_dir: 输出目录，存放生成的RAW文件
        :param extensions: 要处理的文件扩展名元组
        :return: 生成的RAW文件路径列表
        """
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        raw_files = []
        for ext in extensions:
            pattern = os.path.join(input_dir, f"*.{ext}")
            for input_file in glob.glob(pattern):
                print(f"处理文件: {input_file}")
                try:
                    raw_path = self.generate_raw(input_file, output_dir)
                    raw_files.append(raw_path)
                    print(f"生成RAW文件: {raw_path}")
                except Exception as e:
                    print(f"处理 {input_file} 时出错: {str(e)}")
        return raw_files

    @staticmethod
    def get_temperature_from_raw(raw_path, width, height, x, y):
        """
        从RAW文件中读取指定坐标的温度值
        :param raw_path: RAW文件路径
        :param width: 图像宽度(像素)
        :param height: 图像高度(像素)
        :param x: 目标点的x坐标(0-based)
        :param y: 目标点的y坐标(0-based)
        :return: 温度值(浮点数)
        """
        raw_path = Path(raw_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"RAW文件不存在: {raw_path}")
        if x < 0 or x >= width or y < 0 or y >= height:
            raise ValueError(f"坐标({x},{y})超出图像范围({width}x{height})")
        img_data = np.fromfile(raw_path, dtype='uint16')
        temperature_data = img_data.reshape(height, width) / 10.0
        return float(temperature_data[y, x])


def _load_yolo_model() -> YOLO:
    if YOLO is None:
        raise RuntimeError("未安装ultralytics，无法进行目标检测。请安装后重试。")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"模型权重不存在: {WEIGHTS_PATH}")
    return YOLO(str(WEIGHTS_PATH))


def _detect_centers(image_path: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]], object]:
    """使用YOLOv8检测并返回中心点与框坐标，保持与 algorithm_api 的中心点缩放一致。"""
    model = _load_yolo_model()
    results = model(str(image_path), conf=0.5)
    centers: List[Tuple[int, int]] = []
    boxes_xyxy: List[Tuple[int, int, int, int]] = []

    for r in results:
        if getattr(r, 'boxes', None) is None:
            continue
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4]
            # 与 algorithm_api 一致：中心点再除以2（x,y均）
            cx = (x1 + x2) / 2 / 2
            cy = (y1 + y2) / 2 / 2
            centers.append((int(cx), int(cy)))
            boxes_xyxy.append((int(x1), int(y1), int(x2), int(y2)))
    return centers, boxes_xyxy, results


def _read_threshold(box_id: str, task_id: str) -> float:
    task_file = TASK_DIR / f"{box_id}_{task_id}.json"
    if not task_file.exists():
        raise FileNotFoundError(f"任务文件不存在: {task_file}")
    with open(task_file, 'r', encoding='utf-8') as f:
        task_info = json.load(f)
    return float(task_info.get('extendFields', '-273'))


def _ensure_dirs():
    UPLOAD_CEWEN_DIR.mkdir(parents=True, exist_ok=True)


def _draw_and_save(image_path: Path, boxes: List[Tuple[int, int, int, int]], centers: List[Tuple[int, int]], temps: List[float], event_id: str) -> Tuple[Path, Path]:
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"读取图片失败: {image_path}")
    for (x1, y1, x2, y2), temp in zip(boxes, temps):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{temp:.1f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    result_jpeg = UPLOAD_CEWEN_DIR / f"{event_id}.jpeg"
    cv2.imwrite(str(result_jpeg), img)

    orig_jpg = UPLOAD_CEWEN_DIR / f"{event_id}.jpg"
    orig = cv2.imread(str(image_path))
    cv2.imwrite(str(orig_jpg), orig)
    return orig_jpg, result_jpeg


def _build_json(event_id: str, box_id: str, task_id: str, temps: List[float], boxes: List[Tuple[int, int, int, int]], third_group_id: str = "") -> dict:
    import time as _time
    targets = []
    for idx, ((x1, y1, x2, y2), temp) in enumerate(zip(boxes, temps)):
        targets.append({
            "angle": 0,
            "box": {
                "left_top_x": int(x1),
                "left_top_y": int(y1),
                "right_bottom_x": int(x2),
                "right_bottom_y": int(y2)
            },
            "color": [255, 0, 0, 0],
            "cross_label": "",
            "id": idx + 1,
            "label": "hotspot",
            "prob": 1.0,
            "moving": False,
            "ocr": "",
            "region_label": "",
            "roi_id": 0,
            "reserved": ""
        })

    json_data = {
        "event_id": event_id,
        "event_state": 0,
        "device_name": "重庆AI识别",
        "device_id": str(task_id),
        "task_name": "天线资产盘点",
        "task_id": str(task_id),
        "app_name": "天线资产盘点",
        "app_id": "ceWen",
        "src_name": "",
        "src_id": str(task_id),
        "created": int(_time.time()),
        "picNum": 2,
        "location": "",
        "temperature": ",".join([f"{t:.1f}" for t in temps]),
        "thirdGroupId": third_group_id,
        "details": [
            {
                "frame_id": 1,
                "metadata": {},
                "model_id": "YOLO11",
                "model_name": "cewen_v1",
                "model_thres": 0.5,
                "model_type": 1,
                "targets": targets
            }
        ]
    }
    return json_data


def process_image_pipeline(input_image: str, box_id: str, task_id: str, third_group_id: str = "", raw_width: int = 640, raw_height: int = 512) -> int:
    """
    完整流程：YOLO检测 -> 生成RAW -> 点测温 -> 阈值筛选 -> 结果保存
    返回码：0 正常；1 无检测/未过阈值（已清理）；>1 异常
    """
    image_path = Path(input_image)
    if not image_path.exists():
        print(f"输入图片不存在: {image_path}")
        return 2

    centers, boxes, _ = _detect_centers(str(image_path))
    if len(centers) == 0:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            pass
        print("未检测到目标，已删除图片")
        return 1

    processor = DJIThermalProcessor()
    raw_path = processor.generate_raw(str(image_path), str(image_path.parent))

    temps: List[float] = []
    for cx, cy in centers:
        t = processor.get_temperature_from_raw(raw_path, raw_width, raw_height, cx, cy)
        temps.append(t)

    try:
        threshold = _read_threshold(box_id, task_id)
    except Exception as e:
        print(f"读取阈值失败: {e}")
        threshold = -273.0

    valid = [(i, temp) for i, temp in enumerate(temps) if temp is not None and temp > threshold]
    if not valid:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            Path(raw_path).unlink(missing_ok=True)
        except Exception:
            pass
        print("所有点温度低于阈值，已删除图片与RAW")
        return 1

    _ensure_dirs()
    event_id = str(uuid.uuid4())
    selected_boxes = [boxes[i] for i, _ in valid]
    selected_temps = [temps[i] for i, _ in valid]
    orig_jpg, result_jpeg = _draw_and_save(image_path, selected_boxes, centers, selected_temps, event_id)
    json_obj = _build_json(event_id, box_id, task_id, selected_temps, selected_boxes, third_group_id)
    json_path = UPLOAD_CEWEN_DIR / f"{event_id}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=4)

    print(f"已保存: {json_path}, {orig_jpg}, {result_jpeg}")
    try:
        image_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        Path(raw_path).unlink(missing_ok=True)
    except Exception:
        pass

    return 0


def main():
    parser = argparse.ArgumentParser(description="DJI Thermal RAW 工具与完整处理管线")
    sub = parser.add_subparsers(dest='cmd')

    p_raw = sub.add_parser('raw', help='从JPEG生成RAW')
    p_raw.add_argument('--input', required=True, help='输入热成像JPEG路径')
    p_raw.add_argument('--output', required=True, help='输出RAW路径')

    p_temp = sub.add_parser('temp', help='从RAW读取指定点温度')
    p_temp.add_argument('--get_temp', action='store_true', help='启用点测温')
    p_temp.add_argument('x', type=int)
    p_temp.add_argument('y', type=int)
    p_temp.add_argument('raw_path', type=str)
    p_temp.add_argument('--width', type=int, default=640)
    p_temp.add_argument('--height', type=int, default=512)

    p_proc = sub.add_parser('process-image', help='完整流程：检测->RAW->点测温->阈值->保存')
    p_proc.add_argument('--input', required=True, help='输入热成像JPEG路径')
    p_proc.add_argument('--boxId', required=True, help='任务boxId')
    p_proc.add_argument('--task_id', required=True, help='任务task_id')
    p_proc.add_argument('--thirdGroupId', default='', help='thirdGroupId 可选')
    p_proc.add_argument('--raw-width', type=int, default=640)
    p_proc.add_argument('--raw-height', type=int, default=512)

    parser.add_argument('--input', help='兼容旧模式：输入JPEG')
    parser.add_argument('--output', help='兼容旧模式：输出RAW')
    parser.add_argument('--get_temp', action='store_true', help='兼容旧模式：点测温开关')

    args, unknown = parser.parse_known_args()
    if args.cmd is None and args.input and args.output and not args.get_temp:
        processor = DJIThermalProcessor()
        out_dir = Path(args.output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = processor.generate_raw(args.input, str(out_dir))
        return 0

    if args.cmd is None and args.get_temp:
        if len(unknown) >= 3:
            try:
                x = int(unknown[0])
                y = int(unknown[1])
                raw_path = unknown[2]
            except Exception:
                print("参数错误：--get_temp 需要 x y raw_path")
                return 2
        else:
            print("参数错误：--get_temp 需要 x y raw_path")
            return 2
        # 仅进行点测温时无需初始化 SDK 环境，直接使用静态方法
        temp = DJIThermalProcessor.get_temperature_from_raw(raw_path, width=640, height=512, x=x, y=y)
        print(temp)
        return 0

    if args.cmd == 'raw':
        processor = DJIThermalProcessor()
        out_dir = Path(args.output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        processor.generate_raw(args.input, str(out_dir))
        return 0
    elif args.cmd == 'temp':
        # 仅进行点测温时无需初始化 SDK 环境，直接使用静态方法
        temp = DJIThermalProcessor.get_temperature_from_raw(args.raw_path, width=args.width, height=args.height, x=args.x, y=args.y)
        print(temp)
        return 0
    elif args.cmd == 'process-image':
        return process_image_pipeline(args.input, args.boxId, args.task_id, args.thirdGroupId, args.raw_width, args.raw_height)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())