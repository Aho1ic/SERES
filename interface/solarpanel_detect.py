import cv2
import json
import uuid
import time
import numpy as np
import requests
from pathlib import Path
from threading import Thread, Lock, Event
from ultralytics import YOLO
from loguru import logger
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ALLROOT = FILE.parents[1]
for module_path in (ALLROOT, ROOT):
    module_path_str = str(module_path)
    if module_path_str not in sys.path:
        sys.path.insert(0, module_path_str)

try:
    from interface.frame_timestamp import StreamFrameTimestampResolver
    from interface.utils import upload_files, delete_files
except ImportError:
    from frame_timestamp import StreamFrameTimestampResolver
    from utils import upload_files, delete_files

from config import get_auth_headers, SOLARPANEL_MODEL_PATH, get_upload_url, get_upload_dir

class SolarPanelInfer:
    def __init__(self, model_path, video_source, upload_url, upload_dir,
                 task_name=None, task_id=None, box_id=None, app_id=None,
                 video_url=None, type=1):
        self.running = Event()
        self.detection_thread = None
        self.model = None
        self.cap = None
        self.frame_count = 0
        self.frame_interval = 2
        self.last_save_time = 0
        self.last_processed_time = 0
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = Path(model_path)
        self.video_source = video_source
        self.upload_url = upload_url
        self.upload_lock = Lock()
        self.COLORS = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,255,0)]
        self.type = type
        self.frame_timestamp_resolver = StreamFrameTimestampResolver("solarpanel")

        # 新增参数
        self.task_name = task_name or "太阳能板高温点检测"
        self.task_id = task_id or "2"
        self.box_id = box_id or "unknown"
        self.app_id = app_id or "SolarPanel"
        self.video_url = video_url or ""

        # 从视频URL提取src_id
        if self.video_url and "/" in self.video_url:
            self.src_id = self.video_url.split("/")[-1]
        else:
            self.src_id = "unknown"

        logger.info(f"初始化参数: task_name={self.task_name}, task_id={self.task_id}, app_id={self.app_id}, src_id={self.src_id}")

    def upload_files(self, file_group):
        """使用公共工具模块上传文件"""
        return upload_files(file_group, self.upload_url)

    def _is_valid_file(self, file_path):
        """检查文件扩展名是否合法"""
        return file_path.is_file() and file_path.suffix.lower() in ('.json', '.jpeg', '.jpg')

    def run_detection(self):
        """太阳能板高温点检测主循环"""
        logger.info("太阳能板高温点检测启动...")
        try:
            logger.debug(f"加载模型: {self.model_path}")
            self.model = YOLO(str(self.model_path)).cuda()
            # 如果source是整数，视为摄像头ID
            if isinstance(self.video_source, int):
                logger.debug(f"连接摄像头ID: {self.video_source}")
                self.cap = cv2.VideoCapture(self.video_source)
            else:
                # 否则视为URL或文件路径
                logger.debug(f"连接视频流: {self.video_source}")
                self.cap = cv2.VideoCapture(self.video_source)

            self.frame_timestamp_resolver.reset()
            self.frame_count = 0
            self.last_save_time = 0
            self.last_processed_time = 0

            while self.running.is_set():
                success, frame = self.cap.read()
                if not success:
                    logger.warning("视频流中断，尝试重新连接...")
                    time.sleep(2)
                    if self.cap.isOpened():
                        self.cap.release()
                    # 重新连接视频源
                    if isinstance(self.video_source, int):
                        self.cap = cv2.VideoCapture(self.video_source)
                    else:
                        self.cap = cv2.VideoCapture(self.video_source)
                    self.frame_timestamp_resolver.reset()
                    continue

                # 优先使用视频流原始时间轴，后端不支持时回退为取帧时间
                frame_capture_time = time.time()
                frame_stream_timestamp = self.frame_timestamp_resolver.get_timestamp(
                    self.cap,
                    fallback_time=frame_capture_time
                )
                self.frame_count += 1
                if (self.frame_count - 1) % self.frame_interval != 0:
                    continue

                logger.debug(f"处理第 {self.frame_count} 帧")
                # 使用YOLOv8模型进行太阳能板检测
                results = self.model.predict(frame, conf=0.5, verbose=False, iou=0.45)

                # 检查是否有检测结果
                has_detection = False
                try:
                    if len(results) > 0 and hasattr(results[0], 'boxes'):
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            has_detection = True
                            logger.debug(f"检测到 {len(boxes)} 个太阳能板")
                except Exception as e:
                    logger.error(f"检测结果处理异常: {e}", exc_info=True)
                    continue

                # 检查是否有检测结果且距离上次保存已经超过10秒
                if has_detection and (current_time - self.last_save_time >= 10):
                    # 生成event_id，作为三个文件的统一文件名
                    event_id = str(uuid.uuid4())

                    # 准备文件路径
                    orig_path = self.upload_dir / f"{event_id}.jpg"
                    detected_path = self.upload_dir / f"{event_id}.jpeg"
                    json_path = self.upload_dir / f"{event_id}.json"

                    # 保存原始帧
                    cv2.imwrite(str(orig_path), frame)
                    logger.info(f"保存原始帧: {orig_path}")

                    # 创建JSON数据，使用test.json的格式
                    frame_timestamp = frame_stream_timestamp

                    details = []
                    targets = []

                    try:
                        frame_id = self.frame_count

                        # 处理所有检测到的太阳能板
                        target_id = 1
                        boxes = results[0].boxes
                        if boxes is not None:
                            for i in range(len(boxes)):
                                box = boxes[i]
                                if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                    confidence = float(box.conf)
                                    class_id = int(box.cls) if hasattr(box, 'cls') else 0

                                    target = {
                                        "angle": 0,
                                        "box": {
                                            "left_top_x": x1,
                                            "left_top_y": y1,
                                            "right_bottom_x": x2,
                                            "right_bottom_y": y2
                                        },
                                        "color": [255, 0, 0, 0],
                                        "cross_label": "",
                                        "id": target_id,
                                        "label": "solar_panel",
                                        "prob": round(confidence, 5),
                                        "moving": False,
                                        "ocr": "",
                                        "region_label": "",
                                        "roi_id": 0,
                                        "reserved": ""
                                    }
                                    targets.append(target)
                                    target_id += 1

                        if targets:
                            frame_detail = {
                                "frame_id": frame_id,
                                "metadata": {"max_lost_time": 3},
                                "model_id": "YOLO11",
                                "model_name": "solarpanel_v1",
                                "model_thres": 0.5,
                                "model_type": 1,
                                "targets": targets
                            }
                            details.append(frame_detail)

                    except Exception as e:
                        logger.error(f"创建检测详情异常: {e}", exc_info=True)

                    # 按照test.json的格式生成JSON数据
                    json_data = {
                        "event_id": event_id,
                        "event_state": 0,
                        "device_name": self.task_name,
                        "device_id": self.box_id,
                        "task_name": self.task_name,
                        "task_id": self.task_id,
                        "app_name": self.task_name,
                        "app_id": self.app_id,
                        "src_name": self.task_name,
                        "src_id": self.src_id,
                        "created": frame_timestamp,
                        "details": details
                    }

                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"保存JSON: {json_path}")

                    # 在检测结果上绘制边界框
                    detected_frame = frame.copy()
                    boxes = results[0].boxes

                    # 安全地遍历所有检测到的边界框
                    try:
                        if boxes is not None:
                            for i in range(len(boxes)):
                                box = boxes[i]
                                if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                    confidence = float(box.conf)
                                    class_id = int(box.cls) if hasattr(box, 'cls') else 0
                                    color = self.COLORS[class_id % len(self.COLORS)]

                                # 绘制边界框
                                cv2.rectangle(detected_frame, (x1, y1), (x2, y2), color, 2)

                                # 添加标签和置信度
                                label = f"Solar Panel: {confidence:.2f}"
                                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                y1 = max(y1, label_size[1])
                                cv2.rectangle(detected_frame, (x1, y1 - label_size[1] - baseline),
                                            (x1 + label_size[0], y1), color, -1)
                                cv2.putText(detected_frame, label, (x1, y1 - baseline),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    except Exception as e:
                        logger.error(f"绘制边界框异常: {e}", exc_info=True)

                    # 保存检测结果图像
                    cv2.imwrite(str(detected_path), detected_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    logger.info(f"保存检测结果: {detected_path}")

                    # 将三个文件作为一组上传
                    file_group = {
                        'json': json_path,
                        'original': orig_path,
                        'detected': detected_path
                    }

                    # 上传文件组
                    self.upload_files(file_group)

                    # 更新上次保存时间
                    self.last_save_time = current_time
                    logger.info(f"检测结果已保存，下次保存将在10秒后")

        except Exception as e:
            logger.error(f"检测过程中发生错误: {e}", exc_info=True)
        finally:
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'model') and self.model:
                del self.model
            logger.info("太阳能板高温点检测已停止")

    def upload_all_files(self):
        """上传所有未处理的文件（按组上传）"""
        logger.info("开始上传所有未处理文件...")

        # 获取所有文件并按文件名前缀分组
        file_groups = {}
        for file_path in self.upload_dir.iterdir():
            if self._is_valid_file(file_path):
                # 获取文件名前缀（不含扩展名）
                prefix = file_path.stem
                if prefix not in file_groups:
                    file_groups[prefix] = {}

                # 根据扩展名归类
                ext = file_path.suffix.lower()
                if ext == '.jpg':
                    file_groups[prefix]['original'] = file_path
                elif ext == '.jpeg':
                    file_groups[prefix]['detected'] = file_path
                elif ext == '.json':
                    file_groups[prefix]['json'] = file_path

        # 上传所有完整的文件组（包含三种文件）
        uploaded_count = 0
        for prefix, group in file_groups.items():
            if len(group) == 3:  # 只上传完整的文件组
                if self.upload_files(group):
                    uploaded_count += 1

        logger.info(f"已处理 {uploaded_count} 组文件")
        return uploaded_count

    def start(self):
        """启动检测线程"""
        logger.info("启动太阳能板高温点检测服务")
        if self.detection_thread is None or not self.detection_thread.is_alive():
            self.running.set()
            self.detection_thread = Thread(target=self.run_detection)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            logger.info("检测线程已启动")
            return True
        else:
            logger.warning("检测线程已在运行中")
            return False

    def stop(self):
        """停止检测线程"""
        logger.info("停止太阳能板高温点检测服务")
        if self.detection_thread and self.detection_thread.is_alive():
            self.running.clear()
            self.detection_thread.join(timeout=5)
            if self.cap and self.cap.isOpened():
                self.cap.release()
            logger.info("检测线程已停止")
            return True
        else:
            logger.warning("检测线程未在运行")
            return False

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='任务json文件路径')
    args = parser.parse_args()

    logger.info(f"读取任务文件: {args.task}")
    with open(args.task, 'r', encoding='utf-8') as f:
        task = json.load(f)

    model_path = Path(SOLARPANEL_MODEL_PATH)
    rtsp_url = task.get('video_url')
    box_id = task.get('boxId')
    upload_url = get_upload_url(sys_id=11, box_id=box_id, upload_type=1)
    json_task_id = task.get('task_id')
    task_name = task.get('task_name', '太阳能板高温点检测')
    json_app_id = task.get('categoryType', 'SolarPanel')

    # 从视频URL中提取src_id
    video_url = task.get('video_url', '')
    src_id = video_url.split('/')[-1] if video_url and '/' in video_url else 'unknown'

    upload_dir = get_upload_dir("solarpanel")
    logger.info(f"使用上传目录: {upload_dir}")

    logger.info(f"初始化太阳能板高温点检测服务，模型: {model_path}, 视频源: {rtsp_url}")
    logger.info(f"任务参数: task_name={task_name}, task_id={json_task_id}, app_id={json_app_id}, src_id={src_id}")

    infer = SolarPanelInfer(
        model_path=model_path,
        video_source=rtsp_url,
        upload_url=upload_url,
        upload_dir=upload_dir,
        task_name=task_name,
        task_id=json_task_id,
        box_id=box_id,
        app_id=json_app_id,
        video_url=video_url
    )

    infer.running.set()
    logger.info("开始运行太阳能板高温点检测")
    infer.run_detection()
