# -*- coding: UTF-8 -*-
import cv2
import json
import time
import numpy as np
from pathlib import Path
from threading import Thread, Event, Lock
from ultralytics import YOLO
from loguru import logger
import os
import requests
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ALLROOT = FILE.parents[1]
for module_path in (ALLROOT, ROOT):
    module_path_str = str(module_path)
    if module_path_str not in sys.path:
        sys.path.insert(0, module_path_str)

try:
    from interface.tracking_utils import assign_track_ids_by_iou, collect_tracked_objects, get_tracker_config
    from interface.async_verification import AsyncVerificationMixin
    from interface.frame_timestamp import StreamFrameTimestampResolver
    from interface.vllm_client import call_vllm_yes_no
    from interface.utils import upload_files, delete_files
except ImportError:
    from tracking_utils import assign_track_ids_by_iou, collect_tracked_objects, get_tracker_config
    from async_verification import AsyncVerificationMixin
    from frame_timestamp import StreamFrameTimestampResolver
    from vllm_client import call_vllm_yes_no
    from utils import upload_files, delete_files

from config import get_auth_headers, SMOKEFIRE_MODEL_PATH, get_upload_url, get_upload_dir

#nohup python interface/smokefire_detect.py --task task/20_287156217745940480.json > logs/smokefire_20.log 2>&1 &

class SmokefireInfer(AsyncVerificationMixin):
    def __init__(self, model_path, video_source, upload_url, upload_dir,
                 task_name=None, task_id=None, box_id=None, app_id=None,
                 video_url=None, conf_thres=0.5):
        self.running = Event()
        self.detection_thread = None
        self.model = None
        self.cap = None
        self.frame_count = 0
        self.frame_interval = 2
        self.model_path = Path(model_path)
        self.video_source = video_source
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.conf_thres = float(conf_thres)
        self.draw_colors = {
            'fire': (0, 0, 255),
            'smoke': (200, 200, 200)
        }
        self.color_cycle = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,255,0)]
        self.upload_lock = Lock()
        self.upload_url = upload_url
        self.tracker_config = get_tracker_config(ROOT)
        self.frame_timestamp_resolver = StreamFrameTimestampResolver("smokefire")

        self.task_name = task_name or "烟火检测"
        self.task_id = task_id or "smokefire"
        self.box_id = box_id or "unknown"
        self.app_id = app_id or "smokefire"
        self.video_url = video_url or ""
        self._init_async_verification(
            task_type=self.app_id,
            max_queue_size=int(os.getenv("VERIFICATION_QUEUE_MAXSIZE", "50")),
            report_queue_size=int(os.getenv("REPORT_QUEUE_MAXSIZE", "200")),
            verification_worker_count=int(os.getenv("AI_VERIFICATION_WORKERS", "1")),
            report_worker_count=int(os.getenv("REPORT_WORKERS", "1")),
            cache_ttl_seconds=float(os.getenv("TRACK_VERIFY_CACHE_TTL", "300")),
        )

        if self.video_url and "/" in self.video_url:
            video_suffix = self.video_url.split("/")[-1]
            if video_suffix.endswith("1581F8HGX253E00A04A7"):
                self.src_id = "13a8db1f8f8383e384efd56797c4dcd2"
                self.src_name = "2号门机场"
            elif video_suffix.endswith("1581F8HGX253S00A05LN"):
                self.src_id = "13a8db1f8f8383e384efd56797c4dcd7"
                self.src_name = "7号门机场"
            else:
                self.src_id = video_suffix
                self.src_name = self.task_name
        else:
            self.src_id = "unknown"
            self.src_name = self.task_name

        logger.info(f"初始化参数: task_name={self.task_name}, task_id={self.task_id}, app_id={self.app_id}, src_id={self.src_id}")
        logger.info(f"目标追踪配置: {self.tracker_config}")

    def _get_label_color(self, label: str, idx: int):
        if label in self.draw_colors:
            return self.draw_colors[label]
        return self.color_cycle[idx % len(self.color_cycle)]

    def analyze_bbox_with_ai(self, bbox_image_path):
        """
        直接调用本地vLLM服务对裁切的检测框图片进行AI分析
        返回True表示检测到烟雾或火焰，False表示未检测到
        """
        question = """请仔细观察图片的每个角落，判断是否有真实的烟雾或火焰（包括细小的轻烟和微弱的火苗）。

烟雾或火焰的特征（请特别注意细微迹象）：
1. 烟雾：灰色、白色或黑色的烟雾状物体，形状不规则，向上飘散
   - 包括：细小的轻烟、淡淡的烟雾、刚开始冒出的烟
   - 即使烟雾很淡、很小也要识别出来
2. 火焰：橙色、红色或黄色的火光，有燃烧特征
   - 包括：微弱的火苗、小火星、刚起的小火
   - 即使火焰很小、很弱也要识别出来

不是烟雾或火焰的情况：
- 夜晚的车灯和车尾灯
- 保安制服上红色的袖章
- 红色标识桶
- 红色灭火器
- 红色砖块
- 刹车灯的红光
- 地上的白斑不是烟
- 云雾、水汽、灰尘
- 暗处的左右2块红色的灯

请只回答yes或no：
- yes：确认有真实的烟雾或火焰（包括细小轻烟和微弱火苗）
- no：没有烟雾或火焰

答案："""
        confirmed = call_vllm_yes_no(bbox_image_path, question, logger=logger)
        if confirmed:
            logger.debug("AI确认检测到烟雾或火焰")
        else:
            logger.debug("AI确认未检测到烟雾或火焰")
        return confirmed

    def upload_files(self, file_group):
        """使用公共工具模块上传文件"""
        return upload_files(file_group, self.upload_url)

    def delete_files(self, file_group):
        """使用公共工具模块删除文件"""
        return delete_files(file_group)

    def run_detection(self):
        logger.info("烟火检测启动...")
        try:
            logger.debug(f"加载模型: {self.model_path}")
            self.model = YOLO(str(self.model_path)).cuda(device=1)

            if isinstance(self.video_source, int):
                logger.debug(f"连接摄像头ID: {self.video_source}")
                self.cap = cv2.VideoCapture(self.video_source)
            else:
                logger.debug(f"连接视频流: {self.video_source}")
                # 显式指定 FFmpeg 后端以提高兼容性
                self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)

            self.frame_timestamp_resolver.reset()
            self._start_verification_worker()
            self.frame_count = 0

            while self.running.is_set():
                success, frame = self.cap.read()
                if not success or frame is None:
                    logger.warning("视频流中断，尝试重新连接...")
                    time.sleep(2)
                    if self.cap and self.cap.isOpened():
                        self.cap.release()
                    if isinstance(self.video_source, int):
                        self.cap = cv2.VideoCapture(self.video_source)
                    else:
                        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
                    self.frame_timestamp_resolver.reset()
                    continue

                # 优先使用视频流原始时间轴，后端不支持时回退为取帧时间
                frame_capture_time = time.time()
                frame_stream_timestamp = self.frame_timestamp_resolver.get_timestamp(
                    self.cap,
                    fallback_time=frame_capture_time
                )

                # 修复 FFmpeg swscaler 1080p 内存对齐问题
                # 强制 Resize 处理：将高分辨率帧缩小，彻底消除 swscaler 的步长 Bug
                h, w = frame.shape[:2]
                if h >= 1080:
                    # 缩小到 720p 级别，确保尺寸是 32 的倍数（YOLO 喜欢的对齐方式）
                    # 1280x720 都是 32 的倍数
                    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                
                # 确保帧数据是连续内存
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)

                self.frame_count += 1
                if (self.frame_count - 1) % self.frame_interval != 0:
                    continue

                frame_process_start = time.perf_counter()
                logger.debug(f"处理第 {self.frame_count} 帧")

                # 使用 track() 同时进行检测和追踪
                tracked_objects = []
                results = None
                try:
                    track_results = self.model.track(
                        source=frame,
                        conf=self.conf_thres,
                        persist=True,
                        tracker=self.tracker_config,
                        verbose=False
                    )
                    results = track_results  # track 结果包含检测结果
                    tracked_objects = collect_tracked_objects(track_results)
                except Exception as e:
                    logger.error(f"目标追踪推理失败，尝试仅检测: {e}", exc_info=True)
                    # 如果追踪失败，回退到仅检测模式
                    results = self.model(frame, conf=self.conf_thres, verbose=False)


                detections = []
                try:
                    for result in results:
                        if result.boxes is None:
                            continue
                        xyxy = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
                        clss = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)
                        names = result.names if hasattr(result, 'names') and result.names else getattr(self.model, 'names', {})

                        for i in range(xyxy.shape[0]):
                            x1, y1, x2, y2 = xyxy[i].tolist()
                            conf = float(confs[i]) if i < len(confs) else 0.0
                            cls_id = int(clss[i]) if i < len(clss) else 0
                            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'conf': conf,
                                'cls_name': cls_name,
                                'cls_id': cls_id,
                                'track_id': None
                            })
                except Exception as e:
                    logger.error(f"检测结果处理异常: {e}", exc_info=True)
                    continue

                assign_track_ids_by_iou(detections, tracked_objects, bbox_key='bbox', id_key='track_id', class_key='cls_id')

                elapsed_ms = (time.perf_counter() - frame_process_start) * 1000
                self.monitor_metrics.record_yolo_frame(elapsed_ms, len(detections))
                self._log_monitor_metrics_if_needed()

                if not detections:
                    continue

                frame_timestamp = frame_stream_timestamp
                pending_detections, cached_detections, skipped = self._split_by_verification_cache(detections)
                if skipped:
                    logger.debug(f"烟火检测: {skipped} 个候选命中待验证/未通过缓存，跳过重复送检")

                if not pending_detections and not cached_detections:
                    logger.debug("烟火检测: 本帧候选均已被缓存拦截，无需上报")
                    continue

                self._enqueue_verification_event(
                    frame=frame,
                    frame_timestamp=frame_timestamp,
                    pending_detections=pending_detections,
                    cached_detections=cached_detections,
                    confidence_key='conf',
                )

        except Exception as e:
            logger.error(f"检测过程中发生错误: {e}", exc_info=True)
        finally:
            self.running.clear()
            self._stop_verification_worker()
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'model') and self.model:
                del self.model
            logger.info("烟火检测已停止")

    def _process_verification_event(self, event):
        frame = cv2.imread(str(event.orig_path))
        if frame is None:
            logger.error(f"无法读取原始帧，跳过烟火验证事件: {event.orig_path}")
            self._delete_event_files(event)
            return False

        confirmed_detections = list(event.cached_detections)
        if event.cached_detections:
            logger.info(f"烟火检测: {len(event.cached_detections)} 个检测框命中验证缓存，跳过二次验证")

        temp_dir = self.upload_dir / 'temp_bbox'
        temp_dir.mkdir(exist_ok=True)

        for idx, det in enumerate(event.detections):
            x1, y1, x2, y2 = det['bbox']
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))

            if x2 <= x1 or y2 <= y1:
                logger.warning(f"烟火检测: 检测框无效，跳过: {det}")
                self._cache_verification_result(det, False)
                continue

            bbox_img = frame[y1:y2, x1:x2]
            if bbox_img.size == 0:
                logger.warning(f"烟火检测: 检测框为空，跳过: {det}")
                self._cache_verification_result(det, False)
                continue

            bbox_path = temp_dir / f"{event.event_id}_bbox_{idx}.jpg"
            if cv2.imwrite(str(bbox_path), bbox_img):
                logger.info(f"保存烟火二次验证裁切图: {bbox_path}")
            else:
                logger.warning(f"保存烟火二次验证裁切图失败: {bbox_path}")
                self._cache_verification_result(det, False)
                continue

            start = time.perf_counter()
            ai_confirmed = self.analyze_bbox_with_ai(bbox_path)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.monitor_metrics.record_verification(elapsed_ms)

            det['bbox'] = [x1, y1, x2, y2]
            self._cache_verification_result(det, ai_confirmed)
            if ai_confirmed:
                confirmed_detections.append(det)
                logger.info(f"烟火检测: 第{idx + 1}个检测框AI验证通过")
            else:
                logger.info(f"烟火检测: 第{idx + 1}个检测框AI验证未通过")

        if not confirmed_detections:
            logger.info("烟火检测: 所有检测框都未通过AI验证，丢弃本次候选事件")
            self._delete_event_files(event)
            return False

        targets = []
        annotated = frame.copy()
        for idx, det in enumerate(confirmed_detections):
            x1, y1, x2, y2 = det['bbox']
            label = det['cls_name']
            conf = det['conf']
            track_id = det.get('track_id')
            report_id = int(track_id) if track_id is not None else None
            color = self._get_label_color(label, idx)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ID:{report_id} {conf:.2f}" if report_id is not None else f"{label}:{conf:.2f}"
            tsize, tbase = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(y1, tsize[1] + 2)
            cv2.rectangle(annotated, (x1, ty - tsize[1] - tbase), (x1 + tsize[0], ty), color, -1)
            cv2.putText(annotated, text, (x1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            targets.append({
                "angle": 0,
                "box": {
                    "left_top_x": int(x1),
                    "left_top_y": int(y1),
                    "right_bottom_x": int(x2),
                    "right_bottom_y": int(y2)
                },
                "color": [color[2], color[1], color[0], 0],
                "cross_label": "",
                "id": report_id,
                "label": label,
                "prob": round(float(conf), 5),
                "moving": False,
                "ocr": "",
                "region_label": "",
                "roi_id": 0,
                "tracking": 0 if track_id is None else 1
            })

        cv2.imwrite(str(event.detected_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        logger.info(f"保存检测结果: {event.detected_path}")

        json_data = {
            "event_id": event.event_id,
            "event_state": 0,
            "device_name": "重庆AI识别",
            "device_id": self.box_id,
            "task_name": self.task_name,
            "task_id": self.task_id,
            "app_name": self.task_name,
            "app_id": self.app_id,
            "src_name": self.src_name,
            "src_id": self.src_id,
            "created": event.frame_timestamp,
            "picNum": "2",
            "details": [
                {
                    "frame_id": event.frame_id,
                    "metadata": {},
                    "model_id": "YOLOv8",
                    "model_name": "smokefire_v1",
                    "model_thres": self.conf_thres,
                    "model_type": 1,
                    "targets": targets
                }
            ]
        }

        with open(event.json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"保存JSON: {event.json_path}")

        logger.info(f"烟火检测: 异步验证事件处理完成，等待上传线程处理: event_id={event.event_id}")
        return True

    def start(self):
        logger.info("启动烟火检测服务")
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
        logger.info("停止烟火检测服务")
        if self.detection_thread and self.detection_thread.is_alive():
            self.running.clear()
            self.detection_thread.join(timeout=5)
            self._stop_verification_worker()
            if self.cap and self.cap.isOpened():
                self.cap.release()
            logger.info("检测线程已停止")
            return True
        else:
            logger.warning("检测线程未在运行")
            return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='任务json文件路径')
    args = parser.parse_args()

    DEFAULT_CONF = 0.2
    logger.info(f"读取任务文件: {args.task}")
    with open(args.task, 'r', encoding='utf-8') as f:
        task = json.load(f)

    model_path = ALLROOT / 'weights/smokefire.pt'
    video_source = task.get('video_url')
    box_id = task.get('boxId')
    task_name = task.get('task_name', '烟火检测')
    json_task_id = task.get('task_id')
    json_app_id = task.get('categoryType', 'smokefire')
    video_url = task.get('video_url', '')

    try:
        _box_id_str = str(box_id) if box_id is not None else '20'
    except Exception:
        _box_id_str = '20'
    upload_url = get_upload_url(sys_id=11, box_id=_box_id_str, upload_type=1)

    upload_dir = get_upload_dir("smokefire")
    logger.info(f"使用上传目录: {upload_dir}")

    logger.info(f"初始化烟火检测服务，模型: {model_path}, 视频源: {video_source}")
    logger.info(f"任务参数: task_name={task_name}, task_id={json_task_id}, app_id={json_app_id}")

    infer = SmokefireInfer(
        model_path=model_path,
        video_source=video_source,
        upload_url=upload_url,
        upload_dir=upload_dir,
        task_name=task_name,
        task_id=json_task_id,
        box_id=box_id,
        app_id=json_app_id,
        video_url=video_url,
        conf_thres=DEFAULT_CONF
    )

    infer.running.set()
    logger.info("开始运行烟火检测")
    infer.run_detection()
