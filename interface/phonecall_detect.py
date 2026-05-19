import cv2
import json
import time
import numpy as np
import requests
from pathlib import Path
from threading import Thread, Lock, Event
from ultralytics import YOLO
from loguru import logger
import os
import re
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

from config import get_auth_headers, PHONECALL_MODEL_PATH, get_upload_url, get_upload_dir

class PhoneCallInfer(AsyncVerificationMixin):
    def __init__(self, model_path, video_source, upload_url, upload_dir, 
                 task_name=None, task_id=None, box_id=None, app_id=None, 
                 video_url=None, type=1):
        self.running = Event()
        self.detection_thread = None
        self.model = None
        self.track_model = None
        self.cap = None
        self.frame_count = 0
        self.frame_interval = 2
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = Path(model_path)
        self.video_source = video_source
        self.upload_url = upload_url
        self.upload_lock = Lock()
        self.COLORS = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,255,0)]
        self.type = type
        self.tracker_config = get_tracker_config(ROOT)
        self.frame_timestamp_resolver = StreamFrameTimestampResolver("phonecall")
        
        self.task_name = task_name or "打电话姿势检测"
        self.task_id = task_id or "4"
        self.box_id = box_id or "unknown"
        self.app_id = app_id or "phoneCall"
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

    def is_making_phone_call(self, keypoints):
        """
        判断是否为打电话姿势：
        1) 单手：手腕高于手肘（y坐标更小）
        2) 手腕距离同侧耳朵 < 1倍同侧肩膀到肘距离（如果耳朵不可用，则距离头部 < 1倍同侧肩膀到肘距离）
        3) 手腕-肘部与肩膀-肘部的夹角 < 80度
        4) 正在走路（两腿夹角 > 5度）
        
        keypoints: 关键点坐标，格式为[[x0, y0], [x1, y1], ..., [x16, y16]]
        COCO顺序：0鼻子 1左眼 2右眼 3左耳 4右耳 5左肩 6右肩 7左肘 8右肘 9左手腕 10右手腕
        """
        def valid_point(p):
            return p.shape == (2,) and not np.all(p == 0)
        
        # 获取关键点
        left_shoulder = np.array(keypoints[5])
        right_shoulder = np.array(keypoints[6])
        left_elbow = np.array(keypoints[7])
        right_elbow = np.array(keypoints[8])
        left_wrist = np.array(keypoints[9])
        right_wrist = np.array(keypoints[10])
        left_ear = np.array(keypoints[3])
        right_ear = np.array(keypoints[4])
        
        # 检查肩膀关键点是否有效，如果无效则估算
        if not (valid_point(left_shoulder) and valid_point(right_shoulder)):
            logger.debug('肩膀关键点无效，尝试使用估算点')
            # 尝试使用其他关键点估算肩膀位置
            # 如果有手肘和手腕，可以向上估算肩膀
            estimated_shoulders = []
            
            # 估算左肩
            if not valid_point(left_shoulder):
                if valid_point(left_elbow) and valid_point(left_wrist):
                    # 使用手肘向上延伸估算肩膀
                    elbow_wrist_vec = left_wrist - left_elbow
                    left_shoulder = left_elbow - elbow_wrist_vec * 0.8
                    logger.debug(f'估算左肩位置: ({left_shoulder[0]:.1f}, {left_shoulder[1]:.1f})')
                else:
                    logger.debug('无法估算左肩，跳过左手检测')
            
            # 估算右肩
            if not valid_point(right_shoulder):
                if valid_point(right_elbow) and valid_point(right_wrist):
                    # 使用手肘向上延伸估算肩膀
                    elbow_wrist_vec = right_wrist - right_elbow
                    right_shoulder = right_elbow - elbow_wrist_vec * 0.8
                    logger.debug(f'估算右肩位置: ({right_shoulder[0]:.1f}, {right_shoulder[1]:.1f})')
                else:
                    logger.debug('无法估算右肩，跳过右手检测')
            
            # 如果两个肩膀都无法获取或估算，则返回 False
            if not (valid_point(left_shoulder) or valid_point(right_shoulder)):
                logger.debug('左右肩膀都无法获取或估算，无法判断')
                return False
        
        # 获取头部参考点（用于耳朵缺失时的备选）
        head_candidates = [0, 1, 2, 3, 4]
        head_point = None
        for idx in head_candidates:
            pt = np.array(keypoints[idx])
            if valid_point(pt):
                head_point = pt
                break
        
        if head_point is None:
            # 使用肩膀中点上移作为头部
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            head_point = (left_shoulder + right_shoulder) / 2.0
            head_point[1] = max(head_point[1] - shoulder_width * 0.3, 0)
            logger.debug('头部关键点缺失，使用肩膀中点上移作为替代')
        
        # 检查是否在走路（暂时注释掉）
        is_walking = True  # 暂时默认为True，不进行走路判断
        # try:
        #     left_hip = np.array(keypoints[11])
        #     right_hip = np.array(keypoints[12])
        #     left_ankle = np.array(keypoints[15])
        #     right_ankle = np.array(keypoints[16])

        #     if all(map(valid_point, [left_hip, right_hip, left_ankle, right_ankle])):
        #         v_left_leg = left_ankle - left_hip
        #         v_right_leg = right_ankle - right_hip
        #         norms = np.linalg.norm(v_left_leg) * np.linalg.norm(v_right_leg)
        #         if norms > 0:
        #             cos_angle_legs = np.dot(v_left_leg, v_right_leg) / norms
        #             cos_angle_legs = np.clip(cos_angle_legs, -1.0, 1.0)
        #             legs_angle = np.degrees(np.arccos(cos_angle_legs))
        #             if legs_angle > 5:
        #                 is_walking = True
        #                 logger.debug(f'检测到走路姿势，两腿夹角: {legs_angle:.1f}度')
        # except Exception as e:
        #     logger.debug(f"腿部角度计算异常: {e}")

        y_diff_threshold = 1
        
        def check_arm_for_phone_call(wrist, elbow, shoulder, ear, side_name):
            """检查单侧手臂是否为打电话姿势"""
            if not all(map(valid_point, [wrist, elbow, shoulder])):
                return False
            
            # 计算同侧肩膀到肘的距离作为参考单位
            shoulder_to_elbow = np.linalg.norm(shoulder - elbow)
            if shoulder_to_elbow < 10:
                logger.debug(f'{side_name}手: 肩膀到肘距离过小({shoulder_to_elbow:.1f}px)，跳过')
                return False
            
            # 条件1：手腕高于手肘
            if wrist[1] >= elbow[1] - y_diff_threshold:
                return False
            
            # 条件2：手腕距离同侧耳朵很近（< 1.5倍肩膀到肘距离）
            if valid_point(ear):
                # 有耳朵：使用耳朵作为参考点
                dist_to_ear = np.linalg.norm(wrist - ear)
                threshold = shoulder_to_elbow * 1.5  # 距离耳朵 < 1.5倍肩膀到肘距离
                
                if dist_to_ear >= threshold:
                    logger.debug(f'{side_name}手: 距离耳朵{dist_to_ear:.1f}px >= {threshold:.1f}px (1.5倍肩膀到肘距离), 不是打电话')
                    return False
            else:
                # 无耳朵：使用头部参考点，阈值保持一致
                dist_to_head = np.linalg.norm(wrist - head_point)
                threshold = shoulder_to_elbow * 1.5  # 距离头部 < 1.5倍肩膀到肘距离
                
                if dist_to_head >= threshold:
                    logger.debug(f'{side_name}手: 耳朵缺失，距离头部{dist_to_head:.1f}px >= {threshold:.1f}px (1.5倍肩膀到肘距离), 不是打电话')
                    return False
            
            # 条件3：手腕-肘部与肩膀-肘部的夹角 < 80度
            wrist_elbow_vec = wrist - elbow
            shoulder_elbow_vec = shoulder - elbow
            norms = np.linalg.norm(wrist_elbow_vec) * np.linalg.norm(shoulder_elbow_vec)
            if norms > 0:
                cos_angle = np.dot(wrist_elbow_vec, shoulder_elbow_vec) / norms
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                
                if angle >= 80:
                    logger.debug(f'{side_name}手: 手臂角度{angle:.1f}度 >= 80度, 不是打电话')
                    return False
                
                logger.debug(f'{side_name}手: 角度{angle:.1f}度, 距离合适, 判断为打电话')
            
            # 条件4：必须在走路（暂时注释掉）
            # if not is_walking:
            #     logger.debug(f'{side_name}手: 姿势符合但未检测到走路，不判断为打电话')
            #     return False
            
            return True
        
        # 检查左右手
        left_is_phone_call = check_arm_for_phone_call(
            left_wrist, left_elbow, left_shoulder, left_ear, '左'
        )
        right_is_phone_call = check_arm_for_phone_call(
            right_wrist, right_elbow, right_shoulder, right_ear, '右'
        )
        
        # 互斥逻辑：如果双手都满足打电话条件，可能是拍照而不是打电话
        if left_is_phone_call and right_is_phone_call:
            logger.debug("双手都满足打电话条件，判断为拍照姿势而非打电话")
            return False
        
        # 只有单手满足条件才是打电话
        return left_is_phone_call or right_is_phone_call

    def analyze_image_with_ai(self, image_path, bbox=None):
        """
        调用phonecall_analysis.py对图片进行AI分析
        
        Args:
            image_path: 图片文件路径
            bbox: 可选的检测框 [x1, y1, x2, y2]，如果提供则裁切后再分析
            
        返回True表示检测到打电话姿势，False表示未检测到
        """
        try:
            # 如果提供了bbox，先裁切图片
            analysis_image_path = image_path
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                # 读取原图
                img = cv2.imread(str(image_path))
                if img is None:
                    logger.error(f"无法读取图片: {image_path}")
                    return False
                
                # 裁切检测框区域（添加一些边距）
                h, w = img.shape[:2]
                margin = 20  # 边距像素
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                cropped = img[y1:y2, x1:x2]
                
                # 等比放大裁切图片，使长边至少为1000像素
                crop_h, crop_w = cropped.shape[:2]
                max_side = max(crop_h, crop_w)
                
                if max_side < 1000:
                    scale_factor = 1000 / max_side
                    new_w = int(crop_w * scale_factor)
                    new_h = int(crop_h * scale_factor)
                    cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    logger.debug(f"裁切图片从 {crop_w}x{crop_h} 放大到 {new_w}x{new_h} (缩放比例: {scale_factor:.2f})")
                else:
                    logger.debug(f"裁切图片尺寸 {crop_w}x{crop_h} 已满足要求，无需放大")
                
                # 保存裁切后的图片
                crop_path = self.upload_dir / f"{Path(image_path).stem}_crop_{x1}_{y1}.jpg"
                cv2.imwrite(str(crop_path), cropped)
                analysis_image_path = crop_path
                logger.debug(f"裁切检测框区域: ({x1},{y1})-({x2},{y2}), 保存到: {crop_path}")
            
            question = """请仔细观察图片，判断是否有人正在打电话。

打电话的特征：
1. 必须手上拿着手机，这个条件严格限制
2. 人物正在走路
3. 一只手臂抬起，手部靠近耳朵位置（距离头部很近）

不是打电话的情况：
- 只是手臂抬起但没有靠近耳朵
- 在挠头、整理头发、遮阳、吃东西等动作
- 手机在手中但没有贴近耳朵（如看手机、拿着手机走路）
- 手部靠近头部但没有拿手机
- 手拿着东西吃饭
- 用手在抠鼻子
- 用手在揉眼睛
- 整理帽子
- 整理头发
- 手举起来伸懒腰
- 拿着手机走路
- 挠耳朵
- 手上没手机是在抽烟
- 人如果是坐着的/蹲着的一律不通过
- 如果两只手都抬起来,那么一定不是在打电话

请只回答yes或no：
- yes：确认有人正在打电话（手机贴近耳朵）
- no：没有人在打电话或不确定

答案："""
            confirmed = call_vllm_yes_no(analysis_image_path, question, logger=logger)
            if confirmed:
                logger.info("AI确认检测到打电话姿势")
            else:
                logger.info("AI确认未检测到打电话姿势")
            return confirmed
        except Exception as e:
            logger.error(f"AI分析异常: {e}")
            return False

    def upload_files(self, file_group):
        """使用公共工具模块上传文件"""
        return upload_files(file_group, self.upload_url)

    def delete_files(self, file_group):
        """使用公共工具模块删除文件"""
        return delete_files(file_group)

    def run_detection(self):
        """打电话姿势检测主循环"""
        logger.info("打电话姿势检测启动...")
        try:
            logger.debug(f"加载模型: {self.model_path}")
            self.model = YOLO(str(self.model_path)).cuda(device=1)
            self.track_model = YOLO(str(self.model_path)).cuda(device=1)
            if isinstance(self.video_source, int):
                logger.debug(f"连接摄像头ID: {self.video_source}")
                self.cap = cv2.VideoCapture(self.video_source)
            else:
                logger.debug(f"连接视频流: {self.video_source}")
                # 显式指定 FFmpeg 后端以提高兼容性，解决 swscaler 内存对齐问题
                self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
                # 设置后端缓冲区大小，减少延迟
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
            self.frame_timestamp_resolver.reset()
            self._start_verification_worker()
            self.frame_count = 0
            
            # 判断是否为视频文件（非实时流）
            def is_video_file(source):
                if isinstance(source, int):
                    return False  # 摄像头
                if isinstance(source, str):
                    # 检查是否为常见视频文件扩展名
                    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v')
                    if source.lower().endswith(video_extensions):
                        return True
                    # 检查是否为本地文件路径（非rtsp/http等流）
                    if not source.lower().startswith(('rtsp://', 'http://', 'https://', 'rtmp://')):
                        import os
                        if os.path.isfile(source):
                            return True
                return False
            
            is_file_source = is_video_file(self.video_source)
            if is_file_source:
                logger.info("检测到视频文件源，视频结束后将自动停止")
            
            while self.running.is_set():
                success, frame = self.cap.read()
                if not success:
                    if is_file_source:
                        # 视频文件结束，停止检测
                        logger.info("视频文件已播放完毕，停止检测")
                        self.running.clear()
                        break
                    else:
                        # 视频流中断，尝试重新连接
                        logger.warning("视频流中断，尝试重新连接...")
                        time.sleep(2)
                        if self.cap.isOpened():
                            self.cap.release()
                        if isinstance(self.video_source, int):
                            self.cap = cv2.VideoCapture(self.video_source)
                        else:
                            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.frame_timestamp_resolver.reset()
                        continue
                
                # 优先使用视频流原始时间轴，后端不支持时回退为取帧时间
                frame_capture_time = time.time()
                frame_stream_timestamp = self.frame_timestamp_resolver.get_timestamp(
                    self.cap,
                    fallback_time=frame_capture_time
                )
                
                # 修复 FFmpeg swscaler 1080p 内存对齐问题
                # 方案1：确保帧数据是连续内存（优先尝试，开销最小）
                h, w = frame.shape[:2]
                if h == 1080:
                    # 针对 1080p 视频，转换为连续内存以解决 swscaler 对齐问题
                    frame = np.ascontiguousarray(frame)
                    logger.debug(f"检测到1080p帧 ({w}x{h})，已转换为连续内存")
                
                # 方案2：如果仍然出现问题，降级到缩放处理（备选方案）
                # 如果上述方案无效，可以取消下面的注释启用缩放
                # if h >= 1080:
                #     # 缩小到 720p 级别，确保尺寸是 32 的倍数
                #     frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                #     logger.debug(f"高分辨率帧 ({w}x{h}) 已缩放至 720p (1280x720)")
                
                # 通用保障：确保所有帧都是连续内存
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                
                self.frame_count += 1
                if (self.frame_count - 1) % self.frame_interval != 0:
                    continue
                
                frame_process_start = time.perf_counter()
                # logger.debug(f"处理第 {self.frame_count} 帧")
                results = self.model(frame, conf=0.6, verbose=False)
                tracked_objects = []
                try:
                    track_results = self.track_model.track(
                        source=frame,
                        conf=0.6,
                        persist=True,
                        tracker=self.tracker_config,
                        verbose=False
                    )
                    tracked_objects = collect_tracked_objects(track_results)
                except Exception as e:
                    logger.error(f"目标追踪推理失败，本帧仍按检测结果上报，id使用本次上报序号: {e}", exc_info=True)
                
                has_phone_call_pose = False
                all_person_keypoints = []
                
                try:
                    for result in results:
                        if result.keypoints is not None and result.boxes is not None:
                            keypoints = result.keypoints.data.cpu().numpy()
                            boxes = result.boxes.xyxy.cpu().numpy()
                            
                            for person_idx, (person_keypoints, box) in enumerate(zip(keypoints, boxes)):
                                valid_keypoints = []
                                for j, kp in enumerate(person_keypoints):
                                    conf_thr = 0.3 if j in (0, 1, 2, 3, 4) else 0.4
                                    if kp[2] > conf_thr:
                                        valid_keypoints.append([kp[0], kp[1]])
                                    else:
                                        valid_keypoints.append([0, 0])
                                
                                all_person_keypoints.append(valid_keypoints)

                                if self.is_making_phone_call(valid_keypoints):
                                    logger.debug(f"人物 {person_idx+1}: 检测到打电话姿势！")
                                    has_phone_call_pose = True
                except Exception as e:
                    logger.error(f"检测结果处理异常: {e}", exc_info=True)
                    continue

                if not has_phone_call_pose:
                    elapsed_ms = (time.perf_counter() - frame_process_start) * 1000
                    self.monitor_metrics.record_yolo_frame(elapsed_ms, 0)
                    self._log_monitor_metrics_if_needed()
                    continue

                frame_timestamp = frame_stream_timestamp
                phone_call_detections = []

                try:
                    for result in results:
                        if result.keypoints is not None and result.boxes is not None:
                            keypoints = result.keypoints.data.cpu().numpy()
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confs = result.boxes.conf.cpu().numpy()

                            for person_idx, (person_keypoints, box, conf) in enumerate(zip(keypoints, boxes, confs)):
                                valid_keypoints = []
                                for j, kp in enumerate(person_keypoints):
                                    conf_thr = 0.3 if j in (0, 1, 2, 3, 4) else 0.4
                                    if kp[2] > conf_thr:
                                        valid_keypoints.append([kp[0], kp[1]])
                                    else:
                                        valid_keypoints.append([0, 0])

                                if self.is_making_phone_call(valid_keypoints):
                                    x1, y1, x2, y2 = map(int, box)
                                    phone_call_detections.append({
                                        'box': [x1, y1, x2, y2],
                                        'confidence': float(conf),
                                        'person_idx': person_idx,
                                        'track_id': None
                                    })
                except Exception as e:
                    logger.error(f"收集检测框异常: {e}", exc_info=True)
                    continue

                assign_track_ids_by_iou(phone_call_detections, tracked_objects, bbox_key='box', id_key='track_id')
                elapsed_ms = (time.perf_counter() - frame_process_start) * 1000
                self.monitor_metrics.record_yolo_frame(elapsed_ms, len(phone_call_detections))
                self._log_monitor_metrics_if_needed()

                if not phone_call_detections:
                    continue

                pending_detections, cached_detections, skipped = self._split_by_verification_cache(phone_call_detections)
                if skipped:
                    logger.debug(f"打电话姿势检测: {skipped} 个候选命中待验证/未通过缓存，跳过重复送检")

                if not pending_detections and not cached_detections:
                    logger.debug("打电话姿势检测: 本帧候选均已被缓存拦截，无需上报")
                    continue

                self._enqueue_verification_event(
                    frame=frame,
                    frame_timestamp=frame_timestamp,
                    pending_detections=pending_detections,
                    cached_detections=cached_detections,
                    confidence_key='confidence',
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
            if hasattr(self, 'track_model') and self.track_model:
                del self.track_model
            logger.info("打电话姿势检测已停止")

    def _process_verification_event(self, event):
        verified_detections = list(event.cached_detections)

        if event.cached_detections:
            logger.info(f"打电话姿势检测: {len(event.cached_detections)} 个检测框命中验证缓存，跳过二次验证")

        if event.detections:
            logger.info(f"打电话姿势检测: 开始异步AI二次确认 {len(event.detections)} 个检测框")

        for idx, detection in enumerate(event.detections):
            logger.info(f"打电话姿势检测: 正在验证检测框 {idx + 1}/{len(event.detections)}")
            start = time.perf_counter()
            ai_confirmed = self.analyze_image_with_ai(event.orig_path, bbox=detection['box'])
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.monitor_metrics.record_verification(elapsed_ms)

            self._cache_verification_result(detection, ai_confirmed)
            if ai_confirmed:
                logger.info(f"打电话姿势检测: 检测框 {idx + 1} AI确认通过")
                verified_detections.append(detection)
            else:
                logger.info(f"打电话姿势检测: 检测框 {idx + 1} AI确认未通过")

        if not verified_detections:
            logger.info("打电话姿势检测: 没有检测框通过AI验证，丢弃本次候选事件")
            self._delete_event_files(event)
            return False

        details = []
        targets = []
        for detection in verified_detections:
            x1, y1, x2, y2 = detection['box']
            confidence = detection['confidence']
            track_id = detection.get('track_id')
            report_id = int(track_id) if track_id is not None else None
            targets.append({
                "angle": 0,
                "box": {
                    "left_top_x": x1,
                    "left_top_y": y1,
                    "right_bottom_x": x2,
                    "right_bottom_y": y2
                },
                "color": [255, 0, 0, 0],
                "cross_label": "",
                "id": report_id,
                "label": "making_phone_call",
                "prob": round(confidence, 5),
                "moving": False,
                "ocr": "",
                "region_label": "",
                "roi_id": 0,
                "tracking": 0 if track_id is None else 1
            })

        details.append({
            "frame_id": event.frame_id,
            "metadata": {"max_lost_time": 3},
            "model_id": "YOLO11",
            "model_name": "phonecall_v1",
            "model_thres": 0.6,
            "model_type": 1,
            "targets": targets
        })

        json_data = {
            "event_id": event.event_id,
            "event_state": 0,
            "device_name": "重庆AI识别",
            "device_id": self.task_id,
            "task_name": self.task_name,
            "task_id": self.task_id,
            "app_name": self.task_name,
            "app_id": self.app_id,
            "src_name": self.src_name,
            "src_id": self.src_id,
            "created": event.frame_timestamp,
            "picNum": "2",
            "details": details
        }

        with open(event.json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"保存JSON: {event.json_path}")

        detected_frame = cv2.imread(str(event.orig_path))
        if detected_frame is None:
            logger.error(f"无法读取原始帧，跳过上报: {event.orig_path}")
            self._delete_event_files(event)
            return False

        for detection in verified_detections:
            x1, y1, x2, y2 = detection['box']
            track_id = detection.get('track_id')
            report_id = int(track_id) if track_id is not None else None
            color = (0, 0, 255)
            cv2.rectangle(detected_frame, (x1, y1), (x2, y2), color, 2)
            label = f"phoneCall ID:{report_id}" if report_id is not None else "phoneCall"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_label = max(y1, label_size[1])
            cv2.rectangle(
                detected_frame,
                (x1, y1_label - label_size[1] - baseline),
                (x1 + label_size[0], y1_label),
                color,
                -1
            )
            cv2.putText(
                detected_frame,
                label,
                (x1, y1_label - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imwrite(str(event.detected_path), detected_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        logger.info(f"保存检测结果: {event.detected_path}")

        logger.info(f"打电话姿势检测: 异步验证事件处理完成，等待上传线程处理: event_id={event.event_id}")
        return True

    def upload_all_files(self):
        logger.info("开始上传所有未处理文件...")
        file_groups = {}
        for file_path in self.upload_dir.iterdir():
            if self._is_valid_file(file_path):
                prefix = file_path.stem
                if prefix not in file_groups:
                    file_groups[prefix] = {}

                ext = file_path.suffix.lower()
                if ext == '.jpg':
                    file_groups[prefix]['original'] = file_path
                elif ext == '.jpeg':
                    file_groups[prefix]['detected'] = file_path
                elif ext == '.json':
                    file_groups[prefix]['json'] = file_path

        uploaded_count = 0
        for prefix, group in file_groups.items():
            if len(group) == 3:
                if self.upload_files(group):
                    uploaded_count += 1
        
        logger.info(f"已处理 {uploaded_count} 组文件")
        return uploaded_count

    def start(self):
        """启动检测线程"""
        logger.info("启动打电话姿势检测服务")
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
        logger.info("停止打电话姿势检测服务")
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
    
    logger.info(f"读取任务文件: {args.task}")
    with open(args.task, 'r', encoding='utf-8') as f:
        task = json.load(f)
    
    model_path = Path(PHONECALL_MODEL_PATH)
    rtsp_url = task.get('video_url')
    box_id = task.get('boxId')
    upload_url = get_upload_url(sys_id=11, box_id=box_id, upload_type=1)
    json_task_id = task.get('task_id')
    task_name = task.get('task_name', '打电话姿势检测')
    json_app_id = task.get('categoryType', 'phoneCall')

    video_url = task.get('video_url', '')
    if video_url and '/' in video_url:
        video_suffix = video_url.split('/')[-1]
        if video_suffix.endswith("1581F8HGX253E00A04A7"):
            src_id = "13a8db1f8f8383e384efd56797c4dcd2"
        elif video_suffix.endswith("1581F8HGX253S00A05LN"):
            src_id = "13a8db1f8f8383e384efd56797c4dcd7"
        else:
            src_id = video_suffix
    else:
        src_id = 'unknown'

    upload_dir = get_upload_dir("phonecall")
    logger.info(f"使用上传目录: {upload_dir}")
    
    logger.info(f"初始化打电话姿势检测服务，模型: {model_path}, 视频源: {rtsp_url}")
    logger.info(f"任务参数: task_name={task_name}, task_id={json_task_id}, app_id={json_app_id}, src_id={src_id}")
    
    infer = PhoneCallInfer(
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
    logger.info("开始运行打电话姿势检测")
    infer.run_detection()
