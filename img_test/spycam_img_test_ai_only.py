#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拍照姿势检测 - 图片/视频批量推理脚本（仅AI验证版本）
YOLO识别完一张图片后，立即等待大模型二次验证完成，根据验证结果保留或删除
python spycam_img_test_ai_only.py --input /home/algorithm/chongqing/赛力斯测试视频合集 --output /home/algorithm/chongqing/aibox/0316综合测试/spycam_qwen/ --model /home/algorithm/chongqing/weights/spycam.pt --mode video --frame-interval 10
python spycam_img_test_ai_only.py --input /path/to/images --output /path/to/output --model /path/to/model.pt
"""

import cv2
import json
import uuid
import time
import numpy as np
import sys
import subprocess
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import argparse

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ALLROOT = FILE.parents[1]


def is_taking_photo(keypoints):
    """
    判断是否为拍照姿势（简化版）
    只要手腕高于手肘即可
    
    keypoints: 关键点坐标，格式为[[x0, y0], [x1, y1], ..., [x16, y16]]
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
    
    # 检查肩膀关键点是否有效
    if not (valid_point(left_shoulder) and valid_point(right_shoulder)):
        # 尝试估算肩膀位置
        if not valid_point(left_shoulder):
            if valid_point(left_elbow) and valid_point(left_wrist):
                elbow_wrist_vec = left_wrist - left_elbow
                left_shoulder = left_elbow - elbow_wrist_vec * 0.8
        
        if not valid_point(right_shoulder):
            if valid_point(right_elbow) and valid_point(right_wrist):
                elbow_wrist_vec = right_wrist - right_elbow
                right_shoulder = right_elbow - elbow_wrist_vec * 0.8
        
        if not (valid_point(left_shoulder) or valid_point(right_shoulder)):
            return False
    
    y_diff_threshold = 1
    
    def check_arm_for_photo(wrist, elbow, shoulder, side_name):
        """检查单侧手臂是否为拍照姿势"""
        if not all(map(valid_point, [wrist, elbow, shoulder])):
            return False
        
        shoulder_to_elbow = np.linalg.norm(shoulder - elbow)
        if shoulder_to_elbow < 10:
            return False
        
        # 条件：手腕高于手肘
        if wrist[1] >= elbow[1] - y_diff_threshold:
            return False
        
        return True
    
    # 检查左右手
    left_is_photo = check_arm_for_photo(left_wrist, left_elbow, left_shoulder, '左')
    right_is_photo = check_arm_for_photo(right_wrist, right_elbow, right_shoulder, '右')
    
    # 单手或双手满足条件都算拍照
    return left_is_photo or right_is_photo


def analyze_image_with_ai(image_path, bbox, temp_dir):
    """
    调用spycam_analysis.py对图片进行AI分析
    
    Args:
        image_path: 图片文件路径
        bbox: 检测框 [x1, y1, x2, y2]
        temp_dir: 临时目录（用于存放裁切图片）
        
    返回True表示检测到拍照姿势，False表示未检测到
    """
    crop_path = None
    try:
        x1, y1, x2, y2 = map(int, bbox)
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"无法读取图片: {image_path}")
            return False
        
        # 裁切检测框区域（添加边距）
        h, w = img.shape[:2]
        margin = 20
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
        
        # 保存裁切后的图片到临时目录
        crop_path = temp_dir / f"{Path(image_path).stem}_crop_{x1}_{y1}_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(crop_path), cropped)
        
        # 调用AI分析脚本
        image_analysis_path = ALLROOT / 'omni' / 'python' / 'spycam_analysis.py'
        if not image_analysis_path.exists():
            logger.error(f"spycam_analysis.py文件不存在: {image_analysis_path}")
            return False
        
        cmd = [sys.executable, str(image_analysis_path), str(crop_path)]
        logger.info(f"执行AI分析: {crop_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.error(f"AI分析失败: {result.stderr}")
            return False
        
        output = result.stdout.strip()
        logger.info(f"AI分析完整输出:\n{output}")
        
        # 解析输出 - 适配新的简化输出格式
        output_lower = output.lower().strip()
        
        # 查找最终答案
        lines = output.split('\n')
        for line in lines:
            line = line.strip().lower()
            # 查找包含答案的行
            if line.startswith('答案：') or line.startswith('答案:'):
                answer = line.split('：')[-1].split(':')[-1].strip()
                if 'yes' in answer:
                    logger.info("AI确认检测到拍照姿势")
                    return True
                elif 'no' in answer:
                    logger.info("AI确认未检测到拍照姿势")
                    return False
        
        # 如果没有找到标准格式，尝试在整个输出中查找
        if "[{'text': 'yes'}]" in output or "{'text': 'yes'}" in output:
            logger.info("AI确认检测到拍照姿势")
            return True
        elif "[{'text': 'no'}]" in output or "{'text': 'no'}" in output:
            logger.info("AI确认未检测到拍照姿势")
            return False
        
        # 最后尝试简单的yes/no匹配
        if "yes" in output_lower and "no" not in output_lower:
            logger.info("AI确认检测到拍照姿势")
            return True
        elif "no" in output_lower and "yes" not in output_lower:
            logger.info("AI确认未检测到拍照姿势")
            return False
        else:
            logger.warning(f"AI分析结果不明确，默认为否: {output}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("AI分析超时")
        return False
    except Exception as e:
        logger.error(f"AI分析异常: {e}")
        return False
    finally:
        # 无论验证结果如何，都删除临时裁切图片
        if crop_path and crop_path.exists():
            try:
                crop_path.unlink()
                logger.debug(f"已删除临时裁切图片: {crop_path.name}")
            except Exception as e:
                logger.warning(f"删除临时裁切图片失败: {e}")


def process_frame(frame, frame_info, model, output_dir, temp_dir, rejected_dir):
    """
    处理单帧图像：YOLO检测 -> AI验证 -> 根据结果保存到不同目录
    完全同步处理，确保大模型完成验证后再返回
    通过AI验证的保存到主目录，未通过的保存到rejected目录
    
    Args:
        frame: 图像帧（numpy数组）
        frame_info: 帧信息字典，包含 {'source': 'image/video', 'name': '文件名', 'frame_id': 帧号}
        model: YOLO模型
        output_dir: 输出目录（AI确认为拍照的图片）
        temp_dir: 临时目录（用于AI验证时的裁切图片）
        rejected_dir: 拒绝目录（AI确认不是拍照的图片）
    """
    try:
        source_type = frame_info.get('source', 'image')
        source_name = frame_info.get('name', 'unknown')
        frame_id = frame_info.get('frame_id', 0)
        
        if source_type == 'video':
            logger.info(f"开始处理视频帧: {source_name} - 第{frame_id}帧")
        else:
            logger.info(f"开始处理图片: {source_name}")
        
        if frame is None:
            logger.error(f"无效的图像帧")
            return
        
        # YOLO检测
        logger.info(f"YOLO检测中...")
        results = model(frame, conf=0.5)
        
        # 收集所有检测到拍照姿势的检测框
        photo_detections = []
        
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
                    
                    if is_taking_photo(valid_keypoints):
                        x1, y1, x2, y2 = map(int, box)
                        photo_detections.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'person_idx': person_idx
                        })
                        logger.info(f"YOLO检测到拍照姿势: 人物{person_idx+1}, 置信度{conf:.2f}")
        
        if not photo_detections:
            logger.info(f"未检测到拍照姿势，跳过")
            return
        
        logger.info(f"YOLO检测到 {len(photo_detections)} 个拍照姿势")
        
        # AI验证（同步逐个验证）
        verified_detections = []
        rejected_detections = []
        
        for idx, detection in enumerate(photo_detections):
            logger.info(f"正在验证检测框 {idx+1}/{len(photo_detections)}...")
            
            # 需要先保存当前帧为临时图片用于分析
            temp_frame_path = temp_dir / f"temp_frame_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(str(temp_frame_path), frame)
            
            try:
                # 进行AI验证
                ai_result = analyze_image_with_ai(temp_frame_path, detection['box'], temp_dir)
                
                logger.info(f"检测框 {idx+1} - AI: {'✓' if ai_result else '✗'}")
                
                # 根据AI验证结果决定保存到哪个目录
                if ai_result:
                    logger.info(f"✓ 检测框 {idx+1} AI确认为拍照，保存到主目录")
                    detection['ai_confirmed'] = True
                    verified_detections.append(detection)
                else:
                    logger.info(f"✗ 检测框 {idx+1} AI未确认，保存到rejected目录")
                    detection['ai_confirmed'] = False
                    rejected_detections.append(detection)
                    
            finally:
                # 删除临时帧图片
                if temp_frame_path.exists():
                    temp_frame_path.unlink()
        
        # 保存AI验证通过的检测结果到主目录
        if verified_detections:
            logger.info(f"共有 {len(verified_detections)} 个检测框通过AI验证，保存到主目录")
            save_detection_results(frame, frame_info, verified_detections, output_dir, "verified")
        
        # 保存AI验证未通过的检测结果到rejected目录
        if rejected_detections:
            logger.info(f"共有 {len(rejected_detections)} 个检测框未通过AI验证，保存到rejected目录")
            save_detection_results(frame, frame_info, rejected_detections, rejected_dir, "rejected")
        
        if not verified_detections and not rejected_detections:
            logger.info(f"没有检测框需要保存")
            return
        
        if source_type == 'video':
            logger.info(f"✓ 视频帧处理完成: {source_name} - 第{frame_id}帧")
        else:
            logger.info(f"✓ 图片处理完成: {source_name}")
        
    except Exception as e:
        logger.error(f"处理帧时发生错误: {frame_info} - {e}", exc_info=True)


def save_detection_results(frame, frame_info, detections, output_dir, result_type):
    """
    保存检测结果
    
    Args:
        frame: 图像帧
        frame_info: 帧信息
        detections: 检测结果列表
        output_dir: 输出目录
        result_type: 结果类型 ("verified" 或 "rejected")
    """
    try:
        source_type = frame_info.get('source', 'image')
        source_name = frame_info.get('name', 'unknown')
        frame_id = frame_info.get('frame_id', 0)
        
        # 生成唯一ID
        event_id = str(uuid.uuid4())
        
        # 绘制检测框
        detected_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            
            # 根据结果类型选择颜色和标签
            if result_type == "verified":
                color = (0, 255, 0)  # 绿色：AI确认为拍照
                label = "takingPhoto(AI_verified)"
            else:  # rejected
                color = (0, 0, 255)  # 红色：AI确认不是拍照
                label = "notTakingPhoto(AI_rejected)"
                
            cv2.rectangle(detected_frame, (x1, y1), (x2, y2), color, 2)
            
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_label = max(y1, label_size[1])
            cv2.rectangle(detected_frame, (x1, y1_label - label_size[1] - baseline), 
                        (x1 + label_size[0], y1_label), color, -1)
            cv2.putText(detected_frame, label, (x1, y1_label - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 保存检测结果图片
        if source_type == 'video':
            # 视频帧：包含视频名和帧号
            video_stem = Path(source_name).stem
            detected_path = output_dir / f"{video_stem}_frame{frame_id:06d}_{event_id}.jpeg"
        else:
            # 图片：使用事件ID
            detected_path = output_dir / f"{event_id}.jpeg"
            
        cv2.imwrite(str(detected_path), detected_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        logger.info(f"保存检测结果到{result_type}目录: {detected_path.name}")
        
    except Exception as e:
        logger.error(f"保存检测结果时发生错误: {e}", exc_info=True)

def process_single_image(image_path, model, output_dir, temp_dir, rejected_dir):
    """
    处理单张图片：YOLO检测 -> AI验证 -> 保存结果到不同目录
    完全同步处理，确保大模型完成验证后再返回
    通过AI验证的保存到主目录，未通过的保存到rejected目录
    
    Args:
        image_path: 图片路径
        model: YOLO模型
        output_dir: 输出目录（AI确认为拍照的图片）
        temp_dir: 临时目录（用于AI验证时的裁切图片）
        rejected_dir: 拒绝目录（AI确认不是拍照的图片）
    """
    try:
        # 读取图片
        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.error(f"无法读取图片: {image_path}")
            return
        
        # 构建帧信息
        frame_info = {
            'source': 'image',
            'name': image_path.name,
            'frame_id': 0
        }
        
        # 调用通用帧处理函数
        process_frame(frame, frame_info, model, output_dir, temp_dir, rejected_dir)
        
    except Exception as e:
        logger.error(f"处理图片时发生错误: {image_path.name} - {e}", exc_info=True)


def process_video(video_path, model, output_dir, temp_dir, rejected_dir, frame_interval=30):
    """
    处理MP4视频：逐帧检测拍照姿势
    
    Args:
        video_path: 视频文件路径
        model: YOLO模型
        output_dir: 输出目录（AI确认为拍照的图片）
        temp_dir: 临时目录
        rejected_dir: 拒绝目录（AI确认不是拍照的图片）
        frame_interval: 帧间隔，每隔多少帧处理一次（默认30帧，约1秒）
    """
    try:
        logger.info(f"开始处理视频: {video_path.name}")
        
        # 打开视频文件
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"视频信息: 总帧数={total_frames}, FPS={fps:.2f}, 时长={duration:.2f}秒")
        logger.info(f"处理策略: 每{frame_interval}帧处理一次")
        
        frame_count = 0
        processed_count = 0
        detection_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 按间隔处理帧
                if frame_count % frame_interval == 0:
                    processed_count += 1
                    current_time = frame_count / fps if fps > 0 else 0
                    
                    logger.info(f"\n{'='*50}")
                    logger.info(f"处理帧: {frame_count}/{total_frames} (时间: {current_time:.2f}s)")
                    logger.info(f"处理进度: {processed_count} 帧已处理")
                    logger.info(f"{'='*50}")
                    
                    # 构建帧信息
                    frame_info = {
                        'source': 'video',
                        'name': video_path.name,
                        'frame_id': frame_count,
                        'timestamp': current_time
                    }
                    
                    # 记录处理前的检测数量
                    before_count = len(list(output_dir.glob(f"{video_path.stem}_frame*.jpeg")))
                    
                    # 处理当前帧
                    process_frame(frame, frame_info, model, output_dir, temp_dir, rejected_dir)
                    
                    # 检查是否有新的检测结果
                    after_count = len(list(output_dir.glob(f"{video_path.stem}_frame*.jpeg")))
                    if after_count > before_count:
                        detection_count += 1
                        logger.info(f"✓ 第{frame_count}帧检测到拍照姿势")
                
                # 显示总体进度
                if frame_count % (frame_interval * 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"视频处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"视频处理完成: {video_path.name}")
        logger.info(f"总帧数: {total_frames}")
        logger.info(f"处理帧数: {processed_count}")
        logger.info(f"检测到拍照的帧数: {detection_count}")
        logger.info(f"检测率: {(detection_count/processed_count*100):.2f}%" if processed_count > 0 else "检测率: 0%")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"处理视频时发生错误: {video_path.name} - {e}", exc_info=True)


def process_images(input_dir, model, output_dir, temp_dir, rejected_dir):
    """处理图片目录"""
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"输入目录中没有找到图片文件: {input_dir}")
        return
    
    logger.info(f"找到 {len(image_files)} 张图片")
    
    # 逐张处理图片（完全同步）
    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"处理进度: {idx}/{len(image_files)}")
        logger.info(f"{'='*60}")
        
        # 同步处理：YOLO检测 -> AI验证 -> 保存结果
        # 处理完一张图片后才会继续下一张
        process_single_image(image_path, model, output_dir, temp_dir, rejected_dir)


def process_videos(input_dir, model, output_dir, temp_dir, rejected_dir, frame_interval=30):
    """处理视频目录"""
    # 获取所有视频文件
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = [f for f in input_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in video_extensions]
    
    if not video_files:
        logger.warning(f"输入目录中没有找到视频文件: {input_dir}")
        return
    
    logger.info(f"找到 {len(video_files)} 个视频文件")
    
    # 逐个处理视频
    for idx, video_path in enumerate(video_files, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"视频处理进度: {idx}/{len(video_files)}")
        logger.info(f"{'='*80}")
        
        # 处理视频
        process_video(video_path, model, output_dir, temp_dir, rejected_dir, frame_interval)


def main():
    """主函数"""
    try:
        parser = argparse.ArgumentParser(description='拍照姿势检测 - 图片/视频批量推理（仅AI验证版本）')
        parser.add_argument('--input', type=str, required=True, help='输入图片/视频目录')
        parser.add_argument('--output', type=str, required=True, help='输出目录')
        parser.add_argument('--model', type=str, required=True, help='模型路径')
        parser.add_argument('--mode', type=str, choices=['image', 'video', 'auto'], default='auto', 
                          help='处理模式: image(仅图片), video(仅视频), auto(自动检测)')
        parser.add_argument('--frame-interval', type=int, default=30, 
                          help='视频处理帧间隔，每隔多少帧处理一次（默认30帧）')
        args = parser.parse_args()
        
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        model_path = Path(args.model)
        mode = args.mode
        frame_interval = args.frame_interval
        
        # 检查输入目录
        if not input_dir.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录（AI确认）: {output_dir}")
        
        # 创建rejected目录（用于保存AI未确认的图片）
        rejected_dir = output_dir.parent / f"{output_dir.name}_rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"拒绝目录（AI未确认）: {rejected_dir}")
        
        # 创建临时目录（用于AI验证时的裁切图片）
        temp_dir = output_dir.parent / 'temp_spycam_ai'
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"临时目录: {temp_dir}")
        
        # 检查模型文件
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return
        
        # 加载模型
        logger.info(f"加载YOLO模型: {model_path}")
        model = YOLO(str(model_path)).cuda(device=0)
        logger.info("YOLO模型加载完成")
        
        # 根据模式处理文件
        if mode == 'auto':
            # 自动检测模式：同时处理图片和视频
            logger.info("自动检测模式：将处理目录中的所有图片和视频文件")
            
            # 处理图片
            logger.info("\n" + "="*80)
            logger.info("开始处理图片文件")
            logger.info("="*80)
            process_images(input_dir, model, output_dir, temp_dir, rejected_dir)
            
            # 处理视频
            logger.info("\n" + "="*80)
            logger.info("开始处理视频文件")
            logger.info("="*80)
            process_videos(input_dir, model, output_dir, temp_dir, rejected_dir, frame_interval)
            
        elif mode == 'image':
            # 仅处理图片
            logger.info("图片处理模式")
            process_images(input_dir, model, output_dir, temp_dir, rejected_dir)
            
        elif mode == 'video':
            # 仅处理视频
            logger.info(f"视频处理模式 (帧间隔: {frame_interval})")
            process_videos(input_dir, model, output_dir, temp_dir, rejected_dir, frame_interval)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"所有文件处理完成！")
        logger.info(f"主目录: 通过AI验证的拍照姿势图片（绿色框）")
        logger.info(f"  路径: {output_dir}")
        logger.info(f"rejected目录: 未通过AI验证的图片（红色框）")
        logger.info(f"  路径: {rejected_dir}")
        logger.info(f"视频帧命名格式: 视频名_frame帧号_事件ID.jpeg")
        logger.info(f"处理策略: YOLO检测 -> AI二次验证 -> 根据结果保存到不同目录")
        logger.info(f"{'='*60}")
        
        # 清理临时目录
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"已清理临时目录: {temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时目录失败: {e}")
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)


if __name__ == "__main__":
    main()