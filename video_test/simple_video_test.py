# -*- coding: UTF-8 -*-
"""
简化版视频检测推理脚本
快速对单个或多个mp4视频进行推理，不使用AI二次验证
适用于快速测试和预览
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import time
import json
import argparse

# 设置路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ALLROOT = FILE.parents[1]

class SimpleVideoInfer:
    def __init__(self, model_path, input_path, output_dir, conf_thres=0.2, frame_interval=30, save_video=False):
        """
        初始化简化视频推理器
        
        Args:
            model_path: 模型路径
            input_path: 输入视频文件或文件夹路径
            output_dir: 输出结果文件夹
            conf_thres: 置信度阈值
            frame_interval: 帧间隔（每隔多少帧处理一次）
            save_video: 是否保存检测结果视频
        """
        self.model_path = Path(model_path)
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.conf_thres = float(conf_thres)
        self.frame_interval = frame_interval
        self.save_video = save_video
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 绘制颜色
        self.color_cycle = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,255,0), (255,0,255)]
        
        # 统计信息
        self.stats = {
            'total_videos': 0,
            'total_frames': 0,
            'detected_frames': 0,
            'processing_time': 0
        }
        
        logger.info(f"初始化完成: 模型={self.model_path}, 输入={self.input_path}, 输出={self.output_dir}")
        logger.info(f"帧间隔={frame_interval}, 保存视频={save_video}")
    
    def _get_label_color(self, label: str, idx: int):
        """获取标签颜色"""
        return self.color_cycle[idx % len(self.color_cycle)]
    
    def process_single_frame(self, frame, frame_num):
        """
        处理单帧图像
        
        Args:
            frame: 视频帧
            frame_num: 帧号
            
        Returns:
            tuple: (是否检测到目标, 标注后的帧, 检测结果列表)
        """
        try:
            # 修复高分辨率图片的内存对齐问题
            h, w = frame.shape[:2]
            original_frame = frame.copy()
            
            if h >= 1080:
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # 模型推理
            results = self.model(frame, conf=self.conf_thres)
            
            # 解析检测结果
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
                        
                        # 如果原始帧被缩放了，需要将坐标映射回原始尺寸
                        if h >= 1080:
                            scale_x = w / 1280
                            scale_y = h / 720
                            x1, x2 = x1 * scale_x, x2 * scale_x
                            y1, y2 = y1 * scale_y, y2 * scale_y
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'conf': conf,
                            'cls_name': cls_name,
                            'frame_num': frame_num
                        })
            except Exception as e:
                logger.error(f"检测结果处理异常: {e}", exc_info=True)
                return False, original_frame, []
            
            # 如果没有检测到目标，返回原始帧
            if not detections:
                return False, original_frame, []
            
            # 在原始帧上绘制检测框
            annotated = original_frame.copy()
            for idx, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                label = det['cls_name']
                conf = det['conf']
                color = self._get_label_color(label, idx)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                text = f"{label}:{conf:.2f}"
                tsize, tbase = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = max(y1, tsize[1] + 2)
                cv2.rectangle(annotated, (x1, ty - tsize[1] - tbase), (x1 + tsize[0], ty), color, -1)
                cv2.putText(annotated, text, (x1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            return True, annotated, detections
                
        except Exception as e:
            logger.error(f"处理帧时发生错误: {e}", exc_info=True)
            return False, original_frame, []

    def process_single_video(self, video_path):
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            dict: 视频处理结果
        """
        logger.info(f"开始处理视频: {video_path.name}")
        
        # 视频结果字典
        video_results = {
            'video_name': video_path.name,
            'total_frames': 0,
            'processed_frames': 0,
            'detected_frames': 0,
            'detections': [],
            'fps': 30,
            'duration': 0
        }
        
        try:
            # 打开视频
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return video_results
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_results['fps'] = fps
            video_results['total_frames'] = total_frames
            video_results['duration'] = total_frames / fps if fps > 0 else 0
            
            logger.info(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 分辨率={width}x{height}, 时长={video_results['duration']:.2f}秒")
            
            # 如果需要保存视频，初始化视频写入器
            video_writer = None
            if self.save_video:
                output_video_path = self.output_dir / f"{video_path.stem}_detected.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
                logger.info(f"将保存检测结果视频: {output_video_path}")
            
            frame_count = 0
            processed_count = 0
            detected_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 按帧间隔处理
                if frame_count % self.frame_interval == 0:
                    processed_count += 1
                    self.stats['total_frames'] += 1
                    
                    has_detection, annotated_frame, detections = self.process_single_frame(frame, frame_count)
                    
                    if has_detection:
                        detected_count += 1
                        self.stats['detected_frames'] += 1
                        video_results['detections'].extend(detections)
                        
                        # 保存检测到目标的帧
                        frame_output_dir = self.output_dir / video_path.stem
                        frame_output_dir.mkdir(exist_ok=True)
                        frame_filename = f"frame_{frame_count:06d}.jpg"
                        frame_path = frame_output_dir / frame_filename
                        cv2.imwrite(str(frame_path), annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        
                        logger.debug(f"第{frame_count}帧检测到 {len(detections)} 个目标，已保存: {frame_filename}")
                    
                    # 定期输出进度
                    if processed_count % 200 == 0:
                        progress = (frame_count / total_frames) * 100
                        logger.info(f"视频 {video_path.name} 处理进度: {progress:.1f}% "
                                  f"({frame_count}/{total_frames}帧), 检测到目标帧数: {detected_count}")
                
                # 如果需要保存视频，写入当前帧（无论是否有检测结果）
                if video_writer is not None:
                    if frame_count % self.frame_interval == 0 and has_detection:
                        video_writer.write(annotated_frame)
                    else:
                        video_writer.write(frame)
            
            cap.release()
            if video_writer is not None:
                video_writer.release()
            
            video_results['processed_frames'] = processed_count
            video_results['detected_frames'] = detected_count
            
            logger.info(f"视频 {video_path.name} 处理完成: "
                       f"总帧数={total_frames}, 处理帧数={processed_count}, "
                       f"检测到目标帧数={detected_count}")
            
            return video_results
            
        except Exception as e:
            logger.error(f"处理视频时发生错误: {e}", exc_info=True)
            if video_writer is not None:
                video_writer.release()
            return video_results
    
    def run(self):
        """运行视频推理"""
        start_time = time.time()
        
        # 确定输入文件列表
        if self.input_path.is_file():
            if self.input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                video_files = [self.input_path]
            else:
                logger.error(f"不支持的视频格式: {self.input_path}")
                return
        elif self.input_path.is_dir():
            video_files = []
            for ext in ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV']:
                video_files.extend(self.input_path.glob(f'*{ext}'))
        else:
            logger.error(f"输入路径不存在: {self.input_path}")
            return
        
        if not video_files:
            logger.warning(f"未找到视频文件")
            return
        
        self.stats['total_videos'] = len(video_files)
        logger.info(f"找到 {len(video_files)} 个视频文件，开始处理...")
        
        # 加载模型
        logger.info(f"加载模型: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # 处理所有视频
        all_video_results = []
        
        for idx, video_path in enumerate(video_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"视频处理进度: [{idx}/{len(video_files)}] - {video_path.name}")
            
            video_result = self.process_single_video(video_path)
            all_video_results.append(video_result)
        
        self.stats['processing_time'] = time.time() - start_time
        
        # 保存结果摘要
        self.save_results_summary(all_video_results)
        
        # 输出最终统计结果
        logger.info(f"\n{'='*60}")
        logger.info(f"所有视频处理完成！")
        logger.info(f"总视频数: {self.stats['total_videos']} 个")
        logger.info(f"总处理帧数: {self.stats['total_frames']} 帧")
        logger.info(f"检测到目标帧数: {self.stats['detected_frames']} 帧")
        logger.info(f"总耗时: {self.stats['processing_time']:.2f}秒")
        if self.stats['total_frames'] > 0:
            logger.info(f"平均处理速度: {self.stats['total_frames']/self.stats['processing_time']:.2f} 帧/秒")
        logger.info(f"结果保存在: {self.output_dir}")
        logger.info(f"{'='*60}")

    def save_results_summary(self, video_results):
        """保存结果摘要到JSON文件"""
        try:
            summary = {
                'processing_info': {
                    'total_time': self.stats['processing_time'],
                    'frame_interval': self.frame_interval,
                    'conf_threshold': self.conf_thres,
                    'model_path': str(self.model_path),
                    'save_video': self.save_video
                },
                'statistics': self.stats,
                'video_results': video_results
            }
            
            summary_path = self.output_dir / 'simple_processing_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结果摘要已保存: {summary_path}")
            
        except Exception as e:
            logger.error(f"保存结果摘要失败: {e}", exc_info=True)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化版视频检测推理')
    parser.add_argument('--input', type=str, required=True, help='输入视频文件或文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出结果文件夹路径')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--conf', type=float, default=0.2, help='置信度阈值（默认0.2）')
    parser.add_argument('--frame-interval', type=int, default=30, help='帧间隔，每隔多少帧处理一次（默认30）')
    parser.add_argument('--save-video', action='store_true', help='是否保存检测结果视频')
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    # 创建推理器并运行
    infer = SimpleVideoInfer(
        model_path=model_path,
        input_path=args.input,
        output_dir=args.output,
        conf_thres=args.conf,
        frame_interval=args.frame_interval,
        save_video=args.save_video
    )
    
    infer.run()


if __name__ == "__main__":
    main()