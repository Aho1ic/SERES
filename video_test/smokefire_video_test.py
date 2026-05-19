# -*- coding: UTF-8 -*-
"""
烟火检测视频批量推理脚本
对mp4视频进行逐帧推理，使用大模型进行二次验证
使用生产者-消费者模式，避免大模型成为瓶颈
"""
import cv2
import sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import subprocess
import queue
import threading
import time
import json

#python smokefire_video_test.py --input /home/algorithm/chongqing/赛力斯测试视频合集/ --output /home/algorithm/chongqing/aibox/0316综合测试/smokefirecrop/ --model /home/algorithm/chongqing/weights/smokefire.pt --conf 0.2 --frame-interval 10


# 设置路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ALLROOT = FILE.parents[1]

class SmokefireVideoInfer:
    def __init__(self, model_path, input_dir, output_dir, conf_thres=0.2, frame_interval=30, max_queue_size=50, num_workers=2):
        """
        初始化烟火视频推理器
        
        Args:
            model_path: 模型路径
            input_dir: 输入视频文件夹
            output_dir: 输出结果文件夹
            conf_thres: 置信度阈值
            frame_interval: 帧间隔（每隔多少帧处理一次）
            max_queue_size: 待验证队列最大长度
            num_workers: AI验证工作线程数量
        """
        self.model_path = Path(model_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.conf_thres = float(conf_thres)
        self.frame_interval = frame_interval
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建AI验证未通过的视频帧保存目录
        self.rejected_dir = self.output_dir.parent / 'rejected_smokefire_video'
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        
        # 临时目录用于存放裁切的检测框图片
        self.temp_dir = self.output_dir.parent / 'temp_smokefire_video_bbox'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 绘制颜色
        self.draw_colors = {
            'fire': (0, 0, 255),
            'smoke': (200, 200, 200)
        }
        self.color_cycle = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,255,0)]
        
        # 生产者-消费者队列
        self.verification_queue = queue.Queue(maxsize=max_queue_size)
        self.num_workers = num_workers
        self.workers = []
        self.stop_event = threading.Event()
        
        # 统计信息（线程安全）
        self.stats_lock = threading.Lock()
        self.stats = {
            'total_videos': 0,
            'total_frames': 0,
            'detected_frames': 0,
            'confirmed_frames': 0,
            'rejected_frames': 0,
            'confirmed_videos': 0,
            'queue_full_count': 0
        }
        
        logger.info(f"初始化完成: 模型={self.model_path}, 输入={self.input_dir}, 输出={self.output_dir}")
        logger.info(f"AI验证未通过保存目录: {self.rejected_dir}")
        logger.info(f"帧间隔={frame_interval}, 队列大小={max_queue_size}, AI验证工作线程={num_workers}")
    
    def _update_stats(self, key, increment=1):
        """更新统计信息"""
        self.stats[key] += increment
    
    def _get_stats(self):
        """获取统计信息"""
        return self.stats.copy()
    
    def _get_label_color(self, label: str, idx: int):
        """获取标签颜色"""
        if label in self.draw_colors:
            return self.draw_colors[label]
        return self.color_cycle[idx % len(self.color_cycle)]
    def analyze_bbox_with_ai(self, bbox_image_path):
        """
        调用smokefire_analysis.py对裁切的检测框图片进行AI分析
        返回True表示检测到烟雾或火焰，False表示未检测到
        """
        try:
            image_analysis_path = Path('/home/algorithm/chongqing/omni/python/smokefire_analysis.py')
            
            if not image_analysis_path.exists():
                logger.error(f"smokefire_analysis.py文件不存在: {image_analysis_path}")
                return False
            
            cmd = [sys.executable, str(image_analysis_path), str(bbox_image_path)]
            logger.debug(f"执行AI分析命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"AI分析失败: {result.stderr}")
                return False
            
            output = result.stdout.strip()
            logger.debug(f"AI分析输出: {output}")

            # 解析输出结果
            for line in output.split('\n'):
                s = line.strip()
                if s.startswith("[{\'text\': 'yes'}]") or s.startswith("{'text': 'yes'}"):
                    logger.debug("AI确认检测到烟雾或火焰")
                    return True
                if s.startswith("[{\'text\': 'no'}]") or s.startswith("{'text': 'no'}"):
                    logger.debug("AI确认未检测到烟雾或火焰")
                    return False

            if "yes" in output.lower() and "no" not in output.lower():
                logger.debug("AI确认检测到烟雾或火焰")
                return True
            elif "no" in output.lower() and "yes" not in output.lower():
                logger.debug("AI确认未检测到烟雾或火焰")
                return False
            else:
                logger.warning(f"AI分析结果不明确: {output}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("AI分析超时")
            return False
        except Exception as e:
            logger.error(f"AI分析异常: {e}", exc_info=True)
            return False(f"AI分析异常: {e}", exc_info=True)
            return False
    
    def process_single_frame(self, frame, video_name, frame_num, video_results):
        """
        处理单帧图像（YOLO推理 + 检测框级别AI验证）
        同步模式：检测到目标后对每个检测框进行AI验证，等待所有结果后再返回
        
        Args:
            frame: 视频帧
            video_name: 视频文件名
            frame_num: 帧号
            video_results: 视频结果字典
            
        Returns:
            bool: 是否有检测框通过AI验证
        """
        try:
            # 修复高分辨率图片的内存对齐问题
            h, w = frame.shape[:2]
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
                        
                        # 确保检测框在图片范围内
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(frame.shape[1], int(x2))
                        y2 = min(frame.shape[0], int(y2))
                        
                        # 检查检测框是否有效
                        if x2 > x1 and y2 > y1:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'conf': conf,
                                'cls_name': cls_name
                            })
            except Exception as e:
                logger.error(f"检测结果处理异常: {e}", exc_info=True)
                return False
            
            # 如果没有检测到目标，跳过
            if not detections:
                return False
            
            logger.info(f"第{frame_num}帧检测到 {len(detections)} 个目标，开始逐个AI验证...")
            self._update_stats('detected_frames')
            
            # 对每个检测框进行AI验证（同步等待）
            confirmed_detections = []
            for idx, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                
                # 裁切检测框
                bbox_img = frame[y1:y2, x1:x2]
                if bbox_img.size == 0:
                    logger.warning(f"检测框为空，跳过: {det}")
                    continue
                
                # 保存裁切的检测框图片
                base_name = f"{Path(video_name).stem}_frame_{frame_num:06d}_bbox_{idx}"
                bbox_path = self.temp_dir / f"{base_name}.jpg"
                cv2.imwrite(str(bbox_path), bbox_img)
                
                logger.info(f"正在验证第{frame_num}帧第{idx+1}个检测框...")
                
                # AI验证检测框（同步等待结果）
                ai_confirmed = self.analyze_bbox_with_ai(bbox_path)
                
                # 删除临时检测框图片
                if bbox_path.exists():
                    bbox_path.unlink()
                
                if ai_confirmed:
                    confirmed_detections.append(det)
                    logger.info(f"第{frame_num}帧第{idx+1}个检测框AI验证通过")
                else:
                    logger.info(f"第{frame_num}帧第{idx+1}个检测框AI验证未通过")
            
            # 如果有检测框通过验证，保存结果
            if confirmed_detections:
                # 创建视频输出目录
                video_output_dir = self.output_dir / Path(video_name).stem
                video_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 绘制通过验证的检测框
                annotated = frame.copy()
                for idx, det in enumerate(confirmed_detections):
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
                
                # 保存通过验证的结果图片
                confirmed_path = video_output_dir / f"frame_{frame_num:06d}_confirmed.jpeg"
                cv2.imwrite(str(confirmed_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                # 记录到视频结果中
                video_results['confirmed_frames'].append({
                    'frame_num': frame_num,
                    'detected_path': str(confirmed_path),
                    'timestamp': frame_num / video_results.get('fps', 30),
                    'confirmed_count': len(confirmed_detections),
                    'total_detections': len(detections)
                })
                
                logger.info(f"第{frame_num}帧AI验证完成: {len(confirmed_detections)}/{len(detections)} 个检测框通过验证")
                self._update_stats('confirmed_frames')
            
            # 保存所有检测框的验证详情到rejected目录
            if len(detections) > 0:
                # 创建rejected目录
                rejected_video_dir = self.rejected_dir / Path(video_name).stem
                rejected_video_dir.mkdir(parents=True, exist_ok=True)
                
                # 绘制所有检测框（包括未通过验证的）
                rejected_annotated = frame.copy()
                for idx, det in enumerate(detections):
                    x1, y1, x2, y2 = det['bbox']
                    label = det['cls_name']
                    conf = det['conf']
                    
                    # 检查这个检测框是否通过了验证
                    is_confirmed = any(
                        c_det['bbox'] == det['bbox'] and 
                        c_det['cls_name'] == det['cls_name'] and 
                        c_det['conf'] == det['conf'] 
                        for c_det in confirmed_detections
                    )
                    
                    # 通过验证的用绿色，未通过的用红色
                    color = (0, 255, 0) if is_confirmed else (0, 0, 255)
                    
                    cv2.rectangle(rejected_annotated, (x1, y1), (x2, y2), color, 2)
                    text = f"{label}:{conf:.2f}{'✓' if is_confirmed else '✗'}"
                    tsize, tbase = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    ty = max(y1, tsize[1] + 2)
                    cv2.rectangle(rejected_annotated, (x1, ty - tsize[1] - tbase), (x1 + tsize[0], ty), color, -1)
                    cv2.putText(rejected_annotated, text, (x1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                # 保存验证详情图片
                rejected_path = rejected_video_dir / f"frame_{frame_num:06d}_all_detections.jpeg"
                cv2.imwrite(str(rejected_path), rejected_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                if len(confirmed_detections) == 0:
                    self._update_stats('rejected_frames')
            
            # 返回是否有检测框通过验证
            return len(confirmed_detections) > 0
                
        except Exception as e:
            logger.error(f"处理帧时发生错误: {e}", exc_info=True)
            return False

    def process_single_video(self, video_path):
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            dict: 视频处理结果
        """
        logger.info(f"开始处理视频: {video_path.name}")
        
        # 视频结果字典（线程安全）
        video_results = {
            'video_name': video_path.name,
            'total_frames': 0,
            'processed_frames': 0,
            'detected_frames': 0,
            'confirmed_frames': [],
            'fps': 30
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
            video_results['fps'] = fps
            video_results['total_frames'] = total_frames
            
            logger.info(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}")
            
            frame_count = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 按帧间隔处理
                if frame_count % self.frame_interval == 0:
                    processed_count += 1
                    self._update_stats('total_frames')
                    
                    # 同步处理：YOLO检测 + 逐个检测框AI验证（等待所有验证完成）
                    if self.process_single_frame(frame, video_path.name, frame_count, video_results):
                        video_results['detected_frames'] += 1
                    
                    # 定期输出进度
                    if processed_count % 50 == 0:
                        progress = (frame_count / total_frames) * 100
                        logger.info(f"视频 {video_path.name} 处理进度: {progress:.1f}% "
                                  f"({frame_count}/{total_frames}帧), "
                                  f"已确认帧数: {len(video_results['confirmed_frames'])}")
            
            cap.release()
            video_results['processed_frames'] = processed_count
            
            logger.info(f"视频 {video_path.name} 处理完成: "
                       f"总帧数={total_frames}, 处理帧数={processed_count}, "
                       f"检测到目标帧数={video_results['detected_frames']}")
            
            return video_results
            
        except Exception as e:
            logger.error(f"处理视频时发生错误: {e}", exc_info=True)
            return video_results
    
    def run(self):
        """
        批量处理文件夹中的所有mp4视频
        使用生产者-消费者模式，YOLO推理和AI验证并行执行
        """
        # 获取所有mp4文件
        video_files = list(self.input_dir.glob('*.mp4')) + list(self.input_dir.glob('*.MP4'))
        
        if not video_files:
            logger.warning(f"在 {self.input_dir} 中未找到mp4视频文件")
            return
        
        self.stats['total_videos'] = len(video_files)
        logger.info(f"找到 {len(video_files)} 个视频文件，开始批量处理...")
        
        # 加载模型
        logger.info(f"加载模型: {self.model_path}")
        self.model = YOLO(str(self.model_path)).cuda(device=1)
        
        # 处理所有视频
        start_time = time.time()
        all_video_results = []
        
        for idx, video_path in enumerate(video_files, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"视频处理进度: [{idx}/{len(video_files)}] - {video_path.name}")
            
            video_result = self.process_single_video(video_path)
            all_video_results.append(video_result)
            
            # 输出当前视频处理结果
            logger.info(f"视频 {video_path.name} 处理完成: "
                       f"检测帧数={video_result['detected_frames']}, "
                       f"确认帧数={len(video_result['confirmed_frames'])}")
        
        total_time = time.time() - start_time
        
        # 统计有确认结果的视频数量
        confirmed_videos = sum(1 for result in all_video_results if result['confirmed_frames'])
        self.stats['confirmed_videos'] = confirmed_videos
        
        # 保存结果摘要
        self.save_results_summary(all_video_results, total_time)
        
        # 输出最终统计结果
        stats = self._get_stats()
        logger.info(f"\n{'='*80}")
        logger.info(f"所有视频处理完成！")
        logger.info(f"总视频数: {stats['total_videos']} 个")
        logger.info(f"总处理帧数: {stats['total_frames']} 帧")
        logger.info(f"检测到目标帧数: {stats['detected_frames']} 帧")
        logger.info(f"AI确认通过帧数: {stats['confirmed_frames']} 帧")
        logger.info(f"AI确认拒绝帧数: {stats['rejected_frames']} 帧")
        logger.info(f"有确认结果的视频数: {confirmed_videos} 个")
        logger.info(f"总耗时: {total_time:.2f}秒")
        if stats['total_frames'] > 0:
            logger.info(f"平均处理速度: {stats['total_frames']/total_time:.2f} 帧/秒")
        logger.info(f"结果保存在: {self.output_dir}")
        logger.info(f"验证详情（包含未通过的）保存在: {self.rejected_dir}")
        logger.info(f"{'='*80}")

    def save_results_summary(self, video_results, total_time):
        """保存结果摘要到JSON文件"""
        try:
            summary = {
                'processing_info': {
                    'total_time': total_time,
                    'frame_interval': self.frame_interval,
                    'conf_threshold': self.conf_thres,
                    'model_path': str(self.model_path)
                },
                'statistics': self._get_stats(),
                'video_results': video_results
            }
            
            summary_path = self.output_dir / 'processing_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结果摘要已保存: {summary_path}")
            
        except Exception as e:
            logger.error(f"保存结果摘要失败: {e}", exc_info=True)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='烟火检测视频批量推理（检测框级别AI验证）')
    parser.add_argument('--input', type=str, required=True, help='输入视频文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出结果文件夹路径')
    parser.add_argument('--model', type=str, default=None, help='模型路径（默认使用weights/smokefire.pt）')
    parser.add_argument('--conf', type=float, default=0.2, help='置信度阈值（默认0.2）')
    parser.add_argument('--frame-interval', type=int, default=30, help='帧间隔，每隔多少帧处理一次（默认30）')
    parser.add_argument('--queue-size', type=int, default=100, help='验证队列最大长度（默认100）')
    parser.add_argument('--workers', type=int, default=4, help='AI验证工作线程数（默认4）')
    
    args = parser.parse_args()
    
    # 设置默认模型路径
    if args.model is None:
        model_path = ALLROOT / 'weights' / 'smokefire.pt'
    else:
        model_path = Path(args.model)
    
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    # 创建推理器并运行
    infer = SmokefireVideoInfer(
        model_path=model_path,
        input_dir=args.input,
        output_dir=args.output,
        conf_thres=args.conf,
        frame_interval=args.frame_interval,
        max_queue_size=args.queue_size,
        num_workers=args.workers
    )
    
    infer.run()


if __name__ == "__main__":
    main()