# -*- coding: UTF-8 -*-
"""
通用视频检测批量推理脚本
支持多种检测类型：烟火、电话、间谍摄像头等
对mp4视频进行逐帧推理，使用大模型进行二次验证
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
import argparse

# 设置路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ALLROOT = FILE.parents[1]

# 检测类型配置
DETECTION_CONFIGS = {
    'smokefire': {
        'model_name': 'smokefire.pt',
        'analysis_script': 'smokefire_analysis.py',
        'colors': {'fire': (0, 0, 255), 'smoke': (200, 200, 200)}
    },
    'phonecall': {
        'model_name': 'phonecall.pt', 
        'analysis_script': 'phonecall_analysis.py',
        'colors': {'phone': (0, 255, 0), 'call': (255, 0, 0)}
    },
    'spycam': {
        'model_name': 'spycam.pt',
        'analysis_script': 'spycam_analysis.py', 
        'colors': {'camera': (255, 255, 0), 'spy': (255, 0, 255)}
    }
}

class UniversalVideoInfer:
    def __init__(self, detection_type, model_path, input_dir, output_dir, conf_thres=0.2, 
                 max_queue_size=50, num_workers=2, frame_interval=30):
        """
        初始化通用视频推理器
        
        Args:
            detection_type: 检测类型 ('smokefire', 'phonecall', 'spycam')
            model_path: 模型路径
            input_dir: 输入视频文件夹
            output_dir: 输出结果文件夹
            conf_thres: 置信度阈值
            max_queue_size: 待验证队列最大长度
            num_workers: AI验证工作线程数量
            frame_interval: 帧间隔（每隔多少帧处理一次）
        """
        self.detection_type = detection_type
        self.model_path = Path(model_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.conf_thres = float(conf_thres)
        self.frame_interval = frame_interval
        
        # 获取检测配置
        if detection_type not in DETECTION_CONFIGS:
            raise ValueError(f"不支持的检测类型: {detection_type}")
        
        self.config = DETECTION_CONFIGS[detection_type]
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 临时目录用于存放待验证的帧
        self.temp_dir = self.output_dir.parent / f'temp_{detection_type}_video'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 绘制颜色
        self.draw_colors = self.config['colors']
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
            'queue_full_count': 0,
            'confirmed_videos': 0
        }
        
        logger.info(f"初始化完成: 检测类型={detection_type}, 模型={self.model_path}")
        logger.info(f"输入={self.input_dir}, 输出={self.output_dir}")
        logger.info(f"队列大小={max_queue_size}, AI验证工作线程={num_workers}, 帧间隔={frame_interval}")
    
    def _update_stats(self, key, increment=1):
        """线程安全地更新统计信息"""
        with self.stats_lock:
            self.stats[key] += increment
    
    def _get_stats(self):
        """线程安全地获取统计信息"""
        with self.stats_lock:
            return self.stats.copy()
    
    def _get_label_color(self, label: str, idx: int):
        """获取标签颜色"""
        if label in self.draw_colors:
            return self.draw_colors[label]
        return self.color_cycle[idx % len(self.color_cycle)]
    
    def analyze_image_with_ai(self, image_path):
        """
        调用对应的AI分析脚本对图片进行分析
        返回True表示检测到目标，False表示未检测到
        """
        try:
            analysis_script = self.config['analysis_script']
            image_analysis_path = ALLROOT / 'omni' / 'python' / analysis_script
            
            if not image_analysis_path.exists():
                logger.error(f"{analysis_script}文件不存在: {image_analysis_path}")
                return False
            
            cmd = [sys.executable, str(image_analysis_path), str(image_path)]
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
                    logger.debug(f"AI确认检测到{self.detection_type}")
                    return True
                if s.startswith("[{\'text\': 'no'}]") or s.startswith("{'text': 'no'}"):
                    logger.debug(f"AI确认未检测到{self.detection_type}")
                    return False

            if "yes" in output.lower() and "no" not in output.lower():
                logger.debug(f"AI确认检测到{self.detection_type}")
                return True
            elif "no" in output.lower() and "yes" not in output.lower():
                logger.debug(f"AI确认未检测到{self.detection_type}")
                return False
            else:
                logger.warning(f"AI分析结果不明确: {output}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("AI分析超时")
            return False
        except Exception as e:
            logger.error(f"AI分析异常: {e}", exc_info=True)
            return False
    
    def ai_verification_worker(self, worker_id):
        """AI验证工作线程"""
        logger.info(f"AI验证工作线程 {worker_id} 启动")
        
        while not self.stop_event.is_set():
            try:
                try:
                    task = self.verification_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                if task is None:  # 结束信号
                    self.verification_queue.task_done()
                    break
                
                orig_path, detected_path, video_name, frame_num, video_results = task
                
                logger.debug(f"[Worker-{worker_id}] 开始验证: {video_name} 第{frame_num}帧")
                
                # AI二次验证
                ai_confirmed = self.analyze_image_with_ai(orig_path)
                
                if ai_confirmed:
                    # 验证通过，记录到视频结果中
                    video_results['confirmed_frames'].append({
                        'frame_num': frame_num,
                        'detected_path': str(detected_path),
                        'timestamp': frame_num / video_results.get('fps', 30)
                    })
                    logger.debug(f"[Worker-{worker_id}] AI确认通过: {video_name} 第{frame_num}帧")
                    self._update_stats('confirmed_frames')
                else:
                    # 验证不通过，删除检测结果图
                    logger.debug(f"[Worker-{worker_id}] AI确认未通过: {video_name} 第{frame_num}帧")
                    self._update_stats('rejected_frames')
                    if detected_path.exists():
                        detected_path.unlink()
                
                # 删除临时的原始帧文件
                if orig_path.exists():
                    orig_path.unlink()
                
                self.verification_queue.task_done()
                
            except Exception as e:
                logger.error(f"[Worker-{worker_id}] 验证过程异常: {e}", exc_info=True)
                self.verification_queue.task_done()
        
        logger.info(f"AI验证工作线程 {worker_id} 退出")
    
    def start_workers(self):
        """启动AI验证工作线程"""
        self.stop_event.clear()
        for i in range(self.num_workers):
            worker = threading.Thread(target=self.ai_verification_worker, args=(i+1,), daemon=True)
            worker.start()
            self.workers.append(worker)
        logger.info(f"已启动 {self.num_workers} 个AI验证工作线程")
    
    def stop_workers(self):
        """停止AI验证工作线程"""
        logger.info("正在停止AI验证工作线程...")
        
        # 发送结束信号
        for _ in range(self.num_workers):
            self.verification_queue.put(None)
        
        # 等待所有线程结束
        for worker in self.workers:
            worker.join(timeout=10)
        
        self.workers.clear()
        logger.info("所有AI验证工作线程已停止")

    def process_single_frame(self, frame, video_name, frame_num, video_results):
        """处理单帧图像"""
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
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'conf': conf,
                            'cls_name': cls_name
                        })
            except Exception as e:
                logger.error(f"检测结果处理异常: {e}", exc_info=True)
                return False
            
            # 如果没有检测到目标，跳过
            if not detections:
                return False
            
            logger.debug(f"第{frame_num}帧检测到 {len(detections)} 个目标")
            self._update_stats('detected_frames')
            
            # 生成文件名
            base_name = f"{Path(video_name).stem}_frame_{frame_num:06d}"
            orig_path = self.temp_dir / f"{base_name}.jpg"
            detected_path = self.output_dir / video_name.replace('.mp4', '') / f"{base_name}.jpeg"
            
            # 创建视频输出目录
            detected_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存原始帧（用于AI验证）
            cv2.imwrite(str(orig_path), frame)
            
            # 绘制检测框并保存
            annotated = frame.copy()
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
            
            cv2.imwrite(str(detected_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            # 将任务放入验证队列
            try:
                self.verification_queue.put((orig_path, detected_path, video_name, frame_num, video_results), block=True, timeout=60)
                return True
            except queue.Full:
                logger.warning(f"验证队列已满且等待超时，跳过验证: {video_name} 第{frame_num}帧")
                self._update_stats('queue_full_count')
                # 队列满时删除生成的文件
                if orig_path.exists():
                    orig_path.unlink()
                if detected_path.exists():
                    detected_path.unlink()
                return False
                
        except Exception as e:
            logger.error(f"处理帧时发生错误: {e}", exc_info=True)
            return False

    def process_single_video(self, video_path):
        """处理单个视频文件"""
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
                    
                    if self.process_single_frame(frame, video_path.name, frame_count, video_results):
                        video_results['detected_frames'] += 1
                    
                    # 定期输出进度
                    if processed_count % 100 == 0:
                        progress = (frame_count / total_frames) * 100
                        logger.info(f"视频 {video_path.name} 处理进度: {progress:.1f}% "
                                  f"({frame_count}/{total_frames}帧)")
            
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
        """批量处理文件夹中的所有mp4视频"""
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
        
        # 启动AI验证工作线程
        self.start_workers()
        
        # 处理所有视频
        start_time = time.time()
        all_video_results = []
        
        for idx, video_path in enumerate(video_files, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"视频处理进度: [{idx}/{len(video_files)}] - {video_path.name}")
            
            video_result = self.process_single_video(video_path)
            all_video_results.append(video_result)
            
            # 定期输出整体状态
            if idx % 5 == 0:
                stats = self._get_stats()
                logger.info(f"整体状态 - 已处理视频: {idx}, "
                          f"总处理帧数: {stats['total_frames']}, "
                          f"检测到目标帧数: {stats['detected_frames']}, "
                          f"队列长度: {self.verification_queue.qsize()}")
        
        yolo_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"所有视频YOLO推理完成，耗时: {yolo_time:.2f}秒")
        logger.info(f"等待AI验证队列处理完成...")
        
        # 等待队列中的所有任务完成
        self.verification_queue.join()
        
        # 停止工作线程
        self.stop_workers()
        
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
        logger.info(f"检测类型: {self.detection_type}")
        logger.info(f"总视频数: {stats['total_videos']} 个")
        logger.info(f"总处理帧数: {stats['total_frames']} 帧")
        logger.info(f"检测到目标帧数: {stats['detected_frames']} 帧")
        logger.info(f"AI确认通过帧数: {stats['confirmed_frames']} 帧")
        logger.info(f"AI确认拒绝帧数: {stats['rejected_frames']} 帧")
        logger.info(f"有确认结果的视频数: {confirmed_videos} 个")
        if stats['queue_full_count'] > 0:
            logger.warning(f"队列满跳过帧数: {stats['queue_full_count']} 帧")
        logger.info(f"YOLO推理耗时: {yolo_time:.2f}秒")
        logger.info(f"总耗时: {total_time:.2f}秒")
        if stats['total_frames'] > 0:
            logger.info(f"YOLO平均速度: {stats['total_frames']/yolo_time:.2f} 帧/秒")
            logger.info(f"整体平均速度: {stats['total_frames']/total_time:.2f} 帧/秒")
        logger.info(f"结果保存在: {self.output_dir}")
        logger.info(f"{'='*80}")

    def save_results_summary(self, video_results, total_time):
        """保存结果摘要到JSON文件"""
        try:
            summary = {
                'processing_info': {
                    'detection_type': self.detection_type,
                    'total_time': total_time,
                    'frame_interval': self.frame_interval,
                    'conf_threshold': self.conf_thres,
                    'model_path': str(self.model_path)
                },
                'statistics': self._get_stats(),
                'video_results': video_results
            }
            
            summary_path = self.output_dir / f'{self.detection_type}_processing_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结果摘要已保存: {summary_path}")
            
        except Exception as e:
            logger.error(f"保存结果摘要失败: {e}", exc_info=True)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='通用视频检测批量推理')
    parser.add_argument('--type', type=str, required=True, 
                       choices=['smokefire', 'phonecall', 'spycam'],
                       help='检测类型')
    parser.add_argument('--input', type=str, required=True, help='输入视频文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出结果文件夹路径')
    parser.add_argument('--model', type=str, default=None, help='模型路径（默认使用weights/对应模型）')
    parser.add_argument('--conf', type=float, default=0.2, help='置信度阈值（默认0.2）')
    parser.add_argument('--queue-size', type=int, default=100, help='验证队列最大长度（默认100）')
    parser.add_argument('--workers', type=int, default=4, help='AI验证工作线程数（默认4）')
    parser.add_argument('--frame-interval', type=int, default=30, help='帧间隔，每隔多少帧处理一次（默认30）')
    
    args = parser.parse_args()
    
    # 设置默认模型路径
    if args.model is None:
        model_name = DETECTION_CONFIGS[args.type]['model_name']
        model_path = ALLROOT / 'weights' / model_name
    else:
        model_path = Path(args.model)
    
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    # 创建推理器并运行
    infer = UniversalVideoInfer(
        detection_type=args.type,
        model_path=model_path,
        input_dir=args.input,
        output_dir=args.output,
        conf_thres=args.conf,
        max_queue_size=args.queue_size,
        num_workers=args.workers,
        frame_interval=args.frame_interval
    )
    
    infer.run()


if __name__ == "__main__":
    main()