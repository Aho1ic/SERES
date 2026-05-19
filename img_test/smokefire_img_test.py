# -*- coding: UTF-8 -*-
"""
烟火检测图片批量推理脚本
对文件夹中的所有图片进行推理，将检测框裁切后使用大模型进行二次验证
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

#python smokefire_img_test.py --input /path/to/videos --output /path/to/results --model /path/to/smokefire.pt --conf 0.2 --frame-interval 10
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ALLROOT = FILE.parents[1]

class SmokefireImageInfer:
    def __init__(self, model_path, input_dir, output_dir, conf_thres=0.2, max_queue_size=50, num_workers=2):
        """
        初始化烟火图片推理器
        
        Args:
            model_path: 模型路径
            input_dir: 输入图片文件夹
            output_dir: 输出结果文件夹（仅保存验证通过的jpeg）
            conf_thres: 置信度阈值
            max_queue_size: 待验证队列最大长度（防止内存溢出）
            num_workers: AI验证工作线程数量
        """
        self.model_path = Path(model_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.conf_thres = float(conf_thres)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建AI验证未通过的图片保存目录
        self.rejected_dir = self.output_dir.parent / 'rejected_smokefire'
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        
        # 临时目录用于存放裁切的检测框图片
        self.temp_dir = self.output_dir.parent / 'temp_smokefire_bbox'
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
            'total': 0,
            'detected': 0,
            'confirmed': 0,
            'rejected': 0,
            'queue_full_count': 0
        }
        
        logger.info(f"初始化完成: 模型={self.model_path}, 输入={self.input_dir}, 输出={self.output_dir}")
        logger.info(f"AI验证未通过保存目录: {self.rejected_dir}")
        logger.info(f"队列大小={max_queue_size}, AI验证工作线程={num_workers}")
    
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
            logger.info(f"执行AI分析命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"AI分析失败: {result.stderr}")
                return False
            
            output = result.stdout.strip()
            logger.info(f"AI分析输出: {output}")

            # 解析输出结果
            for line in output.split('\n'):
                s = line.strip()
                if s.startswith("[{\'text\': 'yes'}]") or s.startswith("{'text': 'yes'}"):
                    logger.info("AI确认检测到烟雾或火焰")
                    return True
                if s.startswith("[{\'text\': 'no'}]") or s.startswith("{'text': 'no'}"):
                    logger.info("AI确认未检测到烟雾或火焰")
                    return False

            if "yes" in output.lower() and "no" not in output.lower():
                logger.info("AI确认检测到烟雾或火焰")
                return True
            elif "no" in output.lower() and "yes" not in output.lower():
                logger.info("AI确认未检测到烟雾或火焰")
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
        """
        AI验证工作线程
        从队列中取出待验证的检测框进行AI分析
        """
        logger.info(f"AI验证工作线程 {worker_id} 启动")
        
        while not self.stop_event.is_set():
            try:
                # 从队列获取任务，超时1秒避免阻塞
                try:
                    task = self.verification_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                if task is None:  # 结束信号
                    self.verification_queue.task_done()
                    break
                
                frame, detections, image_name = task
                
                logger.info(f"[Worker-{worker_id}] 开始验证: {image_name}, 检测框数量: {len(detections)}")
                
                # 验证每个检测框
                confirmed_detections = []
                for idx, det in enumerate(detections):
                    x1, y1, x2, y2 = det['bbox']
                    
                    # 裁切检测框
                    bbox_img = frame[y1:y2, x1:x2]
                    if bbox_img.size == 0:
                        logger.warning(f"[Worker-{worker_id}] 检测框为空，跳过: {det}")
                        continue
                    
                    # 保存裁切的检测框图片
                    bbox_path = self.temp_dir / f"{image_name}_bbox_{idx}.jpg"
                    cv2.imwrite(str(bbox_path), bbox_img)
                    
                    # AI验证检测框
                    ai_confirmed = self.analyze_bbox_with_ai(bbox_path)
                    
                    # 删除临时检测框图片
                    if bbox_path.exists():
                        bbox_path.unlink()
                    
                    if ai_confirmed:
                        confirmed_detections.append(det)
                        logger.info(f"[Worker-{worker_id}] 检测框 {idx} AI确认通过")
                    else:
                        logger.info(f"[Worker-{worker_id}] 检测框 {idx} AI确认未通过")
                # 如果有检测框通过验证，保存结果图片
                if confirmed_detections:
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
                    
                    # 保存最终结果
                    final_path = self.output_dir / f"{Path(image_name).stem}.jpeg"
                    cv2.imwrite(str(final_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                    logger.info(f"[Worker-{worker_id}] AI确认通过 {len(confirmed_detections)}/{len(detections)} 个检测框: {image_name}")
                    self._update_stats('confirmed')
                
                # 如果有检测框但没有通过验证，或者部分通过验证，都要保存到rejected目录
                if len(detections) > 0:
                    # 绘制所有原始检测框（包括未通过验证的）
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
                    
                    # 保存到rejected目录（显示所有检测框的验证结果）
                    rejected_path = self.rejected_dir / f"{Path(image_name).stem}_all_detections.jpeg"
                    cv2.imwrite(str(rejected_path), rejected_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                    if len(confirmed_detections) == 0:
                        logger.info(f"[Worker-{worker_id}] 所有检测框都未通过AI验证，保存到rejected目录: {image_name}")
                        self._update_stats('rejected')
                    else:
                        logger.info(f"[Worker-{worker_id}] 部分检测框未通过验证，同时保存到rejected目录: {image_name}")
                
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

    def process_single_image(self, image_path):
        """
        处理单张图片（YOLO推理）
        检测到目标后将图片和检测结果放入队列，由工作线程进行AI验证
        
        Args:
            image_path: 图片路径
            
        Returns:
            bool: 是否检测到目标并成功加入验证队列
        """
        try:
            logger.info(f"处理图片: {image_path.name}")
            
            # 读取图片
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.error(f"无法读取图片: {image_path}")
                return False
            
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
                logger.info(f"未检测到目标: {image_path.name}")
                return False
            
            logger.info(f"检测到 {len(detections)} 个目标")
            self._update_stats('detected')
            
            # 将任务放入验证队列（阻塞模式，等待队列有空位）
            try:
                # 使用阻塞模式，超时60秒，避免YOLO推理过快导致队列满
                self.verification_queue.put((frame.copy(), detections, image_path.name), block=True, timeout=60)
                logger.info(f"已加入验证队列，当前队列长度: {self.verification_queue.qsize()}")
                return True
            except queue.Full:
                # 超时后队列仍然满，说明AI验证严重滞后
                logger.warning(f"验证队列已满且等待超时，跳过验证: {image_path.name}")
                self._update_stats('queue_full_count')
                return False
                
        except Exception as e:
            logger.error(f"处理图片时发生错误: {e}", exc_info=True)
            return False
    
    def run(self):
        """
        批量处理文件夹中的所有图片
        使用生产者-消费者模式，YOLO推理和AI验证并行执行
        """
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f'*{ext}'))
            image_files.extend(self.input_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            logger.warning(f"在 {self.input_dir} 中未找到图片文件")
            return
        
        self.stats['total'] = len(image_files)
        logger.info(f"找到 {len(image_files)} 张图片，开始批量处理...")
        
        # 加载模型
        logger.info(f"加载模型: {self.model_path}")
        self.model = YOLO(str(self.model_path)).cuda(device=1)
        
        # 启动AI验证工作线程
        self.start_workers()
        
        # YOLO推理（生产者）
        start_time = time.time()
        for idx, image_path in enumerate(image_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"YOLO推理进度: [{idx}/{len(image_files)}]")
            
            self.process_single_image(image_path)
            
            # 定期输出队列状态
            if idx % 10 == 0:
                stats = self._get_stats()
                logger.info(f"当前状态 - 检测到目标: {stats['detected']}, "
                          f"队列长度: {self.verification_queue.qsize()}, "
                          f"已确认: {stats['confirmed']}, "
                          f"已拒绝: {stats['rejected']}")
        
        yolo_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"YOLO推理完成，耗时: {yolo_time:.2f}秒")
        logger.info(f"等待AI验证队列处理完成...")
        
        # 等待队列中的所有任务完成
        self.verification_queue.join()
        
        # 停止工作线程
        self.stop_workers()
        
        total_time = time.time() - start_time
        
        # 输出最终统计结果
        stats = self._get_stats()
        logger.info(f"\n{'='*60}")
        logger.info(f"处理完成！")
        logger.info(f"总图片数: {stats['total']} 张")
        logger.info(f"检测到目标: {stats['detected']} 张")
        logger.info(f"AI确认通过: {stats['confirmed']} 张")
        logger.info(f"AI确认拒绝: {stats['rejected']} 张")
        if stats['queue_full_count'] > 0:
            logger.warning(f"队列满跳过: {stats['queue_full_count']} 张")
        logger.info(f"YOLO推理耗时: {yolo_time:.2f}秒")
        logger.info(f"总耗时: {total_time:.2f}秒")
        logger.info(f"YOLO平均速度: {stats['total']/yolo_time:.2f} 张/秒")
        logger.info(f"整体平均速度: {stats['total']/total_time:.2f} 张/秒")
        logger.info(f"通过验证的结果保存在: {self.output_dir}")
        logger.info(f"验证详情（包含未通过的）保存在: {self.rejected_dir}")
        logger.info(f"{'='*60}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='烟火检测图片批量推理')
    parser.add_argument('--input', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出结果文件夹路径')
    parser.add_argument('--model', type=str, default=None, help='模型路径（默认使用weights/smokefire.pt）')
    parser.add_argument('--conf', type=float, default=0.2, help='置信度阈值（默认0.2）')
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
    infer = SmokefireImageInfer(
        model_path=model_path,
        input_dir=args.input,
        output_dir=args.output,
        conf_thres=args.conf,
        max_queue_size=args.queue_size,
        num_workers=args.workers
    )
    
    infer.run()


if __name__ == "__main__":
    main()
