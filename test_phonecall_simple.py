#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的打电话姿势检测测试脚本
直接指定MP4视频路径进行测试
支持启用/禁用AI二次验证
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.insert(0, str(ROOT))

from interface.phonecall_detect import PhoneCallInfer
from loguru import logger

def test_video(video_path, enable_ai=True, upload_url=None, model_path=None):
    """
    测试MP4视频
    
    Args:
        video_path: MP4视频文件路径
        enable_ai: 是否启用AI二次验证（默认True）
        upload_url: 上传URL（默认None，不上传）
        model_path: 模型文件路径（默认使用weights/phonecall.pt）
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        return
    
    # 配置参数
    if model_path is None:
        model_path = ROOT / 'weights/phonecall.pt'
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("请确保模型文件路径正确")
        return
    
    # 检查AI分析脚本是否存在
    ai_script_path = ROOT / 'omni' / 'python' / 'phonecall_analysis.py'
    if enable_ai and not ai_script_path.exists():
        logger.warning(f"AI分析脚本不存在: {ai_script_path}")
        logger.warning("将禁用AI二次验证功能")
        enable_ai = False
    
    upload_dir = ROOT / 'upload/phonecall'
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("打电话姿势检测测试")
    logger.info("=" * 60)
    logger.info(f"视频路径: {video_path}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"上传目录: {upload_dir}")
    logger.info(f"上传URL: {upload_url or '不上传'}")
    logger.info(f"AI二次验证: {'启用' if enable_ai else '禁用'}")
    if enable_ai:
        logger.info(f"AI脚本路径: {ai_script_path}")
    logger.info("=" * 60)
    logger.info("按 Ctrl+C 停止测试")
    logger.info("=" * 60)
    
    # 创建检测器实例
    infer = PhoneCallInfer(
        model_path=model_path,
        video_source=str(video_path),  # 使用MP4文件路径
        upload_url=upload_url,
        upload_dir=upload_dir,
        task_name="打电话姿势检测测试",
        task_id="test_001",
        box_id="test_box",
        app_id="phoneCall",
        video_url=str(video_path),
        type=1
    )
    
    # 如果禁用AI验证，修改实例方法
    if not enable_ai:
        logger.warning("AI二次验证已禁用，所有检测结果将直接保存")
        # 替换AI验证方法为始终返回True
        infer.analyze_image_with_ai = lambda *args, **kwargs: True
    
    # 启动检测
    infer.running.set()
    try:
        infer.run_detection()
    except KeyboardInterrupt:
        logger.info("\n收到停止信号，正在退出...")
        infer.running.clear()
    except Exception as e:
        logger.error(f"检测过程出错: {e}", exc_info=True)
    finally:
        logger.info("测试完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="打电话姿势检测测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（启用AI验证）
  python3 %(prog)s /path/to/video.mp4
  
  # 禁用AI二次验证（仅使用YOLO检测）
  python3 %(prog)s /path/to/video.mp4 --no-ai
  
  # 指定自定义模型路径
  python3 %(prog)s /path/to/video.mp4 --model /path/to/model.pt
  
  # 启用上传功能
  python3 %(prog)s /path/to/video.mp4 --upload-url http://example.com/upload
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='视频文件路径（支持MP4、AVI等格式）'
    )
    
    parser.add_argument(
        '--no-ai',
        action='store_true',
        help='禁用AI二次验证（默认启用）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='模型文件路径（默认: weights/phonecall.pt）'
    )
    
    parser.add_argument(
        '--upload-url',
        type=str,
        default=None,
        help='上传URL（默认不上传）'
    )
    
    args = parser.parse_args()
    
    # 执行测试
    test_video(
        video_path=args.video_path,
        enable_ai=not args.no_ai,
        upload_url=args.upload_url,
        model_path=args.model
    )
