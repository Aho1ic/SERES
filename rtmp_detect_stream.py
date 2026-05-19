#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用目标检测 RTMP 推流脚本
读取视频流 -> YOLO检测 -> 绘制检测框 -> 推流到RTMP
"""

import argparse
import signal
import sys
import time
import subprocess
import threading
import queue

import cv2
import numpy as np
from ultralytics import YOLO


stop_flag = {'stop': False}


def run_stream(video_source: str, rtmp_url: str, model_path: str, conf_threshold: float = 0.5):
    """
    单路推流进程：读取视频源，YOLO检测并推送到RTMP
    
    Args:
        video_source: 输入视频流地址（RTMP/RTSP/本地文件）
        rtmp_url: 输出RTMP推流地址
        model_path: YOLO模型权重路径
        conf_threshold: 检测置信度阈值
    """
    # 加载YOLO模型
    print(f'加载模型: {model_path}')
    model = YOLO(model_path)
    print('模型加载完成')

    def open_capture_with_retries(src: str):
        """持续重连打开视频流"""
        attempt = 0
        backoff_base = 0.5
        max_backoff = 8.0
        while not stop_flag['stop']:
            cap_try = cv2.VideoCapture(src)
            if cap_try.isOpened():
                ret, test_frame = cap_try.read()
                if ret and test_frame is not None:
                    print(f'视频流连接成功: {src}')
                    return cap_try, test_frame
                cap_try.release()
            sleep_secs = min(backoff_base * (2 ** min(attempt, 6)), max_backoff)
            print(f'视频流连接失败，{sleep_secs:.1f}s后重试 (尝试#{attempt+1}): {src}', file=sys.stderr)
            time.sleep(sleep_secs)
            attempt += 1
        return None, None

    # 打开视频流
    cap, first_frame = open_capture_with_retries(video_source)
    if cap is None:
        print(f'视频流长时间无法打开，放弃: {video_source}', file=sys.stderr)
        return 1

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if first_frame is not None:
        height, width = first_frame.shape[:2]
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    print(f'视频参数: {width}x{height}, {fps}fps')

    def build_ffmpeg_cmd(out_width, out_height, out_fps, out_url, encoder: str):
        """构建FFmpeg推流命令"""
        return [
            'ffmpeg', '-y',
            '-loglevel', 'warning',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{out_width}x{out_height}', '-r', str(int(out_fps)),
            '-i', '-',
            '-c:v', encoder,
            '-preset', 'llhp' if encoder == 'h264_nvenc' else 'ultrafast',
            '-b:v', '2M',
            '-maxrate', '2M', '-bufsize', '1M',
            '-pix_fmt', 'yuv420p',
            '-g', str(int(out_fps) * 2),
            '-tune', 'zerolatency',
            '-fflags', '+genpts',
            '-flvflags', 'no_duration_filesize',
            '-rtmp_live', 'live',
            '-f', 'flv', out_url
        ]

    log_queue = queue.Queue()
    encoder_preferences = ['h264_nvenc', 'libx264']
    current_encoder_idx = 0

    def start_ffmpeg(encoder: str):
        """启动FFmpeg进程"""
        cmd = build_ffmpeg_cmd(width, height, fps, rtmp_url, encoder)
        print(f"FFmpeg命令: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        def monitor_ffmpeg_logs(p):
            """监控FFmpeg日志"""
            while not stop_flag['stop']:
                try:
                    line = p.stderr.readline()
                    if line:
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        if line_str:
                            log_queue.put(line_str)
                    else:
                        break
                except Exception:
                    break

        t = threading.Thread(target=lambda: monitor_ffmpeg_logs(proc), daemon=True)
        t.start()
        return proc, t

    # 启动FFmpeg
    ffmpeg_proc, log_thread = start_ffmpeg(encoder_preferences[current_encoder_idx])

    def _handle_sigterm(signum, frame):
        stop_flag['stop'] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    frame_count = 0
    restart_attempts = 0
    max_restarts = 5
    last_restart_ts = 0.0

    try:
        while not stop_flag['stop']:
            # 读取视频帧
            ret, frame = cap.read()
            if not ret or frame is None:
                print('视频流读取失败，尝试重连...', file=sys.stderr)
                try:
                    cap.release()
                except Exception:
                    pass
                cap, first_frame = open_capture_with_retries(video_source)
                if cap is None:
                    continue
                frame = first_frame

            # 调整帧大小
            if frame is not None and (frame.shape[1] != width or frame.shape[0] != height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            # 检查FFmpeg进程状态
            if ffmpeg_proc.poll() is not None:
                print(f'FFmpeg进程已退出，退出码: {ffmpeg_proc.returncode}', file=sys.stderr)
                if restart_attempts < max_restarts:
                    restart_attempts += 1
                    last_restart_ts = time.time()
                    try:
                        if ffmpeg_proc.stdin:
                            ffmpeg_proc.stdin.close()
                        ffmpeg_proc.wait(timeout=3)
                    except Exception:
                        try:
                            ffmpeg_proc.kill()
                        except Exception:
                            pass
                    
                    if restart_attempts >= 2 and current_encoder_idx == 0:
                        current_encoder_idx = 1
                        print('切换编码器到 libx264')
                    
                    ffmpeg_proc, log_thread = start_ffmpeg(encoder_preferences[current_encoder_idx])
                    continue
                else:
                    break

            # YOLO检测
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # 绘制检测结果
            annotated_frame = results[0].plot()

            # 推流
            try:
                ffmpeg_proc.stdin.write(annotated_frame.tobytes())
                if frame_count % 30 == 0:
                    ffmpeg_proc.stdin.flush()
                frame_count += 1

                if frame_count % 300 == 0:
                    print(f'已处理 {frame_count} 帧')

                # 重置重启计数
                if last_restart_ts and (time.time() - last_restart_ts > 60):
                    restart_attempts = 0
                    last_restart_ts = 0

            except BrokenPipeError as e:
                print(f'FFmpeg 管道已关闭: {e}', file=sys.stderr)
                if restart_attempts < max_restarts:
                    time.sleep(1)
                    restart_attempts += 1
                    last_restart_ts = time.time()
                    try:
                        if ffmpeg_proc.stdin:
                            ffmpeg_proc.stdin.close()
                        ffmpeg_proc.wait(timeout=3)
                    except Exception:
                        try:
                            ffmpeg_proc.kill()
                        except Exception:
                            pass
                    
                    if restart_attempts >= 2 and current_encoder_idx == 0:
                        current_encoder_idx = 1
                        print('切换编码器到 libx264')
                    
                    ffmpeg_proc, log_thread = start_ffmpeg(encoder_preferences[current_encoder_idx])
                    continue
                else:
                    print('重启次数过多，退出', file=sys.stderr)
                    break
            except Exception as e:
                print(f'推流错误: {e}', file=sys.stderr)
                continue

    finally:
        print(f'清理资源，共处理 {frame_count} 帧')

        # 保存FFmpeg日志
        try:
            with open(f'ffmpeg_debug_{int(time.time())}.log', 'w', encoding='utf-8') as f:
                while not log_queue.empty():
                    try:
                        log_line = log_queue.get_nowait()
                        f.write(log_line + '\n')
                    except queue.Empty:
                        break
            print('FFmpeg调试日志已保存')
        except Exception as e:
            print(f'保存日志失败: {e}')

        # 释放资源
        try:
            cap.release()
        except Exception as e:
            print(f'释放视频捕获器失败: {e}')

        try:
            if ffmpeg_proc and ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
        except Exception as e:
            print(f'关闭FFmpeg输入管道失败: {e}')

        try:
            if ffmpeg_proc:
                ffmpeg_proc.wait(timeout=10)
                print(f'FFmpeg进程已结束，退出码: {ffmpeg_proc.returncode}')
        except subprocess.TimeoutExpired:
            try:
                ffmpeg_proc.kill()
                print('FFmpeg进程已强制终止')
            except Exception as e:
                print(f'强制终止FFmpeg进程失败: {e}')
        except Exception as e:
            print(f'等待FFmpeg进程失败: {e}')


def main():
    parser = argparse.ArgumentParser(description='通用YOLO目标检测RTMP推流')
    parser.add_argument('--video-source', required=True, help='输入视频流地址（RTMP/RTSP/文件路径）')
    parser.add_argument('--rtmp-url', required=True, help='输出RTMP推流地址')
    parser.add_argument('--model-path', required=True, help='YOLO模型权重路径（.pt文件）')
    parser.add_argument('--conf', type=float, default=0.5, help='检测置信度阈值（默认0.5）')
    args = parser.parse_args()

    print('=' * 60)
    print('通用YOLO目标检测RTMP推流')
    print('=' * 60)
    print(f'输入源: {args.video_source}')
    print(f'输出流: {args.rtmp_url}')
    print(f'模型: {args.model_path}')
    print(f'置信度: {args.conf}')
    print('=' * 60)

    return run_stream(args.video_source, args.rtmp_url, args.model_path, conf_threshold=args.conf)


if __name__ == '__main__':
    sys.exit(main() or 0)
