import argparse
import signal
import sys
import time
import math
import subprocess
from multiprocessing import Process

import cv2
import numpy as np
from ultralytics import YOLO


# 头部关键点：0鼻子 1左眼 2右眼 3左耳 4右耳
# 5左肩 6右肩 7左肘 8右肘 9左手腕 10右手腕

def is_taking_photo(keypoints):
    """
    判断是否为拍照姿势（同一只手臂需同时满足以下三个条件）：
    1) 手腕到头部欧氏距离大于5像素
    2) 手腕高于肘部1像素（y坐标更小）
    3) 手腕-肘部 与 肩膀-肘部 的夹角大于45°
    keypoints: 关键点坐标，格式为[[x0, y0], [x1, y1], ..., [x16, y16]]
    """
    # COCO顺序：0鼻子 1左眼 2右眼 3左耳 4右耳 5左肩 6右肩 7左肘 8右肘 9左手腕 10右手腕
    head_candidates = [0, 1, 2, 3, 4]
    nose = None
    for idx in head_candidates:
        pt = np.array(keypoints[idx])
        if not np.all(pt == 0):
            nose = pt
            break
    if nose is None:
        left_shoulder_tmp = np.array(keypoints[5])
        right_shoulder_tmp = np.array(keypoints[6])
        if (left_shoulder_tmp.shape == (2,) and right_shoulder_tmp.shape == (2,)
            and not np.all(left_shoulder_tmp == 0) and not np.all(right_shoulder_tmp == 0)):
            mid = (left_shoulder_tmp + right_shoulder_tmp) / 2.0
            nose = np.array([mid[0], max(mid[1] - 10, 0)])
            print('头部关键点缺失，使用肩膀中点上移10像素作为替代')
        else:
            print('缺少关键点，无法判断姿态')
            return False

    left_shoulder = np.array(keypoints[5])
    right_shoulder = np.array(keypoints[6])
    left_elbow = np.array(keypoints[7])
    right_elbow = np.array(keypoints[8])
    left_wrist = np.array(keypoints[9])
    right_wrist = np.array(keypoints[10])

    dist_threshold = 5
    y_diff_threshold = 1
    angle_threshold_deg = 45.0

    def angle_at_elbow(wrist, elbow, shoulder):
        v1 = wrist - elbow
        v2 = shoulder - elbow
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cosang = np.dot(v1, v2) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    def valid_point(p):
        return p.shape == (2,) and not np.all(p == 0)

    left_ok = False
    if all(map(valid_point, [left_wrist, left_elbow, left_shoulder])):
        lw_dist = np.linalg.norm(left_wrist - nose)
        left_height_ok = left_wrist[1] < left_elbow[1] - y_diff_threshold
        left_angle = angle_at_elbow(left_wrist, left_elbow, left_shoulder)
        left_angle_ok = (left_angle is not None and left_angle > angle_threshold_deg)
        left_ok = (lw_dist > dist_threshold) and left_height_ok and left_angle_ok

    right_ok = False
    if all(map(valid_point, [right_wrist, right_elbow, right_shoulder])):
        rw_dist = np.linalg.norm(right_wrist - nose)
        right_height_ok = right_wrist[1] < right_elbow[1] - y_diff_threshold
        right_angle = angle_at_elbow(right_wrist, right_elbow, right_shoulder)
        right_angle_ok = (right_angle is not None and right_angle > angle_threshold_deg)
        right_ok = (rw_dist > dist_threshold) and right_height_ok and right_angle_ok

    return bool(left_ok or right_ok)


stop_flag = {'stop': False}


def run_stream(video_source: str, rtmp_url: str, model_path: str, conf_threshold: float = 0.5):
    """单路推流进程：读取 RTMP/视频源，推理并推送到 rtmp_url。"""
    model = YOLO(model_path)

    def open_capture_with_retries(src: str):
        """带指数退避的持续重连打开视频流，返回(cap, first_frame)。"""
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

    cap, first_frame = open_capture_with_retries(video_source)
    if cap is None:
        print(f'视频流长时间无法打开，放弃: {video_source}', file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if first_frame is not None:
        height, width = first_frame.shape[:2]
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    def build_ffmpeg_cmd(out_width, out_height, out_fps, out_url, encoder: str):
        return [
            'ffmpeg', '-y',
            '-loglevel', 'debug',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{out_width}x{out_height}', '-r', str(int(out_fps)),
            '-i', '-',
            '-c:v', encoder,
            '-preset', 'llhp' if encoder == 'h264_nvenc' else 'ultrafast',
            '-b:v', '1M',
            '-maxrate', '1M', '-bufsize', '500K',
            '-pix_fmt', 'yuv420p',
            '-g', str(int(out_fps) * 2),
            '-tune', 'zerolatency',
            '-fflags', '+genpts',
            '-flvflags', 'no_duration_filesize',
            '-rtmp_live', 'live',
            '-f', 'flv', out_url
        ]

    import threading
    import queue

    log_queue = queue.Queue()

    encoder_preferences = ['h264_nvenc', 'libx264']
    current_encoder_idx = 0

    def start_ffmpeg(encoder: str):
        cmd = build_ffmpeg_cmd(width, height, fps, rtmp_url, encoder)
        print(f"FFmpeg命令: {' '.join(cmd)}")
        print(f"视频参数: {width}x{height}, {fps}fps")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        def monitor_ffmpeg_logs(p):
            """监控FFmpeg的日志输出"""
            while not stop_flag['stop']:
                try:
                    line = p.stderr.readline()
                    if line:
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        if line_str:
                            print(f"[FFmpeg] {line_str}")
                            log_queue.put(line_str)
                    else:
                        break
                except Exception as e:
                    print(f"FFmpeg日志监控错误: {e}")
                    break

        t = threading.Thread(target=lambda: monitor_ffmpeg_logs(proc), daemon=True)
        t.start()
        return proc, t

    ffmpeg_proc, log_thread = start_ffmpeg(encoder_preferences[current_encoder_idx])

    def _handle_sigterm(signum, frame):
        stop_flag['stop'] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    frame_count = 0
    last_error_time = 0
    error_count = 0

    restart_attempts = 0
    max_restarts = 5
    backoff_base = 1.0
    last_restart_ts = 0.0

    try:
        while not stop_flag['stop']:
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

            if frame is not None and (frame.shape[1] != width or frame.shape[0] != height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            if ffmpeg_proc.poll() is not None:
                print(f'FFmpeg进程已退出，退出码: {ffmpeg_proc.returncode}', file=sys.stderr)
                if restart_attempts < max_restarts:
                    restart_attempts += 1
                    last_restart_ts = time.time()
                    try:
                        try:
                            if ffmpeg_proc.stdin:
                                ffmpeg_proc.stdin.close()
                        except Exception:
                            pass
                        try:
                            ffmpeg_proc.wait(timeout=3)
                        except Exception:
                            try:
                                ffmpeg_proc.kill()
                            except Exception:
                                pass
                        if restart_attempts >= 2 and current_encoder_idx == 0:
                            current_encoder_idx = 1
                            print('切换编码器到 libx264 后重试推流')
                        ffmpeg_proc, log_thread = start_ffmpeg(encoder_preferences[current_encoder_idx])
                        continue
                    except Exception as e:
                        print(f'重启FFmpeg失败: {e}', file=sys.stderr)
                        break
                else:
                    break

            results = model(frame, conf=conf_threshold)
            for result in results:
                if result.keypoints is not None and result.boxes is not None:
                    keypoints = result.keypoints.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for person_keypoints, box in zip(keypoints, boxes):
                        valid_keypoints = []
                        for kp in person_keypoints:
                            if kp[2] > 0.5:
                                valid_keypoints.append([kp[0], kp[1]])
                            else:
                                valid_keypoints.append([0, 0])
                        if is_taking_photo(valid_keypoints):
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, 'taking_photo', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            for x, y in valid_keypoints:
                                if x > 0 and y > 0:
                                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
                if frame_count % 10 == 0:
                    ffmpeg_proc.stdin.flush()
                frame_count += 1

                if frame_count % 100 == 0:
                    print(f'已处理 {frame_count} 帧，FFmpeg进程状态: {ffmpeg_proc.poll()}')

                if last_restart_ts and (time.time() - last_restart_ts > 60):
                    restart_attempts = 0
                    last_restart_ts = 0

            except BrokenPipeError as e:
                current_time = time.time()
                error_count += 1
                print(f'FFmpeg 管道已关闭 (错误 #{error_count}): {e}', file=sys.stderr)

                if restart_attempts < max_restarts:
                    sleep_secs = backoff_base * (2 ** restart_attempts)
                    time.sleep(min(sleep_secs, 5))
                    restart_attempts += 1
                    last_restart_ts = current_time
                    try:
                        try:
                            if ffmpeg_proc.stdin:
                                ffmpeg_proc.stdin.close()
                        except Exception:
                            pass
                        try:
                            ffmpeg_proc.wait(timeout=3)
                        except Exception:
                            try:
                                ffmpeg_proc.kill()
                            except Exception:
                                pass
                        if restart_attempts >= 2 and current_encoder_idx == 0:
                            current_encoder_idx = 1
                            print('Broken pipe 后切换编码器到 libx264')
                        ffmpeg_proc, log_thread = start_ffmpeg(encoder_preferences[current_encoder_idx])
                        continue
                    except Exception as re:
                        print(f'重启FFmpeg失败: {re}', file=sys.stderr)
                        break
                else:
                    print('重启次数过多，退出', file=sys.stderr)
                    break
            except Exception as e:
                print(f'推流错误: {e}', file=sys.stderr)
                continue
    finally:
        print(f'清理资源，共处理 {frame_count} 帧')

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

        try:
            cap.release()
            print('视频捕获器已释放')
        except Exception as e:
            print(f'释放视频捕获器失败: {e}')

        try:
            if ffmpeg_proc and ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
                print('FFmpeg输入管道已关闭')
        except Exception as e:
            print(f'关闭FFmpeg输入管道失败: {e}')

        try:
            if ffmpeg_proc:
                print(f'等待FFmpeg进程结束...')
                ffmpeg_proc.wait(timeout=10)
                print(f'FFmpeg进程已结束，退出码: {ffmpeg_proc.returncode}')
        except subprocess.TimeoutExpired:
            print('FFmpeg进程等待超时，强制终止')
            try:
                ffmpeg_proc.kill()
                print('FFmpeg进程已强制终止')
            except Exception as e:
                print(f'强制终止FFmpeg进程失败: {e}')
        except Exception as e:
            print(f'等待FFmpeg进程失败: {e}')


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 姿势检测 RTMP 推流')
    parser.add_argument('--video-source', required=True, help='输入视频流，如 rtmp://host/app/stream')
    parser.add_argument('--rtmp-url', required=True, help='输出 RTMP 推流地址')
    parser.add_argument('--model-path', default='/home/algorithm/chongqing/weights/spycam.pt', help='YOLO 模型权重路径')
    parser.add_argument('--conf', type=float, default=0.5, help='检测置信度阈值')
    args = parser.parse_args()

    return run_stream(args.video_source, args.rtmp_url, args.model_path, conf_threshold=args.conf)


if __name__ == '__main__':
    sys.exit(main() or 0) 