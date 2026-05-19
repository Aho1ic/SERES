# -*- coding: UTF-8 -*-
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
import threading
import requests
import time
import os
import json
import logging
import logging.handlers
from interface.solarpanel_detect import SolarPanelInfer
from interface.spycam_detect import SpyCamInfer
from interface.phonecall_detect import PhoneCallInfer
from interface.utils import delete_files
from queue import Queue, PriorityQueue, Empty, Full
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import dataclasses
from typing import Optional
import uuid
from PIL import Image
from PIL.ExifTags import TAGS
import subprocess
import sys
import re

from multiprocessing import Process
from config import (
    API_BASE_URL, UPLOAD_URL, AICALLERS_POLLING_URL_1, AICALLERS_POLLING_URL_2,
    TASK_QUEUE_MAXSIZE, WORKER_COUNT, LLM_QUEUE_MAXSIZE, LLM_WORKER_COUNT,
    LLM_MONITOR_INTERVAL, MAX_REQUESTS_PER_SECOND, POLLING_INTERVAL,
    WEIGHTS_DIR, get_upload_url, get_auth_headers, get_rtmp_source_url, get_rtmp_output_url
)

def analyze_image_with_ai(image_path):
    """
    调用smokefire_analysis.py对图片进行AI分析
    返回True表示检测到烟雾或火焰，False表示未检测到
    """
    try:
        image_analysis_path = ROOT / 'omni' / 'python' / 'smokefire_analysis.py'
        
        if not image_analysis_path.exists():
            logger.error(f"smokefire_analysis.py文件不存在: {image_analysis_path}")
            return False
        cmd = [sys.executable, str(image_analysis_path), str(image_path)]
        logger.info(f"执行AI分析命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.error(f"AI分析失败: {result.stderr}")
            return False
        output = result.stdout.strip()
        logger.info(f"AI分析输出: {output}")

        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("[{'text': 'yes'}]"):
                logger.info("AI确认检测到烟雾或火焰")
                return True
            elif line.startswith("[{'text': 'no'}]"):
                logger.info("AI确认未检测到烟雾或火焰")
                return False
            elif line.startswith("{'text': 'yes'}"):
                logger.info("AI确认检测到烟雾或火焰")
                return True
            elif line.startswith("{'text': 'no'}"):
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
        logger.error(f"AI分析异常: {e}")
        return False

@dataclasses.dataclass
class ProcessingTask:
    """处理任务数据结构"""
    priority: int = 0
    timestamp: float = dataclasses.field(default_factory=time.time)
    jpeg_path: str = ""
    boxId: str = ""
    task_id: str = ""
    img_url: str = ""
    video_url: str = ""
    thirdGroupId: str = ""
    
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

@dataclasses.dataclass
class LLMVerificationTask:
    """大模型二次验证任务"""
    event_id: str
    out_dir: str
    orig_jpg_path: str
    result_jpeg_path: str
    json_path: str
    source_image_path: str = ""
    task_id: str = ""
    thirdGroupId: str = ""
    timestamp: float = dataclasses.field(default_factory=time.time)

global_yolo_model = None
model_lock = threading.Lock()

global_smokefire_model = None
smokefire_model_lock = threading.Lock()

test_polling_enabled = False
test_polling_thread = None
TEST_POLLING_INTERVAL = POLLING_INTERVAL
TEST_URL = AICALLERS_POLLING_URL_1
task_queue = PriorityQueue(maxsize=TASK_QUEUE_MAXSIZE)
executor = ThreadPoolExecutor(max_workers=WORKER_COUNT)
processing_files = set()
processing_lock = threading.Lock()

LLM_QUEUE_MAXSIZE = LLM_QUEUE_MAXSIZE
LLM_WORKER_COUNT = LLM_WORKER_COUNT
LLM_MONITOR_INTERVAL = LLM_MONITOR_INTERVAL
llm_verification_queue = Queue(maxsize=LLM_QUEUE_MAXSIZE)
llm_worker_start_lock = threading.Lock()
llm_workers_started = False
llm_monitor_started = False

request_count = 0
request_lock = threading.Lock()
last_reset_time = time.time()

queue_stats = {
    'total_processed': 0,
    'total_failed': 0,
    'queue_size': 0,
    'last_update': time.time()
}
stats_lock = threading.Lock()

llm_stats = {
    'total_submitted': 0,
    'total_processed': 0,
    'total_failed': 0,
    'total_confirmed': 0,
    'total_rejected': 0,
    'total_report_success': 0,
    'total_report_failed': 0,
    'total_fallback': 0,
    'queue_size': 0,
    'queue_full': False,
    'in_flight': 0,
    'last_latency_ms': 0.0,
    'total_latency_ms': 0.0,
    'last_update': time.time(),
    'last_error': ''
}
llm_stats_lock = threading.Lock()

aicallers_polling_enabled = False
aicallers_polling_thread = None
aicallers_last_request_time = 0
AICALLERS_POLLING_INTERVAL = 2
AICALLERS_URL = AICALLERS_POLLING_URL_2

stream_processes_lock = threading.Lock()
stream_processes = {}

def _strip_query_params(url: str) -> str:
    """移除URL中的查询参数（问号后的部分）。"""
    try:
        if not isinstance(url, str):
            return url
        qpos = url.find('?')
        return url if qpos == -1 else url[:qpos]
    except Exception:
        return url

def get_yolo_model():
    """获取全局YOLO模型实例"""
    global global_yolo_model
    if global_yolo_model is None:
        with model_lock:
            if global_yolo_model is None:
                weights_path = str(ROOT / 'weights' / 'solarpanel.pt')
                global_yolo_model = YOLO(weights_path)
    return global_yolo_model

def get_smokefire_model():
    """获取全局smokefire YOLO模型实例"""
    global global_smokefire_model
    if global_smokefire_model is None:
        with smokefire_model_lock:
            if global_smokefire_model is None:
                weights_path = str(ROOT / 'weights' / 'smokefire.pt')
                global_smokefire_model = YOLO(weights_path)
    return global_smokefire_model

def extract_gps_from_image(image_path):
    """
    从图片中提取GPS经纬度信息
    :param image_path: 图片路径
    :return: 经纬度字符串，格式为"经度,纬度"，如果提取失败返回空字符串
    """
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if exif is None:
                logger.warning(f"图片没有EXIF信息: {image_path}")
                return ""

            gps_info = {}
            for tag_id in exif:
                tag = TAGS.get(tag_id, tag_id)
                data = exif.get(tag_id)
                if tag == "GPSInfo":
                    gps_info = data
                    break
            
            if not gps_info:
                logger.warning(f"图片没有GPS信息: {image_path}")
                return ""

            lat = gps_info.get(2)
            lon = gps_info.get(4)
            
            if lat and lon:
                lat_ref = gps_info.get(1, 'N')
                lon_ref = gps_info.get(3, 'E')

                def dms_to_decimal(dms, ref):
                    degrees = float(dms[0])
                    minutes = float(dms[1])
                    seconds = float(dms[2])
                    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                    if ref in ['S', 'W']:
                        decimal = -decimal
                    return decimal
                
                latitude = dms_to_decimal(lat, lat_ref)
                longitude = dms_to_decimal(lon, lon_ref)
                
                location_str = f"{longitude:.6f},{latitude:.6f}"
                logger.info(f"提取到GPS信息: {location_str}")
                return location_str
            else:
                logger.warning(f"GPS信息不完整: {image_path}")
                return ""
                
    except Exception as e:
        logger.error(f"提取GPS信息失败: {image_path}, 错误: {e}")
        return ""

def update_queue_stats(processed=0, failed=0):
    """更新队列统计信息"""
    with stats_lock:
        queue_stats['total_processed'] += processed
        queue_stats['total_failed'] += failed
        queue_stats['queue_size'] = task_queue.qsize()
        queue_stats['last_update'] = time.time()

def update_llm_stats(
    submitted=0,
    processed=0,
    failed=0,
    confirmed=0,
    rejected=0,
    report_success=0,
    report_failed=0,
    fallback=0,
    in_flight_delta=0,
    latency_ms=None,
    last_error=None
):
    """更新大模型验证队列统计信息"""
    with llm_stats_lock:
        llm_stats['total_submitted'] += submitted
        llm_stats['total_processed'] += processed
        llm_stats['total_failed'] += failed
        llm_stats['total_confirmed'] += confirmed
        llm_stats['total_rejected'] += rejected
        llm_stats['total_report_success'] += report_success
        llm_stats['total_report_failed'] += report_failed
        llm_stats['total_fallback'] += fallback
        llm_stats['in_flight'] = max(0, llm_stats['in_flight'] + in_flight_delta)
        llm_stats['queue_size'] = llm_verification_queue.qsize()
        llm_stats['queue_full'] = llm_verification_queue.full()
        if latency_ms is not None:
            llm_stats['last_latency_ms'] = round(float(latency_ms), 3)
            llm_stats['total_latency_ms'] += float(latency_ms)
        if last_error is not None:
            llm_stats['last_error'] = str(last_error)
        llm_stats['last_update'] = time.time()

def get_llm_status_snapshot():
    """获取大模型验证队列实时状态"""
    with llm_stats_lock:
        snapshot = llm_stats.copy()
        snapshot['queue_size'] = llm_verification_queue.qsize()
        snapshot['queue_full'] = llm_verification_queue.full()
        snapshot['max_queue_size'] = LLM_QUEUE_MAXSIZE
        snapshot['worker_count'] = LLM_WORKER_COUNT
        snapshot['monitor_interval'] = LLM_MONITOR_INTERVAL
        snapshot['current_time'] = time.time()
        processed = snapshot.get('total_processed', 0)
        total_latency = snapshot.pop('total_latency_ms', 0.0)
        snapshot['avg_latency_ms'] = round(total_latency / processed, 3) if processed else 0.0
        return snapshot

def get_interface_queue_status_snapshot(stale_seconds=None):
    """读取interface子进程写出的共享队列指标文件。"""
    stale_seconds = float(stale_seconds if stale_seconds is not None else os.getenv("INTERFACE_METRICS_STALE_SECONDS", "120"))
    metrics_dir = Path(os.getenv("INTERFACE_METRICS_DIR", str(ROOT / "logs" / "interface_metrics")))
    now = time.time()
    summary = {
        "metrics_dir": str(metrics_dir),
        "stale_seconds": stale_seconds,
        "process_count": 0,
        "active_process_count": 0,
        "total_verification_queue_length": 0,
        "total_report_queue_length": 0,
        "items": []
    }

    if not metrics_dir.exists():
        return summary

    for metrics_file in sorted(metrics_dir.glob("*.json")):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            updated_at = float(payload.get("updated_at", metrics_file.stat().st_mtime))
            age_seconds = max(0.0, now - updated_at)
            is_stale = age_seconds > stale_seconds
            metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
            verification_queue_length = int(metrics.get("verification_queue_length", 0) or 0)
            report_queue_length = int(metrics.get("report_queue_length", 0) or 0)

            item = {
                "file": metrics_file.name,
                "task_type": payload.get("task_type", ""),
                "app_id": payload.get("app_id", ""),
                "task_id": payload.get("task_id", ""),
                "task_name": payload.get("task_name", ""),
                "box_id": payload.get("box_id", ""),
                "pid": payload.get("pid"),
                "running": payload.get("running", False),
                "reason": payload.get("reason", ""),
                "updated_at": updated_at,
                "age_seconds": round(age_seconds, 3),
                "stale": is_stale,
                "verification_worker_count": payload.get("verification_worker_count", 0),
                "report_worker_count": payload.get("report_worker_count", 0),
                "verification_queue_maxsize": payload.get("verification_queue_maxsize", 0),
                "report_queue_maxsize": payload.get("report_queue_maxsize", 0),
                "metrics": metrics
            }
            summary["items"].append(item)
            summary["process_count"] += 1

            if not is_stale:
                summary["active_process_count"] += 1
                summary["total_verification_queue_length"] += verification_queue_length
                summary["total_report_queue_length"] += report_queue_length
        except Exception as e:
            logger.warning(f"读取interface队列指标失败: {metrics_file}, 错误: {e}")

    return summary

def upload_event_files(save_dir, event_id, upload_url=None):
    """上报事件文件：json、原图jpg、检测结果jpeg。"""
    upload_url = upload_url or UPLOAD_URL
    params = {}
    headers = {}
    save_dir = Path(save_dir)
    json_file = save_dir / (event_id + '.json')
    jpg_file = save_dir / (event_id + '.jpg')
    jpeg_file = save_dir / (event_id + '.jpeg')

    try:
        with open(json_file, 'rb') as f:
            resp = requests.post(upload_url, params=params, files={'file': (json_file.name, f)}, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.error(f"推送JSON失败: {resp.status_code} {resp.text}")
            return False
        logger.info(f"推送JSON成功: {json_file}")
    except Exception as e:
        logger.error(f"推送JSON异常: {e}")
        return False

    time.sleep(2)

    try:
        with open(jpg_file, 'rb') as f:
            resp = requests.post(upload_url, params=params, files={'file': (jpg_file.name, f)}, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.error(f"推送JPG失败: {resp.status_code} {resp.text}")
            return False
        logger.info(f"推送JPG成功: {jpg_file}")
    except Exception as e:
        logger.error(f"推送JPG异常: {e}")
        return False

    try:
        with open(jpeg_file, 'rb') as f:
            resp = requests.post(upload_url, params=params, files={'file': (jpeg_file.name, f)}, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.error(f"推送JPEG失败: {resp.status_code} {resp.text}")
            return False
        logger.info(f"推送JPEG成功: {jpeg_file}")
    except Exception as e:
        logger.error(f"推送JPEG异常: {e}")
        return False

    logger.info(f"推送成功，保留本地上报文件: {json_file}, {jpg_file}, {jpeg_file}")
    return True

def process_llm_verification_task(task: LLMVerificationTask):
    """执行大模型二次验证，确认通过后上报事件。"""
    start_time = time.time()
    update_llm_stats(in_flight_delta=1)
    try:
        queue_size = llm_verification_queue.qsize()
        logger.info(
            f"大模型二次验证开始: event_id={task.event_id}, "
            f"queue_size={queue_size}, in_flight={get_llm_status_snapshot().get('in_flight')}"
        )
        ai_confirmed = analyze_image_with_ai(task.orig_jpg_path)
        latency_ms = (time.time() - start_time) * 1000

        if ai_confirmed:
            update_llm_stats(processed=1, confirmed=1, latency_ms=latency_ms, last_error='')
            logger.info(f"大模型确认通过，开始事件上报: event_id={task.event_id}, latency_ms={latency_ms:.1f}")
            upload_ok = upload_event_files(task.out_dir, task.event_id)
            if upload_ok:
                update_llm_stats(report_success=1)
            else:
                update_llm_stats(report_failed=1, last_error=f"事件上报失败: {task.event_id}")
                logger.warning(f"事件上报失败，保留文件便于重试: event_id={task.event_id}, out_dir={task.out_dir}")
        else:
            update_llm_stats(processed=1, rejected=1, latency_ms=latency_ms, last_error='')
            logger.info(
                f"大模型确认未通过，保留YOLO命中帧用于排查: "
                f"event_id={task.event_id}, orig={task.orig_jpg_path}, result={task.result_jpeg_path}"
            )
    except Exception as e:
        update_llm_stats(failed=1, last_error=e)
        logger.error(f"大模型二次验证任务异常: event_id={task.event_id}, 错误: {e}", exc_info=True)
    finally:
        update_llm_stats(in_flight_delta=-1)

def llm_queue_worker():
    """大模型验证队列工作线程"""
    while True:
        try:
            task = llm_verification_queue.get(timeout=1)
            try:
                update_llm_stats()
                process_llm_verification_task(task)
            finally:
                llm_verification_queue.task_done()
                update_llm_stats()
        except Empty:
            time.sleep(0.1)
        except Exception as e:
            update_llm_stats(failed=1, last_error=e)
            logger.error(f"大模型验证队列工作线程异常: {type(e).__name__}: {e}", exc_info=True)
            time.sleep(0.5)

def llm_monitor_worker():
    """周期性输出大模型负载监控日志"""
    while True:
        time.sleep(LLM_MONITOR_INTERVAL)
        stats = get_llm_status_snapshot()
        logger.info(
            "大模型负载监控: "
            f"queue_size={stats['queue_size']}/{stats['max_queue_size']}, "
            f"in_flight={stats['in_flight']}, "
            f"submitted={stats['total_submitted']}, "
            f"processed={stats['total_processed']}, "
            f"confirmed={stats['total_confirmed']}, "
            f"rejected={stats['total_rejected']}, "
            f"failed={stats['total_failed']}, "
            f"report_success={stats['total_report_success']}, "
            f"report_failed={stats['total_report_failed']}, "
            f"avg_latency_ms={stats['avg_latency_ms']}"
        )

def start_llm_verification_workers():
    """启动大模型验证与负载监控后台线程"""
    global llm_workers_started, llm_monitor_started
    with llm_worker_start_lock:
        if not llm_workers_started:
            for idx in range(max(1, LLM_WORKER_COUNT)):
                worker_thread = threading.Thread(target=llm_queue_worker, name=f"llm-verification-{idx}", daemon=True)
                worker_thread.start()
            llm_workers_started = True
            logger.info(f"大模型验证队列已启动: workers={LLM_WORKER_COUNT}, max_queue_size={LLM_QUEUE_MAXSIZE}")

        if not llm_monitor_started:
            monitor_thread = threading.Thread(target=llm_monitor_worker, name="llm-load-monitor", daemon=True)
            monitor_thread.start()
            llm_monitor_started = True
            logger.info(f"大模型负载监控日志已启动: interval={LLM_MONITOR_INTERVAL}s")

def submit_llm_verification_task(task: LLMVerificationTask):
    """提交大模型二次验证任务，队列满时降级为线程池直接执行。"""
    start_llm_verification_workers()
    update_llm_stats(submitted=1)
    try:
        llm_verification_queue.put(task, timeout=5)
        update_llm_stats()
        logger.info(
            f"YOLO命中帧已提交大模型验证队列: event_id={task.event_id}, "
            f"queue_size={llm_verification_queue.qsize()}/{LLM_QUEUE_MAXSIZE}"
        )
        return True
    except Full:
        update_llm_stats(fallback=1, last_error="大模型验证队列已满")
        logger.warning(
            f"大模型验证队列已满，使用线程池直接验证: event_id={task.event_id}, "
            f"queue_size={llm_verification_queue.qsize()}/{LLM_QUEUE_MAXSIZE}"
        )
        executor.submit(process_llm_verification_task, task)
        return False
    except Exception as e:
        update_llm_stats(failed=1, last_error=e)
        logger.error(f"提交大模型验证任务失败: event_id={task.event_id}, 错误: {e}", exc_info=True)
        return False

def queue_worker():
    """队列工作线程"""
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break
            try:
                has_event = process_downloaded_jpeg(task.jpeg_path, task.boxId, task.task_id, task.video_url, task.thirdGroupId)
                update_queue_stats(processed=1)
                if has_event:
                    logger.info(f"任务处理成功并产生检测事件: {task.jpeg_path}")
            except Exception as e:
                update_queue_stats(failed=1)
                logger.error(f"任务处理失败: {task.jpeg_path}, 错误: {e}")
            finally:
                task_queue.task_done()
        except Empty:
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"队列工作线程异常: {type(e).__name__}: {e}", exc_info=True)
            time.sleep(0.1)
for _ in range(5):
    worker_thread = threading.Thread(target=queue_worker, daemon=True)
    worker_thread.start()

def aicallers_polling_worker():
    """定时轮询aicallers接口的工作线程"""
    global aicallers_polling_enabled, aicallers_last_request_time
    
    while aicallers_polling_enabled:
        try:
            current_time = time.time()
            if current_time - aicallers_last_request_time >= AICALLERS_POLLING_INTERVAL:
                aicallers_last_request_time = current_time
                try:
                    logger.debug(f"向aicallers接口发送POST请求: {AICALLERS_URL}")
                    response = requests.post(AICALLERS_URL, timeout=10)
                    required_fields = ['boxId', 'task_id', 'fileUrl', 'droneSn']
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"aicallers接口返回数据: {data}")
                        if 'data' in data and isinstance(data['data'], dict):
                            data_content = data['data']
                            if all(field in data_content for field in required_fields):
                                boxId = data_content.get('boxId')
                                task_id = data_content.get('task_id')
                                file_url = data_content.get('fileUrl')
                                droneSn = data_content.get('droneSn')
                                thirdGroupId = data_content.get('thirdGroupId', '')
                                logger.info(f"处理aicallers数据: boxId={boxId}, task_id={task_id}, fileUrl={file_url}, droneSn={droneSn}, thirdGroupId={thirdGroupId}")

                                task_file = ROOT / 'task' / f'{boxId}_{task_id}.json'
                                if not os.path.exists(task_file):
                                    logger.warning(f"aicallers任务文件不存在，跳过处理: {task_file}")
                                    continue

                                try:
                                    with open(task_file, 'r', encoding='utf-8') as f:
                                        task_info = json.load(f)
                                    if task_info.get('status') != 'running':
                                        logger.info(f"aicallers任务状态不是running，跳过处理: {task_file}, status={task_info.get('status')}")
                                        continue
                                except Exception as e:
                                    logger.error(f"读取aicallers任务文件失败: {task_file}, 错误: {e}")
                                    continue

                                process_aicallers_image(file_url, boxId, task_id, droneSn, thirdGroupId)
                            else:
                                logger.warning(f"aicallers接口返回数据格式不完整，缺少字段: {[field for field in required_fields if field not in data_content]}")
                                logger.warning(f"完整返回数据: {data}")
                        else:
                            logger.warning(f"aicallers接口返回数据格式错误，缺少data字段或data不是字典类型")
                            logger.warning(f"完整返回数据: {data}")
                    else:
                        logger.warning(f"aicallers接口POST请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"aicallers接口POST请求异常: {e}")
                except json.JSONDecodeError as e:
                    logger.error(f"aicallers接口返回数据解析失败: {e}, 响应内容: {response.text}")
                except Exception as e:
                    logger.error(f"处理aicallers数据异常: {e}")

            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"aicallers轮询工作线程异常: {e}")
            time.sleep(1)

def process_test_smokefire_image(file_url: str, task_id: str = "", drone_sn: str = "", third_group_id: str = ""):
    try:
        # 记录图片下载时间戳（用于事件上报）
        download_timestamp = int(time.time())
        
        clean_url = _strip_query_params(file_url)
        file_name = os.path.basename(clean_url)
        save_dir = ROOT / 'smokefireimg'
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / file_name

        resp = requests.get(clean_url, timeout=10)
        if resp.status_code != 200:
            logger.error(f"smokefire图片下载失败，状态码: {resp.status_code}, url: {clean_url}")
            return
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        model = get_smokefire_model()
        results = model(str(save_path), conf=0.5, verbose=False)

        has_detection = False
        first_result = None
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                has_detection = True
                first_result = r
                break

        if not has_detection or first_result is None:
            return

        out_dir = ROOT / 'upload' / 'smokefire'
        os.makedirs(out_dir, exist_ok=True)

        event_id = str(uuid.uuid4())

        orig_jpg_path = out_dir / f"{event_id}.jpg"
        img_bgr = cv2.imread(str(save_path))
        if img_bgr is None:
            logger.error(f"读取下载图片失败，无法生成原图与结果: {save_path}")
            return
        cv2.imwrite(str(orig_jpg_path), img_bgr)

        annotated_img = first_result.plot()
        result_jpeg_path = out_dir / f"{event_id}.jpeg"
        cv2.imwrite(str(result_jpeg_path), annotated_img)
        logger.info(f"smokefire检测到目标，结果已保存: {result_jpeg_path} 与原图: {orig_jpg_path}")
        targets = []
        try:
            names = first_result.names if hasattr(first_result, 'names') and isinstance(first_result.names, dict) else {}
            for idx, box in enumerate(first_result.boxes):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
                score = float(box.conf[0].item()) if box.conf is not None else 1.0
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                label = names.get(cls_id, 'smoke_fire')
                targets.append({
                    "angle": 0,
                    "box": {
                        "left_top_x": x1,
                        "left_top_y": y1,
                        "right_bottom_x": y2 if False else x2,
                        "right_bottom_y": y2
                    },
                    "color": [255, 0, 0, 0],
                    "cross_label": "",
                    "id": idx + 1,
                    "label": label,
                    "prob": score,
                    "moving": False,
                    "ocr": "",
                    "region_label": "",
                    "roi_id": 0,
                    "reserved": ""
                })
        except Exception as e:
            logger.warning(f"生成targets时异常，将仅包含空targets: {e}")
            targets = []
        location = extract_gps_from_image(str(save_path))
        if drone_sn == "1581F8HGX253E00A04A7":
            mapped_src_id = "13a8db1f8f8383e384efd56797c4dcd2"
        elif drone_sn == "1581F8HGX253S00A05LN":
            mapped_src_id = "13a8db1f8f8383e384efd56797c4dcd7"
        else:
            mapped_src_id = ""

        json_data = {
            "event_id": event_id,
            "event_state": 0,
            "device_name": "重庆AI识别",
            "device_id": str(task_id) if task_id is not None else "",
            "task_name": "烟火检测",
            "task_id": str(task_id) if task_id is not None else "",
            "app_name": "烟火检测",
            "app_id": "smokeFire",
            "src_name": "smokeFire",
            "src_id": mapped_src_id,
            "created": download_timestamp,  # 使用图片下载时间戳而非JSON生成时间戳
            "picNum": 2,
            "location": location,
            "thirdGroupId": third_group_id,
            "details": [
                {
                    "frame_id": 1,
                    "metadata": {},
                    "model_id": "YOLO11",
                    "model_name": "smokefire_v1",
                    "model_thres": 0.5,
                    "model_type": 1,
                    "targets": targets
                }
            ]
        }
        json_path = out_dir / f"{event_id}.json"
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(json_data, jf, ensure_ascii=False, indent=4)
        logger.info(f"结果json已保存: {json_path}")

        llm_task = LLMVerificationTask(
            event_id=event_id,
            out_dir=str(out_dir),
            orig_jpg_path=str(orig_jpg_path),
            result_jpeg_path=str(result_jpeg_path),
            json_path=str(json_path),
            source_image_path=str(save_path),
            task_id=str(task_id) if task_id is not None else "",
            thirdGroupId=str(third_group_id) if third_group_id is not None else ""
        )
        submit_llm_verification_task(llm_task)

    except Exception as e:
        logger.error(f"处理smokefire图片异常: {file_url}, 错误: {e}")


def test_polling_worker():
    global test_polling_enabled
    while test_polling_enabled:
        try:
            logger.debug(f"向test接口发送POST请求: {TEST_URL}")
            response = requests.post(TEST_URL, timeout=10)
            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception as e:
                    logger.error(f"test接口返回数据解析失败: {e}")
                    time.sleep(TEST_POLLING_INTERVAL)
                    continue
                if 'data' in data and isinstance(data['data'], dict):
                    data_content = data['data']
                    required_fields = ['boxId', 'task_id', 'fileUrl', 'droneSn']
                    if all(k in data_content for k in required_fields):
                        file_url = data_content.get('fileUrl')
                        task_id = data_content.get('task_id')
                        drone_sn = data_content.get('droneSn')
                        third_group_id = data_content.get('thirdGroupId', '')
                        process_test_smokefire_image(file_url, task_id, drone_sn, third_group_id)
                    else:
                        logger.warning(f"test接口返回数据格式不完整，缺少字段: {[k for k in required_fields if k not in data_content]}")
                else:
                    logger.warning("test接口返回数据格式错误，缺少data字段或data不是字典类型")
            else:
                logger.warning(f"test接口POST请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
        except Exception as e:
            logger.error(f"test轮询工作线程异常: {e}")
        finally:
            time.sleep(TEST_POLLING_INTERVAL)


def start_test_polling():
    global test_polling_enabled, test_polling_thread
    test_polling_enabled = True
    if test_polling_thread is None or not test_polling_thread.is_alive():
        test_polling_thread = threading.Thread(target=test_polling_worker, daemon=True)
        test_polling_thread.start()
        logger.info("test轮询功能已启动")


def stop_test_polling():
    global test_polling_enabled
    test_polling_enabled = False
    logger.info("test轮询功能已停止")

def process_aicallers_image(file_url, boxId, task_id, droneSn, thirdGroupId):
    """处理aicallers接口返回的图片数据，流程与/jpeg接口相同"""
    try:
        clean_url = _strip_query_params(file_url)
        file_name = os.path.basename(clean_url)
        save_dir = ROOT / 'TSDK' / 'RJPEG'
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / file_name

        resp = requests.get(clean_url, timeout=10)
        if resp.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(resp.content)

            with processing_lock:
                if str(save_path) in processing_files:
                    return
                processing_files.add(str(save_path))
            
            try:
                task_file = ROOT / 'task' / f'{boxId}_{task_id}.json'
                video_url = ""
                if os.path.exists(task_file):
                    try:
                        with open(task_file, 'r', encoding='utf-8') as f:
                            task_info = json.load(f)
                        video_url = task_info.get('video_url', '')
                        logger.info(f"从task文件获取到video_url: {video_url}")
                    except Exception as e:
                        logger.warning(f"读取task文件失败: {task_file}, 错误: {e}")
                
                task = ProcessingTask(
                    priority=0, 
                    jpeg_path=str(save_path), 
                    boxId=boxId, 
                    task_id=task_id, 
                    img_url=file_url,
                    video_url=video_url,
                    thirdGroupId=thirdGroupId
                )
                task_queue.put(task, timeout=5)
            except Exception as queue_error:
                logger.error(f"aicallers任务队列已满或添加失败: {queue_error}")
                executor.submit(process_downloaded_jpeg, str(save_path), boxId, task_id, video_url, thirdGroupId)
                logger.warning(f"aicallers使用线程池直接处理: {save_path}")
        else:
            logger.error(f"aicallers图片下载失败，状态码: {resp.status_code}, url: {clean_url}")
            
    except Exception as e:
        logger.error(f"处理aicallers图片异常: {file_url}, 错误: {e}")
        with processing_lock:
            processing_files.discard(str(save_path))

def start_aicallers_polling():
    """启动aicallers轮询功能"""
    global aicallers_polling_enabled, aicallers_polling_thread
    aicallers_polling_enabled = True
    if aicallers_polling_thread is None or not aicallers_polling_thread.is_alive():
        aicallers_polling_thread = threading.Thread(target=aicallers_polling_worker, daemon=True)
        aicallers_polling_thread.start()
        logger.info("aicallers轮询功能已启动")

def stop_aicallers_polling():
    """停止aicallers轮询功能"""
    global aicallers_polling_enabled
    aicallers_polling_enabled = False
    logger.info("aicallers轮询功能已停止")

log_dir = Path(__file__).resolve().parent / 'logs'
os.makedirs(log_dir, exist_ok=True)

log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format))

detailed_handler = logging.handlers.RotatingFileHandler(
    log_dir / 'algorithm_api_detailed.log',
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'
)
detailed_handler.setLevel(logging.DEBUG)
detailed_handler.setFormatter(logging.Formatter(log_format))

error_handler = logging.handlers.RotatingFileHandler(
    log_dir / 'algorithm_api_error.log',
    maxBytes=5*1024*1024,
    backupCount=3,
    encoding='utf-8'
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(log_format))

logger.addHandler(console_handler)
logger.addHandler(detailed_handler)
logger.addHandler(error_handler)
logger.propagate = False

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT / 'task'
os.makedirs(TASK_DIR, exist_ok=True)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

@app.before_request
def log_request_info():
    if request.path in ['/create', '/query', '/algorithm', '/delete']:
        logger.debug('请求头: %s', dict(request.headers))
        logger.debug('请求体: %s', request.get_data(as_text=True))

@app.errorhandler(400)
def bad_request_handler(error):
    logger.error(f"捕获到400错误: {error}")
    return jsonify({
        "code": 1,
        "error": f"请求格式错误: {str(error)}"
    }), 400

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"捕获到500错误: {error}")
    return jsonify({
        "code": 1,
        "error": f"服务器内部错误: {str(error)}"
    }), 500

@app.route('/create', methods=['POST', 'OPTIONS'])
def create_task():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        if not request.is_json and 'application/json' not in request.content_type:
            logger.warning(f"请求内容类型错误: {request.content_type}")
            return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400
        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "error": "空请求体"}), 400
        required_params = ['boxId', 'task_id', 'categoryType']
        for param in required_params:
            if param not in data:
                return jsonify({"code": 1, "error": f"缺少参数: {param}"}), 400
        status = data.get('status', 'not_started')
        if status not in ['running', 'not_started', 'stream_error']:
            return jsonify({"code": 1, "error": f"无效的状态值: {status}，允许的值为 running, not_started, stream_error"}), 400
        categoryType = data.get('categoryType')
        if categoryType not in ['SolarPanel', 'spyCam', 'ceWen', 'phoneCall', 'smokeFire']:
            return jsonify({"code": 1, "error": f"无效的categoryType值: {categoryType}，允许的值为 SolarPanel, spyCam, ceWen, phoneCall, smokeFire"}), 400
        task_data = {
            "boxId": data.get('boxId'),
            "task_id": data.get('task_id'),
            "status": status,
            "task_name": data.get('task_name', '天线资产盘点'),
            "categoryType": categoryType
        }
        if 'video_url' in data:
            task_data["video_url"] = data.get('video_url')
        if 'extendFields' in data:
            extendFields = data.get('extendFields')
            if not (isinstance(extendFields, str) and extendFields.replace('.', '', 1).isdigit()):
                return jsonify({"code": 1, "error": "extendFields字段必须为字符串数字"}), 400
            task_data["extendFields"] = extendFields
        file_name = f"{task_data['boxId']}_{task_data['task_id']}.json"
        file_path = TASK_DIR / file_name
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, ensure_ascii=False, indent=4)
        logger.info(f"创建任务成功: {file_name}")
        return jsonify({"code": 0, "data": {"message": "任务创建成功", "task_info": task_data}}), 200
    except Exception as e:
        logger.exception(f"创建任务异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

@app.route('/query', methods=['POST', 'OPTIONS'])
def query_tasks():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        if not request.is_json and 'application/json' not in request.content_type:
            logger.warning(f"请求内容类型错误: {request.content_type}")
            return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400
        data = request.get_json()
        if data is None:
            return jsonify({"code": 1, "error": "空请求体"}), 400
        logger.debug(f"查询参数: {data}")
        tasks = []
        for file_path in TASK_DIR.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    task_info = json.load(f)
                match = True
                for k, v in data.items():
                    if k == 'task_name':
                        task_info_name = task_info.get('task_name', '')
                        if task_info_name is None or v not in task_info_name:
                            match = False
                            break
                    else:
                        if str(task_info.get(k, '')) != str(v):
                            match = False
                            break
                if not match:
                    continue
                if 'video_url' in task_info:
                    del task_info['video_url']
                if 'updated_at' in task_info:
                    del task_info['updated_at']
                tasks.append(task_info)
            except Exception as e:
                logger.error(f"读取任务文件错误 {file_path}: {str(e)}")
        logger.info(f"查询任务成功: 找到{len(tasks)}个任务")
        return jsonify({"code": 0, "data": tasks}), 200
    except Exception as e:
        logger.exception(f"查询任务异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

@app.route('/algorithm', methods=['POST', 'OPTIONS'])
def handle_algorithm():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        if not request.is_json and 'application/json' not in request.content_type:
            logger.warning(f"请求内容类型错误: {request.content_type}")
            return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400
        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "error": "空请求体"}), 400
        required_params = ['boxId', 'task_id', 'algorithm_status']
        for param in required_params:
            if param not in data:
                return jsonify({"code": 1, "error": f"缺少参数: {param}"}), 400
        boxId = data.get('boxId')
        task_id = data.get('task_id')
        algorithm_status = data.get('algorithm_status')
        file_name = f"{boxId}_{task_id}.json"
        file_path = TASK_DIR / file_name
        if not os.path.exists(file_path):
            return jsonify({"code": 1, "error": f"任务不存在: {file_name}"}), 404
        with open(file_path, 'r', encoding='utf-8') as f:
            task_info = json.load(f)
        
        categoryType = task_info.get('categoryType')
        
        if algorithm_status == 0 or algorithm_status == "0":
            if categoryType == 'ceWen':
                task_info['status'] = 'running'
                logger.info(f"ceWen任务算法状态更新: 启动 - {file_name}")
            else:
                task_info['status'] = 'running'
                logger.info(f"算法状态更新: 启动 - {file_name}")
                # 立即启动对应的算法进程
                if categoryType in ['SolarPanel', 'spyCam', 'phoneCall', 'smokeFire']:
                    start_success = start_algorithm_process(boxId, task_id, categoryType)
                    if start_success:
                        logger.info(f"算法进程启动成功: {boxId}_{task_id}")
                    else:
                        logger.warning(f"算法进程启动失败: {boxId}_{task_id}")
        elif algorithm_status == 1 or algorithm_status == "1":
            if categoryType == 'ceWen':
                task_info['status'] = 'not_started'
                logger.info(f"ceWen任务算法状态更新: 停止 - {file_name}")
            else:
                task_info['status'] = 'not_started'
                logger.info(f"算法状态更新: 停止 - {file_name}")
        else:
            return jsonify({"code": 1, "error": "无效的algorithm_status值，使用0表示启动或1表示停止"}), 400
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(task_info, f, ensure_ascii=False, indent=4)
        return jsonify({"code": 0, "data": {"message": "算法状态更新成功", "task_info": task_info}}), 200
    except Exception as e:
        logger.exception(f"处理算法状态更新异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

@app.route('/delete', methods=['POST', 'OPTIONS'])
def delete_task():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        if not request.is_json and 'application/json' not in request.content_type:
            logger.warning(f"请求内容类型错误: {request.content_type}")
            return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400
        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "error": "空请求体"}), 400
        required_params = ['boxId', 'task_id']
        for param in required_params:
            if param not in data:
                return jsonify({"code": 1, "error": f"缺少参数: {param}"}), 400
        boxId = data.get('boxId')
        task_id = data.get('task_id')
        file_name = f"{boxId}_{task_id}.json"
        file_path = TASK_DIR / file_name
        if not os.path.exists(file_path):
            return jsonify({"code": 1, "error": f"任务文件不存在: {file_name}"}), 404
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                task_info = json.load(f)
        except Exception as e:
            logger.warning(f"读取任务文件失败，但仍将删除文件: {file_name}, 错误: {str(e)}")
            task_info = {"boxId": boxId, "task_id": task_id}
        try:
            os.remove(file_path)
            logger.info(f"成功删除任务文件: {file_name}")
            return jsonify({
                "code": 0,
                "data": {
                    "message": "任务删除成功",
                    "deleted_file": file_name,
                    "task_info": task_info
                }
            }), 200
        except Exception as e:
            logger.error(f"删除任务文件失败: {file_name}, 错误: {str(e)}")
            return jsonify({
                "code": 1,
                "error": f"删除文件失败: {str(e)}"
            }), 500
    except Exception as e:
        logger.exception(f"删除任务异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

@app.route('/jpeg', methods=['POST', 'OPTIONS'])
def download_jpeg():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    global request_count, last_reset_time
    current_time = time.time()
    with request_lock:
        if current_time - last_reset_time >= 1.0:
            request_count = 0
            last_reset_time = current_time
        
        if request_count >= MAX_REQUESTS_PER_SECOND:
            logger.warning(f"请求频率过高，拒绝请求: {request_count}/s")
            return jsonify({"code": 1, "error": "请求频率过高，请稍后重试"}), 429
        
        request_count += 1
    
    try:
        if not request.is_json and 'application/json' not in request.content_type:
            logger.warning(f"请求内容类型错误: {request.content_type}")
            return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400
        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "error": "空请求体"}), 400
        required_params = ['boxId', 'task_id', 'fileUrl']
        for param in required_params:
            if param not in data:
                return jsonify({"code": 1, "error": f"缺少参数: {param}"}), 400
        boxId = data.get('boxId')
        task_id = data.get('task_id')
        file_url = data.get('fileUrl')
        thirdGroupId = data.get('thirdGroupId', '')
        clean_url = _strip_query_params(file_url)
        file_name = os.path.basename(clean_url)
        save_dir = ROOT / 'TSDK' / 'RJPEG'
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / file_name
        try:
            resp = requests.get(clean_url, timeout=10)
            if resp.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(resp.content)
                try:
                    with processing_lock:
                        if str(save_path) in processing_files:
                            return jsonify({"code": 0, "data": {"message": "图片下载成功，正在处理中", "file_path": str(save_path)}}), 200
                        processing_files.add(str(save_path))
                    
                    try:
                        task = ProcessingTask(
                            priority=0, 
                            jpeg_path=str(save_path), 
                            boxId=boxId, 
                            task_id=task_id, 
                            img_url=file_url,
                            thirdGroupId=thirdGroupId
                        )
                        task_queue.put(task, timeout=5)
                    except Exception as queue_error:
                        logger.error(f"任务队列已满或添加失败: {queue_error}")
                        executor.submit(process_downloaded_jpeg, str(save_path), boxId, task_id, "", thirdGroupId)
                        logger.warning(f"使用线程池直接处理: {save_path}")
                except Exception as e:
                    logger.error(f"图片后处理异常: {e}")
                return jsonify({"code": 0, "data": {"message": "图片下载成功", "file_path": str(save_path)}}), 200
            else:
                logger.error(f"图片下载失败，状态码: {resp.status_code}, url: {clean_url}")
                return jsonify({"code": 1, "error": f"图片下载失败，状态码: {resp.status_code}"}), 500
        except Exception as e:
            logger.error(f"图片下载异常: {clean_url}, 错误: {e}")
            return jsonify({"code": 1, "error": f"图片下载异常: {str(e)}"}), 500
    except Exception as e:
        logger.exception(f"/jpeg接口异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

@app.route('/queue_status', methods=['GET', 'OPTIONS'])
def get_queue_status():
    """获取队列状态信息"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        with stats_lock:
            current_stats = queue_stats.copy()
            current_stats['queue_size'] = task_queue.qsize()
            current_stats['queue_full'] = task_queue.full()
            current_stats['current_time'] = time.time()
        llm_current_stats = get_llm_status_snapshot()
        interface_queue_stats = get_interface_queue_status_snapshot()
        
        return jsonify({
            "code": 0,
            "data": {
                "queue_stats": current_stats,
                "llm_queue_stats": llm_current_stats,
                "interface_queue_stats": interface_queue_stats,
                "max_queue_size": 200,
                "max_workers": 10,
                "request_limit": MAX_REQUESTS_PER_SECOND
            }
        }), 200
    except Exception as e:
        logger.exception(f"获取队列状态异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

@app.route('/llm_queue_status', methods=['GET', 'OPTIONS'])
def get_llm_queue_status():
    """实时获取大模型验证队列状态"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = get_llm_status_snapshot()
        data["interface_process_queues"] = get_interface_queue_status_snapshot()
        return jsonify({
            "code": 0,
            "data": data
        }), 200
    except Exception as e:
        logger.exception(f"获取大模型队列状态异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

@app.route('/interface_queue_status', methods=['GET', 'OPTIONS'])
def get_interface_queue_status():
    """实时获取interface子进程大模型验证队列状态"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        return jsonify({
            "code": 0,
            "data": get_interface_queue_status_snapshot()
        }), 200
    except Exception as e:
        logger.exception(f"获取interface子进程队列状态异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

@app.route('/tempRequest', methods=['POST', 'OPTIONS'])
def control_aicallers_polling():
    """控制aicallers轮询功能"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        if not request.is_json and 'application/json' not in request.content_type:
            logger.warning(f"请求内容类型错误: {request.content_type}")
            return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "error": "空请求体"}), 400
        
        action = data.get('action')
        if action not in ['start', 'stop', 'status']:
            return jsonify({"code": 1, "error": "无效的action值，允许的值为 start, stop, status"}), 400
        
        if action == 'start':
            start_aicallers_polling()
            return jsonify({
                "code": 0, 
                "data": {
                    "message": "aicallers轮询功能已启动",
                    "status": "running",
                    "polling_interval": AICALLERS_POLLING_INTERVAL
                }
            }), 200
        elif action == 'stop':
            stop_aicallers_polling()
            return jsonify({
                "code": 0, 
                "data": {
                    "message": "aicallers轮询功能已停止",
                    "status": "stopped"
                }
            }), 200
        elif action == 'status':
            return jsonify({
                "code": 0, 
                "data": {
                    "enabled": aicallers_polling_enabled,
                    "thread_alive": aicallers_polling_thread.is_alive() if aicallers_polling_thread else False,
                    "polling_interval": AICALLERS_POLLING_INTERVAL,
                    "last_request_time": aicallers_last_request_time,
                    "aicallers_url": AICALLERS_URL,
                    "current_time": time.time(),
                    "time_since_last_request": time.time() - aicallers_last_request_time if aicallers_last_request_time > 0 else None
                }
            }), 200
            
    except Exception as e:
        logger.exception(f"控制温度请求轮询功能异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

@app.route('/tempRequest_stats', methods=['GET', 'OPTIONS'])
def get_aicallers_stats():
    """获取温度请求轮询统计信息"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        return jsonify({
            "code": 0,
            "data": {
                "polling_enabled": aicallers_polling_enabled,
                "thread_alive": aicallers_polling_thread.is_alive() if aicallers_polling_thread else False,
                "polling_interval": AICALLERS_POLLING_INTERVAL,
                "aicallers_url": AICALLERS_URL,
                "last_request_time": aicallers_last_request_time,
                "current_time": time.time(),
                "time_since_last_request": time.time() - aicallers_last_request_time if aicallers_last_request_time > 0 else None,
                "queue_stats": {
                    "queue_size": task_queue.qsize(),
                    "queue_full": task_queue.full(),
                    "processing_files_count": len(processing_files)
                },
                "llm_queue_stats": get_llm_status_snapshot(),
                "interface_queue_stats": get_interface_queue_status_snapshot()
            }
        }), 200
    except Exception as e:
        logger.exception(f"获取温度请求统计信息异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

def process_downloaded_jpeg(jpeg_path, boxId, task_id, video_url="", thirdGroupId=""):
    import shutil
    import cv2
    import torch
    
    try:
        # 记录图片处理开始时间戳（用于事件上报）
        processing_timestamp = int(time.time())
        
        model = get_yolo_model()
        file_lock = threading.Lock()
        with file_lock:
            results = model(jpeg_path, conf=0.5, verbose=False)
        
        centers = []
        boxes = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                cx = (x1 + x2) / 2 / 2
                cy = (y1 + y2) / 2 / 2
                centers.append((int(cx), int(cy)))
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        if not centers:
            try:
                if os.path.exists(jpeg_path):
                    os.remove(jpeg_path)
            except Exception as e:
                logger.warning(f"删除未检测到目标的图片失败: {e}")
            return False
        
        raw_path = jpeg_path.replace('.jpeg', '.raw').replace('.jpg', '.raw').replace('.JPG', '.raw')
        import subprocess
        raw_script_path = ROOT / 'TSDK' / 'read_raw_output_point_temperature.py'
        
        with file_lock:
            subprocess.run(['python3', str(raw_script_path), '--input', jpeg_path, '--output', raw_path], check=True)

        temps = []
        for cx, cy in centers:
            result = subprocess.run([
                'python3', str(raw_script_path),
                'temp', '--get_temp', str(cx), str(cy), raw_path,
                '--width', '640', '--height', '512'
            ], capture_output=True, text=True)
            try:
                out = (result.stdout or '').strip()
                err = (result.stderr or '').strip()
                match = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", out)
                if match:
                    temp = float(match[-1])
                else:
                    logger.warning(f"点测温解析失败，stdout='{out}', stderr='{err}', point=({cx},{cy})")
                    temp = None
            except Exception as parse_e:
                logger.warning(f"点测温异常: {parse_e}, stdout='{result.stdout}', stderr='{result.stderr}', point=({cx},{cy})")
                temp = None
            temps.append(temp)
        try:
            logger.info(f"点测温结果: temps={temps}, centers={centers}")
        except Exception:
            pass
        
        task_file = ROOT / 'task' / f'{boxId}_{task_id}.json'
        if not os.path.exists(task_file):
            logger.warning(f"任务文件不存在: {task_file}")
            return False
        
        with open(task_file, 'r', encoding='utf-8') as f:
            task_info = json.load(f)
        threshold_temp = float(task_info.get('extendFields', '-273'))
        try:
            logger.info(f"阈值: {threshold_temp}")
        except Exception:
            pass
        
        valid_points = [(i, t) for i, t in enumerate(temps) if t is not None and t > threshold_temp]
        if not valid_points:
            try:
                logger.info(f"所有点温度均未超过阈值，详细: temps={temps}, centers={centers}, threshold={threshold_temp}")
            except Exception:
                pass
            try:
                os.remove(jpeg_path)
            except Exception:
                pass
            try:
                os.remove(raw_path)
            except Exception:
                pass
            logger.info(f"所有点温度低于阈值，已删除: {jpeg_path}, {raw_path}")
            return False
        
        img = cv2.imread(jpeg_path)
        save_dir = ROOT / 'upload' / 'cewen'
        os.makedirs(save_dir, exist_ok=True)
        
        event_id = str(uuid.uuid4())

        orig_jpg_name = event_id + '.jpg'
        orig_jpg_path = save_dir / orig_jpg_name
        cv2.imwrite(str(orig_jpg_path), cv2.imread(jpeg_path))
        result_jpeg_name = event_id + '.jpeg'
        result_jpeg_path = save_dir / result_jpeg_name
        targets = []
        temperatures = []
        for idx, temp in valid_points:
            x1, y1, x2, y2 = boxes[idx]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cx, cy = centers[idx]
            cv2.putText(img, f"{temp:.1f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            temperatures.append(f"{temp}°C")
            targets.append({
                "angle": 0,
                "box": {
                    "left_top_x": int(x1),
                    "left_top_y": int(y1),
                    "right_bottom_x": int(x2),
                    "right_bottom_y": int(y2)
                },
                "color": [255, 0, 0, 0],
                "cross_label": "",
                "id": idx+1,
                "label": "hotspot",
                "prob": 1.0,
                "moving": False,
                "ocr": "",
                "region_label": "",
                "roi_id": 0,
                "reserved": ""
            })
        cv2.imwrite(str(result_jpeg_path), img)
        logger.info(f"高温点结果图片已保存: {result_jpeg_path}")
        logger.info(f"原始图片已转换为jpg并保存: {orig_jpg_path}")
        
        location = extract_gps_from_image(jpeg_path)
        
        if not video_url:
            video_url = task_info.get("video_url", "")
        src_id = task_info.get("task_id", "")

        if video_url.endswith("1581F8HGX253Q00A05BQ"):
            src_id = "13a8db1f8f8383e384efd56797c4dcd2"
            src_name = "2号门机场"
        elif video_url.endswith("1581F8HGX253S00A05LN"):
            src_id = "13a8db1f8f8383e384efd56797c4dcd7"
            src_name = "7号门机场"
        elif video_url.endswith("1581F8HGX253E00A04A7"):
            src_id = "13a8db1f8f8383e384efd56797c4dcd2"
            src_name = "2号门机场"
        else:
            src_name = task_info.get("task_name", "")
        
        json_data = {
            "event_id": event_id,
            "event_state": 0,
            "device_name": "重庆AI识别",
            "device_id": task_info.get("task_id", ""),
            "task_name": task_info.get("task_name", ""),
            "task_id": task_info.get("task_id", ""),
            "app_name": task_info.get("task_name", ""),
            "app_id": task_info.get("categoryType", ""),
            "src_name": src_name,
            "src_id": src_id,
            "created": processing_timestamp,  # 使用图片处理开始时间戳而非JSON生成时间戳
            "picNum": 2,
            "location": location,
            "temperature": ",".join(temperatures),
            "thirdGroupId": thirdGroupId,
            "details": [
                {
                    "frame_id": 1,
                    "metadata": {},
                    "model_id": "YOLO11",
                    "model_name": "cewen_v1",
                    "model_thres": 0.5,
                    "model_type": 1,
                    "targets": targets
                }
            ]
        }
        json_save_path = save_dir / (event_id + '.json')
        with open(json_save_path, 'w', encoding='utf-8') as jf:
            json.dump(json_data, jf, ensure_ascii=False, indent=4)
        logger.info(f"结果json已保存: {json_save_path}")
        
        def push_and_cleanup(save_dir, event_id):
            upload_url = UPLOAD_URL
            params = {}
            headers = get_auth_headers()
            json_path = save_dir / (event_id + '.json')
            jpg_path = save_dir / (event_id + '.jpg')
            jpeg_path = save_dir / (event_id + '.jpeg')

            try:
                with open(json_path, 'rb') as f:
                    resp = requests.post(upload_url, params=params, files={'file': (json_path.name, f)}, headers=headers, timeout=30)
                if resp.status_code != 200:
                    logger.error(f"推送JSON失败: {resp.status_code} {resp.text}")
                    return False
                logger.info(f"推送JSON成功: {json_path}")
            except Exception as e:
                logger.error(f"推送JSON异常: {e}")
                return False

            time.sleep(2)

            try:
                with open(jpg_path, 'rb') as f:
                    resp = requests.post(upload_url, params=params, files={'file': (jpg_path.name, f)}, headers=headers, timeout=30)
                if resp.status_code != 200:
                    logger.error(f"推送JPG失败: {resp.status_code} {resp.text}")
                    return False
                logger.info(f"推送JPG成功: {jpg_path}")
            except Exception as e:
                logger.error(f"推送JPG异常: {e}")
                return False
            try:
                with open(jpeg_path, 'rb') as f:
                    resp = requests.post(upload_url, params=params, files={'file': (jpeg_path.name, f)}, headers=headers, timeout=30)
                if resp.status_code != 200:
                    logger.error(f"推送JPEG失败: {resp.status_code} {resp.text}")
                    return False
                logger.info(f"推送JPEG成功: {jpeg_path}")
            except Exception as e:
                logger.error(f"推送JPEG异常: {e}")
                return False

            logger.info(f"推送成功，保留本地上报文件: {json_path}, {jpg_path}, {jpeg_path}")
            return True

        upload_ok = push_and_cleanup(save_dir, event_id)
        
        if upload_ok:
            logger.info(f"上传成功，保留RJPEG与RAW文件: {jpeg_path}, {raw_path}")
        else:
            logger.warning("上传未全部成功，保留RJPEG与RAW以便排查")
        return True
    
    except Exception as e:
        logger.error(f"处理图片异常: {jpeg_path}, 错误: {e}")
        return False
    finally:
        with processing_lock:
            processing_files.discard(jpeg_path)

def check_video_stream_available(video_url):
    import cv2
    try:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            cap.release()
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except Exception:
        return False

def task_watcher_daemon():
    import sys
    import subprocess
    from pathlib import Path
    import json
    import time
    TASK_DIR = Path(__file__).resolve().parent / 'task'
    SOLARPANEL_DETECT_PATH = Path(__file__).resolve().parent / 'interface' / 'solarpanel_detect.py'
    SPYCAM_DETECT_PATH = Path(__file__).resolve().parent / 'interface' / 'spycam_detect.py'
    PHONECALL_DETECT_PATH = Path(__file__).resolve().parent / 'interface' / 'phonecall_detect.py'
    SMOKEFIRE_DETECT_PATH = Path(__file__).resolve().parent / 'interface' / 'smokefire_detect.py'
    task_processes = {}
    last_status_map = {}
    SCAN_INTERVAL = 3
    while True:
        for file in TASK_DIR.glob('*.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    task = json.load(f)
                status = task.get('status')
                categoryType = task.get('categoryType')
                video_url = task.get('video_url')
                task_id = f"{task.get('boxId')}_{task.get('task_id')}"
                prev_status = last_status_map.get(task_id, None)

                new_status = status
                if categoryType == 'ceWen':
                    pass
                elif status in ('stream_error', 'running') and categoryType in ['SolarPanel', 'spyCam', 'phoneCall', 'smokeFire']:
                    available = check_video_stream_available(video_url) if video_url else False
                    if available:
                        new_status = 'running'
                    else:
                        new_status = 'stream_error'
                    if new_status != status:
                        task['status'] = new_status
                        with open(file, 'w', encoding='utf-8') as f:
                            json.dump(task, f, ensure_ascii=False, indent=4)
                        status = new_status
                should_start = False
                if prev_status in ('not_started', 'stream_error') and status == 'running' and task_id not in task_processes:
                    should_start = True
                    logger.info(f"检测到任务{task_id}状态变化为running，准备启动算法...")
                elif status == 'running' and task_id not in task_processes and prev_status is None:
                    should_start = True
                    logger.info(f"检测到任务{task_id}状态为running且进程未启动，准备启动算法...")
                
                if should_start:
                    if categoryType == 'SolarPanel':
                        logger.info(f"启动太阳能板算法: {task_id}")
                        proc = subprocess.Popen([sys.executable, str(SOLARPANEL_DETECT_PATH), '--task', str(file)])
                        task_processes[task_id] = proc
                    elif categoryType == 'spyCam':
                        logger.info(f"启动拍照姿势检测算法: {task_id}")
                        proc = subprocess.Popen([sys.executable, str(SPYCAM_DETECT_PATH), '--task', str(file)])
                        task_processes[task_id] = proc
                    elif categoryType == 'phoneCall':
                        logger.info(f"启动打电话姿势检测算法: {task_id}")
                        proc = subprocess.Popen([sys.executable, str(PHONECALL_DETECT_PATH), '--task', str(file)])
                        task_processes[task_id] = proc
                    elif categoryType == 'smokeFire':
                        logger.info(f"启动烟火检测算法: {task_id}")
                        proc = subprocess.Popen([sys.executable, str(SMOKEFIRE_DETECT_PATH), '--task', str(file)])
                        task_processes[task_id] = proc


                elif status in ('not_started', 'stream_error') and task_id in task_processes:
                    proc = task_processes[task_id]
                    if proc.poll() is None:
                        logger.info(f"停止任务{task_id}...")
                        try:
                            proc.terminate()
                            for _ in range(5):
                                if proc.poll() is not None:
                                    break
                                time.sleep(1)
                            if proc.poll() is None:
                                logger.warning(f"进程未正常退出，强制结束: {task_id}")
                                proc.kill()
                            logger.info(f"任务{task_id}已停止")
                        except Exception as e:
                            logger.error(f"停止进程失败: {task_id}, 错误: {e}")
                        finally:
                            del task_processes[task_id]
                if task_id in task_processes:
                    proc = task_processes[task_id]
                    if proc.poll() is not None:
                        logger.error(f"任务{task_id}对应进程已退出，移除记录")
                        del task_processes[task_id]

                last_status_map[task_id] = status
            except Exception as e:
                logger.error(f"处理任务文件{file}异常: {e}")
        time.sleep(SCAN_INTERVAL)

def start_task_watcher_once():
    if not hasattr(start_task_watcher_once, '_started'):
        t = threading.Thread(target=task_watcher_daemon, daemon=True)
        t.start()
        start_task_watcher_once._started = True

def start_algorithm_process(boxId, task_id, categoryType):
    """立即启动对应的算法进程"""
    import subprocess
    import sys
    from pathlib import Path
    
    try:
        task_file = ROOT / 'task' / f'{boxId}_{task_id}.json'
        if not os.path.exists(task_file):
            logger.warning(f"任务文件不存在，无法启动算法: {task_file}")
            return False
            
        with open(task_file, 'r', encoding='utf-8') as f:
            task_info = json.load(f)
        
        video_url = task_info.get('video_url', '')
        
        if categoryType == 'SolarPanel':
            script_path = ROOT / 'interface' / 'solarpanel_detect.py'
            logger.info(f"立即启动太阳能板算法: {boxId}_{task_id}")
            proc = subprocess.Popen([sys.executable, str(script_path), '--task', str(task_file)])
            return True
        elif categoryType == 'spyCam':
            script_path = ROOT / 'interface' / 'spycam_detect.py'
            logger.info(f"立即启动拍照姿势检测算法: {boxId}_{task_id}")
            proc = subprocess.Popen([sys.executable, str(script_path), '--task', str(task_file)])
            return True
        elif categoryType == 'phoneCall':
            script_path = ROOT / 'interface' / 'phonecall_detect.py'
            logger.info(f"立即启动打电话姿势检测算法: {boxId}_{task_id}")
            proc = subprocess.Popen([sys.executable, str(script_path), '--task', str(task_file)])
            return True
        elif categoryType == 'smokeFire':
            script_path = ROOT / 'interface' / 'smokefire_detect.py'
            logger.info(f"立即启动烟火检测算法: {boxId}_{task_id}")
            proc = subprocess.Popen([sys.executable, str(script_path), '--task', str(task_file)])
            return True
        else:
            logger.warning(f"不支持的算法类型: {categoryType}")
            return False
            
    except Exception as e:
        logger.error(f"启动算法进程失败: {boxId}_{task_id}, 错误: {e}")
        return False

start_task_watcher_once()
start_llm_verification_workers()
# start_aicallers_polling()  # 已废弃，不再自动启动
# start_test_polling()  # 已废弃，不再自动启动

logger.info("算法API服务启动完成")
logger.info(f"aicallers轮询URL: {AICALLERS_URL} (已禁用)")
logger.info(f"aicallers轮询间隔: {AICALLERS_POLLING_INTERVAL}秒")
logger.info("服务监听端口: 20025")
logger.info("温度请求轮询功能初始状态: stopped (已废弃)")
logger.info("TEST轮询功能初始状态: stopped (已废弃)")

@app.route('/apiStream', methods=['POST', 'OPTIONS'])
def api_stream_control():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        if not request.is_json and 'application/json' not in request.content_type:
            logger.warning(f"/apiStream 请求内容类型错误: {request.content_type}")
            return jsonify({"code": 1, "error": "请求内容类型必须是 application/json"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"code": 1, "error": "空请求体"}), 400

        required = ['droneSn', 'task_id', 'status', 'categoryType']
        for k in required:
            if k not in data:
                return jsonify({"code": 1, "error": f"缺少参数: {k}"}), 400

        drone_sn = str(data.get('droneSn')).strip()
        task_id = str(data.get('task_id')).strip()
        status = data.get('status')
        category_type = data.get('categoryType')

        if drone_sn == '' or task_id == '':
            return jsonify({"code": 1, "error": "droneSn 与 task_id 不能为空"}), 400

        if category_type not in ['spyCam', 'phoneCall']:
            return jsonify({"code": 1, "error": "无效的categoryType，允许的值为 spyCam 或 phoneCall"}), 400

        key = f"{drone_sn}_{task_id}"
        video_source = get_rtmp_source_url(drone_sn)
        rtmp_url = get_rtmp_output_url(drone_sn, task_id)

        if category_type == 'spyCam':
            script_path = ROOT / 'rtmp_spycam_stream.py'
            model_path = ROOT / 'weights' / 'spycam.pt'
        elif category_type == 'phoneCall':
            script_path = ROOT / 'rtmp_phonecall_stream.py'
            model_path = ROOT / 'weights' / 'phonecall.pt'

        # 启动
        if status in (0, '0'):
            with stream_processes_lock:
                if key in stream_processes and stream_processes[key].poll() is None:
                    return jsonify({"code": 0, "data": {"message": "进程已在运行", "key": key}}), 200
                try:
                    cmd = [
                        sys.executable,
                        str(script_path),
                        '--video-source', video_source,
                        '--rtmp-url', rtmp_url,
                        '--model-path', str(model_path),
                        '--conf', '0.5'
                    ]
                    logger.info(f"启动推流进程: key={key}, categoryType={category_type}, cmd={' '.join(cmd)}")
                    proc = subprocess.Popen(cmd)
                    stream_processes[key] = proc
                    return jsonify({"code": 0, "data": {"message": "启动成功", "key": key, "pid": proc.pid, "video_source": video_source, "rtmp_url": rtmp_url, "categoryType": category_type}}), 200
                except Exception as e:
                    logger.error(f"启动推流进程失败: key={key}, 错误: {e}")
                    return jsonify({"code": 1, "error": f"启动失败: {e}"}), 500

        # 停止
        elif status in (1, '1'):
            with stream_processes_lock:
                proc = stream_processes.get(key)
                if not proc:
                    return jsonify({"code": 0, "data": {"message": "未找到对应进程", "key": key}}), 200
                try:
                    if proc.poll() is None:
                        logger.info(f"停止推流进程: key={key}, categoryType={category_type}, pid={proc.pid}")
                        proc.terminate()
                        for _ in range(10):
                            if proc.poll() is not None:
                                break
                            time.sleep(0.3)
                        if proc.poll() is None:
                            logger.warning(f"进程未正常退出，强制kill: key={key}")
                            proc.kill()
                    del stream_processes[key]
                    return jsonify({"code": 0, "data": {"message": "停止成功", "key": key, "categoryType": category_type}}), 200
                except Exception as e:
                    logger.error(f"停止推流进程失败: key={key}, 错误: {e}")
                    return jsonify({"code": 1, "error": f"停止失败: {e}"}), 500

        else:
            return jsonify({"code": 1, "error": "无效的status，使用0启动、1停止"}), 400

    except Exception as e:
        logger.exception(f"/apiStream 异常: {str(e)}")
        return jsonify({"code": 1, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=20025, debug=False)
