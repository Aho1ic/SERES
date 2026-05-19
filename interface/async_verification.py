from __future__ import annotations

import json
import os
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from threading import Condition, Lock, Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
from loguru import logger


PENDING = "pending"


@dataclass
class VerificationEvent:
    event_id: str
    frame_id: int
    frame_timestamp: int
    orig_path: Path
    detected_path: Path
    json_path: Path
    detections: List[Dict[str, Any]] = field(default_factory=list)
    cached_detections: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


class VerificationMetrics:
    def __init__(self):
        self._lock = Lock()
        self.yolo_frame_count = 0
        self.yolo_frame_time_ms_total = 0.0
        self.yolo_frame_time_ms_last = 0.0
        self.frame_candidate_count_last = 0
        self.verification_count = 0
        self.verification_time_ms_total = 0.0
        self.verification_time_ms_last = 0.0
        self.queue_length = 0
        self.report_queue_length = 0
        self.queue_drop_count = 0
        self.report_success_count = 0
        self.track_cache_hit_count = 0

    def record_yolo_frame(self, elapsed_ms: float, candidate_count: int) -> None:
        with self._lock:
            self.yolo_frame_count += 1
            self.yolo_frame_time_ms_total += float(elapsed_ms)
            self.yolo_frame_time_ms_last = float(elapsed_ms)
            self.frame_candidate_count_last = int(candidate_count)

    def record_verification(self, elapsed_ms: float) -> None:
        with self._lock:
            self.verification_count += 1
            self.verification_time_ms_total += float(elapsed_ms)
            self.verification_time_ms_last = float(elapsed_ms)

    def set_queue_length(self, queue_length: int) -> None:
        with self._lock:
            self.queue_length = int(queue_length)

    def set_report_queue_length(self, queue_length: int) -> None:
        with self._lock:
            self.report_queue_length = int(queue_length)

    def inc_queue_drop(self, count: int = 1) -> None:
        with self._lock:
            self.queue_drop_count += int(count)

    def inc_report_success(self, count: int = 1) -> None:
        with self._lock:
            self.report_success_count += int(count)

    def inc_track_cache_hit(self, count: int = 1) -> None:
        with self._lock:
            self.track_cache_hit_count += int(count)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            avg_yolo = (
                self.yolo_frame_time_ms_total / self.yolo_frame_count
                if self.yolo_frame_count
                else 0.0
            )
            avg_verification = (
                self.verification_time_ms_total / self.verification_count
                if self.verification_count
                else 0.0
            )
            return {
                "yolo_frame_time_ms_last": round(self.yolo_frame_time_ms_last, 3),
                "yolo_frame_time_ms_avg": round(avg_yolo, 3),
                "frame_candidate_count_last": self.frame_candidate_count_last,
                "verification_time_ms_last": round(self.verification_time_ms_last, 3),
                "verification_time_ms_avg": round(avg_verification, 3),
                "verification_queue_length": self.queue_length,
                "report_queue_length": self.report_queue_length,
                "verification_queue_drop_count": self.queue_drop_count,
                "report_success_count": self.report_success_count,
                "track_cache_hit_count": self.track_cache_hit_count,
            }


class TrackVerificationCache:
    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 4096):
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._lock = Lock()
        self._items: OrderedDict[Tuple[str, int], Tuple[Any, float]] = OrderedDict()

    def get(self, task_type: str, track_id: Any) -> Optional[Any]:
        key = self._make_key(task_type, track_id)
        if key is None:
            return None

        now = time.time()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None

            value, updated_at = item
            if now - updated_at > self.ttl_seconds:
                self._items.pop(key, None)
                return None

            self._items.move_to_end(key)
            return value

    def mark_pending(self, task_type: str, track_id: Any) -> None:
        self._set(task_type, track_id, PENDING)

    def set_result(self, task_type: str, track_id: Any, confirmed: bool) -> None:
        self._set(task_type, track_id, bool(confirmed))

    def release_pending(self, task_type: str, track_id: Any) -> None:
        key = self._make_key(task_type, track_id)
        if key is None:
            return

        with self._lock:
            item = self._items.get(key)
            if item and item[0] == PENDING:
                self._items.pop(key, None)

    def _set(self, task_type: str, track_id: Any, value: Any) -> None:
        key = self._make_key(task_type, track_id)
        if key is None:
            return

        with self._lock:
            self._items[key] = (value, time.time())
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    @staticmethod
    def _make_key(task_type: str, track_id: Any) -> Optional[Tuple[str, int]]:
        if track_id is None:
            return None
        try:
            return str(task_type), int(track_id)
        except (TypeError, ValueError):
            return None


class LowConfidenceDroppingQueue:
    def __init__(self, maxsize: int):
        self.maxsize = max(1, int(maxsize))
        self._items: deque[VerificationEvent] = deque()
        self._condition = Condition()
        self._closed = False

    def put(self, event: VerificationEvent) -> Tuple[bool, Optional[VerificationEvent]]:
        with self._condition:
            if self._closed:
                return False, event

            dropped = None
            if len(self._items) >= self.maxsize:
                min_idx, min_event = min(
                    enumerate(self._items),
                    key=lambda item: item[1].confidence,
                )
                if event.confidence <= min_event.confidence:
                    return False, event

                dropped = min_event
                del self._items[min_idx]

            self._items.append(event)
            self._condition.notify()
            return True, dropped

    def get(self, timeout: float = 1.0) -> VerificationEvent:
        end_time = time.time() + timeout
        with self._condition:
            while not self._items and not self._closed:
                remaining = end_time - time.time()
                if remaining <= 0:
                    raise Empty
                self._condition.wait(timeout=remaining)

            if not self._items:
                raise Empty

            return self._items.popleft()

    def qsize(self) -> int:
        with self._condition:
            return len(self._items)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def is_closed(self) -> bool:
        with self._condition:
            return self._closed


class AsyncVerificationMixin:
    def _init_async_verification(
        self,
        task_type: str,
        max_queue_size: int = 50,
        report_queue_size: int = 200,
        verification_worker_count: int = 1,
        report_worker_count: int = 1,
        cache_ttl_seconds: float = 300.0,
        cache_max_entries: int = 4096,
        metrics_log_interval: float = 30.0,
    ) -> None:
        self.verification_task_type = str(task_type)
        self.verification_queue = LowConfidenceDroppingQueue(max_queue_size)
        self.report_queue = LowConfidenceDroppingQueue(report_queue_size)
        self.verification_worker_count = max(1, int(verification_worker_count))
        self.report_worker_count = max(1, int(report_worker_count))
        self.verification_cache = TrackVerificationCache(
            ttl_seconds=cache_ttl_seconds,
            max_entries=cache_max_entries,
        )
        self.monitor_metrics = VerificationMetrics()
        self.metrics_log_interval = float(metrics_log_interval)
        self._last_metrics_log_time = 0.0
        self.verification_threads = []
        self.report_threads = []
        self.metrics_file_lock = Lock()
        metrics_root = Path(os.getenv(
            "INTERFACE_METRICS_DIR",
            str(Path(__file__).resolve().parents[1] / "logs" / "interface_metrics")
        ))
        metrics_root.mkdir(parents=True, exist_ok=True)
        safe_task_type = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in self.verification_task_type)
        self.metrics_file_path = metrics_root / f"{safe_task_type}_{os.getpid()}.json"
        self._write_metrics_file("init")

    def _start_verification_worker(self) -> None:
        if self.verification_queue.is_closed():
            self.verification_queue = LowConfidenceDroppingQueue(self.verification_queue.maxsize)
        if self.report_queue.is_closed():
            self.report_queue = LowConfidenceDroppingQueue(self.report_queue.maxsize)

        self.verification_threads = [t for t in self.verification_threads if t.is_alive()]
        while len(self.verification_threads) < self.verification_worker_count:
            worker_idx = len(self.verification_threads) + 1
            thread = Thread(
                target=self._verification_worker,
                name=f"{self.verification_task_type}-verification-{worker_idx}",
                daemon=True,
            )
            thread.start()
            self.verification_threads.append(thread)
        logger.info(f"二次验证线程已启动: workers={len(self.verification_threads)}")

        self.report_threads = [t for t in self.report_threads if t.is_alive()]
        while len(self.report_threads) < self.report_worker_count:
            worker_idx = len(self.report_threads) + 1
            thread = Thread(
                target=self._report_worker,
                name=f"{self.verification_task_type}-report-{worker_idx}",
                daemon=True,
            )
            thread.start()
            self.report_threads.append(thread)
        logger.info(f"异步上传线程已启动: workers={len(self.report_threads)}")
        self._write_metrics_file("workers_started")

    def _stop_verification_worker(self, timeout: float = 5.0) -> None:
        if not hasattr(self, "verification_queue"):
            return

        self.verification_queue.close()
        for thread in list(self.verification_threads):
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"二次验证线程仍在处理任务，已超过等待时间: {thread.name}")

        self.report_queue.close()
        for thread in list(self.report_threads):
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"异步上传线程仍在处理任务，已超过等待时间: {thread.name}")
        self._write_metrics_file("workers_stopped")

    def _verification_worker(self) -> None:
        while self.running.is_set() or self.verification_queue.qsize() > 0:
            try:
                event = self.verification_queue.get(timeout=1.0)
                self.monitor_metrics.set_queue_length(self.verification_queue.qsize())
                self.monitor_metrics.set_report_queue_length(self.report_queue.qsize())
                self._write_metrics_file("verification_dequeue")
            except Empty:
                continue

            try:
                report_ready = self._process_verification_event(event)
                if report_ready:
                    self._enqueue_report_event(event)
            except Exception as exc:
                logger.error(f"二次验证事件处理异常: {exc}", exc_info=True)
            finally:
                self._release_pending_detections(event.detections)
                self.monitor_metrics.set_queue_length(self.verification_queue.qsize())
                self.monitor_metrics.set_report_queue_length(self.report_queue.qsize())
                self._write_metrics_file("verification_done")

        logger.info("二次验证线程已退出")

    def _process_verification_event(self, event: VerificationEvent) -> bool:
        raise NotImplementedError

    def _report_worker(self) -> None:
        while (
            self.running.is_set()
            or self.verification_queue.qsize() > 0
            or self._has_live_verification_worker()
            or self.report_queue.qsize() > 0
        ):
            try:
                event = self.report_queue.get(timeout=1.0)
                self.monitor_metrics.set_report_queue_length(self.report_queue.qsize())
                self._write_metrics_file("report_dequeue")
            except Empty:
                continue

            try:
                self._process_report_event(event)
            except Exception as exc:
                logger.error(f"异步上报事件处理异常: {exc}", exc_info=True)
            finally:
                self.monitor_metrics.set_report_queue_length(self.report_queue.qsize())
                self._write_metrics_file("report_done")

        logger.info("异步上报线程已退出")

    def _has_live_verification_worker(self) -> bool:
        return any(thread.is_alive() for thread in self.verification_threads)

    def _process_report_event(self, event: VerificationEvent) -> None:
        file_group = {
            "json": event.json_path,
            "original": event.orig_path,
            "detected": event.detected_path,
        }
        success = self.upload_files(file_group)
        self._record_report_result(success)
        logger.info(f"异步上传事件处理完成: event_id={event.event_id}, upload_success={success}")

    def _split_by_verification_cache(
        self,
        detections: Iterable[Dict[str, Any]],
        track_id_key: str = "track_id",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        pending = []
        cached_verified = []
        skipped = 0
        pending_track_ids = set()

        for detection in detections:
            track_id = detection.get(track_id_key)
            state = self.verification_cache.get(self.verification_task_type, track_id)
            if state is True:
                cached_verified.append(detection)
                self.monitor_metrics.inc_track_cache_hit()
            elif state == PENDING:
                skipped += 1
                self.monitor_metrics.inc_track_cache_hit()
            else:
                cache_key = TrackVerificationCache._make_key(self.verification_task_type, track_id)
                if cache_key is not None and cache_key in pending_track_ids:
                    skipped += 1
                    self.monitor_metrics.inc_track_cache_hit()
                    continue
                if cache_key is not None:
                    pending_track_ids.add(cache_key)
                pending.append(detection)

        return pending, cached_verified, skipped

    def _cache_verification_result(self, detection: Dict[str, Any], confirmed: bool) -> None:
        if confirmed:
            self.verification_cache.set_result(
                self.verification_task_type,
                detection.get("track_id"),
                True,
            )
        else:
            self.verification_cache.release_pending(
                self.verification_task_type,
                detection.get("track_id"),
            )

    def _enqueue_verification_event(
        self,
        frame,
        frame_timestamp: int,
        pending_detections: List[Dict[str, Any]],
        cached_detections: List[Dict[str, Any]],
        confidence_key: str,
    ) -> bool:
        accepted_any = False
        if pending_detections or cached_detections:
            accepted_any = self._enqueue_event_to_queue(
                frame=frame,
                frame_timestamp=frame_timestamp,
                pending_detections=pending_detections,
                cached_detections=cached_detections,
                confidence_key=confidence_key,
                target_queue=self.verification_queue,
                queue_name="二次验证队列",
            ) or accepted_any

        self.monitor_metrics.set_queue_length(self.verification_queue.qsize())
        self.monitor_metrics.set_report_queue_length(self.report_queue.qsize())
        self._write_metrics_file("verification_enqueue")
        return accepted_any

    def _enqueue_event_to_queue(
        self,
        frame,
        frame_timestamp: int,
        pending_detections: List[Dict[str, Any]],
        cached_detections: List[Dict[str, Any]],
        confidence_key: str,
        target_queue: LowConfidenceDroppingQueue,
        queue_name: str,
    ) -> bool:
        all_detections = pending_detections + cached_detections
        if not all_detections:
            return False

        event_id = str(uuid.uuid4())
        orig_path = self.upload_dir / f"{event_id}.jpg"
        detected_path = self.upload_dir / f"{event_id}.jpeg"
        json_path = self.upload_dir / f"{event_id}.json"
        max_confidence = max(float(det.get(confidence_key, 0.0)) for det in all_detections)

        if not cv2.imwrite(str(orig_path), frame):
            logger.error(f"保存原始帧失败: {orig_path}")
            return False

        event = VerificationEvent(
            event_id=event_id,
            frame_id=int(self.frame_count),
            frame_timestamp=int(frame_timestamp),
            orig_path=orig_path,
            detected_path=detected_path,
            json_path=json_path,
            detections=list(pending_detections),
            cached_detections=list(cached_detections),
            confidence=max_confidence,
        )

        self._mark_pending_detections(event.detections)
        accepted, dropped_event = target_queue.put(event)
        if dropped_event is not None:
            self.monitor_metrics.inc_queue_drop()
            self._delete_event_files(dropped_event)
            self._release_pending_detections(dropped_event.detections)
            logger.warning(
                f"{queue_name}已满，丢弃低置信度事件: {dropped_event.event_id}, "
                f"confidence={dropped_event.confidence:.5f}"
            )

        if not accepted:
            self.monitor_metrics.inc_queue_drop()
            self._delete_event_files(event)
            self._release_pending_detections(event.detections)
            logger.warning(
                f"{queue_name}已满，当前事件置信度较低已丢弃: {event.event_id}, "
                f"confidence={event.confidence:.5f}"
            )
            return False

        self.monitor_metrics.set_queue_length(self.verification_queue.qsize())
        self.monitor_metrics.set_report_queue_length(self.report_queue.qsize())
        logger.info(
            f"{queue_name}事件已入队: {event.event_id}, "
            f"待验证={len(event.detections)}, 缓存通过={len(event.cached_detections)}, "
            f"verify_queue={self.verification_queue.qsize()}, report_queue={self.report_queue.qsize()}"
        )
        return True

    def _enqueue_report_event(self, event: VerificationEvent) -> bool:
        accepted, dropped_event = self.report_queue.put(event)
        if dropped_event is not None:
            self.monitor_metrics.inc_queue_drop()
            self._delete_event_files(dropped_event)
            logger.warning(
                f"异步上传队列已满，丢弃低置信度上传事件: {dropped_event.event_id}, "
                f"confidence={dropped_event.confidence:.5f}"
            )

        if not accepted:
            self.monitor_metrics.inc_queue_drop()
            self._delete_event_files(event)
            logger.warning(
                f"异步上传队列已满，当前上传事件置信度较低已丢弃: {event.event_id}, "
                f"confidence={event.confidence:.5f}"
            )
            return False

        self.monitor_metrics.set_report_queue_length(self.report_queue.qsize())
        self._write_metrics_file("report_enqueue")
        logger.info(
            f"异步上传事件已入队: {event.event_id}, "
            f"report_queue={self.report_queue.qsize()}"
        )
        return True

    def _mark_pending_detections(self, detections: Iterable[Dict[str, Any]]) -> None:
        for detection in detections:
            self.verification_cache.mark_pending(
                self.verification_task_type,
                detection.get("track_id"),
            )

    def _release_pending_detections(self, detections: Iterable[Dict[str, Any]]) -> None:
        for detection in detections:
            self.verification_cache.release_pending(
                self.verification_task_type,
                detection.get("track_id"),
            )

    @staticmethod
    def _delete_event_files(event: VerificationEvent) -> None:
        existing_files = [
            str(file_path)
            for file_path in (event.orig_path, event.detected_path, event.json_path)
            if file_path.exists()
        ]
        if existing_files:
            logger.info(f"按配置保留事件文件，不执行删除: {existing_files}")

    def _log_monitor_metrics_if_needed(self) -> None:
        now = time.time()
        if now - self._last_metrics_log_time < self.metrics_log_interval:
            return

        self._last_metrics_log_time = now
        metrics = self.get_monitor_metrics()
        self._write_metrics_file("periodic")
        logger.info(f"运行监控指标: {metrics}")

    def get_monitor_metrics(self) -> Dict[str, Any]:
        self.monitor_metrics.set_queue_length(self.verification_queue.qsize())
        self.monitor_metrics.set_report_queue_length(self.report_queue.qsize())
        return self.monitor_metrics.snapshot()

    def _write_metrics_file(self, reason: str) -> None:
        try:
            metrics = self.get_monitor_metrics()
            payload = {
                "task_type": self.verification_task_type,
                "app_id": getattr(self, "app_id", self.verification_task_type),
                "task_id": getattr(self, "task_id", ""),
                "task_name": getattr(self, "task_name", ""),
                "box_id": getattr(self, "box_id", ""),
                "pid": os.getpid(),
                "running": bool(getattr(self, "running", None).is_set()) if hasattr(getattr(self, "running", None), "is_set") else False,
                "verification_worker_count": self.verification_worker_count,
                "report_worker_count": self.report_worker_count,
                "verification_queue_maxsize": self.verification_queue.maxsize,
                "report_queue_maxsize": self.report_queue.maxsize,
                "updated_at": time.time(),
                "reason": reason,
                "metrics": metrics,
            }
            with self.metrics_file_lock:
                tmp_path = self.metrics_file_path.with_suffix(".tmp")
                tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp_path.replace(self.metrics_file_path)
        except Exception as exc:
            logger.debug(f"写入子进程队列指标失败: {exc}")

    def _record_report_result(self, success: bool) -> None:
        if success:
            self.monitor_metrics.inc_report_success()
