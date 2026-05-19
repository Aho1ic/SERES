import math
import time

import cv2
from loguru import logger


class StreamFrameTimestampResolver:
    """
    将视频后端提供的帧时间轴毫秒值映射为 Unix 秒。

    OpenCV/FFmpeg 对 RTMP 不一定暴露绝对 UTC 时间戳，通常只能拿到当前帧
    在流时间轴上的 POS_MSEC。这里用首帧读到时刻作为基准，将后续帧的
    POS_MSEC 增量换算到 Unix 时间，避免使用 JSON 生成时刻。
    """

    def __init__(self, name="video"):
        self.name = name
        self.base_wall_time = None
        self.base_pos_msec = None
        self.last_pos_msec = None
        self.warned_no_stream_ts = False

    def reset(self):
        self.base_wall_time = None
        self.base_pos_msec = None
        self.last_pos_msec = None
        self.warned_no_stream_ts = False

    def get_timestamp(self, cap, fallback_time=None):
        fallback_time = time.time() if fallback_time is None else float(fallback_time)
        pos_msec = self._read_pos_msec(cap)
        if pos_msec is None:
            if not self.warned_no_stream_ts:
                logger.warning(f"{self.name}: 视频后端未提供有效帧时间轴，created暂时回退为取帧时间")
                self.warned_no_stream_ts = True
            return int(fallback_time)

        if self._need_rebase(pos_msec):
            self.base_wall_time = fallback_time
            self.base_pos_msec = pos_msec
            logger.info(f"{self.name}: 初始化视频帧时间轴基准 pos_msec={pos_msec:.3f}")

        self.last_pos_msec = pos_msec
        return int(self.base_wall_time + (pos_msec - self.base_pos_msec) / 1000.0)

    @staticmethod
    def _read_pos_msec(cap):
        try:
            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        except Exception:
            return None

        if not math.isfinite(pos_msec) or pos_msec <= 0:
            return None
        return pos_msec

    def _need_rebase(self, pos_msec):
        if self.base_wall_time is None or self.base_pos_msec is None:
            return True
        if self.last_pos_msec is not None and pos_msec + 1000.0 < self.last_pos_msec:
            logger.warning(
                f"{self.name}: 检测到视频帧时间轴回退 "
                f"{self.last_pos_msec:.3f}->{pos_msec:.3f}，重新建立时间基准"
            )
            return True
        return False
