# -*- coding: UTF-8 -*-
"""
统一配置管理模块
所有配置优先从环境变量读取，支持 .env 文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# 服务器 API 配置
# ============================================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://10.60.73.189:9052")
UPLOAD_URL = os.getenv("UPLOAD_URL", "http://192.168.0.120:44333/upload")

# 上传接口完整路径
def get_upload_url(sys_id=11, box_id="", upload_type=1):
    """构建上传URL"""
    base = os.getenv("UPLOAD_API_URL", f"{API_BASE_URL}/admin-api/aicallers/upload")
    return f"{base}?sysId={sys_id}&boxId={box_id}&type={upload_type}"

# 轮询接口
AICALLERS_POLLING_URL_1 = os.getenv(
    "AICALLERS_POLLING_URL_1",
    f"{API_BASE_URL}/admin-api/aicallers/aiTask/linshiSendAlgorithmPic/1"
)
AICALLERS_POLLING_URL_2 = os.getenv(
    "AICALLERS_POLLING_URL_2",
    f"{API_BASE_URL}/admin-api/aicallers/aiTask/linshiSendAlgorithmPic/2"
)

# ============================================================
# API 认证凭证
# ============================================================
APP_KEY_ID = os.getenv("APP_KEY_ID", "28813140")
APP_KEY_SECRET = os.getenv("APP_KEY_SECRET", "Rxaq46q8EAP5msGfnhBN")

def get_auth_headers():
    """获取API认证头"""
    return {
        "AppKeyID": APP_KEY_ID,
        "AppKeySecret": APP_KEY_SECRET
    }

# ============================================================
# RTMP 流配置
# ============================================================
RTMP_SERVER = os.getenv("RTMP_SERVER", "10.60.73.189")
RTMP_PORT = int(os.getenv("RTMP_PORT", "1935"))

def get_rtmp_source_url(drone_sn):
    """获取RTMP源流地址"""
    return f"rtmp://{RTMP_SERVER}:{RTMP_PORT}/ly/{drone_sn}"

def get_rtmp_output_url(drone_sn, task_id):
    """获取RTMP输出流地址"""
    return f"rtmp://{RTMP_SERVER}:{RTMP_PORT}/ly/{drone_sn}_{task_id}"

# ============================================================
# VLLM AI 模型配置
# ============================================================
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen3-VL-8B-Instruct")
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "60"))
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "900"))

# ============================================================
# 模型权重路径
# ============================================================
WEIGHTS_DIR = PROJECT_ROOT / "weights"

def get_model_path(model_name):
    """获取模型权重路径"""
    return WEIGHTS_DIR / f"{model_name}.pt"

# 各检测模型路径
SPYCAM_MODEL_PATH = os.getenv("SPYCAM_MODEL_PATH", str(WEIGHTS_DIR / "spycam.pt"))
PHONECALL_MODEL_PATH = os.getenv("PHONECALL_MODEL_PATH", str(WEIGHTS_DIR / "phonecall.pt"))
SMOKEFIRE_MODEL_PATH = os.getenv("SMOKEFIRE_MODEL_PATH", str(WEIGHTS_DIR / "smokefire.pt"))
SOLARPANEL_MODEL_PATH = os.getenv("SOLARPANEL_MODEL_PATH", str(WEIGHTS_DIR / "solarpanel.pt"))

# ============================================================
# 队列和并发配置
# ============================================================
TASK_QUEUE_MAXSIZE = int(os.getenv("TASK_QUEUE_MAXSIZE", "200"))
WORKER_COUNT = int(os.getenv("WORKER_COUNT", "10"))

LLM_QUEUE_MAXSIZE = int(os.getenv("LLM_QUEUE_MAXSIZE", "100"))
LLM_WORKER_COUNT = max(1, int(os.getenv("LLM_WORKER_COUNT", "1")))
LLM_MONITOR_INTERVAL = max(1.0, float(os.getenv("LLM_MONITOR_INTERVAL", "10")))

VERIFICATION_QUEUE_MAXSIZE = int(os.getenv("VERIFICATION_QUEUE_MAXSIZE", "50"))
REPORT_QUEUE_MAXSIZE = int(os.getenv("REPORT_QUEUE_MAXSIZE", "200"))
AI_VERIFICATION_WORKERS = int(os.getenv("AI_VERIFICATION_WORKERS", "1"))
REPORT_WORKERS = int(os.getenv("REPORT_WORKERS", "1"))

# ============================================================
# 检测参数配置
# ============================================================
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", "2"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.5"))
MAX_REQUESTS_PER_SECOND = int(os.getenv("MAX_REQUESTS_PER_SECOND", "1000"))

# 轮询间隔（秒）
POLLING_INTERVAL = float(os.getenv("POLLING_INTERVAL", "2"))

# ============================================================
# 缓存配置
# ============================================================
TRACK_VERIFY_CACHE_TTL = float(os.getenv("TRACK_VERIFY_CACHE_TTL", "300"))

# ============================================================
# 上传目录配置
# ============================================================
UPLOAD_DIR = PROJECT_ROOT / "upload"

def get_upload_dir(sub_dir=""):
    """获取上传目录"""
    if sub_dir:
        path = UPLOAD_DIR / sub_dir
    else:
        path = UPLOAD_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path

# ============================================================
# 日志配置
# ============================================================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
