# SERES - Smart Video Surveillance Analysis System

[中文版本](README.md)

## 📋 Project Overview

SERES is an AI-powered intelligent video surveillance analysis system that focuses on real-time anomaly behavior detection and recognition in video streams. The system integrates multiple visual detection models and supports parallel processing across multiple scenarios and tasks.

## ✨ Core Features

### 🎯 Multi-Scenario Detection

- **Spy Camera Detection (SpyCam)**: Real-time detection of photo-taking behavior in video streams
- **Phone Call Detection (PhoneCall)**: Recognition of phone call postures
- **Smoke & Fire Detection (SmokeFire)**: Detection of smoke and flame safety hazards
- **Solar Panel Detection (SolarPanel)**: High-temperature point detection on solar panels

### 🔧 Technical Features

- **Real-time Video Stream Processing**: Support for RTSP/RTMP video stream input
- **YOLO Object Detection**: High-precision detection based on YOLO11
- **AI Secondary Verification**: Integration with VLLM large language models for intelligent verification
- **Asynchronous Processing Architecture**: Support for high-concurrency task processing
- **Object Tracking**: BoT-SORT based object tracking
- **Auto Upload**: Automatic upload of detection results to management platform

## 🏗️ System Architecture

```
SERES/
├── algorithm_api.py           # Core API service
├── config.py                  # Unified configuration management
├── interface/                 # Detection modules
│   ├── spycam_detect.py      # Photo detection
│   ├── phonecall_detect.py   # Phone call detection
│   ├── smokefire_detect.py   # Smoke & fire detection
│   ├── solarpanel_detect.py  # Solar panel detection
│   ├── async_verification.py # Async verification framework
│   ├── vllm_client.py        # VLLM client
│   └── utils.py              # Common utilities
├── omni/                      # AI analysis modules
├── weights/                   # Model weights (not committed)
└── task/                      # Task configurations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (GPU acceleration)
- FFmpeg (video stream processing)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

```bash
cp .env.example .env
# Edit .env file with actual configuration
```

### Start Service

```bash
# Start API service
python algorithm_api.py

# Run detection modules individually
python interface/spycam_detect.py --task task/your_task.json
python interface/phonecall_detect.py --task task/your_task.json
python interface/smokefire_detect.py --task task/your_task.json
```

## 📖 API Documentation

### Add Detection Task

```http
POST /add_task
Content-Type: application/json

{
    "video_url": "rtsp://...",
    "boxId": "device_001",
    "categoryType": "spyCam",
    "task_id": "task_001",
    "task_name": "Photo Detection Task"
}
```

### Stop Detection Task

```http
POST /stop_task
Content-Type: application/json

{
    "task_id": "task_001"
}
```

### Query Task Status

```http
GET /task_status?task_id=task_001
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | API server address | `http://10.60.73.189:9052` |
| `UPLOAD_URL` | File upload address | `http://192.168.0.120:44333/upload` |
| `APP_KEY_ID` | API authentication ID | - |
| `APP_KEY_SECRET` | API authentication secret | - |
| `VLLM_API_URL` | VLLM service address | `http://localhost:8000/v1/chat/completions` |
| `VLLM_MODEL_NAME` | VLLM model name | `Qwen3-VL-8B-Instruct` |

### Detection Parameters

| Variable | Description | Default |
|----------|-------------|---------|
| `FRAME_INTERVAL` | Frame interval | `2` |
| `CONF_THRESHOLD` | Confidence threshold | `0.5` |
| `WORKER_COUNT` | Worker thread count | `10` |
| `LLM_WORKER_COUNT` | LLM verification thread count | `1` |

## 📊 Performance Metrics

- **Video Stream Processing**: Supports 1080p@30fps real-time processing
- **Detection Latency**: Average < 100ms/frame
- **Concurrent Tasks**: Supports 10+ concurrent detection tasks
- **System Availability**: 99.9%+

## 🔒 Security Notes

- Sensitive configurations (API keys, passwords, etc.) should be configured via environment variables
- Do not commit `.env` files to the code repository
- Production environments recommend using a secrets management service

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## 📝 Development Guidelines

- Follow PEP 8 coding standards
- Use type annotations
- Write clear docstrings
- Keep code simple (KISS, YAGNI, DRY)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 👥 Team

- SERES Development Team

## 📧 Contact

For questions or suggestions, please use GitHub Issues

---

**Note**: Model weight files (*.pt, *.pth) are not included in the repository. Please obtain them separately.
