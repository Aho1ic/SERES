# SERES - 智能视频监控分析系统

[English Version](README_EN.md)

## 📋 项目简介

SERES 是一个基于 AI 的智能视频监控分析系统，专注于实时视频流中的异常行为检测和识别。系统集成了多种视觉检测模型，支持多场景、多任务的并行处理。

## ✨ 核心功能

### 🎯 多场景检测

- **拍照姿势检测 (SpyCam)**: 实时检测视频流中的拍照行为
- **打电话检测 (PhoneCall)**: 识别人员打电话的姿势
- **烟火检测 (SmokeFire)**: 检测烟雾和火焰等安全隐患
- **太阳能板检测 (SolarPanel)**: 太阳能板高温点检测

### 🔧 技术特性

- **实时视频流处理**: 支持 RTSP/RTMP 视频流接入
- **YOLO 目标检测**: 基于 YOLO11 的高精度目标检测
- **AI 二次验证**: 集成 VLLM 大模型进行智能验证
- **异步处理架构**: 支持高并发任务处理
- **目标追踪**: 基于 BoT-SORT 的目标追踪
- **自动上报**: 检测结果自动上传至管理平台

## 🏗️ 系统架构

```
SERES/
├── algorithm_api.py           # 核心 API 服务
├── config.py                  # 统一配置管理
├── interface/                 # 检测模块
│   ├── spycam_detect.py      # 拍照检测
│   ├── phonecall_detect.py   # 打电话检测
│   ├── smokefire_detect.py   # 烟火检测
│   ├── solarpanel_detect.py  # 太阳能板检测
│   ├── async_verification.py # 异步验证框架
│   ├── vllm_client.py        # VLLM 客户端
│   └── utils.py              # 公共工具
├── omni/                      # AI 分析模块
├── weights/                   # 模型权重（不提交）
└── task/                      # 任务配置
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU 加速)
- FFmpeg (视频流处理)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入实际配置
```

### 启动服务

```bash
# 启动 API 服务
python algorithm_api.py

# 单独运行检测模块
python interface/spycam_detect.py --task task/your_task.json
python interface/phonecall_detect.py --task task/your_task.json
python interface/smokefire_detect.py --task task/your_task.json
```

## 📖 API 接口

### 添加检测任务

```http
POST /add_task
Content-Type: application/json

{
    "video_url": "rtsp://...",
    "boxId": "device_001",
    "categoryType": "spyCam",
    "task_id": "task_001",
    "task_name": "拍照检测任务"
}
```

### 停止检测任务

```http
POST /stop_task
Content-Type: application/json

{
    "task_id": "task_001"
}
```

### 查询任务状态

```http
GET /task_status?task_id=task_001
```

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `API_BASE_URL` | API 服务器地址 | `http://10.60.73.189:9052` |
| `UPLOAD_URL` | 文件上传地址 | `http://192.168.0.120:44333/upload` |
| `APP_KEY_ID` | API 认证 ID | - |
| `APP_KEY_SECRET` | API 认证密钥 | - |
| `VLLM_API_URL` | VLLM 服务地址 | `http://localhost:8000/v1/chat/completions` |
| `VLLM_MODEL_NAME` | VLLM 模型名称 | `Qwen3-VL-8B-Instruct` |

### 检测参数

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `FRAME_INTERVAL` | 帧间隔 | `2` |
| `CONF_THRESHOLD` | 置信度阈值 | `0.5` |
| `WORKER_COUNT` | 工作线程数 | `10` |
| `LLM_WORKER_COUNT` | LLM 验证线程数 | `1` |

## 📊 性能指标

- **视频流处理**: 支持 1080p@30fps 实时处理
- **检测延迟**: 平均 < 100ms/帧
- **并发任务**: 支持 10+ 并发检测任务
- **系统可用性**: 99.9%+

## 🔒 安全说明

- 敏感配置（API 密钥、密码等）请通过环境变量配置
- 不要将 `.env` 文件提交到代码仓库
- 生产环境建议使用密钥管理服务

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📝 开发规范

- 遵循 PEP 8 编码规范
- 使用类型注解
- 编写清晰的文档字符串
- 保持代码简洁（KISS、YAGNI、DRY）

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👥 团队

- SERES 开发团队

## 📧 联系方式

如有问题或建议，请通过 GitHub Issues 反馈

---

**注意**: 模型权重文件 (*.pt, *.pth) 不包含在代码仓库中，请单独获取。
