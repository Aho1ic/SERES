#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百炼API配置文件
"""

# 百炼API配置
BAILIAN_CONFIG = {
    "api_key": "sk-your-api-key-here",  # 请替换为你的百炼API key
    "api_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    "model_name": "qwen-vl-plus",  # 可选: qwen-vl-plus, qwen-vl-max
    "timeout": 60,
    "max_tokens": 900,
    "temperature": 0.0
}

# 使用说明:
# 1. 将 api_key 替换为你的实际百炼API key
# 2. 如果需要使用更强的模型，可以将 model_name 改为 "qwen-vl-max"
# 3. 可以根据需要调整 timeout、max_tokens、temperature 等参数