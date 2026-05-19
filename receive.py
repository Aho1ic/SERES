# -*- coding: UTF-8 -*-
from flask import Flask, request, jsonify
from pathlib import Path
from loguru import logger
import os
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import UPLOAD_DIR

app = Flask(__name__)

# 设置保存目录
SAVE_DIR = UPLOAD_DIR / "received"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"文件接收服务启动，保存目录: {SAVE_DIR}")


@app.route('/upload', methods=['POST'])
def upload_files():
    """接收图片和JSON文件"""
    try:
        if 'file' not in request.files:
            return jsonify({"code": 1, "error": "没有文件上传"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"code": 1, "error": "文件名为空"}), 400
        
        # 保存文件
        file_path = SAVE_DIR / file.filename
        file.save(str(file_path))
        
        logger.info(f"接收文件: {file.filename}, 大小: {file_path.stat().st_size} bytes")
        
        return jsonify({
            "code": 0,
            "message": "文件上传成功",
            "filename": file.filename,
            "path": str(file_path)
        }), 200
        
    except Exception as e:
        logger.error(f"文件上传失败: {e}", exc_info=True)
        return jsonify({"code": 1, "error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "running",
        "save_dir": str(SAVE_DIR),
        "files_count": len(list(SAVE_DIR.glob("*")))
    }), 200


if __name__ == "__main__":
    logger.info("启动文件接收服务...")
    logger.info(f"监听端口: 5000")
    logger.info(f"上传接口: http://localhost:5000/upload")
    logger.info(f"健康检查: http://localhost:5000/health")
    
    app.run(host='0.0.0.0', port=44333, debug=False)
