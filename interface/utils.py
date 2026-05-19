# -*- coding: UTF-8 -*-
"""
公共工具模块
提取各检测模块的重复函数
"""
import os
import time
import requests
from pathlib import Path
from loguru import logger

import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ALLROOT = FILE.parents[1]
for module_path in (ALLROOT, ROOT):
    module_path_str = str(module_path)
    if module_path_str not in sys.path:
        sys.path.insert(0, module_path_str)

from config import get_auth_headers


def delete_files(file_group):
    """
    删除文件组中的所有文件

    Args:
        file_group: dict, 文件类型到文件路径的映射

    Returns:
        bool: 是否全部删除成功
    """
    try:
        for file_type, file_path in file_group.items():
            path = Path(file_path) if not isinstance(file_path, Path) else file_path
            if path.exists():
                path.unlink()
                logger.info(f"已删除文件: {path.name}")
            else:
                logger.warning(f"文件不存在，无法删除: {path.name}")
        return True
    except Exception as e:
        logger.error(f"删除文件时发生错误: {e}")
        return False


def upload_files(file_group, upload_url, max_retries=3):
    """
    上传文件组到服务器

    Args:
        file_group: dict, 文件类型到文件路径的映射
        upload_url: str, 上传URL
        max_retries: int, 最大重试次数

    Returns:
        bool: 是否上传成功
    """
    for attempt in range(max_retries):
        try:
            logger.debug(f"开始上传文件组 (尝试 {attempt+1}/{max_retries})")

            # 检查文件是否存在
            for file_type, file_path in file_group.items():
                path = Path(file_path) if not isinstance(file_path, Path) else file_path
                if not path.exists():
                    logger.error(f"文件不存在: {path.name}")
                    return False

            # 按顺序上传文件
            upload_order = ['json', 'original', 'detected']
            upload_success = True
            headers = get_auth_headers()

            for file_type in upload_order:
                if file_type in file_group:
                    file_path = file_group[file_type]
                    path = Path(file_path) if not isinstance(file_path, Path) else file_path

                    with open(path, 'rb') as f:
                        files = {'file': (path.name, f)}

                        logger.debug(f"正在上传文件: {path.name} 到 {upload_url}")
                        logger.debug(f"上传文件名: {path.name}, 大小: {os.path.getsize(path)} 字节")

                        response = requests.post(
                            upload_url,
                            files=files,
                            headers=headers,
                            timeout=30
                        )

                    logger.info(f"上传文件返回状态码: {response.status_code}")

                    if response.status_code not in [200, 201]:
                        logger.error(f"上传失败[{response.status_code}]: {response.text}")
                        upload_success = False
                        break
                    else:
                        logger.info(f"上传成功: {path.name}, 文件类型: {file_type}")
                else:
                    logger.warning(f"文件组中缺少类型: {file_type}")

            if upload_success:
                logger.info("所有文件上传成功")
                return True

            logger.warning("部分文件上传失败，保留本地文件")
            return False

        except Exception as e:
            logger.error(f"上传异常: {str(e)}", exc_info=True)

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    logger.warning("放弃上传文件组")
    return False


def is_valid_file(file_path):
    """
    检查文件扩展名是否合法

    Args:
        file_path: Path, 文件路径

    Returns:
        bool: 是否为合法的检测结果文件
    """
    path = Path(file_path) if not isinstance(file_path, Path) else file_path
    return path.is_file() and path.suffix.lower() in ('.json', '.jpeg', '.jpg')


def collect_file_groups(upload_dir):
    """
    收集上传目录中的文件组

    Args:
        upload_dir: Path, 上传目录

    Returns:
        dict: 文件组字典，key为文件前缀，value为文件类型到路径的映射
    """
    file_groups = {}
    upload_path = Path(upload_dir)

    for file_path in upload_path.iterdir():
        if is_valid_file(file_path):
            prefix = file_path.stem
            if prefix not in file_groups:
                file_groups[prefix] = {}

            ext = file_path.suffix.lower()
            if ext == '.jpg':
                file_groups[prefix]['original'] = file_path
            elif ext == '.jpeg':
                file_groups[prefix]['detected'] = file_path
            elif ext == '.json':
                file_groups[prefix]['json'] = file_path

    return file_groups


def upload_all_files(upload_dir, upload_url):
    """
    上传目录中所有完整的文件组

    Args:
        upload_dir: Path, 上传目录
        upload_url: str, 上传URL

    Returns:
        int: 成功上传的文件组数量
    """
    logger.info("开始上传所有未处理文件...")
    file_groups = collect_file_groups(upload_dir)

    uploaded_count = 0
    for prefix, group in file_groups.items():
        if len(group) == 3:  # 必须有json、original、detected三个文件
            if upload_files(group, upload_url):
                uploaded_count += 1

    logger.info(f"已处理 {uploaded_count} 组文件")
    return uploaded_count
