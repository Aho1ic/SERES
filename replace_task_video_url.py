#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PROJECT_ROOT

TASK_DIR = str(PROJECT_ROOT / "task")
VIDEO_URL = ""  # 需要通过命令行参数指定


def replace_video_url(task_dir: str, video_url: str) -> None:
    task_path = Path(task_dir)
    if not task_path.exists():
        raise FileNotFoundError(f"任务目录不存在: {task_path}")
    if not task_path.is_dir():
        raise NotADirectoryError(f"不是目录: {task_path}")

    json_files = sorted(task_path.glob("*.json"))
    if not json_files:
        print(f"未找到JSON文件: {task_path}")
        return

    updated_count = 0
    failed_count = 0

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            old_video_url = data.get("video_url", "")
            data["video_url"] = video_url

            with json_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.write("\n")

            updated_count += 1
            print(f"[OK] {json_file.name}: {old_video_url} -> {video_url}")
        except Exception as exc:
            failed_count += 1
            print(f"[FAIL] {json_file.name}: {exc}")

    print(f"处理完成: 成功 {updated_count} 个, 失败 {failed_count} 个")


def parse_args():
    parser = argparse.ArgumentParser(description="批量替换任务JSON文件中的video_url")
    parser.add_argument(
        "--task-dir",
        default=TASK_DIR,
        help=f"任务JSON目录，默认: {TASK_DIR}",
    )
    parser.add_argument(
        "--video-url",
        default=VIDEO_URL,
        help=f"新的video_url，默认: {VIDEO_URL}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replace_video_url(args.task_dir, args.video_url)
