import requests
import os
import time
import json
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ALLROOT = FILE.parents[1]
for module_path in (ALLROOT, ROOT):
    module_path_str = str(module_path)
    if module_path_str not in sys.path:
        sys.path.insert(0, module_path_str)

from config import get_upload_url, get_auth_headers, UPLOAD_DIR


def test_upload(box_id='20'):
    """测试上传接口 - 使用固定目录"""
    RESULT_DIR = UPLOAD_DIR / 'helmet'
    os.makedirs(RESULT_DIR, exist_ok=True)

    files_to_upload = []

    json_files = list(RESULT_DIR.glob('*.json'))
    if json_files:
        latest_json = max(json_files, key=os.path.getctime)
        files_to_upload.append(("json", latest_json))
        print(f"找到JSON文件: {latest_json}")
    else:
        print("警告: 未找到JSON文件")

    image_files = list(RESULT_DIR.glob('*.jpeg')) + list(RESULT_DIR.glob('*.jpg'))
    for img_file in image_files:
        files_to_upload.append(("image", img_file))
        print(f"找到图片文件: {img_file}")

    if not files_to_upload:
        print("错误: 未找到任何可上传的文件！")
        return

    print(f"\n总共找到 {len(files_to_upload)} 个文件: {len(json_files)} JSON + {len(image_files)} 图片")

    upload_url = get_upload_url(sys_id=11, box_id=box_id, upload_type=1)
    headers = get_auth_headers()

    print(f"\nAPI配置:")
    print(f"URL: {upload_url}")
    print(f"头部: {headers}")

    print("\n" + "=" * 50)
    print("第一阶段: 上传JSON文件")
    print("=" * 50)

    json_uploaded = False
    for file_type, file_path in files_to_upload:
        if file_type != "json":
            continue

        try:
            print(f"\n上传JSON文件: {file_path.name}")
            with open(file_path, 'rb') as f:
                response = requests.post(
                    upload_url,
                    params=params,
                    files={'file': (file_path.name, f)},
                    headers=headers,
                    timeout=30
                )

                print(f"状态码: {response.status_code}")
                print(f"响应文本: {response.text[:200]}")

                if response.status_code == 200:
                    json_uploaded = True
                    print("✅ JSON上传成功")
                    try:
                        json_resp = response.json()
                        print("解析后的JSON响应:")
                        print(json.dumps(json_resp, indent=2, ensure_ascii=False))
                    except:
                        print("⚠️ 响应不是有效的JSON格式")
                else:
                    print(f"❌ JSON上传失败: {response.status_code}")
        except Exception as e:
            print(f"❌ 上传异常: {str(e)}")

    print("\n等待2秒...")
    time.sleep(2)

    print("\n" + "=" * 50)
    print("第二阶段: 上传图片文件")
    print("=" * 50)

    image_count = 0
    success_count = 0

    for file_type, file_path in files_to_upload:
        if file_type != "image":
            continue

        image_count += 1
        try:
            print(f"\n上传图片 ({image_count}/{len(image_files)}): {file_path.name}")
            with open(file_path, 'rb') as f:
                response = requests.post(
                    upload_url,
                    params=params,
                    files={'file': (file_path.name, f)},
                    headers=headers,
                    timeout=30
                )

                print(f"状态码: {response.status_code}")
                print(f"响应文本: {response.text[:200]}")

                if response.status_code == 200:
                    success_count += 1
                    print("✅ 图片上传成功")
                else:
                    print(f"❌ 图片上传失败: {response.status_code}")
        except Exception as e:
            print(f"❌ 上传异常: {str(e)}")

    print("\n" + "=" * 50)
    print("上传结果总结")
    print("=" * 50)
    print(f"总文件数: {len(files_to_upload)}")
    print(f"JSON文件: {'成功' if json_uploaded else '失败'}")
    print(f"图片文件: {success_count}/{len(image_files)} 成功")

    if not json_uploaded:
        print("\n警告: JSON文件上传失败，请检查接口是否要求先上传JSON文件")

    print("\n测试完成")


if __name__ == "__main__":
    print("=" * 60)
    print("上传接口测试工具 - 使用固定目录")
    print("=" * 60)
    print(f"扫描目录: {RESULT_DIR}")
    print("=" * 60)

    test_upload()