#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import base64
import sys
import requests
import json
from config import BAILIAN_CONFIG

def analyze_image_for_smoke_fire(image_path, question=None, bbox=None):
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        return None
    
    if question is None:
        bbox_hint = ""
        if bbox:
            bbox_hint = f"\n\n重点关注图片中坐标 ({bbox[0]}, {bbox[1]}) 到 ({bbox[2]}, {bbox[3]}) 区域内的人物。"
        
        question = f"""请仔细观察图片，判断是否有人正在打电话。{bbox_hint}

打电话的特征：
1. 必须手上拿着手机，这个条件严格限制
2. 人物正在走路
3. 一只手臂抬起，手部靠近耳朵位置（距离头部很近）

不是打电话的情况：
- 只是手臂抬起但没有靠近耳朵
- 在挠头、整理头发、遮阳、吃东西等动作
- 手机在手中但没有贴近耳朵（如看手机、拿着手机走路）
- 手部靠近头部但没有拿手机
- 手拿着东西吃饭
- 用手在抠鼻子
- 用手在揉眼睛
- 整理帽子
- 整理头发
- 手举起来伸懒腰
- 拿着手机走路
- 挠耳朵
- 手上没手机是在抽烟
- 人如果是坐着的/蹲着的一律不通过
- 如果两只手都抬起来,那么一定不是在打电话

请只回答yes或no：
- yes：确认有人正在打电话（手机贴近耳朵）
- no：没有人在打电话或不确定

答案："""
    
    print(f"正在分析图片: {image_path}")
    print(f"问题: {question}")
    print("-" * 50)
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_image_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 百炼API请求格式
        request_body = {
            "model": BAILIAN_CONFIG["model_name"],
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": f"data:image/jpeg;base64,{encoded_image_string}"
                            },
                            {
                                "text": question
                            }
                        ]
                    }
                ]
            },
            "parameters": {
                "max_tokens": BAILIAN_CONFIG["max_tokens"],
                "temperature": BAILIAN_CONFIG["temperature"]
            }
        }
        
        headers = {
            "Authorization": f"Bearer {BAILIAN_CONFIG['api_key']}",
            "Content-Type": "application/json"
        }
        
        print("正在向百炼API发送请求...")
        response = requests.post(
            BAILIAN_CONFIG["api_url"],
            json=request_body,
            headers=headers,
            timeout=BAILIAN_CONFIG["timeout"]
        )
        response.raise_for_status()
        response_json = response.json()
        print("\nAPI响应：")
        
        if 'output' in response_json and 'choices' in response_json['output']:
            generated_text = response_json['output']['choices'][0]['message']['content'].strip().lower()
            print(f"生成的文本: {generated_text}")
            
            # 更严格的判断逻辑：只有明确的yes才返回yes，其他情况都返回no
            if generated_text.startswith('yes') or generated_text == 'yes':
                formatted_result = "[{'text': 'yes'}]"
            elif generated_text.startswith('no') or generated_text == 'no':
                formatted_result = "[{'text': 'no'}]"
            else:
                # 如果回答不明确，默认为no（保守策略，减少误报）
                print(f"警告: 模型回答不明确，默认判断为no")
                formatted_result = "[{'text': 'no'}]"
            
            print(formatted_result)
            return formatted_result
        else:
            print("未从API响应中获取到有效的文本。")
            print(f"完整响应: {response_json}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"请求失败：{e}")
        return None
    except json.JSONDecodeError:
        print("API响应不是有效的JSON格式。")
        return None
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        return None

def batch_analyze_images(image_folder, output_file=None):
    """
    批量分析文件夹中的图片
    
    Args:
        image_folder: 图片文件夹路径
        output_file: 输出结果文件路径
    """
    if not os.path.exists(image_folder):
        print(f"错误: 文件夹不存在: {image_folder}")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    image_files = []
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    if not image_files:
        print(f"在文件夹 {image_folder} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始批量分析...")
    
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 分析: {os.path.basename(image_path)}")
        result = analyze_image_for_smoke_fire(image_path)
        if result:
            results.append({
                'image': os.path.basename(image_path),
                'analysis': result
            })

    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n分析结果已保存到: {output_file}")
    
    return results

def main():
    """主函数"""
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("1. 分析单张图片: python test_image_analysis.py /home/njly/图片/phone.jpg")
        print("2. 批量分析: python test_image_analysis.py --batch <图片文件夹> [输出文件]")
        print("\n示例:")
        print("  python test_image_analysis.py data/test.jpg")
        print("  python test_image_analysis.py --batch upload/smokefire/ results.json")
        print("\n注意: 请先在config.py中设置你的百炼API key")
        return
    
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("错误: 批量分析需要指定图片文件夹")
            return
        
        image_folder = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else 'analysis_results.json'
        batch_analyze_images(image_folder, output_file)
    else:
        # 检查API key是否设置
        if BAILIAN_CONFIG["api_key"] == "sk-your-api-key-here":
            print("错误: 请先在config.py中设置你的百炼API key")
            print("修改 config.py 中的 api_key 为你的实际API key")
            return
            
        image_path = sys.argv[1]
        analyze_image_for_smoke_fire(image_path)

if __name__ == '__main__':
    main() 