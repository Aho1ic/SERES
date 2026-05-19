#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片分析工具
用于检测图片中的烟雾和火焰
"""

import os
import base64
import sys
import requests
import json

def analyze_image_for_smoke_fire(image_path, question=None):
    """
    分析图片中的烟雾和火焰
    
    Args:
        image_path: 图片文件路径
        question: 自定义问题，默认为烟雾火焰检测
    """
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        return None
    
    if question is None:
        question = """请仔细观察图片的每个角落，判断是否有真实的烟雾或火焰（包括细小的轻烟和微弱的火苗）。

烟雾或火焰的特征（请特别注意细微迹象）：
1. 烟雾：灰色、白色或黑色的烟雾状物体，形状不规则，向上飘散
   - 包括：细小的轻烟、淡淡的烟雾、刚开始冒出的烟
   - 即使烟雾很淡、很小也要识别出来
2. 火焰：橙色、红色或黄色的火光，有燃烧特征
   - 包括：微弱的火苗、小火星、刚起的小火
   - 即使火焰很小、很弱也要识别出来

不是烟雾或火焰的情况：
- 夜晚的车灯和车尾灯
- 保安制服上红色的袖章
- 红色标识桶
- 红色灭火器
- 红色砖块
- 刹车灯的红光
- 地上的白斑不是烟
- 云雾、水汽、灰尘
- 暗处的左右2块红色的灯

请只回答yes或no：
- yes：确认有真实的烟雾或火焰（包括细小轻烟和微弱火苗）
- no：没有烟雾或火焰

答案："""
    
    print(f"正在分析图片: {image_path}")
    print(f"问题: {question}")
    print("-" * 50)
    
    # vLLM API 端点和模型名称
    api_url = "http://localhost:8000/v1/chat/completions"
    model_name = "Qwen3-VL-8B-Instruct"
    
    try:
        # 读取图片并进行Base64编码
        with open(image_path, "rb") as image_file:
            encoded_image_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建请求体
        request_body = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image_string}"
                            }
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ],
            "max_tokens": 900,
            "temperature": 0.0
        }
        
        print("正在向vLLM API发送请求...")
        response = requests.post(
            api_url,
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()  # 如果请求失败，抛出HTTPError
        
        # 处理API响应
        response_json = response.json()
        print("\nAPI响应：")
        
        if 'choices' in response_json and len(response_json['choices']) > 0:
            generated_text = response_json['choices'][0]['message']['content']
            print(f"生成的文本: {generated_text}")
            
            # 格式化输出结果，保持与原来兼容
            if "yes" in generated_text.lower():
                formatted_result = "[{'text': 'yes'}]"
            elif "no" in generated_text.lower():
                formatted_result = "[{'text': 'no'}]"
            else:
                formatted_result = f"[{{'text': '{generated_text}'}}]"
            
            print(formatted_result)
            return formatted_result
        else:
            print("未从API响应中获取到有效的文本。")
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
        print("1. 分析单张图片: python image_analysis.py <图片路径>")
        print("2. 批量分析: python image_analysis.py --batch <图片文件夹> [输出文件]")
        print("\n示例:")
        print("  python image_analysis.py data/test.jpg")
        print("  python image_analysis.py --batch upload/smokefire/ results.json")
        return
    
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("错误: 批量分析需要指定图片文件夹")
            return
        
        image_folder = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else 'analysis_results.json'
        batch_analyze_images(image_folder, output_file)
    else:
        image_path = sys.argv[1]
        analyze_image_for_smoke_fire(image_path)

if __name__ == '__main__':
    main() 