#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import base64
import sys
import requests
import json

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
    
    api_url = "http://localhost:8000/v1/chat/completions"
    model_name = "Qwen3-VL-8B-Instruct"
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_image_string = base64.b64encode(image_file.read()).decode('utf-8')
        
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
        response.raise_for_status()
        response_json = response.json()
        print("\nAPI响应：")
        
        if 'choices' in response_json and len(response_json['choices']) > 0:
            generated_text = response_json['choices'][0]['message']['content'].strip()
            print(f"生成的文本: {generated_text}")
            
            # 解析简化的输出格式
            # 查找 yes 或 no 答案
            final_result = None
            text_lower = generated_text.lower().strip()
            
            # 直接查找yes或no
            if 'yes' in text_lower:
                final_result = 'yes'
            elif 'no' in text_lower:
                final_result = 'no'
            else:
                # 如果完全无法解析，默认为no
                print(f"警告: 无法解析模型输出，默认判断为no")
                final_result = 'no'
            
            formatted_result = f"[{{'text': '{final_result}'}}]"
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
        print("1. 分析单张图片: python phonecall_analysis.py /home/njly/图片/phone.jpg")
        print("2. 分析单张图片(带bbox): python phonecall_analysis.py /home/njly/图片/phone.jpg --bbox x1,y1,x2,y2")
        print("3. 批量分析: python phonecall_analysis.py --batch <图片文件夹> [输出文件]")
        print("\n示例:")
        print("  python phonecall_analysis.py data/test.jpg")
        print("  python phonecall_analysis.py data/test.jpg --bbox 100,50,300,400")
        print("  python phonecall_analysis.py --batch upload/smokefire/ results.json")
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
        bbox = None
        
        # 检查是否有bbox参数
        if len(sys.argv) >= 4 and sys.argv[2] == '--bbox':
            try:
                bbox_str = sys.argv[3]
                bbox_parts = bbox_str.split(',')
                if len(bbox_parts) == 4:
                    bbox = [int(x) for x in bbox_parts]
                    print(f"使用bbox: {bbox}")
                else:
                    print("警告: bbox格式错误，应为 x1,y1,x2,y2")
            except ValueError:
                print("警告: bbox参数解析失败，忽略bbox")
        
        analyze_image_for_smoke_fire(image_path, bbox=bbox)

if __name__ == '__main__':
    main() 