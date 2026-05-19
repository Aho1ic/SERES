import base64
import os
from pathlib import Path
from typing import Optional

import requests


def call_vllm_yes_no(
    image_path,
    question: str,
    logger=None,
    api_url: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> bool:
    image_path = Path(image_path)
    if not image_path.exists():
        if logger:
            logger.error(f"AI分析图片不存在: {image_path}")
        return False

    api_url = api_url or os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
    model_name = model_name or os.getenv("VLLM_MODEL_NAME", "Qwen3-VL-8B-Instruct")
    timeout = float(timeout if timeout is not None else os.getenv("VLLM_TIMEOUT", "60"))
    max_tokens = int(max_tokens if max_tokens is not None else os.getenv("VLLM_MAX_TOKENS", "900"))

    try:
        with image_path.open("rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        request_body = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        response = requests.post(
            api_url,
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        response_json = response.json()
        generated_text = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if logger:
            logger.info(f"vLLM AI分析输出: {generated_text}")
        return parse_yes_no(generated_text)
    except requests.exceptions.RequestException as exc:
        if logger:
            logger.error(f"vLLM请求失败: {exc}")
        return False
    except Exception as exc:
        if logger:
            logger.error(f"vLLM分析异常: {exc}", exc_info=True)
        return False


def parse_yes_no(text) -> bool:
    text_lower = str(text or "").strip().lower()
    if not text_lower:
        return False

    first_token = text_lower.replace("：", ":").split()[0].strip(".,;:!?'\"[]{}()")
    if first_token == "yes":
        return True
    if first_token == "no":
        return False

    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    return has_yes and not has_no
