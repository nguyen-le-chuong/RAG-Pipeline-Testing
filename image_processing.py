import logging
import torch
from PIL import Image

logger = logging.getLogger(__name__)

def process_image(image_path, vision_model, vision_processor, disable_vlm=False):
    if disable_vlm or vision_model is None:
        logger.warning(f"VLM disabled, skipping {image_path}")
        return None, ""
    image = Image.open(image_path).convert("RGB")
    type_prompt = "この画像の種類を特定してください（例：図、写真、表、グラフ、チャート）。<image>"
    inputs = vision_processor(text=type_prompt, images=[image], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = vision_model.generate(**inputs, max_new_tokens=50)
    image_type = vision_processor.decode(outputs[0], skip_special_tokens=True).replace(type_prompt, "").strip()

    if "diagram" in image_type.lower():
        desc_prompt = "この図を詳しく説明し、主要な構成要素（例：ノード、エッジ、ラベル）を特定してください。<image>"
    elif "photo" in image_type.lower():
        desc_prompt = "この写真について説明し、主要な被写体と文脈に焦点を当ててください。<image>"
    elif "table" in image_type.lower():
        desc_prompt = "この表の内容を抽出して説明してください。<image>"
    elif "graph" in image_type.lower() or "chart" in image_type.lower():
        desc_prompt = "このグラフ/チャートについて説明し、その種類と主要なデータポイントを挙げてください。<image>"
    else:
        desc_prompt = "この画像を詳しく説明し、主要な構成要素を特定してください。<image>"

    
    inputs = vision_processor(text=desc_prompt, images=[image], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = vision_model.generate(**inputs, max_new_tokens=300)
    description = vision_processor.decode(outputs[0], skip_special_tokens=True).replace(desc_prompt, "").strip()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return image_type, description