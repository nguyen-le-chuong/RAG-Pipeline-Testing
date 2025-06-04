import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.error("bitsandbytes not installed. Falling back to non-quantized models or CPU for VLM.")

def initialize_llm(args):
    model_name = "hotchpotch/query-crafter-japanese-Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True) if BITSANDBYTES_AVAILABLE else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config
    )
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.2,
        repetition_penalty=1.1,
        max_new_tokens=500,
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm

def initialize_vlm(args):
    if args.disable_vlm:
        return None, None
    vision_model_name = "HuggingFaceM4/idefics2-8b"
    vision_processor = AutoProcessor.from_pretrained(vision_model_name)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True) if BITSANDBYTES_AVAILABLE else None
    vision_model = AutoModelForVision2Seq.from_pretrained(
        vision_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() and BITSANDBYTES_AVAILABLE else "cpu",
        attn_implementation="eager",
        quantization_config=quantization_config
    )
    return vision_model, vision_processor

def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="oshizo/japanese-e5-mistral-1.9b")