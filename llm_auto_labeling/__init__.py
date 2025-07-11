"""
Marine Auto-Labeling Package

A Python package for automated labeling of marine animals using Large Language Models (LLMs).
Supports object detection, classification, and YOLO format annotation generation.
"""

__version__ = "0.1.0"
__author__ = "ziliang"

from .api_configuration import configure_llm_client
from .gemini_labeling import (
    encode_image_to_pil,
    encode_image_to_base64,
    clean_results,
    save_yolo_annotation,
    gemini_inference,
    openrouter_inference
)
from .create_yolo_config import create_yolo_config

__all__ = [
    "configure_llm_client",
    "encode_image_to_pil", 
    "encode_image_to_base64",
    "clean_results",
    "save_yolo_annotation",
    "gemini_inference",
    "openrouter_inference",
    "create_yolo_config"
]