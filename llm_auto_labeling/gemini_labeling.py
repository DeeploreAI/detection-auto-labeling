#!/usr/bin/env python3
# Developed by ziliang

# standard
import json
import base64
import os
from PIL import Image
# third-party lib
import cv2
import ultralytics
from ultralytics.utils.plotting import Annotator, colors
import openai
from google import genai
from google.genai import types
# local


# --- API Key ---
LLM_SOURCE = "openrouter" # "openrouter", "gemini"
GEMINI_API_KEY = "AIzaSyDH9zGFyC2lJD7GKqeWpSnx67_k7EpkKyY"
OPENROUTER_API_KEY = "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"


def encode_image_to_pil(image_path):
    assert image_path is not None, "no input image path."
    # Read image with opencv
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Extract width and height
    h, w = image.shape[:2]

    # Read the image using OpenCV and convert it into the PIL format
    return Image.fromarray(image), w, h


def encode_image_to_base64(image_path):
    assert image_path is not None, "no input image path."
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def clean_results(results):
    """Clean the results for visualization."""
    return results.strip().removeprefix("```json").removesuffix("```").strip()


def save_yolo_annotation(image_path, bbox_xyxy, genus, species, name, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    # 获取图像文件名（不含扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 读取图像尺寸
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # 转换bbox坐标为YOLO格式 (归一化的中心点坐标和宽高)
    x1, y1, x2, y2 = bbox_xyxy
    center_x = (x1 + x2) / 2 / w
    center_y = (y1 + y2) / 2 / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    
    # 管理类别映射
    classes_file = os.path.join(output_dir, "classes.txt")
    genus_classes_file = os.path.join(output_dir, "genus_classes.txt")
    species_classes_file = os.path.join(output_dir, "species_classes.txt")
    
    # 读取现有类别
    genus_classes = []
    species_classes = []
    
    if os.path.exists(genus_classes_file):
        with open(genus_classes_file, 'r', encoding='utf-8') as f:
            genus_classes = [line.strip() for line in f.readlines()]
    
    if os.path.exists(species_classes_file):
        with open(species_classes_file, 'r', encoding='utf-8') as f:
            species_classes = [line.strip() for line in f.readlines()]
    
    # 添加新类别（如果不存在）
    if genus not in genus_classes:
        genus_classes.append(genus)
    if species not in species_classes:
        species_classes.append(species)
    
    # 获取类别ID
    genus_id = genus_classes.index(genus)
    species_id = species_classes.index(species)
    
    # 保存类别文件
    with open(genus_classes_file, 'w', encoding='utf-8') as f:
        for cls in genus_classes:
            f.write(f"{cls}\n")
    
    with open(species_classes_file, 'w', encoding='utf-8') as f:
        for cls in species_classes:
            f.write(f"{cls}\n")
    
    # 创建综合类别文件 (genus_species组合)
    combined_class = f"{genus}_{species}_({name})"
    combined_classes = []
    
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            combined_classes = [line.strip() for line in f.readlines()]
    
    if combined_class not in combined_classes:
        combined_classes.append(combined_class)
    
    combined_id = combined_classes.index(combined_class)
    
    with open(classes_file, 'w', encoding='utf-8') as f:
        for cls in combined_classes:
            f.write(f"{cls}\n")
    
    # 保存YOLO格式的标注文件
    label_file = os.path.join(output_dir, "labels", f"{image_name}.txt")
    with open(label_file, 'w') as f:
        # 主要用combined_id作为类别ID，也可以根据需要使用genus_id或species_id
        f.write(f"{combined_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    # 保存额外的标注信息文件（包含详细信息）
    detailed_label_file = os.path.join(output_dir, "labels", f"{image_name}_detailed.txt")
    with open(detailed_label_file, 'w', encoding='utf-8') as f:
        f.write(f"genus: {genus} (id: {genus_id})\n")
        f.write(f"species: {species} (id: {species_id})\n")
        f.write(f"combined: {combined_class} (id: {combined_id})\n")
        f.write(f"bbox_xywh: {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    return {
        "combined_id": combined_id,
        "genus_id": genus_id, 
        "species_id": species_id,
        "bbox_xywh": [center_x, center_y, width, height]
    }


def gemini_inference(prompt, pil_image, model_name, temp=0.5, max_tokens=1000):
    """
    Performs inference using Google Gemini 2.5 Pro Experimental model.

    Args:
        image (str or genai.types.Blob): The image input, either as a base64-encoded string or Blob object.
        prompt (str): A text prompt to guide the model's response.
        temp (float, optional): Sampling temperature for response randomness. Default is 0.5.

    Returns:
        str: The text response generated by the Gemini model based on the prompt and image.
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, pil_image],  # Provide both the text prompt and image as input
            config=types.GenerateContentConfig(
                temperature=temp,  # Controls creativity vs. determinism in output
                max_output_tokens=max_tokens
            ),
        )
        return response.text
    except Exception as e:
        print(e)


def openrouter_inference(prompt, base64_image, model_name, temp=0.5, max_tokens=1000):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=temp,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        print(f"API request failed: {e}")
    except FileNotFoundError as e:
        print(e)


if __name__ == '__main__':
    # configure llm
    import api_configuration as api
    client = api.configure_llm_client(source=LLM_SOURCE, api_key=OPENROUTER_API_KEY)
    model_name = "gpt-4.1" # gemini-2.5-pro, gpt-4.1
    temp = 0.5
    max_tokens = 2000

    # prompt v1
    # prompt = """
    # **Objective:** Detect the 2d bounding box of the primary marine animal in image, and classify its genus and species.
    # """
    # output_prompt = """
    # **Output Format:** Respond only a JSON object, no additional text.
    # The JSON must contain the following keys in this exact order:
    # 1. "box_2d": An array of four numbers representing the 2d bounding box coordinates [y_min, x_min, y_max, x_max].
    # 2. "genus": A word representing the genus.
    # 3. "species": A word representing the species.
    # """

    # prompt v2
    # prompt = """
    # **Objective** Your task is to act as a marine biology expert AI. From the provided image, you will detect the 2d bounding box of the primary marine animal, and classify its genus and species.
    # **Output** You must respond with only a single JSON object. Do not add any explanatory text before or after the JSON object. The JSON object must contain the following keys in this exact order: "box_2d": An array of four numbers representing the 2d bounding box coordinates [y_min, x_min, y_max, x_max]; "genus": A word representing the genus (scientific name); "species": A word representing the species (scientific name); "name": a word representing the common name of the species.
    # **EXAMPLE**
    # Input: [An image of a orange clownfish]
    # Output: {
    #           "box_2d": [301, 255, 802, 750],
    #           "genus": "Amphiprion",
    #           "species": "percula",
    #           "name": "orange clownfish"
    #         }
    # """

    # TODO: prompt for multi-object detection (currently not working)
    prompt = """
    **Your Task:** Act as a marine biology expert AI. From the provided image, you will identify the primary marine animal, determine its 2d bounding box, and classify its genus, species, and the common name of the species. 
    
    You must respond with only a single JSON object. Do not add any explanatory text or markdown formatting before or after the JSON.
    
    The JSON object must contain the following keys in this exact order:
    1.  "box_2d": An array of four numbers representing the 2d bounding box [x_min, y_min, x_max, y_max].
    2.  "genus": The genus name (scientific).
    3.  "species": The species name (scientific).
    4.  "name": The common name.

    ---

    **EXAMPLE:**

    **Input:** [An image of an orange clownfish]
    **Output:**
    {
      "box_2d": [x_min, y_min, x_max, y_max],
      "genus": "amphiprion",
      "species": "percula",
      "name": "orange clownfish"
    }

    ---

    **TASK:**

    **Input:**
    
    """

    # input image
    image_id = 9
    image_path = f"./test_data/images/test_{image_id}.jpg"
    pil_image, w, h = encode_image_to_pil(image_path)  # read img, width, height

    # llm inference
    if LLM_SOURCE == "gemini":
        results = gemini_inference(prompt, pil_image, model_name, temp, max_tokens)  # inference
    elif LLM_SOURCE == "openrouter":
        if "gemini" in model_name.lower():
            model_name = "google/" + model_name
        elif "gpt" in model_name.lower():
            model_name = "openai/" + model_name
        else:
            model_name = "google/" + model_name
            
        base64_image = encode_image_to_base64(image_path)
        results = openrouter_inference(prompt, base64_image, model_name, temp, max_tokens)
    else:
        print("not supported llm source.")
        results = None

    # clean llm results
    json_label = json.loads(clean_results(results))
    print(json_label)

    # object detection annotator
    annotator = Annotator(pil_image)  # initialize Ultralytics annotator

    # process the bbox coord for visualization
    if "gemini" in model_name.lower():
        # by default, gemini model return output with y coordinates first.
        # bbox coordinates is normalized to (0–1000).
        # refer to: https://cloud.google.com/vertex-ai/generative-ai/docs/bounding-box-detection
        x1, y1, x2, y2 = json_label["box_2d"]
        y1 = y1 / 1000 * h
        x1 = x1 / 1000 * w
        y2 = y2 / 1000 * h
        x2 = x2 / 1000 * w
    elif "gpt" in model_name.lower():
        # by default, gpt model return the original image coord
        x1, y1, x2, y2 = json_label["box_2d"]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

    # swap the coords, in case the order is reversed
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # label
    genus = json_label["genus"]
    species = json_label["species"]
    common_name = json_label["name"]

    # save the annotated visualization image by annotator
    annotator.box_label([x1, y1, x2, y2], label=genus, color=colors(0, True))
    Image.fromarray(annotator.result()).save(f"./test_data/labels/test_{image_id}_annotated.jpg")  # display the output
    
    # save yolo format annotation
    output_dir = "test_data"
    yolo_info = save_yolo_annotation(
        image_path=image_path,
        bbox_xyxy=[x1, y1, x2, y2],
        genus=genus,
        species=species,
        name=common_name,
        output_dir=output_dir
    )
    print(f"\nAnnotated by LLM: {image_path}")
    print(f"Genus: {genus} (ID: {yolo_info['genus_id']})")
    print(f"Species: {species} (ID: {yolo_info['species_id']})")
    print(f"Combined: {genus}_{species} (ID: {yolo_info['combined_id']})")
    print(f"bbox_xywh: {yolo_info['bbox_xywh']}")

    # create yolo config file
    # try:
    #     from create_yolo_config import create_yolo_config
    #     create_yolo_config(output_dir, "marine_animals.yaml")
    # except ImportError:
    #     print("提示: 可以运行 create_yolo_config.py 来生成YOLO训练配置文件")