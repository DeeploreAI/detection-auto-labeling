# Marine Auto-Labeling

A Python package for automated labeling of marine animals using Large Language Models (LLMs). This tool supports object detection, classification, and YOLO format annotation generation for marine biology research and computer vision applications.

## Features

- **Multi-LLM Support**: Works with Google Gemini and OpenRouter APIs
- **Marine Animal Detection**: Automatic detection and classification of marine animals in images
- **YOLO Format Export**: Generates YOLO-compatible annotations for training custom models
- **Species Classification**: Identifies genus, species, and common names
- **Flexible Configuration**: Easy API configuration and model selection

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from llm_auto_labeling import configure_llm_client, gemini_inference, openrouter_inference
from llm_auto_labeling import encode_image_to_pil, encode_image_to_base64, save_yolo_annotation

# Configure API client
api_key = "your_api_key_here"
client = configure_llm_client(source="gemini", api_key=api_key)

# Load and process image
image_path = "path/to/marine_animal.jpg"
pil_image, width, height = encode_image_to_pil(image_path)

# Define prompt for marine animal detection
prompt = """
Act as a marine biology expert AI. Detect the primary marine animal in the image, 
determine its 2d bounding box, and classify its genus, species, and common name.

Respond with only a JSON object containing:
- "box_2d": [x_min, y_min, x_max, y_max]
- "genus": scientific genus name
- "species": scientific species name  
- "name": common name
"""

# Run inference
result = gemini_inference(prompt, pil_image, "gemini-2.5-pro", temp=0.5, max_tokens=1000)

# Save YOLO annotations
save_yolo_annotation(
    image_path=image_path,
    bbox_xyxy=[x1, y1, x2, y2],
    genus="amphiprion",
    species="percula",
    name="orange clownfish",
    output_dir="./annotations"
)
```

## API Configuration

### Supported LLM Providers

- **Google Gemini**: `gemini-2.5-pro` and other Gemini models
- **OpenRouter**: Access to various models including `gpt-4.1`

```python
# Gemini configuration
gemini_client = configure_llm_client(source="gemini", api_key="your_gemini_key")

# OpenRouter configuration  
openrouter_client = configure_llm_client(source="openrouter", api_key="your_openrouter_key")
```

## Output Formats

### YOLO Annotations
The package generates YOLO-compatible annotations with:
- Normalized bounding box coordinates
- Class mappings for genus, species, and combined classifications
- Detailed annotation files with metadata

### File Structure
```
output_dir/
├── images/           # Source images
├── labels/           # YOLO format annotations (.txt)
├── classes.txt       # Combined class names
├── genus_classes.txt # Genus classifications
└── species_classes.txt # Species classifications
```

## Creating YOLO Training Configuration

```python
from llm_auto_labeling import create_yolo_config

# Generate YOLO configuration file
create_yolo_config(dataset_dir="./annotations", config_name="marine_animals.yaml")
```

## Development

### Project Structure
```
marine_auto_labeling/
├── __init__.py
├── api_configuration.py    # LLM API setup
├── gemini_labeling.py      # Core labeling functionality
└── create_yolo_config.py   # YOLO config generation
```

### Dependencies
- PIL (Pillow)
- OpenCV
- Ultralytics
- Google Generative AI
- OpenAI

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Author

ziliang