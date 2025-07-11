#!/usr/bin/env python3
"""
Convert Fishnet dataset format to YOLO format.

Current Fishnet format:
- Images: organized by class folders (e.g., Acanthuridae/*.jpg)
- Labels: organized by class folders (e.g., Acanthuridae/*.txt)
- Label format: "class_name x1 y1 x2 y2" (absolute coordinates)

Target YOLO format:
- Images: train/images/*.jpg, val/images/*.jpg, test/images/*.jpg
- Labels: train/labels/*.txt, val/labels/*.txt, test/labels/*.txt
- Label format: "class_id x_center y_center width height" (normalized coordinates)
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image
import yaml


def convert_fishnet_to_yolo(fishnet_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Convert Fishnet dataset format to YOLO format.
    
    Args:
        fishnet_dir (str): Path to the fishnet dataset directory
        output_dir (str): Path to output YOLO format directory
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
    """
    fishnet_path = Path(fishnet_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get all class folders
    image_dir = fishnet_path / 'images'
    label_dir = fishnet_path / 'bbox'
    
    if not image_dir.exists() or not label_dir.exists():
        raise ValueError(f"Images or labels directory not found in {fishnet_dir}")
    
    class_folders = [d for d in image_dir.iterdir() if d.is_dir()]
    
    # Create class name to ID mapping
    id_to_class = {idx: cls_folder.name for idx, cls_folder in enumerate(sorted(class_folders))}
    
    # Save id_to_class mapping to YAML file
    with open(output_path / 'fishnet_classes.yaml', 'w') as f:
        yaml.dump(id_to_class, f, default_flow_style=False)  # flow_style: all key-value pairs in a single line, block style: each key-value pair in a line
    
    # Process each class
    all_files = []
    
    for class_id in id_to_class.keys():
        class_name = id_to_class[class_id]
        
        # Get all image files for this class
        class_images_files = (image_dir / class_name).iterdir()
        # image_files = list((image_dir / class_name).glob('*.jpg'))  #  glob: finds files matching a specified pattern / image_dir.glob("*/")
        
        for image_file in class_images_files:
            # Check if corresponding label file exists
            label_file = label_dir / class_name / f"{image_file.stem}.txt"  # .stem attribute of a pathlib.Path object returns the filename without the extension
            
            if label_file.exists():
                all_files.append((image_file, label_file, class_name, class_id))
            else:
                print(ValueError(f"Label file {image_file.stem} in class {class_name} not found"))
                continue
    
    # Shuffle files for random split
    random.shuffle(all_files)
    
    # Calculate split indices
    total_files = len(all_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split files
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    
    # Process each split
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        print('='*60)
        print(f"Processing {split_name} split: {len(files)} files")
        
        for image_file, label_file, class_name, class_id in files:
            # Copy image file
            dst_image = output_path / 'images'/ split_name / image_file.name
            shutil.copy2(image_file, dst_image)
            
            # Convert label file (cx, cy, w, h) normalized
            yolo_label_file = output_path / 'labels'/ split_name / f"{image_file.stem}.txt"
            class_to_id = {value:key for key, value in id_to_class.items()}
            convert_label_file(label_file, yolo_label_file, class_to_id, image_file)
    
    print(f"Conversion complete!")
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")
    print(f"Classes: {len(id_to_class)}")


def convert_label_file(input_label_file, output_label_file, class_to_id, image_file):
    """
    Convert a single label file from Fishnet format to YOLO format.
    
    Args:
        input_label_file (Path): Input label file path
        output_label_file (Path): Output label file path
        class_to_id (dict): Mapping from class name to class ID
        image_file (Path): Corresponding image file for getting dimensions
    """
    # Get image dimensions
    with Image.open(image_file) as img:
        img_width, img_height = img.size
    
    # Read and convert labels
    yolo_labels = []
    
    with open(input_label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                print(ValueError("Org label is missing one of the cls, x_min, y_min, x_max, y_max"))
                continue
            
            class_name = parts[0]
            x1, y1, x2, y2 = map(float, parts[1:5])
            assert x2 > x1 and y2 > y1
            
            # Convert to YOLO format
            class_id = class_to_id[class_name]
            
            # Convert absolute coordinates to normalized coordinates
            x_center = (x1 + x2) / 2.0 / img_width
            y_center = (y1 + y2) / 2.0 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Write YOLO format labels
    with open(output_label_file, 'w') as f:
        for label in yolo_labels:
            f.write(label + '\n')


if __name__ == "__main__":
    # Example usage
    fishnet_dir = "/home/ziliang/Projects/Marine Datasets/FishNet"
    output_dir = "/home/ziliang/Projects/Marine Datasets/fishnet_yolo"
    
    convert_fishnet_to_yolo(fishnet_dir, output_dir)