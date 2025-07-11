#!/usr/bin/env python3
"""
创建YOLO训练配置文件的辅助脚本
"""

import os
import yaml
from pathlib import Path


def create_yolo_config(dataset_dir="./yolo_dataset", config_name="marine_animals.yaml"):
    """
    创建YOLO训练配置文件
    
    Args:
        dataset_dir (str): 数据集目录路径
        config_name (str): 配置文件名
    """
    dataset_path = Path(dataset_dir).resolve()
    
    # 读取类别文件
    classes_file = dataset_path / "classes.txt"
    if not classes_file.exists():
        print(f"警告: 类别文件 {classes_file} 不存在")
        return
    
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # 创建YOLO配置
    config = {
        'path': str(dataset_path),  # 数据集根目录
        'train': 'images',  # 训练图像目录(相对于path)
        'val': 'images',    # 验证图像目录(相对于path) 
        'test': 'images',   # 测试图像目录(相对于path)
        
        # 类别数量
        'nc': len(class_names),
        
        # 类别名称
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    # 保存配置文件
    config_file = dataset_path / config_name
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"YOLO配置文件已创建: {config_file}")
    print(f"类别数量: {len(class_names)}")
    print("类别列表:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # 创建数据分割脚本建议
    split_script = dataset_path / "split_dataset.py"
    split_code = '''#!/usr/bin/env python3
"""
将数据集分割为训练集、验证集和测试集
"""

import os
import shutil
from pathlib import Path
import random

def split_dataset(dataset_dir=".", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """分割数据集"""
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    # 创建分割目录
    for split in ['train', 'val', 'test']:
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    random.shuffle(image_files)
    
    # 计算分割点
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 分割文件
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    for split_name, files in splits.items():
        print(f"{split_name}: {len(files)} 文件")
        for img_file in files:
            # 复制图像
            shutil.copy2(img_file, dataset_path / split_name / 'images' / img_file.name)
            
            # 复制对应的标注文件
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, dataset_path / split_name / 'labels' / label_file.name)

if __name__ == '__main__':
    split_dataset()
'''
    
    with open(split_script, 'w', encoding='utf-8') as f:
        f.write(split_code)
    
    print(f"\\n数据分割脚本已创建: {split_script}")
    print("使用方法: python split_dataset.py")
    
    return config_file


if __name__ == '__main__':
    create_yolo_config() 