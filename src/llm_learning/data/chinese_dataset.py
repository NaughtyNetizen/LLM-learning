"""
中文数据集处理

下载和准备中文数据集（西游记）
"""

import os
import urllib.request

def create_chinese_dataset():
    """
    创建中文数据集（西游记）
    
    下载西游记文本作为训练数据
    """
    data_dir = 'data/raw'
    os.makedirs(data_dir, exist_ok=True)
    
    # Update to point to Chinese Wiki
    data_path = os.path.join(data_dir, 'chinese_wiki', 'train.txt')
    
    if not os.path.exists(data_path):
        # Fallback
        fallback_path = os.path.join(data_dir, 'journey_to_the_west.txt')
        if os.path.exists(fallback_path):
            print(f"Chinese Wiki not found, using fallback: {fallback_path}")
            return fallback_path

        print(f"数据集不存在: {data_path}")
        print("请运行: python scripts/download_data.py --dataset chinese")
        raise FileNotFoundError(f"无法找到数据集. 请先运行下载脚本.")
    else:
        print(f"使用中文维基数据集: {data_path}")
    
    return data_path

def create_dummy_chinese_data(path):
    """
    如果下载失败，创建一个小的模拟中文数据集
    """
    text = "白龙马蹄朝西，驮着唐三藏跟着三徒弟。西天取经上大路，一走就是几万里。" * 1000
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"已创建模拟中文数据: {path}")
