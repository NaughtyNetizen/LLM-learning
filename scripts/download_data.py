"""
数据下载脚本

使用 Hugging Face datasets 库下载高质量数据集
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def save_dataset_to_text(dataset, output_path, text_column='text', limit=None):
    """
    将dataset保存为文本文件
    """
    print(f"正在保存到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        count = 0
        for item in tqdm(dataset):
            text = item[text_column]
            if text and len(text.strip()) > 0:
                f.write(text + "\n\n")
                count += 1
                if limit and count >= limit:
                    break
    print(f"✅ 已保存 {count} 条记录")

def download_wikitext(output_dir):
    """下载 WikiText-103"""
    print("\n正在下载 WikiText-103 (English)...")
    try:
        # wikitext-103-raw-v1
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        
        os.makedirs(os.path.join(output_dir, "wikitext"), exist_ok=True)
        
        # 保存训练集
        save_dataset_to_text(
            dataset['train'], 
            os.path.join(output_dir, "wikitext", "train.txt")
        )
        
        # 保存验证集
        save_dataset_to_text(
            dataset['validation'], 
            os.path.join(output_dir, "wikitext", "validation.txt")
        )
        
    except Exception as e:
        print(f"❌ WikiText下载失败: {e}")

def download_chinese_wiki(output_dir):
    """下载中文维基百科"""
    print("\n正在下载 Chinese Wikipedia...")
    try:
        # 使用 pleisto/wikipedia-cn-20230720-filtered
        # 这是一个高质量的中文维基百科过滤版本
        dataset = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split='train')
        
        os.makedirs(os.path.join(output_dir, "chinese_wiki"), exist_ok=True)
        
        # 由于数据集较大，我们只取前100万条或者全部（如果用户允许）
        # 这里为了演示和快速上手，我们先保存全部，但用户可以手动中断
        # 实际上这个数据集大约1GB纯文本，是可以接受的
        
        # 我们手动切分一个验证集 (最后1%)
        total_size = len(dataset)
        train_size = int(total_size * 0.99)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, total_size))
        
        save_dataset_to_text(
            train_dataset,
            os.path.join(output_dir, "chinese_wiki", "train.txt"),
            text_column='completion' # 这个数据集主要内容在 completion 列
        )
        
        save_dataset_to_text(
            val_dataset,
            os.path.join(output_dir, "chinese_wiki", "validation.txt"),
            text_column='completion'
        )
        
    except Exception as e:
        print(f"❌ 中文维基下载失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="下载训练数据 (Hugging Face)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "wikitext", "chinese"], help="要下载的数据集")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="输出目录")
    args = parser.parse_args()
    
    # 检查datasets库
    try:
        import datasets
    except ImportError:
        print("错误: 未安装 datasets 库")
        print("请运行: pip install datasets")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset in ["all", "wikitext"]:
        download_wikitext(args.output_dir)
        
    if args.dataset in ["all", "chinese"]:
        download_chinese_wiki(args.output_dir)
            
    print("\n✨ 下载完成!")

if __name__ == "__main__":
    main()
