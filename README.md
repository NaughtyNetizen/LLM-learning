# 大模型从零实现学习项目 (LLM Learning)

> 一个完整的从零开始学习和实现大语言模型（LLM）的教学项目。支持中文训练、多种数据集自动下载及 GPT-4 Tokenizer。

## 📚 项目简介

本项目提供了从底层原理到完整实现的大语言模型学习资源。经过重构，现在拥有更清晰的项目结构，并原生支持中文模型训练。

主要包含：
- ✅ **Transformer基础组件**：完整实现 Attention 和 Transformer Block
- ✅ **GPT模型架构**：标准的 Decoder-only 架构
- ✅ **多语言支持**：支持英文 (WikiText) 和中文 (Wikipedia) 训练
- ✅ **高级 Tokenizer**：支持 GPT-2 和 GPT-4 (cl100k_base) Tokenizer
- ✅ **完整训练流程**：支持梯度累积、混合精度训练、断点续训
- ✅ **自动化数据管线**：一键下载和预处理高质量数据集

**代码特点：** 每行代码都有详细注释，包含数学公式，易于理解和学习。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

使用自动化脚本下载高质量数据集（支持断点续传）：

```bash
# 下载所有数据 (WikiText + 中文维基)
python scripts/download_data.py --dataset all

# 仅下载中文数据
python scripts/download_data.py --dataset chinese
```

### 3. 训练模型

使用 `scripts/train.py` 进行训练。支持多种配置和数据集。

```bash
# 训练中文模型 (使用 GPT-4 Tokenizer)
python scripts/train.py --model_type gpt-micro --dataset chinese --tokenizer cl100k_base --epochs 5

# 训练英文模型 (使用 GPT-2 Tokenizer)
python scripts/train.py --model_type gpt-mini --dataset shakespeare --tokenizer gpt2
```

**参数说明：**
- `--model_type`: 模型规模 (`gpt-micro`, `gpt-mini`, `gpt-small`, `gpt2` 等)
- `--dataset`: 数据集 (`chinese`, `shakespeare`)
- `--tokenizer`: Tokenizer 类型 (`gpt2`, `cl100k_base`)

### 4. 生成文本

使用 `scripts/generate.py` 加载训练好的模型生成文本。

```bash
# 中文生成
python scripts/generate.py \
    --checkpoint checkpoints/gpt-micro/best_model.pt \
    --prompt "人工智能" \
    --tokenizer cl100k_base \
    --model_type gpt-micro

# 英文生成
python scripts/generate.py \
    --checkpoint checkpoints/gpt-mini/best_model.pt \
    --prompt "To be or not to be" \
    --tokenizer gpt2
```

## 📂 项目结构

```
LLM/
├── src/
│   └── llm_learning/
│       ├── model/          # GPT模型定义 (config.py, model.py)
│       ├── modules/        # Transformer基础组件
│       ├── data/           # 数据处理 (dataset.py, chinese_dataset.py)
│       ├── tokenizer/      # Tokenizer封装 (bpe_tokenizer.py)
│       └── training/       # 训练器实现 (trainer.py)
├── scripts/
│   ├── download_data.py    # 数据下载脚本
│   ├── train.py            # 训练脚本
│   └── generate.py         # 生成脚本
├── data/                   # 数据存放目录
│   └── raw/                # 原始数据 (wikitext, chinese_wiki)
├── checkpoints/            # 模型保存目录
└── requirements.txt
```

## 🎓 学习路径

### 第一阶段：理解组件
阅读 `src/llm_learning/modules/` 下的代码，理解 Self-Attention 和 Transformer Block 的实现细节。

### 第二阶段：理解模型与配置
阅读 `src/llm_learning/model/`，理解 GPT 的整体架构以及 `config.py` 中的参数设计。

### 第三阶段：数据与Tokenizer
阅读 `src/llm_learning/data/` 和 `src/llm_learning/tokenizer/`，了解如何处理大规模文本数据以及 BPE 编码原理。

### 第四阶段：训练与优化
阅读 `src/llm_learning/training/`，掌握训练循环、梯度累积、混合精度训练等工程技巧。

## ❓ 常见问题

**Q: 为什么中文输出是乱码？**
A: 请确保使用 `--tokenizer cl100k_base` 进行训练和生成，GPT-2 的默认 tokenizer 对中文支持较差。同时确保训练数据是中文数据 (`--dataset chinese`)。

**Q: 显存不足怎么办？**
A: 
1. 减小 `--batch_size`
2. 使用更小的模型 (`gpt-micro`)
3. 增加 `--gradient_accumulation_steps` (在代码中调整)

**Q: 如何使用自己的数据？**
A: 将你的文本文件命名为 `train.txt` 和 `validation.txt`，放入 `data/raw/your_dataset/` 目录，并修改 `src/llm_learning/data/dataset.py` 中的加载逻辑。

## 📄 许可证

MIT License
