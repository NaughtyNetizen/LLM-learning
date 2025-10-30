# 大模型从零实现学习项目

> 一个完整的从零开始学习和实现大语言模型（LLM）的教学项目

## 📚 项目简介

本项目提供了从底层原理到完整实现的大语言模型学习资源，包括：
- ✅ Transformer基础组件完整实现
- ✅ GPT模型架构（Decoder-only）
- ✅ 5种文本生成采样策略
- ✅ 完整的训练流程（支持梯度累积、混合精度）
- ✅ LoRA参数高效微调
- ✅ 可运行的训练和微调示例

**代码特点：** 每行代码都有详细注释，包含数学公式，易于理解和学习。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 验证安装
```bash
python verify_setup.py
```

### 3. 快速测试（5分钟）
```bash
# 测试核心组件
python transformer_basics/attention.py
python gpt_model/model.py
python inference/sampling.py

# 训练第一个模型
python examples/train_small_gpt.py

# 尝试LoRA微调
python examples/finetune_with_lora.py
```

## 📂 项目结构

```
LLM/
├── transformer_basics/      # Transformer基础组件
│   ├── attention.py        # Self-Attention, Multi-Head, Causal
│   ├── layers.py           # FFN, LayerNorm, TransformerBlock
│   └── embeddings.py       # Token和Position Embeddings
│
├── gpt_model/              # GPT模型
│   ├── config.py          # 模型配置（micro到GPT-2 XL）
│   └── model.py           # 完整GPT实现
│
├── inference/             # 推理系统
│   ├── sampling.py       # 5种采样策略
│   └── generator.py      # 文本生成器
│
├── training/             # 训练流程
│   ├── dataset.py       # 数据处理
│   └── trainer.py       # 训练器（AMP、梯度累积）
│
├── finetuning/          # 微调技术
│   └── lora.py         # LoRA实现
│
└── examples/           # 实践示例
    ├── train_small_gpt.py      # 从零训练
    └── finetune_with_lora.py   # LoRA微调
```

## 🎓 学习路径

### 第1-2周：Transformer基础
**目标：** 理解注意力机制

```bash
python transformer_basics/attention.py
python transformer_basics/layers.py
python transformer_basics/embeddings.py
```

**关键概念：**
- Self-Attention如何工作：`Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- Multi-Head的作用：并行捕获不同信息
- Causal Mask：GPT只能看到过去的token

### 第3-4周：GPT模型
**目标：** 理解完整架构

```bash
python gpt_model/config.py    # 查看配置
python gpt_model/model.py     # 测试模型
```

**关键收获：**
- ✅ Decoder-only架构原理
- ✅ 模型参数量计算
- ✅ Pre-LN vs Post-LN
- ✅ 权重初始化技巧

### 第5周：推理系统
**目标：** 掌握文本生成

```bash
python inference/sampling.py
python inference/generator.py
```

**5种采样策略：**
1. **Greedy**: 最简单，总选概率最大的
2. **Temperature**: 控制随机性（0.7=保守，1.5=创意）
3. **Top-K**: 限制候选词数量（推荐k=40）
4. **Top-P**: 动态候选集（推荐p=0.9）
5. **Beam Search**: 搜索最优序列

### 第6-8周：训练流程
**目标：** 从零训练模型

```bash
python training/dataset.py
python examples/train_small_gpt.py
```

**训练技巧：**
- 梯度累积：小显存训练大batch
- 混合精度：加速训练，节省显存
- 学习率调度：Cosine with Warmup
- 梯度裁剪：防止梯度爆炸

### 第9-10周：LoRA微调
**目标：** 参数高效微调

```bash
python finetuning/lora.py
python examples/finetune_with_lora.py
```

**LoRA原理：**
```python
# 原始权重(冻结): W [d×d]
# 低秩矩阵(训练): A [d×r], B [r×d], r << d
# 前向传播: y = xW + xBA * (α/r)

# 参数量对比
# 全参数: d² = 768² = 589,824
# LoRA(r=8): 2dr = 2*768*8 = 12,288 (减少98%!)
```

## 💡 核心概念速查

### Attention机制
```python
# Q: 查询"我想要什么信息"
# K: 键"我有什么信息"  
# V: 值"信息的内容"

scores = Q @ K.T / sqrt(d_k)      # 计算相似度
attention = softmax(scores)        # 归一化为权重
output = attention @ V             # 加权求和
```

### GPT架构特点
- **Decoder-only**: 只用解码器，不需要编码器
- **Causal Mask**: 只能看到过去，不能偷看未来
- **Pre-LN**: LayerNorm在attention/FFN之前
- **GELU**: 比ReLU更平滑的激活函数

### 模型配置对比
| 配置 | 参数量 | 层数 | 隐藏维度 | 注意力头 | 用途 |
|------|--------|------|----------|----------|------|
| gpt-micro | ~1M | 4 | 128 | 4 | 学习测试 |
| gpt-mini | ~10M | 6 | 384 | 6 | 实验原型 |
| gpt-small | ~50M | 8 | 512 | 8 | 小规模应用 |
| gpt2 | 117M | 12 | 768 | 12 | 实用模型 |
| gpt2-xl | 1.5B | 48 | 1600 | 25 | 大规模应用 |

## 🔧 实用技巧

### 超参数选择
```python
# 学习率
lr = 3e-4  # Adam的标准值
lr = 1e-3  # LoRA可以更高

# Batch size
batch_size = 32  # 小模型
batch_size = 4   # 显存有限时

# 生成参数
temperature = 0.7  # 更确定
temperature = 1.0  # 平衡
temperature = 1.5  # 更随机

top_k = 40         # 推荐值
top_p = 0.9        # 推荐值
repetition_penalty = 1.2  # 避免重复
```

### 调试技巧
```python
# 1. 检查形状
print(f"Input shape: {x.shape}")

# 2. 验证attention权重
print(f"Attention sum: {attn.sum(dim=-1)}")  # 应该=1

# 3. 监控梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm():.3f}")

# 4. 先过拟合单个batch
# 确保模型有能力学习
```

### 常见问题解决

**Q: CUDA out of memory**
```python
# 解决方案：
batch_size = 4              # 减小batch
gradient_accumulation = 4   # 梯度累积
use_amp = True             # 混合精度
seq_len = 256              # 减小序列长度
```

**Q: Loss不下降**
```python
# 检查清单：
# 1. 学习率是否合适 (试试1e-4到1e-3)
# 2. 数据是否正确 (打印一个batch检查)
# 3. 梯度是否正常 (检查grad norm)
# 4. 先在1个batch上过拟合测试
```

**Q: 生成文本质量差**
```python
# 改进方法：
# 1. 训练更多epochs
# 2. 增加模型规模
# 3. 使用更多/更好的数据
# 4. 调整生成参数(temperature, top_k, top_p)
```

## 🎯 学习检查点

### ✅ 基础理解
- [ ] 能解释Attention的公式
- [ ] 理解Q、K、V的含义
- [ ] 知道为什么需要Causal Mask
- [ ] 能说出GPT的关键特点

### ✅ 代码实践
- [ ] 运行了所有测试代码
- [ ] 修改过超参数观察变化
- [ ] 训练了一个小模型
- [ ] 尝试了不同采样策略

### ✅ 进阶掌握
- [ ] 能手写Attention前向传播
- [ ] 实现了LoRA微调
- [ ] 理解了训练技巧
- [ ] 能在自己数据上训练

## 📚 推荐学习资源

### 必读论文
1. **Attention Is All You Need** - Transformer原论文
2. **Language Models are Few-Shot Learners** - GPT-3
3. **LoRA: Low-Rank Adaptation** - 参数高效微调

### 优质教程
1. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
2. [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
3. [Andrej Karpathy - nanoGPT](https://github.com/karpathy/nanoGPT)
4. [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## � 进阶方向

### 深入某个方向
- **模型优化**: Flash Attention, Gradient Checkpointing
- **推理加速**: KV Cache, 量化(INT8/INT4)
- **扩展能力**: 长文本处理, 多模态

### 参与开源
- Hugging Face Transformers
- vLLM (推理优化)
- DeepSpeed (训练优化)

### 构建应用
- 代码助手
- 写作助手
- 知识问答系统

## 📊 项目统计

- **总代码量**: ~3,300行
- **注释率**: >40%
- **核心模块**: 5个
- **示例代码**: 2个
- **测试覆盖**: 每个模块

## � 使用建议

**适合：**
- 🎓 深入学习大模型原理
- 🔬 进行研究和实验
- 🛠️ 理解工程实践
- 📚 作为教学材料

**不适合：**
- 只想快速调用API（建议用Hugging Face）
- 需要生产级系统（需要额外优化）

## 🎉 完成标志

当你能做到以下几点，说明你已经掌握了：

✅ 能手写Attention的前向传播  
✅ 能解释GPT的每个组件  
✅ 能从零训练一个小模型  
✅ 能使用LoRA微调  
✅ 能在自己的任务上应用

**恭喜你！你已经理解了大模型的底层原理！** 🎓🚀

## 📄 许可证

MIT License

---

**记住：** 理解原理比跑通代码更重要，动手实现比阅读代码更有价值。

**开始你的大模型学习之旅吧！** 🚀

*最后更新: 2025年10月30日*
