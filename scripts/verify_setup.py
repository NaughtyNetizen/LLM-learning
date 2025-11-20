#!/usr/bin/env python3
"""
项目验证脚本

快速验证所有模块是否正常工作
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🚀 大模型学习项目 - 快速验证")
print("=" * 70)

# 检查Python版本
print("\n1️⃣ 检查Python版本...")
print(f"   Python版本: {sys.version}")
if sys.version_info < (3, 8):
    print("   ⚠️  警告: 推荐使用Python 3.8+")
else:
    print("   ✅ Python版本OK")

# 检查依赖
print("\n2️⃣ 检查核心依赖...")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
except ImportError:
    print("   ❌ PyTorch未安装")
    print("   请运行: pip install torch")
    sys.exit(1)

try:
    import numpy as np
    print(f"   ✅ NumPy {np.__version__}")
except ImportError:
    print("   ❌ NumPy未安装")

try:
    from tqdm import tqdm
    print(f"   ✅ tqdm已安装")
except ImportError:
    print("   ⚠️  tqdm未安装（可选）")

# 验证项目结构
print("\n3️⃣ 验证项目结构...")
required_dirs = [
    'transformer_basics',
    'gpt_model',
    'inference',
    'training',
    'finetuning',
    'examples',
]

for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"   ✅ {dir_name}/")
    else:
        print(f"   ❌ {dir_name}/ 缺失")

# 测试导入
print("\n4️⃣ 测试模块导入...")

test_imports = [
    ("transformer_basics.attention", "CausalSelfAttention"),
    ("gpt_model.config", "get_config"),
    ("gpt_model.model", "GPT"),
    ("inference.sampling", "top_p_sampling"),
    ("training.dataset", "TextDataset"),
    ("finetuning.lora", "LoRALayer"),
    ("tokenizer", "CharTokenizer"),
]

import_success = 0
import_total = len(test_imports)

for module_name, class_name in test_imports:
    try:
        module = __import__(module_name, fromlist=[class_name])
        getattr(module, class_name)
        print(f"   ✅ {module_name}.{class_name}")
        import_success += 1
    except Exception as e:
        print(f"   ❌ {module_name}.{class_name}: {e}")

print(f"\n   导入成功率: {import_success}/{import_total}")

# 快速功能测试
if import_success == import_total:
    print("\n5️⃣ 快速功能测试...")
    
    try:
        # 测试创建小模型
        from gpt_model.config import get_config
        from gpt_model.model import GPT
        
        print("   - 创建GPT模型...")
        config = get_config('gpt-micro')
        model = GPT(config)
        print(f"     ✅ 模型创建成功 ({sum(p.numel() for p in model.parameters()):,} 参数)")
        
        # 测试前向传播
        print("   - 测试前向传播...")
        import torch
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        with torch.no_grad():
            loss, logits, _ = model(input_ids, input_ids)
        print(f"     ✅ 前向传播成功 (loss={loss.item():.4f})")
        
        # 测试生成
        print("   - 测试文本生成...")
        with torch.no_grad():
            generated = model.generate(input_ids[:1, :8], max_new_tokens=10)
        print(f"     ✅ 生成成功 (长度: {generated.shape[1]})")
        
    except Exception as e:
        print(f"   ❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()

# 总结
print("\n" + "=" * 70)
print("📊 验证结果总结")
print("=" * 70)

if import_success == import_total:
    print("\n✅ 所有模块验证通过！")
    print("\n🎓 你可以开始学习了！")
    print("\n推荐步骤:")
    print("1. 阅读 README.md 了解项目")
    print("2. 查看 QUICKSTART.md 快速开始")
    print("3. 运行测试: python 01_transformer_basics/attention.py")
    print("4. 训练模型: python examples/train_small_gpt.py")
else:
    print("\n⚠️  部分模块导入失败")
    print("请检查:")
    print("1. 是否安装了所有依赖: pip install -r requirements.txt")
    print("2. 是否在项目根目录运行此脚本")
    print("3. Python路径是否正确")

print("\n" + "=" * 70)
print("🚀 开始你的大模型学习之旅吧！")
print("=" * 70 + "\n")
