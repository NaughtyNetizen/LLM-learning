#!/bin/bash
# 项目任务脚本

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 显示帮助
show_help() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}大模型学习项目 - 任务脚本${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    echo "用法: ./tasks.sh [命令]"
    echo ""
    echo "可用命令:"
    echo ""
    echo -e "${GREEN}setup${NC}        - 安装依赖"
    echo -e "${GREEN}verify${NC}       - 验证项目设置"
    echo -e "${GREEN}test${NC}         - 运行所有测试"
    echo -e "${GREEN}test-basics${NC}  - 测试Transformer基础"
    echo -e "${GREEN}test-model${NC}   - 测试GPT模型"
    echo -e "${GREEN}test-infer${NC}   - 测试推理系统"
    echo -e "${GREEN}test-lora${NC}    - 测试LoRA"
    echo -e "${GREEN}train${NC}        - 训练小型GPT"
    echo -e "${GREEN}finetune${NC}     - LoRA微调示例"
    echo -e "${GREEN}clean${NC}        - 清理生成的文件"
    echo ""
}

# 安装依赖
setup() {
    echo -e "${YELLOW}安装依赖...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✅ 依赖安装完成${NC}"
}

# 验证设置
verify() {
    echo -e "${YELLOW}验证项目设置...${NC}"
    python verify_setup.py
}

# 测试Transformer基础
test_basics() {
    echo -e "${YELLOW}测试Transformer基础组件...${NC}"
    python 01_transformer_basics/attention.py
    echo ""
    python 01_transformer_basics/layers.py
    echo ""
    python 01_transformer_basics/embeddings.py
}

# 测试GPT模型
test_model() {
    echo -e "${YELLOW}测试GPT模型...${NC}"
    python 02_gpt_model/config.py
    echo ""
    python 02_gpt_model/model.py
}

# 测试推理
test_infer() {
    echo -e "${YELLOW}测试推理系统...${NC}"
    python 03_inference/sampling.py
    echo ""
    python 03_inference/generator.py
}

# 测试LoRA
test_lora() {
    echo -e "${YELLOW}测试LoRA...${NC}"
    python 05_finetuning/lora.py
}

# 运行所有测试
test_all() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}运行所有测试${NC}"
    echo -e "${BLUE}================================${NC}"
    test_basics
    echo ""
    test_model
    echo ""
    test_infer
    echo ""
    test_lora
    echo -e "${GREEN}✅ 所有测试完成${NC}"
}

# 训练
train() {
    echo -e "${YELLOW}训练小型GPT...${NC}"
    python examples/train_small_gpt.py
}

# 微调
finetune() {
    echo -e "${YELLOW}LoRA微调...${NC}"
    python examples/finetune_with_lora.py
}

# 清理
clean() {
    echo -e "${YELLOW}清理生成的文件...${NC}"
    rm -rf __pycache__
    rm -rf */__pycache__
    rm -rf */*/__pycache__
    rm -rf .pytest_cache
    rm -rf data/processed/*.npy
    rm -rf checkpoints/
    echo -e "${GREEN}✅ 清理完成${NC}"
}

# 主函数
main() {
    case "$1" in
        setup)
            setup
            ;;
        verify)
            verify
            ;;
        test)
            test_all
            ;;
        test-basics)
            test_basics
            ;;
        test-model)
            test_model
            ;;
        test-infer)
            test_infer
            ;;
        test-lora)
            test_lora
            ;;
        train)
            train
            ;;
        finetune)
            finetune
            ;;
        clean)
            clean
            ;;
        *)
            show_help
            ;;
    esac
}

# 运行
main "$@"
