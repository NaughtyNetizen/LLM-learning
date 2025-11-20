import torch
import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))

from llm_learning.model.model import GPT
from llm_learning.model.config import get_config
from llm_learning.tokenizer import BPETokenizer

def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT model")
    parser.add_argument("--model_type", type=str, default="gpt-mini", help="Model type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="To be or not to be", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer encoding (gpt2, cl100k_base)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Tokenizer
    print(f"Loading Tokenizer ({args.tokenizer})...")
    try:
        if args.tokenizer == "cl100k_base":
            tokenizer = BPETokenizer(encoding_name="cl100k_base")
        else:
            tokenizer = BPETokenizer(args.tokenizer)
    except ImportError:
        print("Error: tiktoken not installed.")
        return

    # 2. Model
    print(f"Loading Model ({args.model_type})...")
    config = get_config(args.model_type)
    config.vocab_size = tokenizer.vocab_size
    model = GPT(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    if os.path.exists(args.checkpoint):
        # Set weights_only=False to allow loading custom classes like GPTConfig
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    model.to(device)
    model.eval()

    # 3. Generate
    print("\nGenerating...")
    print(f"Prompt: {args.prompt}")
    
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()
