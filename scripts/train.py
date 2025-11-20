import torch
import sys
import os
import argparse
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))

from llm_learning.model.model import GPT
from llm_learning.model.config import get_config
from llm_learning.data.dataset import create_shakespeare_dataset, prepare_data
from llm_learning.data.chinese_dataset import create_chinese_dataset
from llm_learning.training.trainer import Trainer, get_cosine_schedule_with_warmup
from llm_learning.tokenizer import BPETokenizer

def main():
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--model_type", type=str, default="gpt-mini", help="Model type (gpt-micro, gpt-mini, gpt-small)")
    parser.add_argument("--dataset", type=str, default="shakespeare", choices=["shakespeare", "chinese"], help="Dataset to use")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer encoding (gpt2, cl100k_base)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--dry_run", action="store_true", help="Run a single batch to verify setup")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Training GPT Model ({args.model_type}) on {args.dataset}")
    print("=" * 70)

    # 1. Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 2. Tokenizer
    print(f"\nInitializing BPE Tokenizer ({args.tokenizer})...")
    try:
        if args.tokenizer == "cl100k_base":
            tokenizer = BPETokenizer(encoding_name="cl100k_base")
        else:
            tokenizer = BPETokenizer(args.tokenizer)
    except ImportError:
        print("Error: tiktoken not installed. Please run 'pip install tiktoken'")
        return

    print(f"Vocab size: {tokenizer.vocab_size}")

    # 3. Data
    print("\nPreparing Data...")
    if args.dataset == "chinese":
        data_path = create_chinese_dataset()
    else:
        data_path = create_shakespeare_dataset()
    
    # Get config to know max_seq_len
    config = get_config(args.model_type)
    config.vocab_size = tokenizer.vocab_size
    
    train_dataset, val_dataset = prepare_data(
        data_path,
        tokenizer,
        seq_len=config.max_seq_len,
        train_split=0.9
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 4. Model
    print("\nInitializing Model...")
    model = GPT(config)
    print(f"Parameters: {model.get_num_params()/1e6:.2f}M")

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=4, # Default to 4 for stability
        save_dir=f'checkpoints/{args.model_type}'
    )

    if args.dry_run:
        print("\n[Dry Run] Training for 1 step...")
        # Manually run one step
        model.to(device)
        model.train()
        input_ids, targets = next(iter(train_dataloader))
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        loss, _, _ = model(input_ids, targets)
        loss.backward()
        optimizer.step()
        print(f"Step complete. Loss: {loss.item():.4f}")
        return

    # 7. Train
    num_training_steps = len(train_dataloader) * args.epochs // 4
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    trainer.train(num_epochs=args.epochs, lr_scheduler=lr_scheduler)

    # 8. Test Generation
    print("\nTesting Generation...")
    model.eval()
    prompt = "To be or not to be"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=40
        )
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {tokenizer.decode(generated_ids[0].tolist())}")

if __name__ == "__main__":
    main()
