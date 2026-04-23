"""
Course training script (simplified from nanoGPT).

Focus:
- Train a small GPT-style model from scratch on a tiny dataset.
- Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import time
import pickle
import argparse
from dataclasses import asdict

import numpy as np
import torch
from codecarbon import EmissionsTracker

from model import GPTConfig, GPT

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the GPT-style language model.")

    parser.add_argument("--out-dir", default="out", help="Directory for checkpoints and emissions logs.")
    parser.add_argument("--data-dir", default="data", help="Directory containing train.bin, val.bin, and meta.pkl.")
    parser.add_argument("--eval-interval", type=int, default=200, help="How often to run validation, in iterations.")
    parser.add_argument("--eval-iters", type=int, default=50, help="Number of batches to average during validation.")
    parser.add_argument("--log-interval", type=int, default=50, help="How often to print training loss, in iterations.")
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Save checkpoints during training.")

    parser.add_argument("--n-layer", type=int, default=4, help="Number of Transformer blocks.")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True, help="Use bias terms in linear and layer norm layers.")

    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument("--dtype", default="float32", help="Recorded training dtype in checkpoints.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of sequences processed in parallel.")
    parser.add_argument("--block-size", type=int, default=256, help="Maximum context length for predictions.")
    parser.add_argument("--max-iters", type=int, default=2000, help="Total number of training iterations.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="L2 regularization strength.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping threshold; set to 0 to disable.")

    return parser

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    # simple, robust memmap loader
    bin_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model: GPT, data_dir: str, block_size: int, batch_size: int, device: str, eval_iters: int):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split, data_dir, block_size, batch_size, device)
            _, loss = model(x, y)
            split_losses[k] = loss
        losses[split] = split_losses.mean().item()
    model.train()
    return losses

def save_checkpoint(out_dir: str, model: GPT, optimizer: torch.optim.Optimizer, iter_num: int, config: dict):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "iter_num": iter_num,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

def main():
    args = build_arg_parser().parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    tracker = EmissionsTracker(
        project_name="gwp_lm_train",
        output_dir=args.out_dir,
        output_file="emissions_train.csv",
        log_level="error",
    )
    tracker.start()

    meta = load_meta(args.data_dir)
    vocab_size = meta["vocab_size"] if meta and "vocab_size" in meta else 50304

    cfg = GPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )

    # create the model and move it to the device
    model = GPT(cfg).to(args.device)

    # create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # (optional) uncomment this for printing model size once
    # print(f"Device: {args.device}")
    # print(f"Model parameters: {model.get_num_params():,}")
    # print(f"Training for {args.max_iters} iterations | batch={args.batch_size} | block={args.block_size}")

    total_emissions = None
    try:
        t0 = time.time()
        for it in range(args.max_iters + 1):
            # periodic evaluation
            if it % args.eval_interval == 0:
                losses = estimate_loss(model, args.data_dir, args.block_size, args.batch_size, args.device, args.eval_iters)
                dt = time.time() - t0
                print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | elapsed {dt:.1f}s")

                if args.save_checkpoint and it > 0:
                    config_dump = {
                        "data_dir": args.data_dir,
                        "train": {
                            "batch_size": args.batch_size,
                            "block_size": args.block_size,
                            "max_iters": args.max_iters,
                            "learning_rate": args.learning_rate,
                            "weight_decay": args.weight_decay,
                            "grad_clip": args.grad_clip,
                            "dtype": args.dtype,
                            "device": args.device,
                        },
                        "model": asdict(cfg),
                    }
                    save_checkpoint(args.out_dir, model, optimizer, it, config_dump)

            # training step
            x, y = get_batch("train", args.data_dir, args.block_size, args.batch_size, args.device)
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            if it % args.log_interval == 0:
                print(f"iter {it:5d} | loss {loss.item():.4f}")

        print("Training completed.")

        # Save final checkpoint
        if args.save_checkpoint:
            config_dump = {
                "data_dir": args.data_dir,
                "train": {
                    "batch_size": args.batch_size,
                    "block_size": args.block_size,
                    "max_iters": args.max_iters,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "grad_clip": args.grad_clip,
                    "dtype": args.dtype,
                    "device": args.device,
                },
                "model": asdict(cfg),
            }
            save_checkpoint(args.out_dir, model, optimizer, args.max_iters, config_dump)
    finally:
        total_emissions = tracker.stop()
        if total_emissions is not None:
            print(f"Total training emissions: {total_emissions:.6f} kg CO2eq")

if __name__ == "__main__":
    main()
