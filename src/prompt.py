"""
Inference / prompting script (Tiny Shakespeare, char-level).
Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import pickle
import argparse
import torch
from codecarbon import EmissionsTracker

from model import GPT, GPTConfig

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--out-dir", default="out", help="Directory for emissions logs and default checkpoint location.")
    parser.add_argument("--ckpt-path", default=None, help="Path to a checkpoint file. Defaults to <out-dir>/ckpt.pt.")
    parser.add_argument("--prompt", default="To be, or not to be", help="Prompt text to seed generation.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering threshold; use 0 to disable.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device.")
    return parser


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def main():
    args = build_arg_parser().parse_args()
    ckpt_path = args.ckpt_path or os.path.join(args.out_dir, "ckpt.pt")

    os.makedirs(args.out_dir, exist_ok=True)
    tracker = EmissionsTracker(
        project_name="gwp_lm_infer",
        output_dir=args.out_dir,
        output_file="emissions_infer.csv",
    )
    tracker.start()

    try:
        ckpt = torch.load(ckpt_path, map_location=args.device)

        # train.py should store config with model parameters and data_dir
        data_dir = ckpt["config"]["data_dir"]
        model_cfg = ckpt["config"]["model"]

        meta = load_meta(data_dir)
        stoi = meta["stoi"]         # char to index mapping
        itos = meta["itos"]         # index to char mapping

        def encode(s: str):
            # map unknown chars to a safe fallback if needed
            return [stoi.get(ch, stoi[" "]) for ch in s]

        def decode(tokens):
            return "".join([itos[t] for t in tokens])

        config = GPTConfig(**model_cfg)
        model = GPT(config).to(args.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=args.device)

        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )

        print(decode(out[0].tolist()))
    finally:
        total_emissions = tracker.stop()
        if total_emissions is not None:
            print(f"Total inference emissions: {total_emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
