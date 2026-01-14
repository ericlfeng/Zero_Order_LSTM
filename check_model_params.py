#!/usr/bin/env python3
"""
Check parameter count for LSTM models with flexible configuration.

Usage:
    python check_model_params.py --scale 16
    python check_model_params.py --scale 32 --n_layers 3
    python check_model_params.py --scale 64 --n_layers 2
    python check_model_params.py --scale 16 32 64 --n_layers 3
"""
import argparse
import torch
import torch.nn as nn
from models.models import LSTM


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def create_model(scale, n_layers, input_size=128, output_size=256):
    """Create an LSTM model with given configuration."""
    # Base config from sweep script
    hidden_size = 111 * scale
    memory_size = 111 * scale
    head_size = 0
    num_heads = 1
    
    embed = nn.Embedding(output_size, input_size)
    
    model = LSTM(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        memory_size=memory_size,
        head_size=head_size,
        num_heads=num_heads,
        embed=embed,
        device=torch.device("cpu"),
        dtype=torch.float32,
        n_layers=n_layers,
    )
    
    return model, hidden_size


def main():
    parser = argparse.ArgumentParser(description='Check LSTM parameter counts')
    parser.add_argument('--scale', type=int, nargs='+', default=[16, 32, 64],
                        help='Model scale(s) to check (default: 16 32 64)')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of LSTM layers (default: 3)')
    parser.add_argument('--input_size', type=int, default=128,
                        help='Input embedding size (default: 128)')
    parser.add_argument('--output_size', type=int, default=256,
                        help='Output vocab size (default: 256)')
    parser.add_argument('--breakdown', action='store_true',
                        help='Show parameter breakdown by layer')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"LSTM Parameter Count Check ({args.n_layers}-Layer)")
    print("="*70)
    
    for scale in args.scale:
        model, hidden_size = create_model(scale, args.n_layers, args.input_size, args.output_size)
        total_params, trainable_params = count_parameters(model)
        
        print(f"\n{'─'*70}")
        print(f"Scale: {scale}")
        print(f"{'─'*70}")
        print(f"  hidden_size:      {hidden_size:,}")
        print(f"  n_layers:         {args.n_layers}")
        print(f"  input_size:       {args.input_size}")
        print(f"  output_size:      {args.output_size}")
        print(f"  {'─'*66}")
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Params (M):       {total_params / 1e6:.2f}M")
        print(f"  Params (B):       {total_params / 1e9:.3f}B")
        
        if args.breakdown:
            print(f"\n  Parameter breakdown:")
            for name, p in model.named_parameters():
                print(f"    {name:40s}: {p.numel():>12,} ({list(p.shape)})")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

