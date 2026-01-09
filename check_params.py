#!/usr/bin/env python3
"""Quick parameter count check for LSTM"""
import torch
import torch.nn as nn
from models.models import LSTM

# Config from lstm_nlayer_overfit_sweep.sh at scale 64
n_layers = 3
hidden_size = 111 * 64  # 7104
memory_size = 111 * 64
head_size = 0
num_heads = 1
input_size = 128
output_size = 256  # typical vocab size

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

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*50}")
print(f"3-Layer LSTM at Scale 64")
print(f"{'='*50}")
print(f"hidden_size: {hidden_size}")
print(f"n_layers:    {n_layers}")
print(f"input_size:  {input_size}")
print(f"output_size: {output_size}")
print(f"{'='*50}")
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Params (B):       {total_params / 1e9:.3f}B")
print(f"{'='*50}")

# Breakdown by layer
print("\nParameter breakdown:")
for name, p in model.named_parameters():
    print(f"  {name}: {p.numel():,} ({p.shape})")
