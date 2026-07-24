"""
run_demo.py — Continuous Severity Scoring Demo
================================================

Demonstrates:
  1. Model instantiation (regression-only, score in [0.0, 3.0]).
  2. Dummy forward pass with shape verification.
  3. Training step with SmoothL1Loss on continuous severity labels.
  4. Forward pass without metadata (graceful degradation).
  5. Parameter count summary.

Run:
    py -3 run_demo.py
"""

import torch
import torch.nn as nn

from skin_severity import SkinSeverityModel
from skin_severity.fusion import SEVERITY_BANDS


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 70)
    print("  CONTINUOUS SEVERITY SCORING DEMO")
    print("  Score range: [0.0, 3.0]")
    print("=" * 70)

    # ── Config ──
    B = 4            # batch size
    H, W = 256, 256  # input image resolution
    M = 10           # metadata features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Model ──
    model = SkinSeverityModel(
        roi_size=224,
        base_channels=64,
        embed_dim=128,
        metadata_dim=M,
        num_locations=20,
    ).to(device)

    print(f"Total trainable parameters: {count_parameters(model):,}\n")

    # ── Severity interpretation bands ──
    print("  Severity interpretation bands:")
    for low, high, label in SEVERITY_BANDS:
        print(f"    {low:.1f} - {high:.1f}  ->  {label}")
    print()

    # ── Dummy inputs ──
    image = torch.rand(B, 3, H, W, device=device)           # [B, 3, H, W]
    metadata = torch.rand(B, M, device=device)               # [B, M]
    location = torch.tensor([0, 3, -1, 7], device=device)    # [B]

    # Continuous severity labels in [0.0, 3.0]
    labels = torch.tensor(
        [[0.3], [1.4], [2.1], [2.8]], device=device
    )  # [B, 1] — very_mild, moderate, severe, critical

    # ── Forward pass (eval) ──
    model.eval()
    with torch.no_grad():
        out = model(image, metadata=metadata, location=location)

    print("Forward pass outputs:")
    print(f"  severity_score : {out['severity_score'].shape}  ->  {out['severity_score'].squeeze().tolist()}")
    print(f"  severity_bands : {out['severity_bands']}")
    print(f"  gate_weights   : {out['gate_weights'].shape}")
    print(f"  fused_embed    : {out['fused_embedding'].shape}")
    print(f"  explanations   : {list(out['explanations'].keys())}")
    print()

    # Gate weights sum check
    gw_sum = out["gate_weights"].sum(dim=1)
    print(f"  gate_weights sum (should be ~1.0): {gw_sum.tolist()}")
    print()

    # Per-gate contributions for first sample
    print(f"  Per-gate contributions (sample 0, score={out['severity_score'][0].item():.2f}):")
    for name, val in out["explanations"].items():
        print(f"    {name:20s}: {val[0].item():.4f}")
    print()

    # ── Training step with SmoothL1Loss ──
    print("-" * 70)
    print("  TRAINING STEP (SmoothL1Loss)")
    print("-" * 70)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()

    optimizer.zero_grad()
    out = model(image, metadata=metadata, location=location)

    # ONLY final severity loss — no auxiliary losses
    loss = criterion(out["severity_score"], labels)

    loss.backward()
    optimizer.step()

    print(f"\n  Loss (SmoothL1): {loss.item():.6f}")
    print(f"  Predicted      : {[f'{s:.2f}' for s in out['severity_score'].squeeze().tolist()]}")
    print(f"  Target         : {[f'{s:.2f}' for s in labels.squeeze().tolist()]}")
    print(f"  Bands          : {out['severity_bands']}")
    print()

    # ── Forward pass WITHOUT metadata/location ──
    print("-" * 70)
    print("  FORWARD PASS WITHOUT METADATA / LOCATION")
    print("-" * 70)

    model.eval()
    with torch.no_grad():
        out_no_meta = model(image)

    print(f"\n  severity_score : {out_no_meta['severity_score'].squeeze().tolist()}")
    print(f"  severity_bands : {out_no_meta['severity_bands']}")
    print(f"  gate_weights   : {out_no_meta['gate_weights'][0].tolist()}")
    print()

    # ── Multi-step training loop example ──
    print("-" * 70)
    print("  MINI TRAINING LOOP (5 steps)")
    print("-" * 70)

    model.train()
    for step in range(5):
        # Simulate a batch
        img_batch = torch.rand(B, 3, H, W, device=device)
        lbl_batch = torch.rand(B, 1, device=device) * 3.0  # random labels in [0, 3]

        optimizer.zero_grad()
        out = model(img_batch)
        loss = criterion(out["severity_score"], lbl_batch)
        loss.backward()
        optimizer.step()

        pred = out["severity_score"].squeeze().tolist()
        print(f"  Step {step}: loss={loss.item():.4f}  pred=[{', '.join(f'{p:.2f}' for p in pred)}]")

    print()
    print("=" * 70)
    print("  ALL DEMOS PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
