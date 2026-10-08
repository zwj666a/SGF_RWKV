"""
Download CLIP ViT-B-32 pretrained weights.

Run once before training:
  python scripts/download_clip_weights.py
"""

import os
import sys

# Add project root for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clip

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pretrained")
os.makedirs(SAVE_DIR, exist_ok=True)

save_path = os.path.join(SAVE_DIR, "ViT-B-32.pt")

if os.path.exists(save_path):
    print(f"[Skip] Already exists: {save_path}")
else:
    print("Downloading CLIP ViT-B-32 weights...")
    model, _ = clip.load("ViT-B/32", device="cpu", download_root=SAVE_DIR)
    print(f"Saved to: {save_path}")
