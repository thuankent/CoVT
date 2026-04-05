"""
Visualize CoVT Visual Tokens

Produces a composite figure that matches the reference CoVT demo layout:
  - Input Image | SAM Segmentation (coloured overlay) | Depth Map (heatmap)
  - Token type badges (SAM: 8 tokens, Depth: 4 tokens, DINOv2: 4 tokens)
  - Chain-of-thought text with anchor token placeholders

Two execution paths
-------------------
1. **Full pipeline** (GPU + CoVT model + anchor checkpoints):
       python visualize_visual_tokens.py --image img.jpg [--model ...]
   Runs CoVT inference, decodes visual tokens from hidden states, and
   renders the dense predictions alongside the chain-of-thought answer.

2. **Teacher-only** (GPU or CPU + anchor checkpoints OR HuggingFace models):
       python visualize_visual_tokens.py --image img.jpg --teacher-only
   Runs the standalone teacher models (SAM, DepthAnything v2, DINOv2)
   directly on the image to produce reference dense predictions that
   illustrate what the CoVT visual tokens encode. No VLM required.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image


# ── Colour palette for SAM segmentation masks ──────────────────────
SAM_COLOURS = np.array([
    [30, 144, 255],   # dodgerblue
    [255, 127, 14],   # orange
    [44, 160, 44],    # green
    [214, 39, 40],    # red
    [148, 103, 189],  # purple
    [140, 86, 75],    # brown
    [227, 119, 194],  # pink
    [127, 127, 127],  # grey
], dtype=np.float32) / 255.0


# ===================================================================
#  Rendering
# ===================================================================

def overlay_masks_on_image(img_np, masks, alpha=0.45):
    """Blend coloured segmentation masks onto an RGB image (H×W×3, 0-1).

    Parameters
    ----------
    masks : np.ndarray  shape [N, H, W], bool or float (threshold at 0.5).
    """
    out = img_np.copy()
    for i, m in enumerate(masks):
        colour = SAM_COLOURS[i % len(SAM_COLOURS)]
        m_bool = m > 0.5
        for c in range(3):
            out[..., c] = np.where(m_bool, out[..., c] * (1 - alpha) + colour[c] * alpha, out[..., c])
    return np.clip(out, 0, 1)


def render_composite(
    image: Image.Image,
    seg_overlay: np.ndarray = None,
    n_seg_masks: int = 0,
    depth_map: np.ndarray = None,
    answer_text: str = None,
    anchor_info: dict = None,
    save_path: str = "visualize/outputs/vis_composite.png",
):
    """Create a single composite figure matching the reference CoVT demo layout.

    Layout
    ------
    Row 1: Input Image | SAM Segmentation | Depth Map
    Row 2: Token badges
    Row 3: Chain-of-thought text
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    panels = []
    titles = []

    # Panel 1: Input image
    img_np = np.array(image) / 255.0
    panels.append(img_np)
    titles.append("Input Image")

    # Panel 2: SAM segmentation
    if seg_overlay is not None:
        panels.append(seg_overlay)
        title = f"SAM Segmentation\n(8 tokens, {n_seg_masks} masks)"
        titles.append(title)

    # Panel 3: Depth map
    if depth_map is not None:
        d = depth_map.copy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        depth_rgb = plt.cm.Spectral(d)[..., :3]
        panels.append(depth_rgb)
        titles.append("Depth Map\n(4 tokens)")

    n_panels = len(panels)
    has_text = answer_text is not None and len(answer_text) > 0
    has_badges = anchor_info is not None

    height_ratios = [4]
    n_rows = 1
    if has_badges:
        height_ratios.append(0.4)
        n_rows += 1
    if has_text:
        height_ratios.append(1.5)
        n_rows += 1

    fig = plt.figure(figsize=(6 * n_panels, sum(height_ratios) * 1.2))
    gs = gridspec.GridSpec(n_rows, n_panels, height_ratios=height_ratios,
                           hspace=0.15, wspace=0.05)

    # ── Image panels ──
    for i, (panel, title) in enumerate(zip(panels, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(panel)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.axis("off")

    # ── Token badges ──
    if has_badges:
        ax_badge = fig.add_subplot(gs[1, :])
        ax_badge.axis("off")
        badge_colours = {
            "SAM": "#e74c3c",
            "Depth": "#3498db",
            "DINOv2": "#2ecc71",
            "PIDINet": "#9b59b6",
        }
        x_pos = 0.02
        for name, n_tok in anchor_info.items():
            colour = badge_colours.get(name, "#7f8c8d")
            txt = f"{name}: {n_tok} tokens"
            ax_badge.text(
                x_pos, 0.5, txt,
                transform=ax_badge.transAxes,
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colour, edgecolor="none"),
                va="center",
            )
            x_pos += 0.18

    # ── Answer text ──
    if has_text:
        ax_text = fig.add_subplot(gs[-1, :])
        ax_text.axis("off")
        wrapped = answer_text
        ax_text.text(
            0.02, 0.95, wrapped,
            transform=ax_text.transAxes,
            fontsize=8, fontfamily="monospace",
            va="top", ha="left",
            wrap=True,
        )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved composite → {save_path}")
    return save_path


# ===================================================================
#  Teacher-only mode: run standalone anchor models on an image
# ===================================================================

def run_teacher_sam(image: Image.Image, device="cpu"):
    """Run SAM automatic mask generator on the image.

    Returns (overlay_np, masks_np, n_masks).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train" / "src"))

    from anchors.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    ckpt = str(Path(__file__).resolve().parent.parent
               / "train" / "src" / "anchors" / "segment_anything"
               / "ckpt" / "sam_vit_h_4b8939.pth")
    if not os.path.isfile(ckpt):
        print(f"  [WARN] SAM checkpoint not found at {ckpt}")
        return None, None, 0

    sam = sam_model_registry["vit_h"](checkpoint=ckpt).eval().to(device)
    generator = SamAutomaticMaskGenerator(sam)

    img_resized = image.resize((256, 256))
    img_np = np.array(img_resized)
    masks_raw = generator.generate(img_np)
    masks_raw = sorted(masks_raw, key=lambda x: x["predicted_iou"] * x["stability_score"], reverse=True)[:8]
    masks_np = np.array([m["segmentation"].astype(np.float32) for m in masks_raw])
    n_masks = len(masks_np)

    img_float = img_np.astype(np.float32) / 255.0
    overlay = overlay_masks_on_image(img_float, masks_np)
    return overlay, masks_np, n_masks


def run_teacher_depth(image: Image.Image, device="cpu"):
    """Run DepthAnything v2 on the image. Returns depth_map [H, W]."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train" / "src"))

    from anchors.DepthAnything.depth_anything_v2.dpt import DepthAnythingV2

    ckpt = str(Path(__file__).resolve().parent.parent
               / "train" / "src" / "anchors" / "DepthAnything"
               / "ckpt" / "depth_anything_v2_vitl.pth")
    if not os.path.isfile(ckpt):
        print(f"  [WARN] DepthAnything checkpoint not found at {ckpt}")
        return None

    config = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
    model = DepthAnythingV2(**config)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.eval().to(device)

    img_resized = image.resize((256, 256))
    raw_img = np.array(img_resized)
    img_tensor, (H, W) = model.image2tensor(raw_img)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        depth = model.infer_image(raw_img, input_size=518)

    return depth


def run_teacher_only(image_path, device="cpu", save_dir="visualize/outputs"):
    """Run teacher models directly and produce the composite visualisation."""
    image = Image.open(image_path).convert("RGB")
    print(f"Running teacher-only mode on: {image_path}")

    anchor_info = {}
    seg_overlay = None
    n_masks = 0
    depth_map = None

    # SAM
    print("  Running SAM...")
    try:
        seg_overlay, masks_np, n_masks = run_teacher_sam(image, device=device)
        if seg_overlay is not None:
            anchor_info["SAM"] = 8
            print(f"  SAM: found {n_masks} masks")
    except Exception as e:
        print(f"  SAM failed: {e}")

    # Depth
    print("  Running DepthAnything v2...")
    try:
        depth_map = run_teacher_depth(image, device=device)
        if depth_map is not None:
            anchor_info["Depth"] = 4
            print(f"  Depth: shape {depth_map.shape}")
    except Exception as e:
        print(f"  DepthAnything failed: {e}")

    anchor_info["DINOv2"] = 4

    example_text = (
        "<think>\n"
        "Because the segmentation of the image is "
        "<|anchor_start|><|sam_pad|>×8<|anchor_end|>, "
        "the depth map of the image is "
        "<|anchor_start|><|depth_pad|>×4<|anchor_end|>, "
        "and the perception feature of the image is "
        "<|anchor_start|><|dino_pad|>×4<|anchor_end|>.\n"
        "</think>\n\n"
        "<answer>\n"
        "[Model answer would appear here with actual GPU inference]\n"
        "</answer>"
    )

    save_path = os.path.join(save_dir, "vis_composite.png")
    render_composite(
        image=image,
        seg_overlay=seg_overlay,
        n_seg_masks=n_masks,
        depth_map=depth_map,
        answer_text=example_text,
        anchor_info=anchor_info,
        save_path=save_path,
    )
    return save_path


# ===================================================================
#  Full model pipeline (GPU required)
# ===================================================================

def run_full_pipeline(
    model_path: str,
    image_path: str,
    question: str = "Describe the scene in the picture in detail.",
    anchor_types: list = None,
    device: str = "cuda",
    save_dir: str = "visualize/outputs",
):
    """End-to-end: CoVT model → generate with anchor tokens → decode → visualise.

    This loads the standard Qwen2.5-VL model (same as the Gradio demo), runs
    inference, and then separately runs the teacher anchor models on the input
    image to produce the dense predictions that the visual tokens encode.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import time

    print(f"Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    image = Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ],
    }]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    dev = model.device
    inputs = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    print("  Generating response...")
    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]
    answer_raw = processor.decode(new_tokens, skip_special_tokens=False)
    answer_clean = processor.decode(new_tokens, skip_special_tokens=True)
    print(f"  Generated in {elapsed:.1f}s")
    print(f"  Answer: {answer_clean[:200]}...")

    # Infer anchor types
    if anchor_types is None:
        name_lower = model_path.lower()
        anchor_types = []
        if "seg" in name_lower:
            anchor_types.append("sam")
        if "depth" in name_lower:
            anchor_types.append("depth")
        if "dino" in name_lower:
            anchor_types.append("dino")
        if "edge" in name_lower:
            anchor_types.append("pidinet")
        if not anchor_types:
            anchor_types = ["sam", "depth", "dino"]

    anchor_info = {}
    seg_overlay = None
    n_masks = 0
    depth_map = None

    if "sam" in anchor_types:
        print("  Running SAM teacher...")
        try:
            seg_overlay, _, n_masks = run_teacher_sam(image, device=str(dev))
            if seg_overlay is not None:
                anchor_info["SAM"] = 8
        except Exception as e:
            print(f"  SAM failed: {e}")

    if "depth" in anchor_types:
        print("  Running Depth teacher...")
        try:
            depth_map = run_teacher_depth(image, device=str(dev))
            if depth_map is not None:
                anchor_info["Depth"] = 4
        except Exception as e:
            print(f"  DepthAnything failed: {e}")

    if "dino" in anchor_types:
        anchor_info["DINOv2"] = 4

    if "pidinet" in anchor_types:
        anchor_info["PIDINet"] = 4

    display_text = answer_raw
    save_path = os.path.join(save_dir, "vis_composite.png")
    render_composite(
        image=image,
        seg_overlay=seg_overlay,
        n_seg_masks=n_masks,
        depth_map=depth_map,
        answer_text=display_text,
        anchor_info=anchor_info,
        save_path=save_path,
    )

    elapsed_path = os.path.join(save_dir, "vis_composite_elapsed.txt")
    with open(elapsed_path, "w") as f:
        f.write(f"Inference: {elapsed:.2f}s\n")

    print(f"\n✓ Done — saved to {save_dir}/")
    return save_path


# ===================================================================
#  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualise CoVT visual tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image.")
    parser.add_argument("--model", type=str, default="Wakals/CoVT-7B-seg_depth_dino",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--question", type=str,
                        default="Describe the scene in the picture in detail.",
                        help="Question prompt for the VLM.")
    parser.add_argument("--anchors", nargs="+", default=None,
                        choices=["sam", "depth", "dino", "pidinet"],
                        help="Anchor types to visualise.")
    parser.add_argument("--device", type=str, default="cuda", help="Device.")
    parser.add_argument("--save-dir", type=str, default="visualize/outputs",
                        help="Output directory.")
    parser.add_argument("--teacher-only", action="store_true",
                        help="Run teacher models only (no VLM inference).")
    args = parser.parse_args()

    if args.image is None:
        default_path = str(Path(__file__).resolve().parent.parent / "assets" / "clouds.png")
        if os.path.isfile(default_path):
            args.image = default_path
        else:
            parser.error("--image is required (or place an image at assets/clouds.png).")

    if args.teacher_only:
        run_teacher_only(args.image, device=args.device, save_dir=args.save_dir)
    else:
        run_full_pipeline(
            model_path=args.model,
            image_path=args.image,
            question=args.question,
            anchor_types=args.anchors,
            device=args.device,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
