"""
Visualize CoVT Visual Tokens

Produces a composite figure showing all four anchor-teacher outputs:
  - Input Image | SAM Segmentation | Depth Map | DINO PCA | Edge Map
  - Token type badges (SAM: 8 tokens, Depth: 4 tokens, DINOv2: 4 tokens, PIDINet: 4 tokens)
  - Chain-of-thought text with anchor token placeholders

Two execution paths
-------------------
1. **Full pipeline** (GPU + CoVT model + anchor checkpoints):
       python visualize_visual_tokens.py --image img.jpg [--model ...]
   Runs CoVT inference, then runs teacher models to visualise what
   the visual tokens encode.

2. **Teacher-only** (GPU or CPU + anchor checkpoints / HuggingFace models):
       python visualize_visual_tokens.py --image img.jpg --teacher-only
   Runs the standalone teacher models (SAM, DepthAnything v2, DINOv2,
   PIDINet) directly on the image. No VLM required.
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
from PIL import Image

TRAIN_ROOT = str(Path(__file__).resolve().parent.parent / "train")
TRAIN_SRC = os.path.join(TRAIN_ROOT, "src")


def _ensure_train_path():
    for p in (TRAIN_ROOT, TRAIN_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)


# ── Colour palette for SAM segmentation masks ──────────────────────
SAM_COLOURS = np.array([
    [30, 144, 255], [255, 127, 14], [44, 160, 44], [214, 39, 40],
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
], dtype=np.float32) / 255.0


# ===================================================================
#  Rendering
# ===================================================================

def overlay_masks_on_image(img_np, masks, alpha=0.45):
    """Blend coloured segmentation masks onto an RGB image (H,W,3 float 0-1)."""
    out = img_np.copy()
    for i, m in enumerate(masks):
        colour = SAM_COLOURS[i % len(SAM_COLOURS)]
        m_bool = m > 0.5
        for c in range(3):
            out[..., c] = np.where(
                m_bool,
                out[..., c] * (1 - alpha) + colour[c] * alpha,
                out[..., c],
            )
    return np.clip(out, 0, 1)


def dino_pca_rgb(features, patch_hw=None):
    """PCA-project DINO patch features → (H, W, 3) RGB in [0, 1].

    Parameters
    ----------
    features : torch.Tensor [1, N, C]  or  [N, C]
        N = 1 (CLS) + num_patches.
    """
    feats = features.squeeze(0).float().cpu()
    patch_feats = feats[1:] if feats.shape[0] > 1 else feats
    N, C = patch_feats.shape
    if patch_hw is None:
        side = int(np.sqrt(N))
        patch_hw = (side, side)
    H, W = patch_hw
    mean = patch_feats.mean(0, keepdim=True)
    centred = patch_feats - mean
    _, _, V = torch.pca_lowrank(centred, q=3)
    proj = centred @ V[:, :3]
    proj = proj - proj.min(0).values
    proj = proj / (proj.max(0).values + 1e-8)
    return proj.reshape(H, W, 3).numpy()


def render_composite(
    image: Image.Image,
    seg_overlay: np.ndarray = None,
    n_seg_masks: int = 0,
    depth_map: np.ndarray = None,
    dino_rgb: np.ndarray = None,
    edge_map: np.ndarray = None,
    answer_text: str = None,
    anchor_info: dict = None,
    save_path: str = "visualize/outputs/vis_composite.png",
):
    """Create a single composite figure with all available anchor panels."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    panels = []
    titles = []

    img_np = np.array(image) / 255.0
    panels.append(img_np)
    titles.append("Input Image")

    if seg_overlay is not None:
        panels.append(seg_overlay)
        titles.append(f"SAM Segmentation\n(8 tokens, {n_seg_masks} masks)")

    if depth_map is not None:
        d = depth_map.copy().astype(np.float64)
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        panels.append(plt.cm.Spectral(d)[..., :3])
        titles.append("Depth Map\n(4 tokens)")

    if dino_rgb is not None:
        H_d, W_d = dino_rgb.shape[:2]
        target_h, target_w = img_np.shape[:2]
        up = np.array(Image.fromarray(
            (np.clip(dino_rgb, 0, 1) * 255).astype(np.uint8)
        ).resize((target_w, target_h), Image.BILINEAR)) / 255.0
        panels.append(up)
        titles.append(f"DINOv2 PCA\n(4 tokens, {H_d}×{W_d} patches)")

    if edge_map is not None:
        e = edge_map.copy().astype(np.float64)
        e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        panels.append(np.stack([e] * 3, axis=-1))
        titles.append("Edge Map (PIDINet)\n(4 tokens)")

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

    fig = plt.figure(figsize=(5 * n_panels, sum(height_ratios) * 1.2))
    gs = gridspec.GridSpec(
        n_rows, n_panels,
        height_ratios=height_ratios,
        hspace=0.15, wspace=0.05,
    )

    for i, (panel, title) in enumerate(zip(panels, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(panel)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

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
        spacing = min(0.18, 0.9 / max(len(anchor_info), 1))
        for name, n_tok in anchor_info.items():
            colour = badge_colours.get(name, "#7f8c8d")
            ax_badge.text(
                x_pos, 0.5, f"{name}: {n_tok} tokens",
                transform=ax_badge.transAxes,
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=colour, edgecolor="none",
                ),
                va="center",
            )
            x_pos += spacing

    if has_text:
        ax_text = fig.add_subplot(gs[-1, :])
        ax_text.axis("off")
        ax_text.text(
            0.02, 0.95, answer_text,
            transform=ax_text.transAxes,
            fontsize=8, fontfamily="monospace",
            va="top", ha="left", wrap=True,
        )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved composite → {save_path}")
    return save_path


# ===================================================================
#  Teacher runners
# ===================================================================

def _ckpt_path(*parts):
    return str(Path(__file__).resolve().parent.parent.joinpath(*parts))


def run_teacher_sam(image: Image.Image, device="cpu"):
    """Run SAM automatic mask generator. Returns (overlay, masks, n_masks)."""
    _ensure_train_path()
    from anchors.segment_anything import (
        sam_model_registry, SamAutomaticMaskGenerator,
    )

    ckpt = _ckpt_path(
        "train", "src", "anchors", "segment_anything",
        "ckpt", "sam_vit_h_4b8939.pth",
    )
    if not os.path.isfile(ckpt):
        print(f"  [WARN] SAM checkpoint not found: {ckpt}")
        return None, None, 0

    sam = sam_model_registry["vit_h"](checkpoint=ckpt).eval().to(device)
    gen = SamAutomaticMaskGenerator(sam)

    img256 = image.resize((256, 256))
    img_np = np.array(img256)
    masks_raw = gen.generate(img_np)
    masks_raw = sorted(
        masks_raw,
        key=lambda x: x["predicted_iou"] * x["stability_score"],
        reverse=True,
    )[:8]
    masks_np = np.array([m["segmentation"].astype(np.float32) for m in masks_raw])
    n = len(masks_np)
    overlay = overlay_masks_on_image(
        img_np.astype(np.float32) / 255.0, masks_np,
    )
    return overlay, masks_np, n


def run_teacher_depth(image: Image.Image, device="cpu"):
    """Run DepthAnything v2.  Returns depth [H, W] numpy array."""
    _ensure_train_path()
    from anchors.DepthAnything.depth_anything_v2.dpt import DepthAnythingV2

    ckpt = _ckpt_path(
        "train", "src", "anchors", "DepthAnything",
        "ckpt", "depth_anything_v2_vitl.pth",
    )
    if not os.path.isfile(ckpt):
        print(f"  [WARN] DepthAnything checkpoint not found: {ckpt}")
        return None

    cfg = {
        "encoder": "vitl", "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    }
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.eval().to(device)

    raw = np.array(image.resize((256, 256)))
    with torch.no_grad():
        depth = model.infer_image(raw, input_size=518)
    return depth


def run_teacher_dino(image: Image.Image, device="cpu"):
    """Run DINOv2 ViT-L/14.  Returns PCA RGB [H_p, W_p, 3] numpy."""
    dino = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14",
        verbose=False,
    )
    dino = dino.eval().to(device)

    extracted = {}

    def _hook(module, inp, out):
        extracted["norm"] = out

    handle = dino.norm.register_forward_hook(_hook)

    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-large",
        crop_size={"height": 518, "width": 518},
    )
    pixel_values = proc(images=image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        dino(pixel_values)
    handle.remove()

    feats = extracted["norm"].detach()  # [1, 1+N, 1024]
    patch_feats = feats[0, 1:]          # [N, 1024]
    side = int(np.sqrt(patch_feats.shape[0]))
    rgb = dino_pca_rgb(feats, patch_hw=(side, side))
    return rgb


def run_teacher_pidinet(image: Image.Image, device="cpu"):
    """Run PIDINet edge detector.  Returns edge map [H, W] numpy in [0,1]."""
    _ensure_train_path()
    import anchors.pidinet.models as pidinet_model
    from anchors.pidinet.models.convert_pidinet import convert_pidinet
    from torchvision import transforms as T

    ckpt = _ckpt_path(
        "train", "src", "anchors", "pidinet",
        "ckpt", "table5_baseline.pth",
    )
    if not os.path.isfile(ckpt):
        print(f"  [WARN] PIDINet checkpoint not found: {ckpt}")
        return None

    class _Args:
        model = "pidinet_converted"
        config = "carv4"
        sa = True
        dil = True

    args = _Args()
    net = pidinet_model.pidinet_converted(args)
    state = torch.load(ckpt, map_location="cpu")
    sd = state["state_dict"] if "state_dict" in state else state
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(convert_pidinet(sd, args.config))
    net = net.eval().to(device)

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    img256 = image.resize((256, 256))
    x = normalize(T.ToTensor()(img256)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, e1, _ = net(x)
    edge = torch.sigmoid(e1).squeeze().cpu().numpy()
    return edge


# ===================================================================
#  Teacher-only mode
# ===================================================================

def run_teacher_only(image_path, device="cpu", save_dir="visualize/outputs"):
    """Run all teacher models and produce composite visualisation."""
    image = Image.open(image_path).convert("RGB")
    print(f"Running teacher-only mode on: {image_path}")

    anchor_info = {}
    seg_overlay = None
    n_masks = 0
    depth_map = None
    dino_rgb = None
    edge_map = None

    print("  Running SAM...")
    try:
        seg_overlay, _, n_masks = run_teacher_sam(image, device=device)
        if seg_overlay is not None:
            anchor_info["SAM"] = 8
            print(f"    → {n_masks} masks")
    except Exception as e:
        print(f"    SAM failed: {e}")

    print("  Running DepthAnything v2...")
    try:
        depth_map = run_teacher_depth(image, device=device)
        if depth_map is not None:
            anchor_info["Depth"] = 4
            print(f"    → shape {depth_map.shape}")
    except Exception as e:
        print(f"    DepthAnything failed: {e}")

    print("  Running DINOv2...")
    try:
        dino_rgb = run_teacher_dino(image, device=device)
        if dino_rgb is not None:
            anchor_info["DINOv2"] = 4
            print(f"    → PCA map {dino_rgb.shape[:2]}")
    except Exception as e:
        print(f"    DINOv2 failed: {e}")

    print("  Running PIDINet...")
    try:
        edge_map = run_teacher_pidinet(image, device=device)
        if edge_map is not None:
            anchor_info["PIDINet"] = 4
            print(f"    → edge map {edge_map.shape}")
    except Exception as e:
        print(f"    PIDINet failed: {e}")

    cot = (
        "<think>\n"
        "Because the segmentation of the image is "
        "<|anchor_start|><|sam_pad|>×8<|anchor_end|>, "
        "the depth map of the image is "
        "<|anchor_start|><|depth_pad|>×4<|anchor_end|>, "
        "the perception feature of the image is "
        "<|anchor_start|><|dino_pad|>×4<|anchor_end|>, "
        "and the edge map of the image is "
        "<|anchor_start|><|pidinet_pad|>×4<|anchor_end|>.\n"
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
        dino_rgb=dino_rgb,
        edge_map=edge_map,
        answer_text=cot,
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
    """CoVT model inference + teacher model visualisation."""
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

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    dev = model.device
    inputs = {
        k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

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
            anchor_types = ["sam", "depth", "dino", "pidinet"]

    anchor_info = {}
    seg_overlay, n_masks, depth_map, dino_rgb, edge_map = (
        None, 0, None, None, None,
    )
    gpu_dev = str(dev)

    if "sam" in anchor_types:
        print("  Running SAM teacher...")
        try:
            seg_overlay, _, n_masks = run_teacher_sam(image, device=gpu_dev)
            if seg_overlay is not None:
                anchor_info["SAM"] = 8
        except Exception as e:
            print(f"    SAM failed: {e}")

    if "depth" in anchor_types:
        print("  Running Depth teacher...")
        try:
            depth_map = run_teacher_depth(image, device=gpu_dev)
            if depth_map is not None:
                anchor_info["Depth"] = 4
        except Exception as e:
            print(f"    DepthAnything failed: {e}")

    if "dino" in anchor_types:
        print("  Running DINOv2 teacher...")
        try:
            dino_rgb = run_teacher_dino(image, device=gpu_dev)
            if dino_rgb is not None:
                anchor_info["DINOv2"] = 4
        except Exception as e:
            print(f"    DINOv2 failed: {e}")

    if "pidinet" in anchor_types:
        print("  Running PIDINet teacher...")
        try:
            edge_map = run_teacher_pidinet(image, device=gpu_dev)
            if edge_map is not None:
                anchor_info["PIDINet"] = 4
        except Exception as e:
            print(f"    PIDINet failed: {e}")

    save_path = os.path.join(save_dir, "vis_composite.png")
    render_composite(
        image=image,
        seg_overlay=seg_overlay,
        n_seg_masks=n_masks,
        depth_map=depth_map,
        dino_rgb=dino_rgb,
        edge_map=edge_map,
        answer_text=answer_raw,
        anchor_info=anchor_info,
        save_path=save_path,
    )
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
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image.")
    parser.add_argument("--model", type=str,
                        default="Wakals/CoVT-7B-seg_depth_dino",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--question", type=str,
                        default="Describe the scene in detail.",
                        help="Question prompt for the VLM.")
    parser.add_argument("--anchors", nargs="+", default=None,
                        choices=["sam", "depth", "dino", "pidinet"],
                        help="Anchor types to visualise.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device.")
    parser.add_argument("--save-dir", type=str, default="visualize/outputs",
                        help="Output directory.")
    parser.add_argument("--teacher-only", action="store_true",
                        help="Run teacher models only (no VLM inference).")
    args = parser.parse_args()

    if args.image is None:
        default = str(
            Path(__file__).resolve().parent.parent / "assets" / "clouds.png"
        )
        if os.path.isfile(default):
            args.image = default
        else:
            parser.error("--image required (or place assets/clouds.png).")

    if args.teacher_only:
        run_teacher_only(
            args.image, device=args.device, save_dir=args.save_dir,
        )
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
