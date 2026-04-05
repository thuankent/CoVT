"""
Visualize CoVT Visual Tokens

This script decodes the hidden-state representations at CoVT's special
anchor-token positions back into dense predictions (segmentation masks,
depth maps, edge maps, DINO feature maps) and renders publication-quality
composite visualisations.

Two modes
---------
1. **Full model mode** (requires GPU + model weights + anchor checkpoints):
       python visualize_visual_tokens.py --image path/to/img.jpg
   Loads CoVTForConditionalGeneration, runs inference, decodes anchor tokens.

2. **Demo / synthetic mode** (CPU only, no weights needed):
       python visualize_visual_tokens.py --demo
   Generates synthetic dense predictions to verify the rendering pipeline.
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
from matplotlib.colors import ListedColormap
from PIL import Image


# ---------------------------------------------------------------------------
# Colour palette for segmentation overlay (up to 8 masks)
# ---------------------------------------------------------------------------
MASK_COLOURS = [
    (0.12, 0.47, 0.71, 0.50),   # blue
    (1.00, 0.50, 0.05, 0.50),   # orange
    (0.17, 0.63, 0.17, 0.50),   # green
    (0.84, 0.15, 0.16, 0.50),   # red
    (0.58, 0.40, 0.74, 0.50),   # purple
    (0.55, 0.34, 0.29, 0.50),   # brown
    (0.89, 0.47, 0.76, 0.50),   # pink
    (0.50, 0.50, 0.50, 0.50),   # grey
]


# ===================================================================
# Rendering helpers (work on plain numpy/torch tensors, no model needed)
# ===================================================================

def render_segmentation(image: Image.Image,
                        masks: torch.Tensor,
                        title: str = "Segmentation Masks") -> plt.Figure:
    """Overlay predicted segmentation masks on *image*.

    Parameters
    ----------
    image : PIL.Image
        Original image.
    masks : torch.Tensor
        Shape ``[N, H, W]`` – binary or logit masks (thresholded at 0).
    """
    img_np = np.array(image.resize((masks.shape[-1], masks.shape[-2]))) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1) original
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # 2) individual masks tiled
    n_masks = masks.shape[0]
    combined = np.zeros((*img_np.shape[:2], 4))
    for i in range(n_masks):
        m = (masks[i] > 0).float().cpu().numpy()
        colour = MASK_COLOURS[i % len(MASK_COLOURS)]
        overlay = np.zeros((*m.shape, 4))
        overlay[..., :3] = colour[:3]
        overlay[..., 3] = m * colour[3]
        combined = np.maximum(combined, overlay)

    axes[1].imshow(img_np)
    axes[1].imshow(combined)
    axes[1].set_title(f"Masks Overlay ({n_masks} tokens)", fontsize=14)
    axes[1].axis("off")

    # 3) mask-coloured composite
    composite = img_np.copy()
    for i in range(n_masks):
        m = (masks[i] > 0).float().cpu().numpy()
        colour = np.array(MASK_COLOURS[i % len(MASK_COLOURS)][:3])
        alpha = 0.45
        for c in range(3):
            composite[..., c] = np.where(
                m > 0.5,
                composite[..., c] * (1 - alpha) + colour[c] * alpha,
                composite[..., c],
            )
    axes[2].imshow(composite)
    axes[2].set_title("Blended Composite", fontsize=14)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def render_depth(image: Image.Image,
                 token_depths: torch.Tensor,
                 depth_avg: torch.Tensor,
                 title: str = "Depth Reconstruction") -> plt.Figure:
    """Visualise per-token depth maps and the averaged depth.

    Parameters
    ----------
    token_depths : torch.Tensor
        Shape ``[T, H, W]`` – one depth map per token.
    depth_avg : torch.Tensor
        Shape ``[1, H, W]`` or ``[H, W]`` – averaged depth.
    """
    img_np = np.array(image.resize(
        (depth_avg.shape[-1], depth_avg.shape[-2]))) / 255.0
    n_tokens = token_depths.shape[0]

    fig, axes = plt.subplots(1, n_tokens + 2, figsize=(4 * (n_tokens + 2), 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    for t in range(n_tokens):
        d = token_depths[t].cpu().numpy()
        axes[t + 1].imshow(d, cmap="magma")
        axes[t + 1].set_title(f"Token {t}", fontsize=11)
        axes[t + 1].axis("off")

    avg = depth_avg.squeeze().cpu().numpy()
    im = axes[-1].imshow(avg, cmap="magma")
    axes[-1].set_title("Averaged Depth", fontsize=11)
    axes[-1].axis("off")
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def render_edge(image: Image.Image,
                edge_map: torch.Tensor,
                title: str = "Edge Detection (PIDINet)") -> plt.Figure:
    """Side-by-side original vs predicted edge map.

    Parameters
    ----------
    edge_map : torch.Tensor
        Shape ``[1, H, W]`` or ``[H, W]`` – values in [0, 1].
    """
    img_np = np.array(image.resize(
        (edge_map.shape[-1], edge_map.shape[-2]))) / 255.0
    e = edge_map.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(e, cmap="gray")
    axes[1].set_title("Predicted Edge Map", fontsize=13)
    axes[1].axis("off")

    # overlay
    overlay = img_np.copy()
    edge_3ch = np.stack([e] * 3, axis=-1)
    overlay = overlay * 0.6 + edge_3ch * 0.4
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Edge Overlay", fontsize=13)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def render_dino_features(image: Image.Image,
                         dino_features: torch.Tensor,
                         patch_hw: tuple = None,
                         title: str = "DINO Feature Map (PCA)") -> plt.Figure:
    """PCA-visualise DINO patch features as an RGB map.

    Parameters
    ----------
    dino_features : torch.Tensor
        Shape ``[1, N, C]`` where N = 1 (CLS) + num_patches.
    patch_hw : tuple, optional
        ``(H_patches, W_patches)`` for reshaping. Inferred if None.
    """
    feats = dino_features.squeeze(0).float().cpu()
    # drop CLS if present (first token when N > 1)
    if feats.shape[0] > 1:
        patch_feats = feats[1:]  # [num_patches, C]
    else:
        patch_feats = feats

    N, C = patch_feats.shape
    if patch_hw is None:
        side = int(np.sqrt(N))
        patch_hw = (side, side)
    H, W = patch_hw

    # PCA → 3 components for RGB
    mean = patch_feats.mean(0, keepdim=True)
    centred = patch_feats - mean
    U, S, V = torch.pca_lowrank(centred, q=3)
    proj = centred @ V[:, :3]  # [N, 3]
    # normalise to [0, 1]
    proj = proj - proj.min(0).values
    proj = proj / (proj.max(0).values + 1e-8)
    rgb = proj.reshape(H, W, 3).numpy()

    img_np = np.array(image.resize((W * 14, H * 14))) / 255.0
    rgb_up = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (W * 14, H * 14), Image.BILINEAR)) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(rgb)
    axes[1].set_title("DINO PCA (patch grid)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(rgb_up)
    axes[2].set_title("DINO PCA (upscaled)", fontsize=13)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def render_all_tokens(image: Image.Image,
                      results: dict,
                      save_dir: str = "visualize/outputs") -> list:
    """Dispatch to per-anchor renderers and save figures.

    Parameters
    ----------
    results : dict
        Keys: ``"seg_masks"``, ``"token_depths"``, ``"depth_avg"``,
        ``"edge_map"``, ``"dino_features"`` (all optional).
    save_dir : str
        Directory for saving PNG outputs.

    Returns
    -------
    list[str]  – paths of saved figures.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved = []

    if "seg_masks" in results and results["seg_masks"] is not None:
        fig = render_segmentation(image, results["seg_masks"])
        p = os.path.join(save_dir, "vis_segmentation.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
        print(f"  Saved segmentation  → {p}")

    if "depth_avg" in results and results["depth_avg"] is not None:
        token_d = results.get("token_depths", results["depth_avg"].unsqueeze(0))
        fig = render_depth(image, token_d, results["depth_avg"])
        p = os.path.join(save_dir, "vis_depth.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
        print(f"  Saved depth         → {p}")

    if "edge_map" in results and results["edge_map"] is not None:
        fig = render_edge(image, results["edge_map"])
        p = os.path.join(save_dir, "vis_edge.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
        print(f"  Saved edge          → {p}")

    if "dino_features" in results and results["dino_features"] is not None:
        fig = render_dino_features(image, results["dino_features"])
        p = os.path.join(save_dir, "vis_dino_pca.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
        print(f"  Saved DINO PCA      → {p}")

    # ---- summary panel ----
    if saved:
        _render_summary_panel(image, results, save_dir)

    return saved


def _render_summary_panel(image: Image.Image, results: dict, save_dir: str):
    """Create a single summary image with all available visualisations."""
    panels = []
    labels = []

    img_np = np.array(image) / 255.0
    panels.append(img_np)
    labels.append("Original")

    if "seg_masks" in results and results["seg_masks"] is not None:
        masks = results["seg_masks"]
        h, w = masks.shape[-2], masks.shape[-1]
        composite = np.array(image.resize((w, h))) / 255.0
        for i in range(masks.shape[0]):
            m = (masks[i] > 0).float().cpu().numpy()
            colour = np.array(MASK_COLOURS[i % len(MASK_COLOURS)][:3])
            for c in range(3):
                composite[..., c] = np.where(
                    m > 0.5,
                    composite[..., c] * 0.55 + colour[c] * 0.45,
                    composite[..., c],
                )
        panels.append(composite)
        labels.append("Segmentation")

    if "depth_avg" in results and results["depth_avg"] is not None:
        d = results["depth_avg"].squeeze().cpu().numpy()
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        cmap = plt.cm.magma
        depth_rgb = cmap(d_norm)[..., :3]
        panels.append(depth_rgb)
        labels.append("Depth")

    if "edge_map" in results and results["edge_map"] is not None:
        e = results["edge_map"].squeeze().cpu().numpy()
        panels.append(np.stack([e] * 3, axis=-1))
        labels.append("Edges")

    if "dino_features" in results and results["dino_features"] is not None:
        feats = results["dino_features"].squeeze(0).float().cpu()
        patch_feats = feats[1:] if feats.shape[0] > 1 else feats
        N, C = patch_feats.shape
        side = int(np.sqrt(N))
        mean = patch_feats.mean(0, keepdim=True)
        centred = patch_feats - mean
        U, S, V = torch.pca_lowrank(centred, q=3)
        proj = centred @ V[:, :3]
        proj = proj - proj.min(0).values
        proj = proj / (proj.max(0).values + 1e-8)
        rgb = proj.reshape(side, side, 3).numpy()
        rgb_up = np.array(Image.fromarray(
            (rgb * 255).astype(np.uint8)).resize(
                (panels[0].shape[1], panels[0].shape[0]),
                Image.BILINEAR)) / 255.0
        panels.append(rgb_up)
        labels.append("DINO PCA")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, panel, label in zip(axes, panels, labels):
        ax.imshow(np.clip(panel, 0, 1))
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.axis("off")
    fig.suptitle("CoVT Visual Token Visualisation", fontsize=16,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    p = os.path.join(save_dir, "vis_summary.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary panel → {p}")


# ===================================================================
# Full-model decoding (requires GPU + weights)
# ===================================================================

def decode_visual_tokens(
    model_path: str,
    image_path: str,
    question: str = "Describe the scene in the picture in detail.",
    anchor_types: list = None,
    device: str = "cuda",
    save_dir: str = "visualize/outputs",
):
    """End-to-end: load model → generate → decode anchors → visualise.

    Parameters
    ----------
    model_path : str
        HuggingFace hub ID or local path (e.g. ``Wakals/CoVT-7B-seg_depth_dino``).
    image_path : str
        Path to the input image.
    question : str
        User question for the VLM.
    anchor_types : list[str]
        Which anchors to decode, e.g. ``["sam", "depth", "dino"]``.
        Inferred from model name if None.
    device : str
        ``"cuda"`` or ``"cpu"`` (CPU will be extremely slow for 7B).
    save_dir : str
        Where to save visualisation PNGs.
    """
    from transformers import AutoProcessor

    # --- add training source to path for CoVT imports ---
    train_root = str(Path(__file__).resolve().parent.parent / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
        sys.path.insert(0, os.path.join(train_root, "src"))

    from src.training.covt_qwen2_5_vl import CoVTForConditionalGeneration
    from src.training.constants import (
        SAM_PAD_TOKEN, DINO_PAD_TOKEN, DEPTH_PAD_TOKEN,
        PIDINET_PAD_TOKEN, ANCHOR_START_TOKEN, ANCHOR_END_TOKEN,
    )

    # infer anchor types from model name
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

    print(f"Anchor types: {anchor_types}")
    print(f"Loading model from: {model_path}")

    # load processor and add special tokens
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    add_tokens = [
        SAM_PAD_TOKEN, DINO_PAD_TOKEN, DEPTH_PAD_TOKEN,
        "<|sd_pad|>", "<|intern_pad|>", PIDINET_PAD_TOKEN,
        "<|siglip_pad|>", "<|metaclip_pad|>",
        "<think>", "</think>", "<answer>", "</answer>",
    ]
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [ANCHOR_START_TOKEN, ANCHOR_END_TOKEN]}
    )
    processor.tokenizer.add_tokens(add_tokens)

    # resolve token IDs
    def _tid(tok):
        return processor.tokenizer(tok, add_special_tokens=False).input_ids[0]

    sam_token_idx = _tid(SAM_PAD_TOKEN)
    dino_token_idx = _tid(DINO_PAD_TOKEN)
    depth_token_idx = _tid(DEPTH_PAD_TOKEN)
    pidinet_token_idx = _tid(PIDINET_PAD_TOKEN)

    # load model
    model = CoVTForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    model.resize_token_embeddings(len(processor.tokenizer))
    model.get_anchor_token_idx(
        sam_token_idx, dino_token_idx, depth_token_idx,
        _tid("<|sd_pad|>"), _tid("<|intern_pad|>"),
        pidinet_token_idx, _tid("<|siglip_pad|>"), _tid("<|metaclip_pad|>"),
    )

    # set up anchor models
    anchor_id_str = "_".join(anchor_types)
    model.get_anchor_model_ids(anchor_id_str)

    # load image
    pil_image = Image.open(image_path).convert("RGB")

    # build conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt], images=[pil_image], return_tensors="pt"
    ).to(device)

    # run forward with hidden states
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0]
    text_answer = processor.decode(
        generated_ids[inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print(f"\nModel answer: {text_answer}\n")

    # --- Extract hidden states at anchor positions and decode ---
    # For full decoding we re-run a single forward pass with the generated sequence
    full_ids = generated_ids.unsqueeze(0)
    with torch.no_grad():
        fwd_out = model(
            input_ids=full_ids,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True,
            image_files=[[pil_image]],
        )

    results = {}
    anchor_out = fwd_out.anchor_outputs

    if anchor_out[0] is not None:
        results["seg_masks"] = anchor_out[0]["pred_masks"]
    if anchor_out[2] is not None:
        results["depth_avg"] = anchor_out[2]["pred_depth"].squeeze(0)
        results["token_depths"] = anchor_out[2].get("token_depths")
    if anchor_out[1] is not None:
        results["dino_features"] = anchor_out[1]["pred_dino"]
    if anchor_out[5] is not None:
        results["edge_map"] = anchor_out[5].get("pred_edge")

    # visualise
    saved = render_all_tokens(pil_image, results, save_dir=save_dir)
    print(f"\n✓ Saved {len(saved)} visualisation(s) to {save_dir}/")
    return results


# ===================================================================
# Demo / synthetic mode
# ===================================================================

def generate_synthetic_results(image: Image.Image) -> dict:
    """Create plausible synthetic dense predictions for pipeline testing."""
    W, H = image.size
    rng = np.random.default_rng(42)

    # --- segmentation masks (8 tokens) ---
    masks = torch.zeros(8, H, W)
    for i in range(8):
        cx = rng.integers(W // 4, 3 * W // 4)
        cy = rng.integers(H // 4, 3 * H // 4)
        rx = rng.integers(W // 10, W // 4)
        ry = rng.integers(H // 10, H // 4)
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        masks[i] = (ellipse < 1.0).float()

    # --- depth maps (4 tokens) ---
    yy_norm = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    xx_norm = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    token_depths = torch.stack([
        yy_norm * 0.8 + xx_norm * 0.2,
        (1 - yy_norm) * 0.6 + xx_norm * 0.4,
        yy_norm * 0.5 + (1 - xx_norm) * 0.5,
        torch.sqrt(yy_norm ** 2 + xx_norm ** 2) / np.sqrt(2),
    ])
    depth_avg = token_depths.mean(0)

    # --- edge map ---
    img_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    from scipy.ndimage import sobel
    sx = sobel(img_np, axis=0)
    sy = sobel(img_np, axis=1)
    edge = np.sqrt(sx ** 2 + sy ** 2)
    edge = edge / (edge.max() + 1e-8)
    edge_map = torch.from_numpy(edge).float()

    # --- DINO features (1025 × 1024) ---
    n_patches = 32 * 32
    dino_features = torch.randn(1, n_patches + 1, 1024)
    # make spatially correlated
    patch_feats = dino_features[0, 1:]
    grid = patch_feats.reshape(32, 32, 1024)
    for _ in range(3):
        grid = (
            grid
            + F.pad(grid[1:], (0, 0, 0, 0, 0, 1))
            + F.pad(grid[:-1], (0, 0, 0, 0, 1, 0))
            + F.pad(grid[:, 1:], (0, 0, 0, 1, 0, 0))
            + F.pad(grid[:, :-1], (0, 0, 1, 0, 0, 0))
        ) / 5.0
    dino_features[0, 1:] = grid.reshape(n_patches, 1024)

    return {
        "seg_masks": masks,
        "token_depths": token_depths,
        "depth_avg": depth_avg,
        "edge_map": edge_map,
        "dino_features": dino_features,
    }


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualise CoVT visual tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to input image (required in full-model mode).",
    )
    parser.add_argument(
        "--model", type=str, default="Wakals/CoVT-7B-seg_depth_dino",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--question", type=str,
        default="Describe the scene in the picture in detail.",
        help="Question prompt for the VLM.",
    )
    parser.add_argument(
        "--anchors", nargs="+", default=None,
        choices=["sam", "depth", "dino", "pidinet"],
        help="Anchor types to decode (inferred from model name if omitted).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for full-model mode.",
    )
    parser.add_argument(
        "--save-dir", type=str, default="visualize/outputs",
        help="Output directory for saved visualisations.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode with synthetic data (no model needed).",
    )
    args = parser.parse_args()

    if args.demo:
        img_path = args.image or os.path.join(
            os.path.dirname(__file__), "..", "assets", "clouds.png"
        )
        if not os.path.isfile(img_path):
            print(f"Image not found at {img_path}, generating a test image.")
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
        else:
            img = Image.open(img_path).convert("RGB")

        print(f"Running in DEMO mode with image: {img_path}")
        results = generate_synthetic_results(img)
        saved = render_all_tokens(img, results, save_dir=args.save_dir)
        print(f"\n✓ Demo complete — saved {len(saved)} file(s) to {args.save_dir}/")
    else:
        if args.image is None:
            parser.error("--image is required in full-model mode (or use --demo).")
        decode_visual_tokens(
            model_path=args.model,
            image_path=args.image,
            question=args.question,
            anchor_types=args.anchors,
            device=args.device,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
