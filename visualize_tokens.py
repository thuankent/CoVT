"""
Visualize and decode CoVT visual tokens (segmentation and depth).

This script loads a CoVT model checkpoint, runs inference on an image,
extracts the hidden states at visual-token positions (<|sam_pad|> and
<|depth_pad|>), projects them through the learned projection layers,
and decodes them with the frozen anchor models (SAM for segmentation,
DepthAnything V2 for depth) to produce dense visual predictions.

Usage
-----
    python visualize_tokens.py \
        --image_path assets/clouds.png \
        --model_name Wakals/CoVT-7B-seg_depth_dino \
        --sam_ckpt  train/src/anchors/segment_anything/ckpt/sam_vit_h_4b8939.pth \
        --depth_ckpt train/src/anchors/DepthAnything/ckpt/depth_anything_v2_vitl.pth \
        --output_dir outputs/visual_tokens \
        --question "Describe the scene and analyze the spatial layout."

The script produces:
    - Individual segmentation masks (one per SAM token)
    - A combined overlay of all masks on the original image
    - The reconstructed depth map
    - A side-by-side comparison panel
"""

import argparse
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Path setup – anchor modules live under train/src/
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SRC = os.path.join(REPO_ROOT, "train", "src")
if TRAIN_SRC not in sys.path:
    sys.path.insert(0, TRAIN_SRC)

from anchors.segment_anything import build_sam_vit_h, SamPredictor
from anchors.segment_anything.utils.transforms import ResizeLongestSide
from anchors.DepthAnything.depth_anything_v2.dpt import DepthAnythingV2

# ---------------------------------------------------------------------------
# Special token strings (mirrored from training/constants.py)
# ---------------------------------------------------------------------------
SAM_PAD_TOKEN = "<|sam_pad|>"
DEPTH_PAD_TOKEN = "<|depth_pad|>"
DINO_PAD_TOKEN = "<|dino_pad|>"
ANCHOR_START_TOKEN = "<|anchor_start|>"
ANCHOR_END_TOKEN = "<|anchor_end|>"

NUM_SAM_TOKENS = 8
NUM_DEPTH_TOKENS = 4

# Qwen2.5-VL hidden dimension
HIDDEN_DIM = 3584
SAM_EMBED_DIM = 256
DEPTH_EMBED_DIM = 1024
DEPTH_LAYER_IDX = [4, 11, 17, 23]


# ═══════════════════════════════════════════════════════════════════════════
# Helper: custom RoPE identical to CoVTForConditionalGeneration.apply_rope_custome
# ═══════════════════════════════════════════════════════════════════════════
def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply rotary position encoding to visual-token hidden states."""
    N, K = x.shape[-2], x.shape[-1]
    pad = K % 2 == 1
    if pad:
        x = F.pad(x, (0, 1))
        K += 1
    half = K // 2
    x1, x2 = x[..., :half], x[..., half:]

    idx = torch.arange(half, device=x.device, dtype=x.dtype)
    theta = torch.exp(
        -torch.log(torch.tensor(10000.0, device=x.device, dtype=x.dtype))
        * (2 * idx / K)
    )
    pos = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(-1)
    ang = pos * theta
    cos, sin = torch.cos(ang), torch.sin(ang)

    while cos.dim() < x1.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.cat([y1, y2], dim=-1)
    return y[..., : K - int(pad)]


# ═══════════════════════════════════════════════════════════════════════════
# Segmentation decoder
# ═══════════════════════════════════════════════════════════════════════════
class SegmentationDecoder:
    """Decode SAM-token hidden states into per-token segmentation masks."""

    def __init__(self, sam_ckpt: str, device: torch.device):
        self.device = device
        self.sam = build_sam_vit_h(checkpoint=sam_ckpt).to(device).eval()
        self.predictor = SamPredictor(self.sam)
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)

    @torch.no_grad()
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Run SAM image encoder → [1, 256, 64, 64]."""
        img = pil_image.resize((256, 256))
        img_np = np.array(img)
        self.predictor.set_image(img_np)
        return self.predictor.get_image_embedding().detach()

    @torch.no_grad()
    def decode_tokens(
        self,
        sam_embed: torch.Tensor,
        pil_image: Image.Image,
        token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode projected token embeddings into masks.

        Parameters
        ----------
        sam_embed : [1, 256, 64, 64]  SAM image embedding
        pil_image : original PIL image
        token_embeddings : [n_tokens, 256]  projected SAM token vectors

        Returns
        -------
        masks : [n_tokens, H, W]  full-resolution binary-ish masks
        """
        img = pil_image.resize((256, 256))
        img_np = np.array(img)
        original_h, original_w = img_np.shape[:2]
        input_size = self.transform.apply_image(img_np).shape[:2]

        preds = []
        for tok in token_embeddings:
            text_embeds = tok.unsqueeze(0).unsqueeze(0)  # [1, 1, 256]
            sparse, dense = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=text_embeds
            )
            low_res, _ = self.sam.mask_decoder(
                image_embeddings=sam_embed,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            up = self.sam.postprocess_masks(
                low_res,
                input_size=input_size,
                original_size=(original_h, original_w),
            )[0]
            preds.append(up.squeeze(0))
        return torch.stack(preds, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Depth decoder
# ═══════════════════════════════════════════════════════════════════════════
class DepthDecoder:
    """Decode depth-token hidden states into a depth map."""

    def __init__(self, depth_ckpt: str, device: torch.device):
        self.device = device
        cfg = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        }
        self.model = DepthAnythingV2(**cfg)
        self.model.load_state_dict(
            torch.load(depth_ckpt, map_location="cpu"), strict=True
        )
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def get_features_and_gt(self, pil_image: Image.Image):
        """
        Get multi-layer patch features and teacher depth GT.

        Returns (patch_feats, depth_gt, patch_hw, img_hw)
        """
        raw = np.array(pil_image.resize((256, 256)))
        img_tensor, (h, w) = self.model.image2tensor(raw)
        img_tensor = img_tensor.to(self.device)

        patch_h = img_tensor.shape[-2] // 14
        patch_w = img_tensor.shape[-1] // 14

        feats_raw = self.model.pretrained.get_intermediate_layers(
            img_tensor, DEPTH_LAYER_IDX, return_class_token=True
        )
        patch_feats = [f[0] for f in feats_raw]

        depth_gt = self.model.depth_head(feats_raw, patch_h, patch_w)
        depth_gt = F.relu(depth_gt)
        depth_gt = F.interpolate(
            depth_gt, size=(h, w), mode="bilinear", align_corners=True
        )
        return patch_feats, depth_gt, (patch_h, patch_w), (h, w)

    @staticmethod
    def reconstruct_depth(
        tokens: torch.Tensor,
        patch_feats: list,
        patch_hw: tuple,
        img_hw: tuple,
    ) -> torch.Tensor:
        """
        Reconstruct depth map from 4 depth tokens via bilinear attention.

        Parameters
        ----------
        tokens : [4, 1024]  projected depth token vectors
        patch_feats : list of 4 tensors, each [1, N, 1024]
        patch_hw, img_hw : spatial sizes

        Returns
        -------
        depth_pred : [1, 1, H, W]
        """
        tokens = tokens.unsqueeze(0)  # [1, 4, 1024]
        B, T, C = tokens.shape
        Hf, Wf = patch_hw
        outs = []
        for i in range(T):
            tok = tokens[:, i, :].unsqueeze(1)  # [1, 1, C]
            f = patch_feats[i]  # [1, N, C]
            tok = tok.to(f.dtype)
            score = torch.bmm(tok, f.transpose(1, 2))  # [1, 1, N]
            score = score.squeeze(1).view(B, 1, Hf, Wf)
            up = F.interpolate(score, size=img_hw, mode="bilinear", align_corners=False)
            outs.append(up.squeeze(1))
        token_depths = torch.stack(outs, dim=1)  # [1, 4, H, W]
        depth_avg = token_depths.mean(1, keepdim=True)  # [1, 1, H, W]
        return depth_avg


# ═══════════════════════════════════════════════════════════════════════════
# Projection layers (loaded from the CoVT checkpoint)
# ═══════════════════════════════════════════════════════════════════════════
class VisualTokenProjector(nn.Module):
    """
    Standalone module holding the learned projection layers that map
    VLM hidden states to anchor-model embedding spaces.
    """

    def __init__(self):
        super().__init__()
        # SAM projection: hidden_dim → 256, cross-attn, learned queries
        self.sam_projection = nn.Linear(HIDDEN_DIM, SAM_EMBED_DIM)
        self.sam_query_vectors = nn.Parameter(
            torch.randn(NUM_SAM_TOKENS, SAM_EMBED_DIM, dtype=torch.bfloat16)
        )
        self.sam_cross_attention = nn.MultiheadAttention(
            embed_dim=SAM_EMBED_DIM, num_heads=8, batch_first=True
        )

        # Depth projection + token generator + cross-attn
        self.depth_projection = nn.Linear(HIDDEN_DIM, DEPTH_EMBED_DIM)
        self.depth_query_vectors = nn.Parameter(
            torch.randn(1369, DEPTH_EMBED_DIM, dtype=torch.bfloat16)
        )
        self.depth_cross_attention = nn.MultiheadAttention(
            embed_dim=DEPTH_EMBED_DIM, num_heads=8, batch_first=True
        )
        self.depth_token_generator = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, DEPTH_EMBED_DIM),
        )

    def project_sam(self, hidden_features: torch.Tensor) -> torch.Tensor:
        """
        Map VLM hidden states at SAM-token positions to SAM prompt space.

        Parameters
        ----------
        hidden_features : [B, 8, 3584]

        Returns
        -------
        sam_tokens : [B, 8, 256]  ready for SAM PromptEncoder
        """
        hidden_features = apply_rope(hidden_features)
        sam_proj = F.normalize(self.sam_projection(hidden_features))
        B = hidden_features.shape[0]
        query = self.sam_query_vectors.unsqueeze(0).expand(B, -1, -1)
        query = query.to(sam_proj.dtype)
        attn_out, _ = self.sam_cross_attention(
            query=query, key=sam_proj, value=sam_proj
        )
        return attn_out.reshape(sam_proj.shape)

    def project_depth(self, hidden_features: torch.Tensor) -> torch.Tensor:
        """
        Map VLM hidden states at depth-token positions to depth space.

        Parameters
        ----------
        hidden_features : [B, 4, 3584]

        Returns
        -------
        depth_tokens : [B, 4, 1024]  for DepthReconstructor
        """
        tokens = self.depth_token_generator(hidden_features)
        return tokens


def load_projector_from_checkpoint(
    ckpt_path: str, device: torch.device
) -> VisualTokenProjector:
    """
    Load the projection layers from a CoVT checkpoint.

    The checkpoint contains the full CoVTForConditionalGeneration state dict.
    We extract only the projection-related keys.
    """
    projector = VisualTokenProjector()

    if not os.path.isfile(ckpt_path):
        print(f"[WARN] Checkpoint not found at {ckpt_path}. Using random projections.")
        return projector.to(device).eval()

    print(f"Loading projection weights from {ckpt_path} ...")
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    prefix_map = {
        "sam_projection.": "sam_projection.",
        "sam_query_vectors": "sam_query_vectors",
        "sam_cross_attention.": "sam_cross_attention.",
        "depth_projection.": "depth_projection.",
        "depth_query_vectors": "depth_query_vectors",
        "depth_cross_attention.": "depth_cross_attention.",
        "depth_token_generator.": "depth_token_generator.",
    }

    proj_state = {}
    for full_key, value in state.items():
        for prefix in prefix_map:
            if full_key.endswith(prefix) or prefix in full_key:
                short_key = full_key.split(prefix)[-1]
                short_key = prefix + short_key if short_key else full_key
                for p in prefix_map:
                    if p in full_key:
                        start = full_key.index(p)
                        short_key = full_key[start:]
                        break
                proj_state[short_key] = value
                break

    if proj_state:
        missing, unexpected = projector.load_state_dict(proj_state, strict=False)
        print(f"  Loaded {len(proj_state)} keys.  Missing: {missing}  Unexpected: {unexpected}")
    else:
        print("  No matching projection keys found; using random weights.")

    return projector.to(device).eval()


# ═══════════════════════════════════════════════════════════════════════════
# VLM inference: run the base Qwen2.5-VL and extract hidden states
# ═══════════════════════════════════════════════════════════════════════════
def run_vlm_inference(
    model_name: str,
    pil_image: Image.Image,
    question: str,
    device: torch.device,
    anchor_ids: list[str],
):
    """
    Run inference with a CoVT/Qwen2.5-VL model and extract hidden states
    at the visual-token positions.

    Returns
    -------
    generated_text : str
    sam_hidden : [1, 8, 3584] or None
    depth_hidden : [1, 4, 3584] or None
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"Loading VLM: {model_name} ...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    # Register the special tokens used by CoVT
    add_tokens = [
        SAM_PAD_TOKEN, DINO_PAD_TOKEN, DEPTH_PAD_TOKEN,
        "<|sd_pad|>", "<|intern_pad|>", "<|pidinet_pad|>",
        "<|siglip_pad|>", "<|metaclip_pad|>",
        "<think>", "</think>", "<answer>", "</answer>",
    ]
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [ANCHOR_START_TOKEN, ANCHOR_END_TOKEN]}
    )
    processor.tokenizer.add_tokens(add_tokens)

    sam_token_id = processor.tokenizer(SAM_PAD_TOKEN, add_special_tokens=False).input_ids[0]
    depth_token_id = processor.tokenizer(DEPTH_PAD_TOKEN, add_special_tokens=False).input_ids[0]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device if device.type == "cuda" else "cpu",
    ).eval()
    model.resize_token_embeddings(len(processor.tokenizer))

    # Build the prompt that includes visual-token placeholders in the
    # assistant response (chain-of-visual-thought style).
    anchor_pad_str = ""
    for aid in anchor_ids:
        if aid == "sam":
            anchor_pad_str += ANCHOR_START_TOKEN + SAM_PAD_TOKEN * NUM_SAM_TOKENS + ANCHOR_END_TOKEN
        elif aid == "depth":
            anchor_pad_str += ANCHOR_START_TOKEN + DEPTH_PAD_TOKEN * NUM_DEPTH_TOKENS + ANCHOR_END_TOKEN

    anchor_names = []
    for aid in anchor_ids:
        if aid == "sam":
            anchor_names.append("segmentation")
        elif aid == "depth":
            anchor_names.append("depth map")

    cot_prefix = "Because "
    parts = anchor_pad_str.split(ANCHOR_END_TOKEN)
    for i, name in enumerate(anchor_names):
        pad_part = parts[i] + ANCHOR_END_TOKEN
        if i == len(anchor_names) - 1 and i > 0:
            cot_prefix += f"and the {name} of the image is {pad_part}. "
        elif i > 0:
            cot_prefix += f"the {name} of the image is {pad_part}, "
        else:
            cot_prefix += f"the {name} of the image is {pad_part}"
            if len(anchor_names) > 1:
                cot_prefix += ", "
            else:
                cot_prefix += ". "

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Inject the CoVT chain-of-thought anchor pads into the assistant prefix
    prompt = prompt + cot_prefix

    inputs = processor(
        text=[prompt], images=[pil_image.convert("RGB")], return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    last_hidden = outputs.hidden_states[-1]  # [1, seq_len, 3584]
    input_ids = inputs["input_ids"]

    # Extract hidden states at SAM token positions
    sam_hidden = None
    if "sam" in anchor_ids:
        sam_mask = input_ids == sam_token_id
        if sam_mask.any():
            sam_hidden = last_hidden[sam_mask].unsqueeze(0)
            print(f"  SAM hidden states shape: {sam_hidden.shape}")

    depth_hidden = None
    if "depth" in anchor_ids:
        depth_mask = input_ids == depth_token_id
        if depth_mask.any():
            depth_hidden = last_hidden[depth_mask].unsqueeze(0)
            print(f"  Depth hidden states shape: {depth_hidden.shape}")

    # Also do generation to get the textual answer
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_text = processor.decode(gen_ids[0, input_len:], skip_special_tokens=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return generated_text, sam_hidden, depth_hidden


# ═══════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════════════
MASK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]


def visualize_segmentation_masks(
    pil_image: Image.Image,
    masks: torch.Tensor,
    output_dir: str,
):
    """
    Save individual masks and a combined overlay.

    Parameters
    ----------
    masks : [n_tokens, H, W]  raw logits from SAM decoder
    """
    os.makedirs(output_dir, exist_ok=True)
    img = pil_image.resize((256, 256))
    img_np = np.array(img)

    binary_masks = (masks > 0).float().cpu().numpy()

    # Individual masks
    for i, m in enumerate(binary_masks):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(img_np)
        color = np.array(MASK_COLORS[i % len(MASK_COLORS)]) / 255.0
        mask_rgba = np.zeros((*m.shape, 4))
        mask_rgba[..., :3] = color
        mask_rgba[..., 3] = m * 0.55
        axes[1].imshow(mask_rgba)
        axes[1].set_title(f"SAM Token {i} Mask")
        axes[1].axis("off")

        plt.tight_layout()
        path = os.path.join(output_dir, f"seg_mask_token_{i}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")

    # Combined overlay
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_np)
    for i, m in enumerate(binary_masks):
        color = np.array(MASK_COLORS[i % len(MASK_COLORS)]) / 255.0
        mask_rgba = np.zeros((*m.shape, 4))
        mask_rgba[..., :3] = color
        mask_rgba[..., 3] = m * 0.45
        ax.imshow(mask_rgba)
    ax.set_title("All SAM Token Masks (overlay)")
    ax.axis("off")
    path = os.path.join(output_dir, "seg_masks_combined.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    return binary_masks


def visualize_depth_map(
    pil_image: Image.Image,
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor | None,
    per_token_depths: torch.Tensor | None,
    output_dir: str,
):
    """
    Save depth visualizations.

    Parameters
    ----------
    depth_pred : [1, 1, H, W]  predicted depth (averaged over tokens)
    depth_gt   : [1, 1, H, W]  teacher depth from DepthAnything (optional)
    per_token_depths : [1, 4, H, W]  individual token depth maps (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    img = pil_image.resize((256, 256))
    img_np = np.array(img)

    pred_np = depth_pred.squeeze().cpu().numpy()
    pred_norm = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)

    # Main depth map
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(pred_norm, cmap="inferno")
    axes[1].set_title("Predicted Depth (averaged)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(output_dir, "depth_predicted.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # Teacher GT comparison (if available)
    if depth_gt is not None:
        gt_np = depth_gt.squeeze().cpu().numpy()
        gt_norm = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(gt_norm, cmap="inferno")
        axes[1].set_title("Teacher Depth (DepthAnything V2)")
        axes[1].axis("off")

        axes[2].imshow(pred_norm, cmap="inferno")
        axes[2].set_title("CoVT Predicted Depth")
        axes[2].axis("off")

        plt.tight_layout()
        path = os.path.join(output_dir, "depth_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")

    # Per-token depth maps
    if per_token_depths is not None:
        n_tokens = per_token_depths.shape[1]
        fig, axes = plt.subplots(1, n_tokens + 1, figsize=(4 * (n_tokens + 1), 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Input")
        axes[0].axis("off")
        for i in range(n_tokens):
            d = per_token_depths[0, i].cpu().numpy()
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
            axes[i + 1].imshow(d_norm, cmap="inferno")
            axes[i + 1].set_title(f"Depth Token {i}")
            axes[i + 1].axis("off")
        plt.tight_layout()
        path = os.path.join(output_dir, "depth_per_token.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")


def visualize_combined_panel(
    pil_image: Image.Image,
    seg_masks: np.ndarray | None,
    depth_pred: torch.Tensor | None,
    depth_gt: torch.Tensor | None,
    output_dir: str,
):
    """Create a combined panel with all visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    img = pil_image.resize((256, 256))
    img_np = np.array(img)

    n_cols = 1  # input
    if seg_masks is not None:
        n_cols += 1
    if depth_pred is not None:
        n_cols += 1
    if depth_gt is not None:
        n_cols += 1

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    col = 0
    axes[col].imshow(img_np)
    axes[col].set_title("Input Image", fontsize=14)
    axes[col].axis("off")
    col += 1

    if seg_masks is not None:
        axes[col].imshow(img_np)
        for i, m in enumerate(seg_masks):
            color = np.array(MASK_COLORS[i % len(MASK_COLORS)]) / 255.0
            rgba = np.zeros((*m.shape, 4))
            rgba[..., :3] = color
            rgba[..., 3] = m * 0.5
            axes[col].imshow(rgba)
        axes[col].set_title("Segmentation Masks", fontsize=14)
        axes[col].axis("off")
        col += 1

    if depth_gt is not None:
        gt = depth_gt.squeeze().cpu().numpy()
        gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        axes[col].imshow(gt, cmap="inferno")
        axes[col].set_title("Teacher Depth (GT)", fontsize=14)
        axes[col].axis("off")
        col += 1

    if depth_pred is not None:
        dp = depth_pred.squeeze().cpu().numpy()
        dp = (dp - dp.min()) / (dp.max() - dp.min() + 1e-8)
        axes[col].imshow(dp, cmap="inferno")
        axes[col].set_title("CoVT Depth Prediction", fontsize=14)
        axes[col].axis("off")
        col += 1

    plt.suptitle("CoVT Visual Token Decoding", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "combined_panel.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Decode and visualize CoVT segmentation & depth visual tokens."
    )
    parser.add_argument(
        "--image_path", type=str, default="assets/clouds.png",
        help="Path to input image.",
    )
    parser.add_argument(
        "--model_name", type=str, default="Wakals/CoVT-7B-seg_depth_dino",
        help="HuggingFace model name or local path for the CoVT checkpoint.",
    )
    parser.add_argument(
        "--projector_ckpt", type=str, default=None,
        help="Path to a full CoVT checkpoint to load projection layer weights. "
             "If not given, the script attempts to extract them from the HF model.",
    )
    parser.add_argument(
        "--sam_ckpt", type=str,
        default="train/src/anchors/segment_anything/ckpt/sam_vit_h_4b8939.pth",
        help="Path to SAM ViT-H checkpoint.",
    )
    parser.add_argument(
        "--depth_ckpt", type=str,
        default="train/src/anchors/DepthAnything/ckpt/depth_anything_v2_vitl.pth",
        help="Path to DepthAnything V2 Large checkpoint.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/visual_tokens",
        help="Directory to save visualizations.",
    )
    parser.add_argument(
        "--question", type=str,
        default="Describe the scene in the picture in detail, and find out how "
                "many clouds are in the sky. Use segmentation, depth map, and "
                "perception feature information of the image to answer this question.",
        help="Question to ask the model.",
    )
    parser.add_argument(
        "--anchors", type=str, nargs="+", default=["sam", "depth"],
        choices=["sam", "depth"],
        help="Which visual-token types to decode.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--use_random_hidden", action="store_true",
        help="Use random hidden states instead of running VLM inference "
             "(useful for testing the decoding pipeline without GPU).",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    pil_image = Image.open(args.image_path).convert("RGB")
    print(f"Loaded image: {args.image_path}  ({pil_image.size})")

    # ------------------------------------------------------------------
    # Step 1: Get hidden states at visual-token positions
    # ------------------------------------------------------------------
    sam_hidden = None
    depth_hidden = None
    generated_text = ""

    if args.use_random_hidden:
        print("\n[Using random hidden states for pipeline testing]")
        if "sam" in args.anchors:
            sam_hidden = torch.randn(1, NUM_SAM_TOKENS, HIDDEN_DIM, device=device, dtype=torch.bfloat16)
        if "depth" in args.anchors:
            depth_hidden = torch.randn(1, NUM_DEPTH_TOKENS, HIDDEN_DIM, device=device, dtype=torch.bfloat16)
        generated_text = "(random hidden states – no VLM inference)"
    else:
        generated_text, sam_hidden, depth_hidden = run_vlm_inference(
            args.model_name, pil_image, args.question, device, args.anchors,
        )
    print(f"\nGenerated text:\n  {generated_text}\n")

    # ------------------------------------------------------------------
    # Step 2: Load projection layers
    # ------------------------------------------------------------------
    if args.projector_ckpt:
        projector = load_projector_from_checkpoint(args.projector_ckpt, device)
    else:
        print("No projector checkpoint specified; using randomly-initialized projections.")
        projector = VisualTokenProjector().to(device).eval()

    # ------------------------------------------------------------------
    # Step 3: Decode segmentation tokens
    # ------------------------------------------------------------------
    seg_binary = None
    if "sam" in args.anchors and sam_hidden is not None:
        print("--- Decoding segmentation tokens ---")
        if not os.path.isfile(args.sam_ckpt):
            print(f"  [WARN] SAM checkpoint not found: {args.sam_ckpt}")
            print("  Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        else:
            seg_decoder = SegmentationDecoder(args.sam_ckpt, device)
            with torch.no_grad():
                sam_tokens = projector.project_sam(sam_hidden.to(torch.float32))
                sam_embed = seg_decoder.encode_image(pil_image)
                masks = seg_decoder.decode_tokens(
                    sam_embed, pil_image, sam_tokens[0]
                )
            print(f"  Decoded {masks.shape[0]} segmentation masks, shape {masks.shape}")
            seg_binary = visualize_segmentation_masks(pil_image, masks, args.output_dir)

    # ------------------------------------------------------------------
    # Step 4: Decode depth tokens
    # ------------------------------------------------------------------
    depth_pred = None
    depth_gt = None
    if "depth" in args.anchors and depth_hidden is not None:
        print("\n--- Decoding depth tokens ---")
        if not os.path.isfile(args.depth_ckpt):
            print(f"  [WARN] Depth checkpoint not found: {args.depth_ckpt}")
            print("  Download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Large")
        else:
            depth_decoder = DepthDecoder(args.depth_ckpt, device)
            patch_feats, depth_gt, patch_hw, img_hw = depth_decoder.get_features_and_gt(pil_image)
            with torch.no_grad():
                depth_tokens = projector.project_depth(depth_hidden.to(torch.float32))
                depth_tokens_squeezed = depth_tokens[0]  # [4, 1024]

                # Reconstruct per-token depth maps and averaged depth
                tokens_unsqueezed = depth_tokens_squeezed.unsqueeze(0)
                B, T, C = tokens_unsqueezed.shape
                Hf, Wf = patch_hw
                per_token_list = []
                for i in range(T):
                    tok = tokens_unsqueezed[:, i, :].unsqueeze(1)
                    f = patch_feats[i]
                    tok = tok.to(f.dtype)
                    score = torch.bmm(tok, f.transpose(1, 2))
                    score = score.squeeze(1).view(B, 1, Hf, Wf)
                    up = F.interpolate(score, size=img_hw, mode="bilinear", align_corners=False)
                    per_token_list.append(up.squeeze(1))
                per_token_depths = torch.stack(per_token_list, dim=1)
                depth_pred = per_token_depths.mean(1, keepdim=True)

            print(f"  Depth prediction shape: {depth_pred.shape}")
            print(f"  Teacher depth shape: {depth_gt.shape}")
            visualize_depth_map(
                pil_image, depth_pred, depth_gt, per_token_depths, args.output_dir
            )

    # ------------------------------------------------------------------
    # Step 5: Combined panel
    # ------------------------------------------------------------------
    visualize_combined_panel(pil_image, seg_binary, depth_pred, depth_gt, args.output_dir)

    # Save the generated text
    txt_path = os.path.join(args.output_dir, "generated_text.txt")
    with open(txt_path, "w") as f:
        f.write(generated_text)
    print(f"\n  Saved generated text to {txt_path}")

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
