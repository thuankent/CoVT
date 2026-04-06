# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

CoVT (Chain-of-Visual-Thought) is a research project that enhances Vision-Language Models with continuous visual tokens for vision-centric reasoning. It has three components:

- **Training** (`train/`) – Multi-stage LoRA fine-tuning of Qwen2.5-VL with DeepSpeed and vision anchor models (SAM, DepthAnything, DINOv2, PIDINet).
- **Evaluation** (`VLMEvalKit/`) – Forked VLMEvalKit with CoVT model definitions.
- **Gradio Demo** (`gradio/`) – Interactive web UI for inference.

### Key technical notes

- **No GPU in Cloud Agent VMs.** All three components require GPU for meaningful inference/training. Use `--use_random_hidden` flag on `visualize_tokens.py` for CPU-only pipeline testing.
- **`flash-attn` and `xformers` cannot be built without GPU.** Skip them when installing on CPU-only environments. The models fall back to standard attention automatically.
- The VLMEvalKit is installed as an editable package (`pip install -e .` in `VLMEvalKit/`).
- Anchor model checkpoints (SAM ViT-H ~2.4GB, DepthAnything V2 Large ~1.3GB) are required for the visualization script and training but are **not** committed to the repo. Download instructions are in `docs/Train.md`.

### Running lint

```bash
flake8 --max-line-length=120 --ignore=F401,F403,F405,E402,E722,E741,W503,E231,E702 train/ gradio/ visualize_tokens.py
```

VLMEvalKit has its own `.pre-commit-config.yaml` with flake8 + yapf.

### Running the visualization script

```bash
# Full pipeline (requires GPU + model weights + anchor checkpoints):
python visualize_tokens.py --image_path assets/clouds.png

# CPU-only pipeline test (random hidden states, needs anchor checkpoints):
python visualize_tokens.py --use_random_hidden --device cpu

# CPU-only without anchor checkpoints (tests code paths, skips decode):
python visualize_tokens.py --use_random_hidden --device cpu  # warns about missing ckpts
```

### Running the Gradio demo

```bash
cd gradio && python gradio_demo.py  # requires GPU
```

### Dependency installation

See `train/install_training.sh` and `VLMEvalKit/install_eval.sh` for the canonical install scripts. On CPU-only environments, install PyTorch CPU variant first, then the requirements.
