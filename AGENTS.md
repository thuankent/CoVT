# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

CoVT (Chain-of-Visual-Thought) is a research project with two Python sub-projects and a demo:

- **`VLMEvalKit/`** — Forked VLMEvalKit for evaluation. Installed via `pip install -e .` from that directory.
- **`train/`** — Training pipeline. Dependencies from `train/requirements.txt`. Uses `PYTHONPATH=src` when running scripts (see `train/scripts/run.sh`).
- **`gradio/`** — Standalone Gradio demo (`python3 gradio/gradio_demo.py`, serves on port 7860). Model loads lazily on first inference request.

### Key Dependency Notes

- `transformers==4.50.1` is required — the training code uses `ALL_LAYERNORM_LAYERS` from `transformers.trainer` (removed in newer versions), and VLMEvalKit uses `AutoModelForVision2Seq` (removed in v5+).
- `liger-kernel==0.5.5` is required to match `transformers==4.50.1`.
- The strict pinned versions in `train/requirements.txt` include conda-only packages (`mkl-fft`, `mkl-service`, `mkl-random`) that cannot be pip-installed. Install key packages individually instead of using `pip install -r train/requirements.txt`.
- GPU-specific packages (`flash-attn`, `xformers`) require CUDA and won't install in CPU-only environments.

### Linting

- VLMEvalKit uses flake8 + yapf (see `VLMEvalKit/.pre-commit-config.yaml`).
- Lint command: `flake8 --max-line-length=120 --ignore=F401,F403,F405,E402,E722,E741,W503,E231,E702`.
- Pre-existing lint warnings exist in the repo code; `vlmeval/config.py` is excluded in their pre-commit config.

### Testing

- No formal automated test suite exists. This is a research project.
- The `VLMEvalKit/vlmeval/dataset/olmOCRBench/tests.py` is a utility module, not a pytest test.
- Validation is done by running evaluation benchmarks or the Gradio demo.

### Running Services

- **Gradio demo**: `cd gradio && python3 gradio_demo.py` — starts on port 7860. Requires GPU for actual inference, but the UI loads without one.
- **Training**: Requires GPU (4-8x A6000 recommended). Run from `train/` with `bash scripts/finetune.sh`.
- **Evaluation**: Requires GPU. Run from `VLMEvalKit/` with `python3 run.py --data <benchmark> --model <model> --verbose`.

### Environment

- VLMEvalKit expects a `.env` file at `VLMEvalKit/.env` (can be empty if no API keys needed).
- 4 CoVT models are registered: `CoVT-7B-seg`, `CoVT-7B-depth`, `CoVT-7B-seg_depth_dino`, `CoVT-7B-seg_depth_dino_edge`.
