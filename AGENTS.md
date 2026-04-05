# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

CoVT (Chain-of-Visual-Thought) is an ML research project with three components:

| Component | Location | Purpose |
|---|---|---|
| Training | `train/` | Fine-tune VLMs with LoRA using visual expert anchors |
| Evaluation | `VLMEvalKit/` | Benchmark CoVT models via forked VLMEvalKit |
| Gradio Demo | `gradio/` | Interactive web UI for CoVT model inference |

### Python environment

- A virtualenv lives at `/workspace/.venv`. Activate with `source /workspace/.venv/bin/activate`.
- `transformers==4.50.1` is pinned — newer versions break VLMEvalKit imports (e.g. `AutoModelForVision2Seq` removed).
- `torch` is installed CPU-only on Cloud VMs (no GPU). Model inference, training, and evaluation all require NVIDIA GPU and will error at runtime on CPU-only environments.

### Running services

- **Gradio demo**: `source /workspace/.venv/bin/activate && cd /workspace/gradio && python gradio_demo.py` — launches on port 7860. Will load the UI but inference fails without GPU.
- **Evaluation**: `source /workspace/.venv/bin/activate && cd /workspace/VLMEvalKit && python run.py --help` — see `docs/Eval.md` for full usage.
- **Training**: Requires GPU, model checkpoints, and dataset. See `docs/Train.md`.

### Linting

- VLMEvalKit uses flake8 (v6.1.0) with pre-commit config at `VLMEvalKit/.pre-commit-config.yaml`. Due to Python 3.12 incompatibilities with invalid escape sequences in VLMEvalKit code, flake8 crashes when run on the full `vlmeval/` directory. Run on individual files or on the main CoVT code (`train/src/training/`, `gradio/`).
- Lint CoVT code: `flake8 --max-line-length=120 --ignore=F401,F403,F405,E402,E722,E741,W503,E231,E702 train/src/training/ gradio/`

### Key gotchas

- VLMEvalKit is installed in editable mode (`pip install -e .` from `VLMEvalKit/`). The `.env` file at `VLMEvalKit/.env` is optional (API keys for LLM-as-judge evaluation).
- Training code uses `PYTHONPATH=/workspace/train/src:/workspace/train` for imports (set automatically in `train/scripts/run.sh`).
- `flash-attn` and `xformers` are CUDA-only and are not installed on CPU environments; their absence produces harmless warnings.
- No automated test suite exists in this repository.
