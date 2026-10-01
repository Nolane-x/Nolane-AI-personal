# Local models

Model weights live here at runtime but are deliberately excluded from Git.

Run:

```bash
python -m pip install -r requirements-model.txt
python scripts/download_model.py
```

This downloads the checkpoint pinned in `model.lock.json` to `models/Qwen3-0.6B/`.

Do not commit `.safetensors` weights to normal Git history. Keeping the upstream checkpoint reproducible and external makes architecture experiments, pruning, distillation, and model surgery much easier to manage.
