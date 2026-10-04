# Local models

Model weights live here at runtime but are deliberately excluded from Git.

Run:

```bash
python -m pip install -r requirements-model.txt
python scripts/download_model.py
```

This downloads the research-compatibility checkpoint pinned in `model.lock.json` to `models/Qwen3-1.7B/`. The Windows production product does not use that checkpoint; it uses the hash-pinned Qwen3.5-2B GGUF declared in `config/windows-software-runtime-lock.json`. Historical Qwen3-0.6B courts use the separate `config/legacy-qwen3-0.6b-model.lock.json` lock and are not part of any active product runtime.

Do not commit `.safetensors` weights to normal Git history. Keeping the upstream checkpoint reproducible and external makes architecture experiments, pruning, distillation, and model surgery much easier to manage.
