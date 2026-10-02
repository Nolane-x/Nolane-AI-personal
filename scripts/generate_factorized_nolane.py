from __future__ import annotations

import argparse
from pathlib import Path

from nolane_personal.factorized_artifact import load_factorized_model
from nolane_personal.latent import LatentStore
from nolane_personal.qwen import SYSTEM_PROMPT

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--factorized",default="runtime-data/l17-factorized-trained/factorized-nolane.pt")
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json")
    p.add_argument("--tokenizer",default=str(ROOT/"models/Qwen3-0.6B"))
    p.add_argument("--device",default="cpu")
    p.add_argument("--prompt",required=True)
    p.add_argument("--max-new-tokens",type=int,default=128)
    a=p.parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install tokenizer support with: pip install -e '.[qwen]'") from exc
    latent=LatentStore(a.latent).load()
    if latent is None: raise SystemExit(f"persistent latent not found: {a.latent}")
    model,meta=load_factorized_model(a.factorized,latent.values,device=a.device)
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    msgs=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":a.prompt}]
    try:r=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:r=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=tok(r,return_tensors="pt")
    ids=inp["input_ids"].to(a.device)
    out=model.generate(input_ids=ids,max_new_tokens=a.max_new_tokens,do_sample=False,eos_token_id=tok.eos_token_id)
    print(f"[UNPROMOTED L17 FACTORIZED BOUNDARY | {meta['checkpoint_sha256'][:12]}]")
    print(tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip())
    return 0
if __name__=="__main__": raise SystemExit(main())
