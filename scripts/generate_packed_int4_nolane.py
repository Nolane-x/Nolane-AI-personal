from __future__ import annotations

import argparse
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.packed_int4_artifact import load_packed_int4_model
from nolane_personal.qwen import SYSTEM_PROMPT

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--int4",default="runtime-data/l20-packed-int4/packed-int4-nolane.pt")
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json")
    p.add_argument("--tokenizer",default=str(ROOT/"models/Qwen3-0.6B"))
    p.add_argument("--device",default="cpu")
    p.add_argument("--prompt",required=True)
    p.add_argument("--max-new-tokens",type=int,default=128)
    a=p.parse_args()
    from transformers import AutoTokenizer
    latent=LatentStore(a.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {a.latent}")
    model,meta=load_packed_int4_model(a.int4,latent.values,device=a.device)
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":a.prompt}]
    try:
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ids=tok(rendered,return_tensors="pt")["input_ids"].to(a.device)
    out=model.generate(
        input_ids=ids,max_new_tokens=a.max_new_tokens,do_sample=False,eos_token_id=tok.eos_token_id
    )
    print(f"[UNPROMOTED L20 PACKED INT4 | {meta['checkpoint_sha256'][:12]}]")
    print(tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip())
    return 0


if __name__=="__main__":
    raise SystemExit(main())
