from __future__ import annotations
import argparse
from pathlib import Path
from nolane_personal.anchor_artifact import build_anchor_model
from nolane_personal.latent import LatentStore
from nolane_personal.qwen import QwenCortex,SYSTEM_PROMPT
from nolane_personal.surgery_candidate import load_model_lock,model_lock_fingerprint
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-lock",default=str(ROOT/"model.lock.json")); p.add_argument("--model",default=str(ROOT/"models/Qwen3-0.6B")); p.add_argument("--anchor-cortex",default="runtime-data/l14-minimal-anchor/minimal-qwen-anchor-cortex.pt"); p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json"); p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"]); p.add_argument("--prompt",required=True); p.add_argument("--max-new-tokens",type=int,default=128); a=p.parse_args()
    latent=LatentStore(a.latent).load(); fp=model_lock_fingerprint(load_model_lock(a.model_lock)); q=QwenCortex(a.model,device=a.device); model,meta=build_anchor_model(q.model,a.anchor_cortex,latent.values,expected_base_model_fingerprint=fp)
    msgs=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":a.prompt}]
    try:r=q.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:r=q.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=q.tokenizer(r,return_tensors="pt").to(q.device); pad=q.tokenizer.eos_token_id or 0
    out=model.generate_cached(**dict(inp),max_new_tokens=a.max_new_tokens,do_sample=False,pad_token_id=pad)
    print(f"[UNPROMOTED L14 MINIMAL ANCHOR | {meta['checkpoint_sha256'][:12]}]"); print(q.tokenizer.decode(out[0,inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()); return 0
if __name__=="__main__": raise SystemExit(main())
