from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path

from nolane_personal.factorized_artifact import load_factorized_model,save_factorized_artifact
from nolane_personal.factorized_training import FactorizedTrainingConfig,train_factorized_boundary
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example,load_jsonl
from nolane_personal.personal_protocol import examples_for_split,load_protocol,verify_personalization_protocol
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.standalone_artifact import load_standalone_model

def sha256_file(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): d.update(c)
    return d.hexdigest()

def encode(tok,examples,latent,max_length):
    rows=[]
    for e in examples:
        ids,labels=encode_chat_example(tok,e,system_prompt=SYSTEM_PROMPT,max_length=max_length)
        rows.append((ids,labels,e.latent or latent,e.weight))
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--l16",default="runtime-data/l16-standalone/standalone-nolane.pt")
    p.add_argument("--factorized",default="runtime-data/l17-factorized-boundary/factorized-nolane.pt")
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json")
    p.add_argument("--dataset",default="runtime-data/personalization.jsonl")
    p.add_argument("--protocol",default="runtime-data/personalization-protocol-v1.json")
    p.add_argument("--tokenizer",default="models/Qwen3-0.6B")
    p.add_argument("--output-dir",default="runtime-data/l17-factorized-trained")
    p.add_argument("--device",default="cpu")
    p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--learning-rate",type=float,default=2e-3)
    p.add_argument("--distill-weight",type=float,default=1.0)
    p.add_argument("--temperature",type=float,default=2.0)
    p.add_argument("--max-length",type=int,default=384)
    a=p.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install tokenizer support with: pip install -e '.[qwen]'") from exc

    dp=Path(a.dataset); examples=load_jsonl(dp); protocol=load_protocol(a.protocol)
    verify_personalization_protocol(protocol,dataset_sha256=sha256_file(dp))
    train_examples=examples_for_split(examples,protocol,"train")
    latent=LatentStore(a.latent).load()
    if latent is None: raise SystemExit(f"persistent latent not found: {a.latent}")
    teacher,tmeta=load_standalone_model(a.l16,latent.values,device=a.device,expected_dataset_fingerprint=protocol["protocol_sha256"])
    student,smeta=load_factorized_model(a.factorized,latent.values,device=a.device,expected_source_l16_checkpoint_sha256=tmeta["checkpoint_sha256"],expected_dataset_fingerprint=protocol["protocol_sha256"])
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    encoded=encode(tok,train_examples,latent.values,a.max_length)
    receipt=train_factorized_boundary(student,teacher,encoded,config=FactorizedTrainingConfig(
        epochs=a.epochs,learning_rate=a.learning_rate,distill_weight=a.distill_weight,distill_temperature=a.temperature,
    ))
    manifest=save_factorized_artifact(
        a.output_dir,student,smeta["factorization_receipt"],source_meta=tmeta,
        dataset_fingerprint=protocol["protocol_sha256"],training_receipt=receipt.to_dict(),
    )
    print(json.dumps({"training":receipt.to_dict(),"artifact":manifest},indent=2,sort_keys=True))
    return 0 if receipt.cortex_unchanged and receipt.cortex_gradients_seen==0 else 2
if __name__=="__main__": raise SystemExit(main())
