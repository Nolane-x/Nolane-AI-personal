from __future__ import annotations

from dataclasses import asdict,dataclass
from typing import Any

from .surgery import module_parameter_digest


@dataclass(slots=True)
class FactorizedTrainingConfig:
    epochs:int=4
    learning_rate:float=2e-3
    distill_weight:float=1.0
    distill_temperature:float=2.0
    max_grad_norm:float=1.0

    def validate(self):
        if self.epochs<1: raise ValueError("epochs must be >=1")
        if self.learning_rate<=0: raise ValueError("learning_rate must be positive")
        if self.distill_weight<0: raise ValueError("distill_weight must be non-negative")
        if self.distill_temperature<=0: raise ValueError("distill_temperature must be positive")
        if self.max_grad_norm<=0: raise ValueError("max_grad_norm must be positive")


@dataclass(slots=True)
class FactorizedTrainingReceipt:
    schema:str
    authority:str
    examples:int
    epochs:int
    optimizer_steps:int
    task_initial_loss:float
    task_final_loss:float
    distill_initial_loss:float
    distill_final_loss:float
    boundary_digest_before:str
    boundary_digest_after:str
    boundary_changed:bool
    cortex_digest_before:str
    cortex_digest_after:str
    cortex_unchanged:bool
    cortex_gradients_seen:int
    boundary_gradients_seen:int
    config:dict[str,Any]
    def to_dict(self): return asdict(self)


def _nll(model,examples):
    torch=model.cortex.torch
    device=next(model.boundary.module.parameters()).device
    vals=[]
    model.eval()
    with torch.no_grad():
        for ids,labels,latent,weight in examples:
            x=torch.tensor([ids],dtype=torch.long,device=device)
            y=torch.tensor([labels],dtype=torch.long,device=device)
            if latent is not None: model.set_latent(latent)
            vals.append(float(model.forward(input_ids=x,labels=y,state=None).loss.detach().cpu())*float(weight))
    return sum(vals)/max(1,len(vals))


def _kl(student,teacher,temp):
    torch=__import__("torch"); t=float(temp)
    s=torch.nn.functional.log_softmax(student.float()/t,dim=-1)
    q=torch.nn.functional.softmax(teacher.float()/t,dim=-1)
    return torch.nn.functional.kl_div(s,q,reduction="batchmean")*(t*t)


def train_factorized_boundary(student,teacher,examples,*,config=None):
    config=config or FactorizedTrainingConfig(); config.validate()
    if not examples: raise ValueError("factorized-boundary training examples are empty")
    torch=student.cortex.torch
    device=next(student.boundary.module.parameters()).device
    for p in student.cortex.module.parameters(): p.requires_grad_(False)
    for p in student.boundary.module.parameters(): p.requires_grad_(True)
    # Preserve exact source norm; train only low-rank factors.
    for p in student.boundary.final_norm.parameters(): p.requires_grad_(False)

    before_b=module_parameter_digest(student.boundary.module)
    before_c=module_parameter_digest(student.cortex.module)
    initial=_nll(student,examples)
    params=[p for p in student.boundary.module.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=config.learning_rate,weight_decay=0.0)
    steps=0; bg=0; cg=0; first_kl=None; last=[]
    teacher.eval(); student.eval()
    for _ in range(config.epochs):
        epoch=[]
        for ids,labels,latent,weight in examples:
            x=torch.tensor([ids],dtype=torch.long,device=device)
            y=torch.tensor([labels],dtype=torch.long,device=device)
            if latent is not None:
                student.set_latent(latent); teacher.set_latent(latent)
            with torch.no_grad():
                tlog=teacher.forward(input_ids=x,state=None).logits.detach()
            opt.zero_grad(set_to_none=True)
            out=student.forward(input_ids=x,labels=y,state=None)
            kl=_kl(out.logits,tlog,config.distill_temperature)
            if first_kl is None: first_kl=float(kl.detach().cpu())
            loss=out.loss*float(weight)+config.distill_weight*kl
            loss.backward()
            cg += sum(1 for p in student.cortex.module.parameters() if p.grad is not None)
            if cg: raise RuntimeError("cortex received gradients during boundary distillation")
            bg += sum(1 for p in params if p.grad is not None and torch.isfinite(p.grad).all())
            torch.nn.utils.clip_grad_norm_(params,config.max_grad_norm)
            opt.step(); steps+=1; epoch.append(float(kl.detach().cpu()))
        last=epoch
    final=_nll(student,examples)
    after_b=module_parameter_digest(student.boundary.module)
    after_c=module_parameter_digest(student.cortex.module)
    return FactorizedTrainingReceipt(
        schema="NOLANE-L17-FACTORIZED-BOUNDARY-TRAINING-V1",
        authority="TRAINED_FACTORIZED_BOUNDARY_CANDIDATE_ONLY",
        examples=len(examples),epochs=config.epochs,optimizer_steps=steps,
        task_initial_loss=initial,task_final_loss=final,
        distill_initial_loss=float(first_kl or 0.0),
        distill_final_loss=sum(last)/max(1,len(last)),
        boundary_digest_before=before_b,boundary_digest_after=after_b,boundary_changed=before_b!=after_b,
        cortex_digest_before=before_c,cortex_digest_after=after_c,cortex_unchanged=before_c==after_c,
        cortex_gradients_seen=cg,boundary_gradients_seen=bg,config=asdict(config),
    )
