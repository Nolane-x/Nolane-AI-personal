from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class FactorizedBoundaryConfig:
    vocab_size: int
    hidden_size: int
    rank: int = 128
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    padding_idx: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    boundary_dtype: str = "float32"

    def validate(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        if not 1 <= self.rank <= min(self.vocab_size, self.hidden_size):
            raise ValueError("rank must be in [1,min(vocab_size,hidden_size)]")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.boundary_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported boundary_dtype")


def analytical_factorized_boundary_parameters(config: FactorizedBoundaryConfig) -> int:
    config.validate()
    v,h,r=config.vocab_size,config.hidden_size,config.rank
    input_params=v*r+r*h
    norm_params=h
    output_params=0 if config.tie_word_embeddings else v*r+r*h
    return input_params+norm_params+output_params


def dense_boundary_parameters(config: FactorizedBoundaryConfig) -> int:
    v,h=config.vocab_size,config.hidden_size
    return v*h+h+(0 if config.tie_word_embeddings else v*h)


def _dtype(torch, name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


class FactorizedBoundaryModule:
    """Low-rank language boundary independent of Transformers/Qwen runtime."""

    def __init__(self, config: FactorizedBoundaryConfig, *, dtype=None, device=None):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc
        config.validate()
        self.torch=torch
        self.nn=nn
        self.config=config
        if dtype is None:
            dtype=_dtype(torch,config.boundary_dtype)
        factory={}
        if dtype is not None: factory["dtype"]=dtype
        if device is not None: factory["device"]=device

        class FactorizedEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.codes=nn.Embedding(config.vocab_size,config.rank,padding_idx=config.padding_idx,**factory)
                self.basis=nn.Parameter(torch.empty(config.rank,config.hidden_size,**factory))
                nn.init.normal_(self.basis,mean=0.0,std=0.02)
            def forward(self,input_ids):
                return self.codes(input_ids) @ self.basis

        class RMSNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=nn.Parameter(torch.ones(config.hidden_size,**factory))
                self.variance_epsilon=float(config.rms_norm_eps)
            def forward(self,x):
                dt=x.dtype
                y=x.float()
                y=y*torch.rsqrt(y.pow(2).mean(-1,keepdim=True)+self.variance_epsilon)
                return self.weight*y.to(dt)

        self.embed_tokens=FactorizedEmbedding()
        self.final_norm=RMSNorm()
        self.output_codes=None
        self.output_basis=None
        if not config.tie_word_embeddings:
            self.output_codes=nn.Parameter(torch.empty(config.vocab_size,config.rank,**factory))
            self.output_basis=nn.Parameter(torch.empty(config.rank,config.hidden_size,**factory))
            nn.init.normal_(self.output_codes,mean=0.0,std=0.02)
            nn.init.normal_(self.output_basis,mean=0.0,std=0.02)

        class Module(nn.Module):
            def __init__(self,outer):
                super().__init__()
                self.embed_tokens=outer.embed_tokens
                self.final_norm=outer.final_norm
                if outer.output_codes is not None:
                    self.output_codes=outer.output_codes
                    self.output_basis=outer.output_basis
        self.module=Module(self)

    def to(self,device:str):
        self.module.to(device); return self
    def eval(self):
        self.module.eval(); return self
    def train(self):
        self.module.train(); return self
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.module.parameters())
    def state_dict(self):
        return self.module.state_dict()
    def load_state_dict(self,state):
        return self.module.load_state_dict(state)

    def reconstructed_input_weight(self):
        return self.embed_tokens.codes.weight @ self.embed_tokens.basis

    def reconstructed_output_weight(self):
        if self.config.tie_word_embeddings:
            return self.reconstructed_input_weight()
        return self.output_codes @ self.output_basis

    def logits(self,hidden_states):
        torch=self.torch
        normalized=self.final_norm(hidden_states)
        if self.config.tie_word_embeddings:
            rank_hidden=normalized @ self.embed_tokens.basis.t()
            return (rank_hidden @ self.embed_tokens.codes.weight.t()).float()
        rank_hidden=normalized @ self.output_basis.t()
        return (rank_hidden @ self.output_codes.t()).float()


def _truncated_svd(weight, rank:int, *, seed:int=0, oversample:int=8):
    torch=__import__("torch")
    matrix=weight.detach().float().cpu()
    q=min(min(matrix.shape),int(rank)+max(2,int(oversample)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if min(matrix.shape) <= 256 or matrix.shape[0] <= 4096:
            u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
            u=u[:,:rank]; s=s[:rank]; vh=vh[:rank,:]
        else:
            u,s,v=torch.pca_lowrank(matrix,q=q,center=False,niter=4)
            u=u[:,:rank]; s=s[:rank]; vh=v[:,:rank].t()
    codes=u*s.unsqueeze(0)
    basis=vh
    recon=codes@basis
    denom=matrix.norm().clamp_min(1e-12)
    rel=float(((matrix-recon).norm()/denom).cpu())
    return codes,basis,rel


def factorize_standalone_boundary(source_boundary, *, rank:int, seed:int=0):
    torch=__import__("torch")
    source=source_boundary.config
    config=FactorizedBoundaryConfig(
        vocab_size=int(source.vocab_size),
        hidden_size=int(source.hidden_size),
        rank=int(rank),
        rms_norm_eps=float(source.rms_norm_eps),
        tie_word_embeddings=bool(source.tie_word_embeddings),
        padding_idx=source.padding_idx,
        bos_token_id=source.bos_token_id,
        eos_token_id=source.eos_token_id,
        pad_token_id=source.pad_token_id,
        boundary_dtype=str(source.boundary_dtype),
    )
    boundary=FactorizedBoundaryModule(config)
    in_codes,in_basis,in_error=_truncated_svd(
        source_boundary.embed_tokens.weight,rank,seed=seed
    )
    out_error=in_error
    with torch.no_grad():
        boundary.embed_tokens.codes.weight.copy_(in_codes.to(boundary.embed_tokens.codes.weight.dtype))
        boundary.embed_tokens.basis.copy_(in_basis.to(boundary.embed_tokens.basis.dtype))
        boundary.final_norm.weight.copy_(source_boundary.final_norm.weight.detach().cpu().to(boundary.final_norm.weight.dtype))
        if not config.tie_word_embeddings:
            out_codes,out_basis,out_error=_truncated_svd(
                source_boundary.lm_head.weight,rank,seed=seed+1
            )
            boundary.output_codes.copy_(out_codes.to(boundary.output_codes.dtype))
            boundary.output_basis.copy_(out_basis.to(boundary.output_basis.dtype))
    receipt={
        "schema":"NOLANE-L17-BOUNDARY-FACTORIZATION-V1",
        "rank":int(rank),
        "input_relative_frobenius_error":float(in_error),
        "output_relative_frobenius_error":float(out_error),
        "dense_parameters":dense_boundary_parameters(config),
        "factorized_parameters":boundary.parameter_count(),
        "parameter_ratio":boundary.parameter_count()/dense_boundary_parameters(config),
        "tie_word_embeddings":config.tie_word_embeddings,
        "config":asdict(config),
    }
    return boundary,receipt
