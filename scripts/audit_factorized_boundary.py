from __future__ import annotations
import json
from nolane_personal.factorized_boundary import FactorizedBoundaryConfig,analytical_factorized_boundary_parameters,dense_boundary_parameters

def main():
    cfg=FactorizedBoundaryConfig(vocab_size=151936,hidden_size=1024,rank=128,tie_word_embeddings=True)
    fact=analytical_factorized_boundary_parameters(cfg)
    dense=dense_boundary_parameters(cfg)
    result={"schema":"NOLANE-L17-FACTORIZED-BOUNDARY-AUDIT-V1","vocab_size":cfg.vocab_size,"hidden_size":cfg.hidden_size,"rank":cfg.rank,"tied":cfg.tie_word_embeddings,"dense_parameters":dense,"factorized_parameters":fact,"parameter_ratio":fact/dense}
    print(json.dumps(result,indent=2,sort_keys=True))
    if fact>=dense*0.35: raise SystemExit("factorized boundary did not meet 0.35 parameter-ratio gate")
    return 0
if __name__=="__main__": raise SystemExit(main())
