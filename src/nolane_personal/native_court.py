from __future__ import annotations

from .surgery import resolve_transformer_layers


def run_with_decoder_call_count(qwen_model, fn):
    layers = resolve_transformer_layers(qwen_model)
    calls = [0 for _ in layers]
    handles = []
    for index, layer in enumerate(layers):
        def hook(_module, _args, _output, *, _index=index):
            calls[_index] += 1
        handles.append(layer.register_forward_hook(hook))
    try:
        result = fn()
    finally:
        for handle in reversed(handles):
            handle.remove()
    return result, sum(calls), calls
