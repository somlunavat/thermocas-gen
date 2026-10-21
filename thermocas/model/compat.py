"""Loading ProGen2 weights into the vendored model code.

The model definition lives in ``thermocas.model.progen2`` (see that package's
docstring for why it is vendored rather than loaded with
``trust_remote_code=True``). This module loads the published weights into it
and verifies the result.

The verification is the point. ProGen2's upstream code declared
``lm_head.weight`` ignorable when missing, which means a checkpoint whose
output layer failed to load would have produced a working model with a random
head and no warning. Under LoRA the base weights are frozen, so that head
could never be relearned: the run would train, the loss would fall, and Gate 2
would fail for a reason unrelated to temperature conditioning. Loading
therefore refuses to return a model unless every parameter came from the
checkpoint.
"""

from __future__ import annotations

# Buffers a ProGen2 checkpoint legitimately does not contain. The vendored
# code no longer registers these, but a checkpoint saved from the upstream
# implementation still lists them, so they stay recognized as benign.
_EXPECTED_MISSING_SUBSTRINGS = (".attn.bias", ".attn.masked_bias", ".causal_mask")


def _is_expected_missing(key: str) -> bool:
    return any(part in key for part in _EXPECTED_MISSING_SUBSTRINGS)


def unloaded_parameter_keys(loading_info: dict) -> list[str]:
    """Parameters that did not come from the checkpoint, benign ones removed."""
    keys = list(loading_info.get("missing_keys", ()))
    keys += [entry[0] for entry in loading_info.get("mismatched_keys", ())]
    return sorted(key for key in keys if not _is_expected_missing(key))


def meta_device_parameters(model) -> list[str]:
    """Names of parameters or buffers still on the meta device.

    A tensor left on ``meta`` holds no values. It is the signature of the
    second failure described in the vendored package's docstring, and it
    surfaces as a confusing device error deep inside a forward pass rather
    than at load time -- so it is checked here instead.
    """
    named = list(model.named_parameters()) + list(model.named_buffers())
    return sorted(name for name, tensor in named if tensor.is_meta)


def checkpoint_weight_files(checkpoint: str) -> list[str]:
    """Local paths to the checkpoint's safetensors shards, downloading if needed."""
    import json

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        index_path = hf_hub_download(checkpoint, "model.safetensors.index.json")
    except EntryNotFoundError:
        return [hf_hub_download(checkpoint, "model.safetensors")]

    with open(index_path) as handle:
        shards = sorted(set(json.load(handle)["weight_map"].values()))
    return [hf_hub_download(checkpoint, shard) for shard in shards]


def weights_differing_from_checkpoint(model, checkpoint: str) -> list[str]:
    """Parameter names whose values do not match the checkpoint file.

    This compares against the file on disk rather than trusting
    ``from_pretrained``'s report, because the report was wrong. See
    ``_init_weights`` in the vendored ``modeling_progen.py``: transformers 5.x
    re-applies weight initialization after loading, and an initializer that
    writes through ``.data`` overwrites loaded weights while every key still
    counts as loaded. Byte-identical is the right standard -- loading does not
    do arithmetic, so anything other than equality means something intervened.
    """
    import torch
    from safetensors import safe_open

    state = model.state_dict()
    differing: list[str] = []
    for path in checkpoint_weight_files(checkpoint):
        with safe_open(path, framework="pt") as shard:
            for key in shard.keys():
                stored = shard.get_tensor(key)
                live = state.get(key)
                if live is None:
                    differing.append(f"{key} (absent from model)")
                elif tuple(live.shape) != tuple(stored.shape):
                    differing.append(
                        f"{key} (shape {tuple(live.shape)} vs {tuple(stored.shape)})"
                    )
                elif not torch.equal(live.detach().cpu().to(stored.dtype), stored):
                    differing.append(key)
    return sorted(differing)


def load_progen2(checkpoint: str, verify_weights: bool = True, **kwargs):
    """Load ProGen2 weights into the vendored implementation.

    Raises if any parameter was randomly initialized, left on the meta device,
    or does not match the checkpoint file, naming the keys. ``verify_weights``
    turns off only the file comparison, for the case where a caller has
    deliberately modified the weights before this runs.
    """
    from thermocas.model.progen2 import ProGenForCausalLM

    model, loading_info = ProGenForCausalLM.from_pretrained(
        checkpoint, output_loading_info=True, **kwargs
    )

    unloaded = unloaded_parameter_keys(loading_info)
    if unloaded:
        raise RuntimeError(
            f"{checkpoint} loaded with {len(unloaded)} randomly initialized "
            f"parameter(s): {unloaded[:10]}. Training on this would discard "
            "pretrained weights silently."
        )

    on_meta = meta_device_parameters(model)
    if on_meta:
        raise RuntimeError(
            f"{checkpoint} loaded with {len(on_meta)} tensor(s) on the meta "
            f"device: {on_meta[:10]}. These hold no values."
        )

    if verify_weights:
        differing = weights_differing_from_checkpoint(model, checkpoint)
        if differing:
            raise RuntimeError(
                f"{len(differing)} parameter(s) of {checkpoint} do not match the "
                f"checkpoint file after loading: {differing[:10]}. The weights "
                "were loaded and then overwritten; see _init_weights in the "
                "vendored modeling_progen.py."
            )
    return model
