"""ProGen2 model code, vendored.

Provenance
----------
Source      https://huggingface.co/hugohrban/progen2-base
Revision    71228cbfca5960f9fab5775f378bba3673af9f00
Files       modeling_progen.py, configuration_progen.py
Vendored    2026-10-21
Licence     Apache-2.0 for this code (it derives from HuggingFace's GPT-J
            implementation and carries that header); the ProGen2 *weights* are
            BSD-3-Clause from Salesforce Research. Apache-2.0 section 4(b)
            requires stating that the files were changed: they were, and each
            change is marked with a ``THERMOCAS:`` comment.

Why vendored rather than loaded with ``trust_remote_code=True``
--------------------------------------------------------------
The upstream file targets transformers 4.40. Under transformers 5.x it fails
in three separate places, and the failures were found one at a time by fixing
the previous one:

1. ``AttributeError: 'ProGenForCausalLM' object has no attribute
   'all_tied_weights_keys'`` -- it calls the 4.x ``init_weights()`` rather than
   5.x's ``post_init()``.
2. ``RuntimeError: Tensor on device meta is not on the expected device cpu``
   -- its non-persistent causal-mask buffer and its ``scale_attn`` tensor are
   built during 5.x's meta-device initialization and nothing in the checkpoint
   materializes them.
3. ``AttributeError: 'ProGenModel' object has no attribute 'get_head_mask'``
   -- removed from ``PreTrainedModel`` in 5.x.

Three accumulating runtime monkeypatches against a file downloaded at import
time is a worse foundation than one copy of the code in the repository. The
alternative, pinning transformers 4.x, would also mean pinning the Python
interpreter, since the 4.x line does not support Python 3.14.

Vendoring also fixed two defects that are not version drift at all, and these
mattered more than the crashes:

* ``lm_head.weight`` was listed in ``_keys_to_ignore_on_load_missing``. ProGen2's
  head is an untied ``nn.Linear`` whose weight is in the checkpoint, so that
  entry meant a lost output layer would load silently as a random one.
* ``supports_gradient_checkpointing`` was False and the forward pass only read
  ``config.gradient_checkpointing``, so ``gradient_checkpointing_enable()``
  would not have taken effect. Stage A's memory budget assumes it does.

Updating the vendored copy
--------------------------
Re-download both files at the pinned revision, diff against these, and
re-apply the ``THERMOCAS:``-marked changes. ``tests/test_progen2.py`` checks
the invariants that matter: weights all load, no parameter is left on the meta
device, gradient checkpointing is honoured, and the model assigns real protein
sequences markedly lower loss than shuffled ones.
"""

from thermocas.model.progen2.configuration_progen import ProGenConfig
from thermocas.model.progen2.modeling_progen import ProGenForCausalLM, ProGenModel

__all__ = ["ProGenConfig", "ProGenForCausalLM", "ProGenModel"]
