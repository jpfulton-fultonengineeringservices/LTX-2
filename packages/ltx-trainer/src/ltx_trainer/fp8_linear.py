# Hand-rolled fp8 (e4m3) matmul path for the FROZEN base transformer's Linear
# layers, used for LoRA training on GB10 (sm_121).
#
# WHY THIS EXISTS (and why it is NOT `quantization.py`)
# -----------------------------------------------------
# `quantization.py` uses optimum-quanto, which is a MEMORY optimization: it stores
# weights in int8/fp8 but DEQUANTIZES them back to bf16 for every matmul, so it does
# not use the fp8 tensor cores and adds dequant overhead. On GB10 the bottleneck is
# COMPUTE, not memory: the bf16 tensor-core path is badly underutilized (~25 TFLOPS
# measured at the DiT's 4096/16384 GEMM shapes), while `torch._scaled_mm` fp8×fp8→bf16
# hits ~65-90 TFLOPS (~2.6-3.3x measured on GB10 under torch 2.9/cu130).
#
# This module swaps the large frozen Linears (FF + attention projections — the bulk of
# the 22B FLOPs) for `Fp8Linear`, which runs the matmul in fp8 and returns bf16. It is
# only meaningful for LoRA training:
#   - The base is FROZEN, so fp8 weight rounding is a fixed error the trainable bf16
#     LoRA adapters learn to correct (no accuracy regression in practice).
#   - Gradients flow only to the bf16 LoRA params, so we never need fp8 GRADIENT
#     scaling — this is fp8 inference-style matmul inside a training forward, which is
#     the robust, tractable subset of "fp8 training".
#
# Scaling recipe (chosen for speed/simplicity): per-tensor STATIC weight scale computed
# once at conversion, per-tensor DYNAMIC activation scale computed per-forward from the
# running max. e4m3 for both operands.

from __future__ import annotations

import torch
from torch import nn

from ltx_trainer import logger

# e4m3 max representable magnitude; used to map a real-valued tensor's amax onto the
# fp8 dynamic range: scale = amax / F8_E4M3_MAX, then x_fp8 = (x / scale).to(e4m3).
_F8_E4M3_MAX = 448.0
_EPS = 1e-12


def _amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    """Map an absolute-max value to a positive fp8 scale, clamped away from zero.

    `_scaled_mm` computes ``out = (a_fp8 * scale_a) @ (b_fp8 * scale_b)``, so the scale
    is the multiplier that turns the stored fp8 code back into the real value. We store
    ``x_fp8 = (x / scale)`` (bringing amax to the e4m3 max), hence ``scale = amax / MAX``.

    The scale is always fp32 (``_scaled_mm`` requires fp32 scales), but the amax itself
    is computed in the activation's native dtype to avoid an extra fp32 upcast pass over
    the (large) activation tensor.
    """
    return (amax.to(torch.float32) / _F8_E4M3_MAX).clamp(min=_EPS)


def _quantize_to_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale into the e4m3 range and cast, without an fp32 round-trip.

    The division/clamp run in ``x``'s own dtype (bf16 for activations); only the scalar
    ``scale`` is fp32. Casting bf16->e4m3 is well-defined and keeps the quant a single
    cheap pass rather than the fp32 upcast + downcast that dominated the naive version.
    """
    inv_scale = (1.0 / scale).to(x.dtype)
    q = (x * inv_scale).clamp(-_F8_E4M3_MAX, _F8_E4M3_MAX)
    return q.to(torch.float8_e4m3fn)


def _fp8_forward(
    x2d: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Dynamic-activation-quant + fp8 scaled matmul, returning bf16 (NO autograd).

    Pure forward math, used inside the custom autograd Function below. Kept as a free
    function so a single ``torch.compile`` wrapper is shared across every ``Fp8Linear``
    instance and specializes per unique (M, K, N) shape. Compiling fuses the pointwise quant
    (amax/scale/cast) with the matmul epilogue.
    """
    x_scale = _amax_to_scale(x2d.detach().abs().amax())
    x_fp8 = _quantize_to_fp8(x2d, x_scale)
    return torch._scaled_mm(
        x_fp8,
        weight_fp8.t(),
        scale_a=x_scale,
        scale_b=weight_scale,
        bias=None,
        out_dtype=torch.bfloat16,
    )


# Lazily-built compiled variant of `_fp8_forward`, shared process-wide. `None` until first
# use; set to the eager function if compilation is unavailable (e.g. missing python3-dev
# headers -> inductor C++ codegen fails) so we degrade gracefully.
_compiled_impl = None


def _get_impl(compile_enabled: bool):
    global _compiled_impl
    if not compile_enabled:
        return _fp8_forward
    if _compiled_impl is None:
        try:
            _compiled_impl = torch.compile(_fp8_forward)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"fp8_base_linear: torch.compile unavailable ({type(e).__name__}); using eager fp8.")
            _compiled_impl = _fp8_forward
    return _compiled_impl


class _Fp8MatmulFn(torch.autograd.Function):
    """Autograd wrapper: fp8 forward, bf16 backward w.r.t. the ACTIVATION only.

    WHY THIS EXISTS
    ---------------
    ``torch._scaled_mm`` has NO registered autograd backward ("derivative for aten::
    _scaled_mm is not implemented"). In a real training forward the activation ``x`` is on
    the gradient path (it is produced by upstream trainable/LoRA layers), so autograd tries
    to differentiate the base matmul and crashes — even though the fp8 WEIGHT is frozen and
    needs no gradient. (A pure inference micro-benchmark never hits this, which is why the
    naive version looked fine until a full forward+backward.)

    We therefore define the base matmul as: forward = fp8 ``_scaled_mm`` (fast), backward =
    ``grad_x = grad_out @ W`` computed in bf16 from the DEQUANTIZED weight. The frozen fp8
    weight/scale receive ``None`` gradients. This is exactly the frozen-base LoRA regime: no
    fp8 gradient scaling, no weight grad — just a correct activation gradient so the
    upstream bf16 LoRA params still learn.
    """

    @staticmethod
    def forward(ctx, x2d, weight_fp8, weight_scale, compile_enabled):  # noqa: ANN001
        impl = _get_impl(compile_enabled)
        out = impl(x2d, weight_fp8, weight_scale)
        # Save the fp8 weight + scale to reconstruct a bf16 weight for the backward matmul.
        ctx.save_for_backward(weight_fp8, weight_scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: ANN001
        weight_fp8, weight_scale = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            # Dequantize the frozen fp8 weight back to bf16 and do a plain bf16 matmul for the
            # activation gradient: d/dx (x @ W.T) = grad_out @ W. Correct to fp8-rounding of W,
            # which is the fixed base error the LoRA adapters are learning to correct anyway.
            w_bf16 = (weight_fp8.to(torch.bfloat16)) * weight_scale.to(torch.bfloat16)
            grad_x = grad_out.to(torch.bfloat16) @ w_bf16
        # No gradient for the frozen weight_fp8 / weight_scale / compile flag.
        return grad_x, None, None, None


class Fp8Linear(nn.Module):
    """Drop-in replacement for a FROZEN ``nn.Linear`` that runs the matmul in fp8.

    The weight is stored pre-quantized to e4m3 with a per-tensor static scale. In
    forward, the (bf16) activation is dynamically quantized to e4m3 (per-tensor) and the
    product is computed with ``torch._scaled_mm`` returning bf16. Bias is kept in bf16
    and added after the scaled matmul.

    This module has NO trainable parameters (the base is frozen); the fp8 weight and its
    scale are registered as buffers so they move with ``.to(device)`` and are excluded
    from the optimizer. When PEFT wraps this as a LoRA target, PEFT adds its own bf16
    ``lora_A``/``lora_B`` and computes ``base(x) + lora(x)`` — so the frozen bulk runs
    fp8 while the trainable delta stays bf16.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        in_features: int,
        out_features: int,
        compile_matmul: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compile_matmul = compile_matmul

        # Static per-tensor weight scale from the bf16 weight's amax, computed once.
        w = weight.detach()
        w_scale = _amax_to_scale(w.abs().amax())
        w_fp8 = _quantize_to_fp8(w, w_scale)

        # Buffers (not Parameters): frozen, not optimized, move with .to().
        self.register_buffer("weight_fp8", w_fp8.contiguous(), persistent=True)
        self.register_buffer("weight_scale", w_scale.reshape(()), persistent=True)
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(torch.bfloat16), persistent=True)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ARG002
        # PEFT invokes the wrapped base as ``base_layer(x, *args, **kwargs)`` (see
        # peft/tuners/lora/layer.py). A plain nn.Linear ignores the extras; we do too, so
        # this stays a drop-in replacement for the frozen base inside a lora.Linear.
        orig_shape = x.shape
        # _scaled_mm needs 2D operands; fold any leading (batch, seq, ...) dims.
        x2d = x.reshape(-1, self.in_features)
        in_dtype = x.dtype

        # Route through the custom autograd Function: fp8 forward, bf16 activation-grad
        # backward (torch._scaled_mm has no autograd backward of its own).
        out = _Fp8MatmulFn.apply(x2d, self.weight_fp8, self.weight_scale, self.compile_matmul)

        if self.bias is not None:
            out = out + self.bias
        out = out.reshape(*orig_shape[:-1], self.out_features)
        return out.to(in_dtype)


# Substring patterns that must STAY bf16 (mirrors quantization.EXCLUDE_PATTERNS intent):
# numerically sensitive / small layers where fp8 rounding hurts and saves little.
_FP8_EXCLUDE_SUBSTRINGS = (
    "norm",
    "adaln",
    "time_proj",
    "timestep_embedder",
    "caption_projection",
    "audio_caption_projection",
    "patchify_proj",
    "audio_patchify_proj",
    "proj_out",
    "audio_proj_out",
)


def _fp8_from_linear(lin: nn.Linear, compile_matmul: bool) -> "Fp8Linear":
    """Build an ``Fp8Linear`` mirroring an existing (frozen) ``nn.Linear``."""
    return Fp8Linear(
        lin.weight,
        lin.bias,
        in_features=lin.in_features,
        out_features=lin.out_features,
        compile_matmul=compile_matmul,
    )


def _should_convert(module_path: str) -> bool:
    """Only convert Linears inside transformer blocks, excluding sensitive layers."""
    if "transformer_blocks." not in module_path:
        return False
    return not any(s in module_path for s in _FP8_EXCLUDE_SUBSTRINGS)


def convert_base_linears_to_fp8(model: nn.Module, compile_matmul: bool = True) -> int:
    """In-place swap eligible frozen ``nn.Linear`` layers in ``model`` for ``Fp8Linear``.

    Use this on a RAW (pre-PEFT) transformer where every target is a bare ``nn.Linear``.
    NOTE: PEFT/LoRA cannot wrap an ``Fp8Linear`` (its ``_create_new_module`` only accepts
    ``nn.Linear`` & friends), so when LoRA is attached you must instead call
    ``convert_lora_base_linears_to_fp8`` AFTER ``get_peft_model``. This function remains for
    non-LoRA / standalone use.

    Returns the number of layers converted. Only the large GEMMs inside
    ``transformer_blocks.*`` (FF + attention projections) are converted; norms and small
    projections are left bf16.
    """
    to_convert: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full = f"{name}.{child_name}" if name else child_name
                if _should_convert(full):
                    to_convert.append((module, child_name, child))

    converted = 0
    for parent, child_name, lin in to_convert:
        setattr(parent, child_name, _fp8_from_linear(lin, compile_matmul))
        converted += 1

    logger.info(
        f"fp8_base_linear: converted {converted} frozen Linear layers to fp8 "
        f"(_scaled_mm, bf16 out, compile={compile_matmul})"
    )
    return converted


def convert_lora_base_linears_to_fp8(peft_model: nn.Module, compile_matmul: bool = True) -> int:
    """Swap the FROZEN base ``nn.Linear`` inside each PEFT ``lora.Linear`` for ``Fp8Linear``.

    MUST be called AFTER ``get_peft_model``. PEFT wraps each targeted base Linear in a
    ``lora.Linear`` that holds the frozen base at ``.base_layer`` and adds trainable bf16
    ``lora_A``/``lora_B``. Its forward is ``base_layer(x) + lora_B(lora_A(x)) * scaling``.
    We replace ONLY ``base_layer`` with ``Fp8Linear`` so the frozen bulk matmul runs fp8
    while the LoRA delta stays bf16 — this is the piece PEFT rejects if you try to convert
    before attach (it refuses to LoRA-wrap a non-``nn.Linear``).

    We detect a LoRA-wrapped Linear structurally (a module exposing ``base_layer`` +
    ``lora_A``) rather than importing ``peft.tuners.lora.Linear`` so we stay robust across
    PEFT versions. Only wrappers whose module path is under ``transformer_blocks.*`` and not
    in the exclude list are converted; ``lora_A``/``lora_B`` are left untouched.

    Returns the number of base layers converted.
    """
    to_convert: list[tuple[nn.Module, nn.Linear]] = []
    for name, module in peft_model.named_modules():
        base = getattr(module, "base_layer", None)
        has_lora = hasattr(module, "lora_A")
        if base is None or not has_lora:
            continue
        if not isinstance(base, nn.Linear):
            # Already converted, or a non-Linear target (Embedding/Conv) we don't touch.
            continue
        if not _should_convert(name):
            continue
        to_convert.append((module, base))

    converted = 0
    for wrapper, base in to_convert:
        wrapper.base_layer = _fp8_from_linear(base, compile_matmul)
        converted += 1

    logger.info(
        f"fp8_base_linear: converted {converted} LoRA base_layer Linears to fp8 "
        f"(_scaled_mm, bf16 out, compile={compile_matmul})"
    )
    return converted
