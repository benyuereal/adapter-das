# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Detect and repair a bf16 down_proj GEMM that returned an unwritten buffer.

hipBLASLt on this HCU stack occasionally returns a buffer it never wrote
(stale 1e37/NaN values) for the Qwen2MoeMLP down_proj GEMM. The failure is
visible as NaN logits and a multinomial device assert. This is a default-off
diagnostic/repair hook: a healthy output is returned untouched; a corrupt one
is recomputed with the same operands through F.linear (rocBLAS), up to three
times. Enable with SGLANG_FIX_MLP_DOWNPROJ_GUARD=1.

This is a workaround, not a fix: the root cause is in the vendor GEMM. The
healthy path is the hbird bf16 provider (plugin) or a vendor fix.
"""

import logging
import os

import torch

from . import registry

LOG = logging.getLogger("hcu.adapter")

_TARGET = "sglang.srt.models.qwen2_moe.Qwen2MoeMLP.forward"
_ABSMAX = 1e20
_installed = False


def _enabled() -> bool:
    return os.environ.get("SGLANG_FIX_MLP_DOWNPROJ_GUARD") == "1"


def _healthy(t) -> bool:
    return bool(torch.isfinite(t).all()) and float(t.abs().max()) <= _ABSMAX


def _around(original, self, x):
    out = original(self, x)
    if not _enabled() or torch.cuda.is_current_stream_capturing():
        return out
    if _healthy(out) or not _healthy(x):
        return out
    down = self.down_proj
    if getattr(down, "tp_size", 1) != 1 or getattr(down, "quant_method", None):
        return out
    try:
        gate_up, _ = self.gate_up_proj(x)
        x_act = self.act_fn(gate_up)
        for _ in range(3):
            repaired = torch.nn.functional.linear(x_act, down.weight)
            if _healthy(repaired):
                LOG.warning("HCU adapter: repaired a corrupt down_proj output")
                return repaired
    except Exception:  # noqa: BLE001 - keep the original output on any repair failure
        LOG.warning("HCU adapter: down_proj guard repair failed", exc_info=True)
    return out


def install() -> bool:
    global _installed
    if _installed:
        return True
    if torch.version.hip is None:
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(_TARGET, _around, HookType.AROUND)
    registry.register(
        "platform_detect",
        "bf16 down_proj write-loss guard (hipBLASLt, default off)",
        _TARGET,
        registry.MECH_SEAM,
    )
    _installed = True
    return True
