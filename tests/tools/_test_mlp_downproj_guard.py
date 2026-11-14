# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Gate for the default-off down_proj write-loss guard.

Run: python tests/tools/_test_mlp_downproj_guard.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter import mlp_downproj_guard as guard  # noqa: E402

FAILS = []


def ok(tag, cond, detail=""):
    print(f"{tag}: {'OK' if cond else 'FAIL ' + detail}", flush=True)
    if not cond:
        FAILS.append(tag)


class _Down:
    tp_size = 1
    quant_method = None

    def __init__(self):
        self.weight = torch.randn(8, 4)


class _FakeMLP:
    def __init__(self):
        self.down_proj = _Down()

    def gate_up_proj(self, x):
        return x, None

    def act_fn(self, x):
        return x


def main():
    old = os.environ.get("SGLANG_FIX_MLP_DOWNPROJ_GUARD")
    os.environ["SGLANG_FIX_MLP_DOWNPROJ_GUARD"] = "1"
    capture = torch.cuda.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = lambda: False
    try:
        x = torch.randn(3, 4)
        mlp = _FakeMLP()
        expected = torch.nn.functional.linear(x, mlp.down_proj.weight)

        healthy = expected.clone()
        got = guard._around(lambda _self, _x: healthy, mlp, x)
        ok("healthy-output-untouched", got is healthy)

        for name, bad in (
            ("nan", expected.clone().fill_(float("nan"))),
            ("1e37", expected.clone().fill_(1e37)),
        ):
            got = guard._around(lambda _self, _x, b=bad: b, mlp, x)
            ok(f"repairs-{name}", torch.isfinite(got).all().item())
            ok(f"repairs-{name}-value", torch.allclose(got, expected))

        bad_input = torch.full_like(x, float("nan"))
        got = guard._around(lambda _self, _x: expected, mlp, bad_input)
        ok("bad-input-skips-recompute", got is expected)

        os.environ["SGLANG_FIX_MLP_DOWNPROJ_GUARD"] = "0"
        got = guard._around(lambda _self, _x: expected.clone().fill_(1e37), mlp, x)
        ok(
            "disabled-returns-original",
            bool(torch.isnan(got).any().item() or (got > 1e20).any().item()),
        )
    finally:
        torch.cuda.is_current_stream_capturing = capture
        if old is None:
            os.environ.pop("SGLANG_FIX_MLP_DOWNPROJ_GUARD", None)
        else:
            os.environ["SGLANG_FIX_MLP_DOWNPROJ_GUARD"] = old

    if FAILS:
        print("FAILED:", FAILS)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
