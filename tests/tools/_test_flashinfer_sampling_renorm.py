# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Gate for the flashinfer.sampling plural renormalization faces.

Run: python tests/tools/_test_flashinfer_sampling_renorm.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter import flashinfer as fi  # noqa: E402

FAILS = []


def ok(tag, cond, detail=""):
    print(f"{tag}: {'OK' if cond else 'FAIL ' + detail}", flush=True)
    if not cond:
        FAILS.append(tag)


def main():
    fi.register()
    from flashinfer.sampling import (
        top_k_renorm_probs,
        top_p_renorm_probs,
    )
    from sgl_kernel.sampling import (
        top_k_renorm_prob,
        top_p_renorm_prob,
    )

    probs = torch.tensor(
        [[0.5, 0.3, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]], dtype=torch.float32
    )
    top_ks = torch.tensor([2, 3], dtype=torch.int32)
    top_ps = torch.tensor([0.6, 0.9], dtype=torch.float32)

    k = top_k_renorm_probs(probs, top_ks)
    ok("top_k_sum", torch.allclose(k.sum(-1), torch.ones(2)))
    ok("top_k_nonzero", int((k[0] > 0).sum()) == 2 and int((k[1] > 0).sum()) == 3)
    ok("top_k_matches_singular", torch.allclose(k, top_k_renorm_prob(probs, top_ks)))

    p = top_p_renorm_probs(probs, top_ps)
    ok("top_p_sum", torch.allclose(p.sum(-1), torch.ones(2)))
    ok("top_p_nonzero", int((p[0] > 0).sum()) == 2 and int((p[1] > 0).sum()) == 4)
    ok("top_p_matches_singular", torch.allclose(p, top_p_renorm_prob(probs, top_ps)))

    if FAILS:
        print("FAILED:", FAILS)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
