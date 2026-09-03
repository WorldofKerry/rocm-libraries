# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Solution-validation guard for TLU=1 subtile K window counts (gfx950).

A TLU=1 operand fetches one K window per MatrixInstK, and a window count
divisible by five reads out of bounds for some K.  Measured on gfx950 NT at
M=N=384: bf16 DepthU 160/320/480 (5/10/15 windows) fault with
hipErrorIllegalAddress or return wrong results, and non-MX fp4 DepthU 640
(5 windows, MatrixInstK 128) fails the same way, while the window counts either
side are clean.  ``Tensile/SolutionStructs/Solution.py`` therefore rejects::

    kWindows = DepthU // MatrixInstK
    if kWindows % 5 == 0: reject(...)

These tests pin that behaviour.  They matter because the condition is empirical
rather than derived, so it is exactly the kind of guard a later refactor might
"simplify" away: the accept cases fail if it is widened, the reject cases fail
if it is dropped.

Two properties are worth protecting beyond the raw reject:

* It keys off the *window count*, not DepthU.  bf16 320 and fp4 640 are both
  multiples of their own ``numSubIterK * MatrixInstK`` unit, so the DepthU rule
  immediately above does not catch either.
* It applies only when an operand is TLU=1.  TN reaches none of this, and a
  TN case at the same DepthU must stay valid.

The harness mirrors ``test_prefetchgl2_streamk_guard``: real gfx950 capability
maps from ``makeIsaInfoMap`` (needs ``amdclang++``; skipped if the toolchain
cannot target gfx950) plus a real assembler feed ``Solution.__init__``, which
runs ``assignDerivedParameters`` end-to-end.  The reject reason is captured from
stdout via ``capsys``.
"""

import copy

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


GUARD_REASON = "is a multiple of five"


# Snapshot the pristine process-global defaultSolution at import time (collection
# runs before any test executes). Sibling unit tests mutate it in place, which
# makes Solution.__init__'s `for key in defaultSolution` loop overwrite derived
# objects and break Solution construction in an order-dependent way.
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


@pytest.fixture(scope="module")
def gfx950_iim():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx950")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx950")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx950(gfx950_iim):
    """Assign process-global parameters for gfx950; restore after module."""
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx950_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


# Base solutions: the MT64x64 wg1x1 NT bf16 candidate from
# subtile_bf16_tlu1.yaml, and its fp4 counterpart. Each test varies only DepthU
# (and, for the control, the transposes).
_BF16_MI = [16, 16, 32, 1, 1, 4, 4, 1, 1]    # MT64x64 wg1x1, MatrixInstK 32
_FP4_MI = [16, 16, 128, 1, 1, 2, 2, 1, 1]    # MT32x32 wg1x1, MatrixInstK 128


def _make_params(gfx950_iim, mi, dtype, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx950")
    pt = overrides.pop("ProblemType", {})
    problem_type = {
        "OperationType": "GEMM",
        "DataType": dtype,
        "DestDataType": "b" if dtype == "b" else "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        # NT: both operands free-dim contiguous, so TLUA and TLUB are both True.
        "TransposeA": False,
        "TransposeB": True,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,
    }
    problem_type.update(pt)

    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 4, 1],
        "WavefrontSize": 64,
        "DepthU": 64,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 0,
        "PrefetchLocalRead": 0,
        "ScheduleIterAlg": 3,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "InnerUnroll": 1,
        "DirectToLds": 1,
        "TransposeLDS": -1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "SourceSwap": False,
        "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "StreamK": 0,
        "UseSubtileImpl": True,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
        "ClusterLocalRead": 0,
    }
    params.update(overrides)
    mi_params = matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx950_iim
    )
    params.update(mi_params)
    return params


def _derive(gfx950_iim, assembler, capsys, mi, dtype, **overrides):
    """Construct a Solution with reject printing on; return (sol, stdout)."""
    params = _make_params(gfx950_iim, mi, dtype, **overrides)
    sol = Solution(params, False, True, False, assembler, gfx950_iim)
    out = capsys.readouterr().out
    return sol, out


# ---------------------------------------------------------------------------
# bf16, MatrixInstK 32. DepthU 160/320/480 are 5/10/15 windows and must reject.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("depthu", [160, 320, 480])
def test_bf16_tlu1_rejects_multiple_of_five_windows(
    _gp_gfx950, gfx950_iim, assembler, capsys, depthu
):
    sol, out = _derive(gfx950_iim, assembler, capsys, _BF16_MI, "b", DepthU=depthu)
    assert sol.get("Valid") is False, f"expected reject at DepthU={depthu}"
    assert GUARD_REASON in out, f"rejected for the wrong reason: {out!r}"


# The window counts either side must stay valid; a widened guard breaks these.
@pytest.mark.parametrize("depthu", [64, 96, 128, 192, 224, 256])
def test_bf16_tlu1_accepts_other_window_counts(
    _gp_gfx950, gfx950_iim, assembler, capsys, depthu
):
    sol, out = _derive(gfx950_iim, assembler, capsys, _BF16_MI, "b", DepthU=depthu)
    assert sol.get("Valid") is True, f"expected accept at DepthU={depthu}, got: {out!r}"
    assert GUARD_REASON not in out


# ---------------------------------------------------------------------------
# fp4, MatrixInstK 128. The guard is not dtype-scoped: DepthU 640 is 5 windows
# and must reject, while 512 (4) and 768 (6) stay valid.
# ---------------------------------------------------------------------------
def test_fp4_tlu1_rejects_multiple_of_five_windows(
    _gp_gfx950, gfx950_iim, assembler, capsys
):
    sol, out = _derive(gfx950_iim, assembler, capsys, _FP4_MI, "F4", DepthU=640)
    assert sol.get("Valid") is False, "expected reject at fp4 DepthU=640 (5 windows)"
    assert GUARD_REASON in out, f"rejected for the wrong reason: {out!r}"


@pytest.mark.parametrize("depthu", [512, 768])
def test_fp4_tlu1_accepts_other_window_counts(
    _gp_gfx950, gfx950_iim, assembler, capsys, depthu
):
    sol, out = _derive(gfx950_iim, assembler, capsys, _FP4_MI, "F4", DepthU=depthu)
    assert sol.get("Valid") is True, f"expected accept at DepthU={depthu}, got: {out!r}"
    assert GUARD_REASON not in out


# ---------------------------------------------------------------------------
# The guard must key off the window count, not DepthU, and only for TLU=1.
# ---------------------------------------------------------------------------
def test_tn_at_same_depthu_is_not_touched(_gp_gfx950, gfx950_iim, assembler, capsys):
    """TN has no TLU=1 operand, so the K window guard must not fire.

    DepthU 320 is 10 MatrixInstK on this instruction -- the count that rejects
    under NT -- and is a multiple of TN's own 2 * MatrixInstK unit, so if this
    ever rejects it is the window guard overreaching rather than the DepthU rule.
    """
    sol, out = _derive(
        gfx950_iim, assembler, capsys, _BF16_MI, "b", DepthU=320,
        ProblemType={"TransposeA": True, "TransposeB": False},
    )
    assert GUARD_REASON not in out, f"window guard fired on TN: {out!r}"
    assert sol.get("Valid") is True, f"expected TN accept, got: {out!r}"
