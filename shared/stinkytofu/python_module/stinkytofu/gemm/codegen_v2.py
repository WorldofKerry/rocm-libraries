# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tree-walking codegen using TileContext and TileLevel.

Each tile level has a default emit function that can be replaced via
``tree.replace("level_name", emit=my_fn)``.  Custom functions receive
``(level, ctx)`` and use named bindings -- never raw register indices.

Generated kernel structure::

    workgroup_start:
      thread/wave index computation
      init accumulators to zero
      global_load A, B -> VGPRs
      lds_write A, B -> LDS
      barrier
      k_loop:
        for ki in k_iterations:
          lds_read A, B operands
          for mi, ni: MFMA
      epilogue:
        acc -> VGPR -> convert -> global store D
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .context import TileContext, Lifetime
from .problem import GemmProblem, MfmaConfig, TileConfig
from .tile import TileLevel, build_gemm_tile_tree, walk_tile_tree
from .transforms import Dim, Embed

__all__ = [
    "generate_from_tree",
    "default_visitor",
    "EMIT_REGISTRY",
    "GenerateResult",
]


# ===================================================================
# Permanent register allocation
# ===================================================================

def _alloc_kernel_args(ctx: TileContext, problem: GemmProblem,
                       tile: TileConfig) -> None:
    """Allocate permanent kernel-argument and index registers."""
    ctx.alloc_sgpr_permanent(2, "srd_A")
    ctx.alloc_sgpr_permanent(2, "srd_B")
    ctx.alloc_sgpr_permanent(2, "srd_D")
    ctx.alloc_sgpr_permanent(1, "s_M")
    ctx.alloc_sgpr_permanent(1, "s_N")
    ctx.alloc_sgpr_permanent(1, "s_K")
    ctx.alloc_sgpr_permanent(1, "s_lda")
    ctx.alloc_sgpr_permanent(1, "s_ldb")
    ctx.alloc_sgpr_permanent(1, "s_ldd")
    ctx.alloc_sgpr_permanent(1, "s_alpha")
    ctx.alloc_sgpr_permanent(1, "s_beta")

    ctx.alloc_vgpr_permanent(1, "v_tid")
    ctx.alloc_vgpr_permanent(1, "v_wave_id")
    ctx.alloc_vgpr_permanent(1, "v_lane_id")
    ctx.alloc_vgpr_permanent(1, "v_wave_m")
    ctx.alloc_vgpr_permanent(1, "v_wave_n")

    # Global-load buffers (permanent -- reused every K-tile)
    elem = problem.element_bytes
    a_elems_per_thread = (tile.wg_m * tile.unroll_k) // tile.block_size
    b_elems_per_thread = (tile.wg_n * tile.unroll_k) // tile.block_size
    a_vgprs = max(1, (a_elems_per_thread * elem + 3) // 4)
    b_vgprs = max(1, (b_elems_per_thread * elem + 3) // 4)
    ctx.alloc_vgpr_permanent(a_vgprs, "v_gload_a")
    ctx.alloc_vgpr_permanent(b_vgprs, "v_gload_b")

    # Address computation VGPRs
    ctx.alloc_vgpr_permanent(2, "v_addr_a")
    ctx.alloc_vgpr_permanent(2, "v_addr_b")
    ctx.alloc_vgpr_permanent(2, "v_addr_d")
    ctx.alloc_vgpr_permanent(1, "v_lds_write_a")
    ctx.alloc_vgpr_permanent(1, "v_lds_write_b")
    ctx.alloc_vgpr_permanent(1, "v_lds_read_a")
    ctx.alloc_vgpr_permanent(1, "v_lds_read_b")

    # MFMA operand VGPRs (permanent -- filled by LDS read each K-iter)
    ctx.alloc_vgpr_permanent(tile.mfma.a_vgprs, "v_a")
    ctx.alloc_vgpr_permanent(tile.mfma.b_vgprs, "v_b")

    # Store temporary
    ctx.alloc_vgpr_permanent(1, "v_store_tmp")

    # Accumulators
    acc_total = tile.mfma_m_repeat * tile.mfma_n_repeat * tile.mfma.acc_vgprs
    ctx.alloc_acc_permanent(acc_total, "acc_C")


# ===================================================================
# Emit helpers (used by default level functions)
# ===================================================================

def _label(ctx, text):
    if ctx.module is None:
        return
    import stinkytofu as st
    ctx.module.add(st.Label(text))


def _emit_thread_indices(ctx: TileContext, tile: TileConfig) -> None:
    if ctx.module is None:
        return
    import stinkytofu as st
    log2_ws = int(math.log2(tile.wave_size))
    ctx.module.add(st.VLShiftRightB32(
        ctx.vgpr("v_wave_id"), ctx.vgpr("v_tid"),
        st.Register(log2_ws), comment="wave_id = tid >> log2(wave_size)",
    ))
    ctx.module.add(st.VAndB32(
        ctx.vgpr("v_lane_id"), ctx.vgpr("v_tid"),
        st.Register(tile.wave_size - 1), comment="lane_id = tid & (wave_size-1)",
    ))
    if tile.waves_n > 1:
        ctx.module.add(st.VAndB32(
            ctx.vgpr("v_wave_n"), ctx.vgpr("v_wave_id"),
            st.Register(tile.waves_n - 1), comment="wave_n = wave_id % waves_n",
        ))
        ctx.module.add(st.VLShiftRightB32(
            ctx.vgpr("v_wave_m"), ctx.vgpr("v_wave_id"),
            st.Register(int(math.log2(tile.waves_n))),
            comment="wave_m = wave_id / waves_n",
        ))
    else:
        ctx.module.add(st.VMovB32(
            ctx.vgpr("v_wave_m"), ctx.vgpr("v_wave_id"),
            comment="wave_m = wave_id (waves_n=1)",
        ))


def _emit_init_acc(ctx: TileContext) -> None:
    if ctx.module is None:
        return
    import stinkytofu as st
    b = ctx.get("acc_C")
    for i in range(b.count):
        ctx.module.add(st.VAccvgprWriteB32(
            ctx.acc("acc_C", i, 1), st.Register(0),
            comment=f"acc[{i}] = 0",
        ))


def _emit_buffer_loads(ctx, buf_name, addr_name, tag):
    if ctx.module is None:
        return
    import stinkytofu as st
    b = ctx.get(buf_name)
    n = b.count
    chunk = 4
    for i in range(0, n, chunk):
        cnt = min(chunk, n - i)
        fn = {4: st.BufferLoadB128, 2: st.BufferLoadB64, 1: st.BufferLoadB32}[cnt]
        ctx.module.add(fn(
            ctx.vgpr(buf_name, i, cnt), ctx.vgpr(addr_name, 0, 1),
            comment=f"global load {tag}[{i}:{i+cnt}]",
        ))


def _emit_ds_stores(ctx, buf_name, addr_name, tag):
    if ctx.module is None:
        return
    import stinkytofu as st
    b = ctx.get(buf_name)
    n = b.count
    chunk = 4
    for i in range(0, n, chunk):
        cnt = min(chunk, n - i)
        fn = {4: st.DSStoreB128, 2: st.DSStoreB64, 1: st.DSStoreB32}[cnt]
        ctx.module.add(fn(
            ctx.vgpr(addr_name), ctx.vgpr(buf_name, i, cnt),
            comment=f"LDS write {tag}[{i}:{i+cnt}]",
        ))


def _emit_barrier(ctx):
    if ctx.module is None:
        return
    import stinkytofu as st
    ctx.module.add(st.SBarrier(comment="LDS barrier"))


def _emit_lds_read(ctx, mfma):
    """Read MFMA operands from LDS."""
    if ctx.module is None:
        return
    import stinkytofu as st
    # A operand
    if mfma.a_vgprs >= 4:
        ctx.module.add(st.DSLoadB128(
            ctx.vgpr("v_a", 0, 4), ctx.vgpr("v_lds_read_a"),
            comment="LDS read A",
        ))
    else:
        for r in range(mfma.a_vgprs):
            ctx.module.add(st.DSLoadB32(
                ctx.vgpr("v_a", r, 1), ctx.vgpr("v_lds_read_a"),
                comment=f"LDS read A[{r}]",
            ))
    # B operand
    if mfma.b_vgprs >= 4:
        ctx.module.add(st.DSLoadB128(
            ctx.vgpr("v_b", 0, 4), ctx.vgpr("v_lds_read_b"),
            comment="LDS read B",
        ))
    else:
        for r in range(mfma.b_vgprs):
            ctx.module.add(st.DSLoadB32(
                ctx.vgpr("v_b", r, 1), ctx.vgpr("v_lds_read_b"),
                comment=f"LDS read B[{r}]",
            ))


def _emit_store_d(ctx, tile):
    """Epilogue: move accumulators to VGPRs, convert f32->f16, store."""
    if ctx.module is None:
        return
    import stinkytofu as st
    mfma = tile.mfma
    acc_per = mfma.acc_vgprs
    n_tiles = tile.mfma_m_repeat * tile.mfma_n_repeat
    for t_idx in range(n_tiles):
        for r in range(acc_per):
            ctx.module.add(st.VAccvgprReadB32(
                ctx.vgpr("v_store_tmp"), ctx.acc("acc_C", t_idx * acc_per + r, 1),
                comment=f"acc->vgpr tile{t_idx}[{r}]",
            ))
            ctx.module.add(st.VCvtF32toF16(
                ctx.vgpr("v_store_tmp"), ctx.vgpr("v_store_tmp"),
                comment="cvt f32->f16",
            ))
            ctx.module.add(st.FlatStoreB32(
                ctx.vgpr("v_addr_d", 0, 2),
                ctx.vgpr("v_store_tmp"),
                ctx.vgpr("v_store_tmp"),
                comment=f"store D tile{t_idx}[{r}]",
            ))


# ===================================================================
# Default emit functions per level
# ===================================================================

def _emit_workgroup_prologue(ctx: TileContext, tile: TileConfig) -> None:
    """Prologue: thread indices, init acc, global load, LDS write, barrier."""
    """Workgroup level: prologue, global load, LDS write, barrier,
    K-loop (via inner walk), epilogue."""

    _label(ctx, "prologue")
    _emit_thread_indices(ctx, tile)
    _emit_init_acc(ctx)

    _label(ctx, "global_load")
    _emit_buffer_loads(ctx, "v_gload_a", "v_addr_a", "A")
    _emit_buffer_loads(ctx, "v_gload_b", "v_addr_b", "B")

    _label(ctx, "lds_write")
    _emit_ds_stores(ctx, "v_gload_a", "v_lds_write_a", "A")
    _emit_ds_stores(ctx, "v_gload_b", "v_lds_write_b", "B")
    _emit_barrier(ctx)

    # workgroup.ki and recursing into the wave level.


def _emit_wave(level: TileLevel, ctx: TileContext) -> None:
    """Wave level: grouping only, LDS read + MFMA happen at inner levels."""
    ki = ctx.indices.get("workgroup.ki", 0)
    _label(ctx, f"k_iter_{ki}")
    tile = ctx._metadata["tile"]
    _emit_lds_read(ctx, tile.mfma)


def _emit_mfma_leaf(level: TileLevel, ctx: TileContext) -> None:
    """Leaf: emit one MFMA instruction."""
    if ctx.module is None:
        return
    import stinkytofu as st

    tile = ctx._metadata["tile"]
    mfma = tile.mfma

    mi = ctx.indices.get("wave.mi", 0)
    ni = ctx.indices.get("wave.ni", 0)
    ki = ctx.indices.get("workgroup.ki", 0)
    acc_per = mfma.acc_vgprs
    acc_offset = (mi * tile.mfma_n_repeat + ni) * acc_per

    ctx.module.add(st.MFMA(
        instType=mfma.input_type, accType=mfma.acc_type,
        m=mfma.m, n=mfma.n, k=mfma.k,
        blocks=mfma.blocks, mfma1k=False,
        acc=ctx.acc("acc_C", acc_offset, acc_per),
        a=ctx.vgpr("v_a", 0, mfma.a_vgprs),
        b=ctx.vgpr("v_b", 0, mfma.b_vgprs),
        comment=f"MFMA m{mi}_n{ni} k{ki}",
    ))


EMIT_REGISTRY: Dict[str, Callable] = {
    "wave": _emit_wave,
    "mfma": _emit_mfma_leaf,
}


# ===================================================================
# Visitor and entry point
# ===================================================================

def default_visitor(level: TileLevel, ctx: TileContext) -> None:
    handler = EMIT_REGISTRY.get(level.name)
    if handler:
        handler(level, ctx)


@dataclass
class GenerateResult:
    module: object          # LogicalModule or None
    ctx: TileContext
    tree: TileLevel
    problem: GemmProblem
    tile: TileConfig

    def summary(self) -> str:
        grid_m, grid_n = self.problem.grid_dims(self.tile)
        lines = [
            f"=== Kernel: {self.problem.dtype.value} "
            f"{self.tile.wg_m}x{self.tile.wg_n}x{self.tile.unroll_k} "
            f"mfma{self.tile.mfma.m}x{self.tile.mfma.n}x{self.tile.mfma.k} ===",
            f"Problem : {self.problem.m}x{self.problem.n}x{self.problem.k}",
            f"Grid    : {grid_m}x{grid_n} workgroups",
            "",
            self.tile.summary(),
            "",
            "Tree:",
            self.tree.summary(indent=1),
            "",
            "Registers:",
            self.ctx.summary(),
        ]
        return "\n".join(lines)


def generate_from_tree(
    problem: GemmProblem,
    tile: Optional[TileConfig] = None,
    tile_tree: Optional[TileLevel] = None,
    *,
    dry_run: bool = False,
) -> GenerateResult:
    """Generate a GEMM kernel by walking a tile tree with TileContext.

    Args:
        problem: GEMM problem specification.
        tile: Tile configuration. Defaults chosen if None.
        tile_tree: Custom TileLevel tree. Built from *tile* if None.
        dry_run: Skip stinkytofu emission; allocation and index tracking
            still happen.

    Returns:
        ``GenerateResult`` with module, context, tree, and metadata.

    Example::

        result = generate_from_tree(GemmProblem(4096, 4096, 4096))
        print(result.module.dump())

    Custom MFMA::

        def my_mfma(level, ctx):
            ...
        tree = build_gemm_tile_tree(...)
        tree = tree.replace("mfma", emit=my_mfma)
        result = generate_from_tree(problem, tile_tree=tree)
    """
    if tile is None:
        tile = TileConfig()
    problem.validate(tile)

    if tile_tree is None:
        tile_tree = build_gemm_tile_tree(
            wg_m=tile.wg_m, wg_n=tile.wg_n, unroll_k=tile.unroll_k,
            waves_m=tile.waves_m, waves_n=tile.waves_n,
            mfma_m=tile.mfma.m, mfma_n=tile.mfma.n, mfma_k=tile.mfma.k,
        )
    tile_tree.validate()

    module = None
    if not dry_run:
        import stinkytofu as st
        name = (
            f"gemm_{problem.dtype.value}"
            f"_{tile.wg_m}x{tile.wg_n}x{tile.unroll_k}"
            f"_mfma{tile.mfma.m}x{tile.mfma.n}x{tile.mfma.k}"
        )
        module = st.LogicalModule(name)

    ctx = TileContext(module=module)
    ctx._metadata = {"tile": tile, "problem": problem}

    _alloc_kernel_args(ctx, problem, tile)
    # Workgroup-level setup (before tree walk)
    _emit_workgroup_prologue(ctx, tile)

    # Tree walk covers one wave K-loop: LDS read + MFMAs
    walk_tile_tree(tile_tree, ctx, default_visitor)

    # Epilogue
    _label(ctx, "epilogue")
    _emit_store_d(ctx, tile)
    _label(ctx, "kernel_end")

    return GenerateResult(
        module=module, ctx=ctx, tree=tile_tree,
        problem=problem, tile=tile,
    )
