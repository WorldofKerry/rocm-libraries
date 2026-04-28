# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Software-pipelined K-loop using double-buffered prefetch.

The classic (non-pipelined) K-loop is::

    for ki in range(K // unroll_k):
        global_load(A[ki], B[ki])
        lds_write(A, B)
        barrier
        for inner_ki in range(k_iterations):
            lds_read(A, B)
            mfma(A, B, acc)

The software-pipelined version overlaps global loads with compute::

    # Prolog: load first tile
    global_load(A[0], B[0])
    lds_write(A, B)
    barrier
    
    for ki in range(1, K // unroll_k):
        # Overlap: load next tile while computing current
        global_load(A[ki], B[ki])       # into buffer 1
        for inner_ki in range(k_iterations):
            lds_read(A, B)              # from buffer 0
            mfma(A, B, acc)
        lds_write(A, B)                 # buffer 1 -> LDS
        barrier
    
    # Epilog: compute last tile (no more loads)
    for inner_ki in range(k_iterations):
        lds_read(A, B)
        mfma(A, B, acc)

This is implemented as a custom ``emit`` function for the wave level
that can be plugged into the TileLevel tree::

    tree = build_gemm_tile_tree(...)
    tree = tree.replace("wave", emit=emit_pipelined_k_loop)
    result = generate_from_tree(problem, tile_tree=tree)

The prefetch buffers use HELD lifetime so they survive across
K-loop iterations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import TileContext
    from .tile import TileLevel

__all__ = ["emit_pipelined_k_loop"]


def emit_pipelined_k_loop(level: TileLevel, ctx: TileContext) -> None:
    """Software-pipelined K-loop with double-buffered global loads.

    Replaces the default wave-level emit.  Expects ctx to have:
      - ``v_gload_a``, ``v_gload_b``: global load buffers (permanent)
      - ``v_addr_a``, ``v_addr_b``: global load addresses
      - ``v_lds_write_a``, ``v_lds_write_b``: LDS write addresses
      - ``v_lds_read_a``, ``v_lds_read_b``: LDS read addresses
      - ``v_a``, ``v_b``: MFMA operand registers
      - ``acc_C``: accumulator registers
      - ``_metadata.tile``, ``_metadata.problem`` in ctx._metadata
    """
    if ctx.module is None:
        # Dry run: just walk the tree normally for index tracking
        _walk_inner_tiles(level, ctx)
        return

    import stinkytofu as st

    tile = ctx._metadata["tile"]
    problem = ctx._metadata["problem"]
    mfma = tile.mfma
    k_tiles = problem.k // tile.unroll_k

    if k_tiles <= 1:
        # Not worth pipelining -- fall back to non-pipelined
        _walk_inner_tiles(level, ctx)
        return

    # -- Allocate prefetch buffers with HELD lifetime --
    gload_a_count = ctx.get("v_gload_a").count
    gload_b_count = ctx.get("v_gload_b").count
    ctx.alloc_vgpr(gload_a_count, "v_prefetch_a", held=True)
    ctx.alloc_vgpr(gload_b_count, "v_prefetch_b", held=True)

    # -- Prolog: first tile is already loaded (by workgroup prologue) --
    # The workgroup prologue already did: global_load -> lds_write -> barrier
    # So buffer 0 is in LDS and ready to read.

    ctx.module.add(st.Label("k_loop_pipelined"))

    # -- Main loop: ki = 1..k_tiles-1 --
    # For each iteration: start loading next tile, compute current tile
    for ki_outer in range(1, k_tiles):
        ctx.module.add(st.Label(f"k_outer_{ki_outer}"))

        # Start global load for NEXT tile (into prefetch buffers)
        # In real code this would advance the address; here we just
        # emit the load instructions to show the structure
        _emit_global_loads(ctx, "v_prefetch_a", "v_addr_a", "A(prefetch)")
        _emit_global_loads(ctx, "v_prefetch_b", "v_addr_b", "B(prefetch)")

        # Compute CURRENT tile: LDS read + MFMAs
        _walk_inner_tiles(level, ctx)

        # Write prefetched data to LDS
        _emit_ds_stores(ctx, "v_prefetch_a", "v_lds_write_a", "A(pf)")
        _emit_ds_stores(ctx, "v_prefetch_b", "v_lds_write_b", "B(pf)")

        # Barrier
        ctx.module.add(st.SBarrier(comment="wait for prefetched LDS"))

    # -- Epilog: compute last tile (no more loads) --
    ctx.module.add(st.Label("k_epilog"))
    _walk_inner_tiles(level, ctx)

    # Free prefetch buffers
    ctx.free("v_prefetch_a")
    ctx.free("v_prefetch_b")


def _walk_inner_tiles(level: TileLevel, ctx: TileContext) -> None:
    """Walk the inner tile levels for one K-unroll iteration."""
    from .tile import walk_tile_tree
    from .codegen_v2 import default_visitor

    if level.inner is None:
        return

    # The level's K dimension gives us k_iterations
    mfma_k = level.inner.k if level.inner else 16
    k_iters = (level.k or 32) // mfma_k if level.k and mfma_k else 1
    m_reps = level.repeats_m
    n_reps = level.repeats_n

    if ctx.module is not None:
        import stinkytofu as st
        tile = ctx._metadata["tile"]
        mfma = tile.mfma

        for ki in range(k_iters):
            ctx.set_index("wave", "ki", ki)

            # LDS read for this K-step
            if mfma.a_vgprs >= 4:
                ctx.module.add(st.DSLoadB128(
                    ctx.vgpr("v_a", 0, 4), ctx.vgpr("v_lds_read_a"),
                    comment=f"LDS read A k={ki}",
                ))
            else:
                for r in range(mfma.a_vgprs):
                    ctx.module.add(st.DSLoadB32(
                        ctx.vgpr("v_a", r, 1), ctx.vgpr("v_lds_read_a"),
                        comment=f"LDS read A[{r}] k={ki}",
                    ))
            if mfma.b_vgprs >= 4:
                ctx.module.add(st.DSLoadB128(
                    ctx.vgpr("v_b", 0, 4), ctx.vgpr("v_lds_read_b"),
                    comment=f"LDS read B k={ki}",
                ))
            else:
                for r in range(mfma.b_vgprs):
                    ctx.module.add(st.DSLoadB32(
                        ctx.vgpr("v_b", r, 1), ctx.vgpr("v_lds_read_b"),
                        comment=f"LDS read B[{r}] k={ki}",
                    ))

            # MFMAs for this K-step
            for mi in range(m_reps):
                for ni in range(n_reps):
                    ctx.set_index("wave", "mi", mi)
                    ctx.set_index("wave", "ni", ni)
                    acc_per = mfma.acc_vgprs
                    acc_off = (mi * n_reps + ni) * acc_per
                    ctx.module.add(st.MFMA(
                        instType=mfma.input_type, accType=mfma.acc_type,
                        m=mfma.m, n=mfma.n, k=mfma.k,
                        blocks=mfma.blocks, mfma1k=False,
                        acc=ctx.acc("acc_C", acc_off, acc_per),
                        a=ctx.vgpr("v_a", 0, mfma.a_vgprs),
                        b=ctx.vgpr("v_b", 0, mfma.b_vgprs),
                        comment=f"MFMA m{mi}_n{ni} k{ki}",
                    ))
    else:
        # Dry run: just update indices
        for ki in range(k_iters):
            ctx.set_index("wave", "ki", ki)
            for mi in range(m_reps):
                for ni in range(n_reps):
                    ctx.set_index("wave", "mi", mi)
                    ctx.set_index("wave", "ni", ni)


def _emit_global_loads(ctx, buf_name, addr_name, tag):
    """Emit buffer loads."""
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
    """Emit LDS stores."""
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
