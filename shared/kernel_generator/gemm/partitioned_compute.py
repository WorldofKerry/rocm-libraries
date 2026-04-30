"""Partition-scoped MFMA compute with VGPR recycling.

Splits the wave tile into partitions (e.g., 2x2 MFMA tiles per partition).
Each partition loads only its B tiles, computes, then frees B VGPRs for
the next partition.  A operands are double-buffered across mi-groups
within each partition.

This replaces _emit_scheduled_compute for large tiles (256x256) where
loading ALL B tiles upfront creates an unacceptable preamble stall.

Structure per partition (pm x pn MFMAs, ki K-iterations):
  1. Load B tiles for this partition (pn * ki reads)
  2. Load A[first_mi] for this partition (ki reads)
  3. Wait for all reads
  4. For each mi in partition:
     a. Prefetch A[next_mi] (if not last mi in partition)
     b. Execute pn * ki MFMAs
     c. Wait for A prefetch
  5. (VGPRs for this partition's B are dead after step 4)
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

from .asm_context import AsmContext
from .problem import TileConfig, GemmProblem

__all__ = ["emit_partitioned_compute"]


def emit_partitioned_compute(
    ctx: AsmContext,
    tile: TileConfig,
    problem: GemmProblem,
    partition_m: int = 2,
    partition_n: int = 2,
) -> None:
    """Emit partitioned MFMA compute with VGPR recycling.

    Args:
        ctx: Assembly context
        tile: Tile configuration
        problem: GEMM problem
        partition_m: MFMA tiles per partition along M
        partition_n: MFMA tiles per partition along N
    """
    mfma = tile.mfma
    elem = problem.element_bytes
    mr = tile.mfma_m_repeat
    nr = tile.mfma_n_repeat
    ki_count = tile.k_iterations
    av = mfma.a_vgprs
    bv = mfma.b_vgprs
    acc_per = mfma.acc_vgprs

    num_part_m = mr // partition_m
    num_part_n = nr // partition_n
    total_parts = num_part_m * num_part_n
    mfma_per_part = partition_m * partition_n * ki_count
    total_mfma = mr * nr * ki_count

    ctx.comment(f"Partitioned compute: {total_mfma} MFMAs in {total_parts} "
                f"partitions ({partition_m}x{partition_n}x{ki_count})")

    def b_off(ni, ki):
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (ni * mfma.n * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    def a_off(mi, ki):
        pad_e_ = tile.lds_pad // elem if tile.lds_pad > 0 else 0; return (mi * mfma.m * (tile.unroll_k + pad_e_) + ki * mfma.k) * elem

    # Allocate A double-buffer (persistent across partitions)
    a_names = {}
    for buf in range(2):
        for ki in range(ki_count):
            name = f"v_a_b{buf}k{ki}"
            if not ctx.has(name):
                ctx.alloc_vgpr_permanent(av, name)
            a_names[(buf, ki)] = name

    # B registers are allocated/freed per partition
    # We reuse the same register names across partitions
    b_pool_names = []
    for slot in range(partition_n * ki_count):
        name = f"v_bp_{slot}"
        if not ctx.has(name):
            ctx.alloc_vgpr_permanent(bv, name)
        b_pool_names.append(name)

    # Iterate partitions in column-major order (all mi for each ni group)
    for pn_idx in range(num_part_n):
        for pm_idx in range(num_part_m):
            part_id = pn_idx * num_part_m + pm_idx
            mi_start = pm_idx * partition_m
            ni_start = pn_idx * partition_n

            ctx.comment(f"--- Partition {part_id}: "
                        f"mi=[{mi_start}:{mi_start+partition_m}] "
                        f"ni=[{ni_start}:{ni_start+partition_n}] ---")

            # Map (local_ni, ki) -> B pool register name
            b_map: Dict[Tuple[int,int], str] = {}
            slot = 0
            for ki in range(ki_count):
                for local_ni in range(partition_n):
                    b_map[(local_ni, ki)] = b_pool_names[slot]
                    slot += 1

            # 1. Load B tiles for this partition
            for ki in range(ki_count):
                for local_ni in range(partition_n):
                    ni = ni_start + local_ni
                    name = b_map[(local_ni, ki)]
                    ctx.ds_read(ctx.vreg(name, 0, bv),
                                ctx.vreg("v_lds_rd_b"),
                                offset=b_off(ni, ki), width=bv,
                                comment=f"LR B n{ni}k{ki} p{part_id}")

            # 2. Load A[first_mi] for this partition
            cur_a = 0  # reset A double-buffer for each partition
            first_mi = mi_start
            for ki in range(ki_count):
                ctx.ds_read(ctx.vreg(a_names[(cur_a, ki)], 0, av),
                            ctx.vreg("v_lds_rd_a"),
                            offset=a_off(first_mi, ki), width=av,
                            comment=f"LR A m{first_mi}k{ki} b{cur_a} p{part_id}")

            # 3. Wait for all reads
            ctx.s_waitcnt("lgkmcnt(0)",
                          comment=f"wait partition {part_id} preamble")

            # 4. Per-mi loop within partition
            for local_mi in range(partition_m):
                mi = mi_start + local_mi

                # Prefetch A for next mi in partition (if not last)
                has_prefetch = local_mi < partition_m - 1
                if has_prefetch:
                    next_a = 1 - cur_a
                    next_mi = mi + 1
                    for ki in range(ki_count):
                        ctx.ds_read(
                            ctx.vreg(a_names[(next_a, ki)], 0, av),
                            ctx.vreg("v_lds_rd_a"),
                            offset=a_off(next_mi, ki), width=av,
                            comment=f"LR A m{next_mi}k{ki} b{next_a}")

                # MFMAs: iterate ki then ni
                for ki in range(ki_count):
                    for local_ni in range(partition_n):
                        ni = ni_start + local_ni
                        acc_off = (mi * nr + ni) * acc_per
                        ctx.inst(
                            mfma.instruction_name,
                            ctx.areg("acc_C", acc_off, acc_per),
                            ctx.vreg(a_names[(cur_a, ki)], 0, av),
                            ctx.vreg(b_map[(local_ni, ki)], 0, bv),
                            ctx.areg("acc_C", acc_off, acc_per),
                            comment=f"MFMA m{mi}_n{ni}_k{ki}")

                # Wait for A prefetch
                if has_prefetch:
                    ctx.s_waitcnt("lgkmcnt(0)",
                                  comment=f"wait A[{mi+1}]")
                    cur_a = next_a

            ctx.raw("")
