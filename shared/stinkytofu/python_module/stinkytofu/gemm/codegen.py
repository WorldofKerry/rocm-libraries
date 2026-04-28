# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Code generation with layered, overridable abstractions.

Architecture
------------
The codegen is split into composable layers.  Each layer handles one
concern and can be subclassed or replaced independently:

    RegisterAllocator   -- bump-pointer register pool
    ThreadMapping       -- coordinate-transform-driven index computation
    Emitter             -- thin wrappers that produce stinkytofu instructions
    GemmSchedule        -- structural decisions (k-loop, prefetch, barriers)
    GemmCodegen         -- assembles layers into a full LogicalModule

A user who wants to hand-optimise the MFMA block keeps the default
schedule but overrides ``Emitter.emit_mfma_block``.  A user who wants
a different k-loop structure overrides ``GemmSchedule.emit_k_loop``.
Raw stinkytofu ``LogicalModule`` sections can be injected anywhere via
``GemmCodegen.inject(label, module)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .problem import GemmProblem, TileConfig
from .transforms import Dim, Tile, Flatten, TileDescriptor

__all__ = [
    "RegisterAllocator",
    "ThreadMapping",
    "Emitter",
    "GemmSchedule",
    "GemmCodegen",
]


# ===================================================================
# Layer 1 -- Register allocator
# ===================================================================

class RegisterAllocator:
    """Bump-pointer register allocator with named ranges.

    Each ``alloc_*`` call returns the start index and records
    ``(pool, start, count)`` under *name* for later lookup.
    """

    def __init__(self) -> None:
        self._next = {"v": 0, "s": 0, "acc": 0}
        self._named: Dict[str, Tuple[str, int, int]] = {}

    # -- allocation ---------------------------------------------------------

    def alloc_vgpr(self, count: int, name: str = "") -> int:
        return self._alloc("v", count, name)

    def alloc_sgpr(self, count: int, name: str = "") -> int:
        return self._alloc("s", count, name)

    def alloc_acc(self, count: int, name: str = "") -> int:
        return self._alloc("acc", count, name)

    def _alloc(self, pool: str, count: int, name: str) -> int:
        start = self._next[pool]
        self._next[pool] += count
        if name:
            self._named[name] = (pool, start, count)
        return start

    # -- queries ------------------------------------------------------------

    @property
    def vgpr_count(self) -> int:
        return self._next["v"]

    @property
    def sgpr_count(self) -> int:
        return self._next["s"]

    @property
    def acc_count(self) -> int:
        return self._next["acc"]

    def get(self, name: str) -> Tuple[str, int, int]:
        """``(pool, start, count)`` for a named allocation."""
        return self._named[name]

    def summary(self) -> str:
        lines = [
            f"VGPRs: {self._next['v']}",
            f"SGPRs: {self._next['s']}",
            f"ACCs : {self._next['acc']}",
        ]
        for n, (pool, s, c) in self._named.items():
            lines.append(f"  {n}: {pool}[{s}:{s + c}]")
        return "\n".join(lines)


# ===================================================================
# Layer 2 -- Thread mapping  (transforms -> index arithmetic)
# ===================================================================

class ThreadMapping:
    """Translates coordinate transforms into concrete thread-to-element maps.

    Given a ``TileConfig`` it builds transform chains for:

    * **Global load A/B** -- which elements each thread loads from HBM.
    * **LDS layout** -- how the workgroup tile is stored in LDS.
    * **MFMA operand fetch** -- which LDS elements feed each MFMA.

    All mappings are expressed as ``TileDescriptor`` chains so the index
    arithmetic can be inspected, printed, or modified before emission.
    """

    def __init__(self, tile: TileConfig) -> None:
        self.tile = tile
        self._build()

    def _build(self) -> None:
        t = self.tile

        # -- Global-load cluster for A[M_wg, K_unroll] ----
        # Threads are arranged as a 2-D cluster:
        #   cluster_m x cluster_k  where cluster_k = vector_width
        #   cluster_m = block_size // cluster_k  (but bounded by wg_m)
        cluster_k = t.vector_width
        cluster_m = min(t.block_size, t.wg_m)
        # How many loads each thread must issue to cover the full tile
        elems_a = t.wg_m * t.unroll_k
        elems_per_load = t.block_size * cluster_k
        self.a_loads_per_thread = max(1, elems_a // elems_per_load)

        self.a_tile = TileDescriptor("A_global_load", [
            Dim("M_wg", t.wg_m),
            Dim("K_unroll", t.unroll_k),
        ])
        # Tile K into vector-width chunks
        self.a_tile.add_transform(Tile(
            Dim("K_unroll", t.unroll_k), cluster_k,
            outer_name="K_vec_id", inner_name="K_vec",
        ))

        # -- Global-load cluster for B[K_unroll, N_wg] ----
        cluster_n = t.vector_width
        elems_b = t.wg_n * t.unroll_k
        elems_per_load_b = t.block_size * cluster_n
        self.b_loads_per_thread = max(1, elems_b // elems_per_load_b)

        self.b_tile = TileDescriptor("B_global_load", [
            Dim("K_unroll", t.unroll_k),
            Dim("N_wg", t.wg_n),
        ])
        self.b_tile.add_transform(Tile(
            Dim("N_wg", t.wg_n), cluster_n,
            outer_name="N_vec_id", inner_name="N_vec",
        ))

        # -- MFMA tile descriptors ----
        self.m_desc = t.build_m_descriptor()
        self.n_desc = t.build_n_descriptor()
        self.k_desc = t.build_k_descriptor()

    # -- LDS size -----------------------------------------------------------

    @property
    def lds_size_bytes(self) -> int:
        """Total LDS required (A tile + B tile), in bytes."""
        elem = 2  # f16 = 2 bytes
        a = self.tile.wg_m * self.tile.unroll_k * elem
        b = self.tile.unroll_k * self.tile.wg_n * elem
        return a + b

    @property
    def lds_offset_b(self) -> int:
        """Byte offset where B's LDS region starts."""
        return self.tile.wg_m * self.tile.unroll_k * 2


# ===================================================================
# Layer 3 -- Emitter  (instruction-level, individually overridable)
# ===================================================================

class Emitter:
    """Emits stinkytofu instructions for each micro-operation.

    Every ``emit_*`` method writes into a caller-provided
    ``stinkytofu.LogicalModule``.  Override any method to inject
    hand-tuned assembly for that specific operation while keeping
    the rest of the kernel auto-generated.
    """

    def __init__(
        self,
        problem: GemmProblem,
        tile: TileConfig,
        regs: RegisterAllocator,
        mapping: ThreadMapping,
    ) -> None:
        self.problem = problem
        self.tile = tile
        self.regs = regs
        self.mapping = mapping

    # -- register helpers (import stinkytofu lazily) ------------------------

    def _v(self, name: str, offset: int = 0, count: int = 1):
        import stinkytofu as st
        _, start, _ = self.regs.get(name)
        return st.vgpr(start + offset, count)

    def _s(self, name: str, offset: int = 0, count: int = 1):
        import stinkytofu as st
        _, start, _ = self.regs.get(name)
        return st.sgpr(start + offset, count)

    def _acc(self, name: str, offset: int = 0, count: int = 1):
        import stinkytofu as st
        _, start, _ = self.regs.get(name)
        return st.accvgpr(start + offset, count)

    # -- individual operations ----------------------------------------------

    def emit_thread_indices(self, module) -> None:
        """Compute wave_id and lane_id from the thread ID."""
        import stinkytofu as st
        log2_ws = int(math.log2(self.tile.wave_size))
        module.add(st.VLShiftRightB32(
            self._v("v_wave_id"), self._v("v_tid"), st.Register(log2_ws),
            comment="wave_id = tid >> log2(wave_size)",
        ))
        module.add(st.VAndB32(
            self._v("v_lane_id"), self._v("v_tid"),
            st.Register(self.tile.wave_size - 1),
            comment="lane_id = tid & (wave_size - 1)",
        ))
        # wave_id -> (wave_m, wave_n)
        if self.tile.waves_n > 1:
            module.add(st.VAndB32(
                self._v("v_wave_n"), self._v("v_wave_id"),
                st.Register(self.tile.waves_n - 1),
                comment="wave_n = wave_id % waves_n",
            ))
            module.add(st.VLShiftRightB32(
                self._v("v_wave_m"), self._v("v_wave_id"),
                st.Register(int(math.log2(self.tile.waves_n))),
                comment="wave_m = wave_id / waves_n",
            ))
        else:
            module.add(st.VMovB32(
                self._v("v_wave_m"), self._v("v_wave_id"),
                comment="wave_m = wave_id (waves_n == 1)",
            ))

    def emit_global_load_a(self, module) -> None:
        """Emit buffer loads for the A tile."""
        import stinkytofu as st
        _, _, buf_count = self.regs.get("v_gload_a")
        self._emit_buffer_loads(module, "v_gload_a", "v_addr_a",
                                buf_count, "A")

    def emit_global_load_b(self, module) -> None:
        """Emit buffer loads for the B tile."""
        import stinkytofu as st
        _, _, buf_count = self.regs.get("v_gload_b")
        self._emit_buffer_loads(module, "v_gload_b", "v_addr_b",
                                buf_count, "B")

    def _emit_buffer_loads(self, module, buf: str, addr: str,
                           n_vgprs: int, tag: str) -> None:
        import stinkytofu as st
        chunk = 4  # buffer_load_b128
        for i in range(0, n_vgprs, chunk):
            cnt = min(chunk, n_vgprs - i)
            load_fn = {4: st.BufferLoadB128,
                       2: st.BufferLoadB64,
                       1: st.BufferLoadB32}[cnt]
            module.add(load_fn(
                self._v(buf, i, cnt), self._v(addr, 0, 1),
                comment=f"global load {tag}[{i}:{i+cnt}]",
            ))

    def emit_lds_write_a(self, module) -> None:
        """Store the globally-loaded A tile into LDS."""
        self._emit_ds_stores(module, "v_gload_a", "v_lds_write_a", "A")

    def emit_lds_write_b(self, module) -> None:
        self._emit_ds_stores(module, "v_gload_b", "v_lds_write_b", "B")

    def _emit_ds_stores(self, module, buf: str, addr: str, tag: str) -> None:
        import stinkytofu as st
        _, _, n = self.regs.get(buf)
        chunk = 4
        for i in range(0, n, chunk):
            cnt = min(chunk, n - i)
            store_fn = {4: st.DSStoreB128,
                        2: st.DSStoreB64,
                        1: st.DSStoreB32}[cnt]
            module.add(store_fn(
                self._v(addr), self._v(buf, i, cnt),
                comment=f"LDS write {tag}[{i}:{i+cnt}]",
            ))

    def emit_barrier(self, module) -> None:
        import stinkytofu as st
        module.add(st.SBarrier(comment="LDS barrier"))

    def emit_lds_read(self, module, k_iter: int) -> None:
        """Read MFMA operands from LDS for K-iteration *k_iter*."""
        import stinkytofu as st
        mfma = self.tile.mfma

        # A operand
        if mfma.a_vgprs >= 4:
            module.add(st.DSLoadB128(
                self._v("v_a", 0, 4), self._v("v_lds_read_a"),
                comment=f"LDS read A k={k_iter}",
            ))
        else:
            for r in range(mfma.a_vgprs):
                module.add(st.DSLoadB32(
                    self._v("v_a", r), self._v("v_lds_read_a"),
                    comment=f"LDS read A[{r}] k={k_iter}",
                ))

        # B operand
        if mfma.b_vgprs >= 4:
            module.add(st.DSLoadB128(
                self._v("v_b", 0, 4), self._v("v_lds_read_b"),
                comment=f"LDS read B k={k_iter}",
            ))
        else:
            for r in range(mfma.b_vgprs):
                module.add(st.DSLoadB32(
                    self._v("v_b", r), self._v("v_lds_read_b"),
                    comment=f"LDS read B[{r}] k={k_iter}",
                ))

    def emit_mfma_block(self, module, k_iter: int) -> None:
        """Emit MFMAs for all (m_repeat, n_repeat) tiles at K-step *k_iter*.

        Override this to inject hand-scheduled MFMA sequences.
        """
        import stinkytofu as st
        mfma = self.tile.mfma
        acc_per = mfma.acc_vgprs

        for mi in range(self.tile.mfma_m_repeat):
            for ni in range(self.tile.mfma_n_repeat):
                acc_off = (mi * self.tile.mfma_n_repeat + ni) * acc_per
                module.add(st.MFMA(
                    instType=mfma.input_type,
                    accType=mfma.acc_type,
                    m=mfma.m, n=mfma.n, k=mfma.k,
                    blocks=mfma.blocks, mfma1k=False,
                    acc=self._acc("acc_C", acc_off, acc_per),
                    a=self._v("v_a", 0, mfma.a_vgprs),
                    b=self._v("v_b", 0, mfma.b_vgprs),
                    comment=f"MFMA m{mi}_n{ni} k{k_iter}",
                ))

    def emit_init_acc(self, module) -> None:
        """Zero-initialise accumulator registers."""
        import stinkytofu as st
        _, start, count = self.regs.get("acc_C")
        for i in range(count):
            module.add(st.VAccvgprWriteB32(
                st.accvgpr(start + i), st.Register(0),
                comment=f"acc[{i}] = 0",
            ))

    def emit_store_d(self, module) -> None:
        """Move accumulators to VGPRs, convert, store to global D.

        Override this for a custom epilogue (e.g. fused activation).
        """
        import stinkytofu as st
        mfma = self.tile.mfma
        acc_per = mfma.acc_vgprs
        n_tiles = self.tile.mfma_m_repeat * self.tile.mfma_n_repeat
        for t_idx in range(n_tiles):
            for r in range(acc_per):
                module.add(st.VAccvgprReadB32(
                    self._v("v_store_tmp"),
                    self._acc("acc_C", t_idx * acc_per + r),
                    comment=f"acc->vgpr tile{t_idx}[{r}]",
                ))
                module.add(st.VCvtF32toF16(
                    self._v("v_store_tmp"), self._v("v_store_tmp"),
                    comment="cvt f32->f16",
                ))
                module.add(st.FlatStoreB16(
                    self._v("v_addr_d", 0, 2), self._v("v_store_tmp"),
                    comment=f"store D tile{t_idx}[{r}]",
                ))


# ===================================================================
# Layer 4 -- Schedule  (structural: what runs in what order)
# ===================================================================

class GemmSchedule:
    """Controls the macro-structure of the kernel.

    Override individual methods to change loop structure, prefetching
    strategy, or barrier placement without touching instruction emission.
    """

    def __init__(self, emitter: Emitter) -> None:
        self.emitter = emitter
        self.tile = emitter.tile

    def emit_prologue(self, module) -> None:
        """Thread indexing + accumulator init."""
        import stinkytofu as st
        module.add(st.Label("prologue"))
        self.emitter.emit_thread_indices(module)

        module.add(st.Label("init_acc"))
        self.emitter.emit_init_acc(module)

    def emit_global_load(self, module) -> None:
        """Load one K-tile of A and B from HBM."""
        import stinkytofu as st
        module.add(st.Label("global_load"))
        self.emitter.emit_global_load_a(module)
        self.emitter.emit_global_load_b(module)

    def emit_lds_store(self, module) -> None:
        """Write loaded data into LDS."""
        import stinkytofu as st
        module.add(st.Label("lds_write"))
        self.emitter.emit_lds_write_a(module)
        self.emitter.emit_lds_write_b(module)
        self.emitter.emit_barrier(module)

    def emit_k_loop(self, module) -> None:
        """The inner K-loop: LDS-read + MFMA for each K-step.

        Override this to implement software pipelining, double-buffering,
        or split-K strategies.
        """
        import stinkytofu as st
        module.add(st.Label("k_loop"))
        for ki in range(self.tile.k_iterations):
            self.emitter.emit_lds_read(module, ki)
            self.emitter.emit_mfma_block(module, ki)

    def emit_epilogue(self, module) -> None:
        """Store accumulators back to global memory."""
        import stinkytofu as st
        module.add(st.Label("epilogue"))
        self.emitter.emit_store_d(module)

    def emit_kernel(self, module) -> None:
        """Assemble the full kernel body.  Calls each phase in order."""
        self.emit_prologue(module)
        self.emit_global_load(module)
        self.emit_lds_store(module)
        self.emit_k_loop(module)
        self.emit_epilogue(module)


# ===================================================================
# Layer 5 -- GemmCodegen  (top-level orchestrator)
# ===================================================================

class GemmCodegen:
    """Top-level kernel generator.

    Composes all layers and supports injection of user-provided code
    at any labelled point.

    Basic usage::

        cg = GemmCodegen(problem, tile)
        module = cg.generate()       # -> stinkytofu.LogicalModule

    Custom MFMA block::

        class MyEmitter(Emitter):
            def emit_mfma_block(self, module, k_iter):
                # hand-optimised MFMA sequence
                ...

        cg = GemmCodegen(problem, tile, emitter_cls=MyEmitter)
        module = cg.generate()

    Custom K-loop::

        class MySchedule(GemmSchedule):
            def emit_k_loop(self, module):
                # software-pipelined loop
                ...

        cg = GemmCodegen(problem, tile, schedule_cls=MySchedule)
        module = cg.generate()

    Inject raw assembly::

        cg = GemmCodegen(problem, tile)
        cg.inject_before("k_loop", my_prefetch_module)
        module = cg.generate()
    """

    def __init__(
        self,
        problem: GemmProblem,
        tile: TileConfig,
        emitter_cls: type = Emitter,
        schedule_cls: type = GemmSchedule,
    ) -> None:
        problem.validate(tile)
        self.problem = problem
        self.tile = tile

        # Build layers
        self.regs = RegisterAllocator()
        self.mapping = ThreadMapping(tile)
        self._allocate_registers()

        self.emitter = emitter_cls(problem, tile, self.regs, self.mapping)
        self.schedule = schedule_cls(self.emitter)

        self._injections_before: Dict[str, list] = {}
        self._injections_after: Dict[str, list] = {}

    # -- register layout ----------------------------------------------------

    def _allocate_registers(self) -> None:
        t = self.tile
        mfma = t.mfma
        elem = self.problem.element_bytes

        # SGPRs: kernel arguments
        self.regs.alloc_sgpr(2, "srd_A")
        self.regs.alloc_sgpr(2, "srd_B")
        self.regs.alloc_sgpr(2, "srd_D")
        self.regs.alloc_sgpr(1, "s_M")
        self.regs.alloc_sgpr(1, "s_N")
        self.regs.alloc_sgpr(1, "s_K")
        self.regs.alloc_sgpr(1, "s_lda")
        self.regs.alloc_sgpr(1, "s_ldb")
        self.regs.alloc_sgpr(1, "s_ldd")
        self.regs.alloc_sgpr(1, "s_alpha")
        self.regs.alloc_sgpr(1, "s_beta")
        self.regs.alloc_sgpr(1, "s_k_iter")

        # VGPRs: thread / wave indices
        self.regs.alloc_vgpr(1, "v_tid")
        self.regs.alloc_vgpr(1, "v_wave_id")
        self.regs.alloc_vgpr(1, "v_lane_id")
        self.regs.alloc_vgpr(1, "v_wave_m")
        self.regs.alloc_vgpr(1, "v_wave_n")

        # VGPRs: global-load buffers
        a_elems = self.mapping.a_loads_per_thread * t.vector_width
        b_elems = self.mapping.b_loads_per_thread * t.vector_width
        a_vgprs = max(1, (a_elems * elem + 3) // 4)
        b_vgprs = max(1, (b_elems * elem + 3) // 4)
        self.regs.alloc_vgpr(a_vgprs, "v_gload_a")
        self.regs.alloc_vgpr(b_vgprs, "v_gload_b")

        # VGPRs: address computation
        self.regs.alloc_vgpr(2, "v_addr_a")
        self.regs.alloc_vgpr(2, "v_addr_b")
        self.regs.alloc_vgpr(2, "v_addr_d")
        self.regs.alloc_vgpr(1, "v_lds_write_a")
        self.regs.alloc_vgpr(1, "v_lds_write_b")
        self.regs.alloc_vgpr(1, "v_lds_read_a")
        self.regs.alloc_vgpr(1, "v_lds_read_b")

        # VGPRs: MFMA operand registers
        self.regs.alloc_vgpr(mfma.a_vgprs, "v_a")
        self.regs.alloc_vgpr(mfma.b_vgprs, "v_b")

        # VGPRs: store temporaries
        self.regs.alloc_vgpr(1, "v_store_tmp")

        # Accumulators
        acc_total = t.mfma_m_repeat * t.mfma_n_repeat * mfma.acc_vgprs
        self.regs.alloc_acc(acc_total, "acc_C")

    # -- injection API ------------------------------------------------------

    def inject_before(self, label: str, instructions) -> None:
        """Insert *instructions* (a LogicalModule or list) before *label*.

        *label* is one of: ``prologue``, ``init_acc``, ``global_load``,
        ``lds_write``, ``k_loop``, ``epilogue``.
        """
        self._injections_before.setdefault(label, []).append(instructions)

    def inject_after(self, label: str, instructions) -> None:
        """Insert *instructions* after *label*."""
        self._injections_after.setdefault(label, []).append(instructions)

    # -- generation ---------------------------------------------------------

    def kernel_name(self) -> str:
        t = self.tile
        return (
            f"gemm_{self.problem.dtype.value}"
            f"_{t.wg_m}x{t.wg_n}x{t.unroll_k}"
            f"_mfma{t.mfma.m}x{t.mfma.n}x{t.mfma.k}"
        )

    def generate(self):
        """Build the full kernel as a ``stinkytofu.LogicalModule``."""
        import stinkytofu as st

        module = st.LogicalModule(self.kernel_name())
        self.schedule.emit_kernel(module)
        module.add(st.Label("kernel_end"))
        return module

    def generate_dry(self) -> dict:
        """Return kernel metadata without importing stinkytofu.

        Useful for inspecting register usage, tile config, and transform
        chains without needing a built stinkytofu binary.
        """
        return {
            "name": self.kernel_name(),
            "registers": self.regs.summary(),
            "tile": self.tile.summary(),
            "mapping": {
                "a_loads_per_thread": self.mapping.a_loads_per_thread,
                "b_loads_per_thread": self.mapping.b_loads_per_thread,
                "lds_bytes": self.mapping.lds_size_bytes,
            },
            "m_desc": repr(self.mapping.m_desc),
            "n_desc": repr(self.mapping.n_desc),
            "k_desc": repr(self.mapping.k_desc),
        }
