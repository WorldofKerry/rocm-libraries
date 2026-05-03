"""Global memory -> LDS data loading strategies.

Building blocks for modular K-loop composition. Each loader handles
how A/B tile data moves from global memory into LDS, including
double-buffer toggling and synchronization.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from ..emit.context import AsmContext
from ..problem import GemmProblem, TileConfig

__all__ = ["GlobalLoader", "DTLLoader", "BufferLoader"]


class GlobalLoader(ABC):
    """Base class for global-to-LDS data movement strategies."""

    def __init__(self, ctx: AsmContext, tile: TileConfig, problem: GemmProblem) -> None:
        self.ctx = ctx
        self.tile = tile
        self.problem = problem
        self.elem = problem.element_bytes
        self.mfma = tile.mfma

        tpr = int(tile.unroll_k * self.elem) // 16
        rpl = tile.block_size // tpr
        self.num_loads_a = tile.wg_m // rpl
        self.num_loads_b = tile.wg_n // rpl
        self.k_stride = int(tile.unroll_k * self.elem)

    @abstractmethod
    def emit_first_tile(self, extra_vmcnt: int = 0) -> None:
        """Load first tile, wait, barrier. extra_vmcnt leaves that many in-flight."""

    @abstractmethod
    def emit_loads(self) -> None:
        """Issue loads for next K-tile (within K-loop, after advance)."""

    @abstractmethod
    def advance(self) -> None:
        """Advance global addresses/SRDs by unroll_k."""

    @abstractmethod
    def toggle_write(self) -> None:
        """Toggle LDS write bases for double-buffering."""

    @abstractmethod
    def emit_sync(self) -> None:
        """Wait for loads to land in LDS (strategy-specific)."""

    @property
    @abstractmethod
    def num_inflight(self) -> int:
        """Number of vmcnt-tracked loads in-flight after emit_loads."""

    def precompute_soffsets(self) -> None:
        """Precompute scalar offsets for loads. Override if needed."""
        pass


class DTLLoader(GlobalLoader):
    """Direct-To-LDS loader: buffer_load_dwordx4 ... lds.

    Data goes directly from global memory to LDS without touching VGPRs.
    Requires SRDs (s_srd_a, s_srd_b) and per-thread LDS write offsets
    set up by the setup phase. Uses vmcnt for synchronization.
    """

    def precompute_soffsets(self) -> None:
        """Precompute DTL scalar offsets to avoid cumulative s_add in K-loop."""
        ctx = self.ctx
        ctx.comment("Precompute DTL soffsets")
        for i in range(1, self.num_loads_a):
            name = f"s_dtl_soff_a{i}"
            ctx.alloc_sgpr_permanent(1, name)
            ctx.s_mul(ctx.sreg(name), ctx.sreg("s_soffset_a"), str(i),
                      comment=f"dtl_soff_a[{i}] = {i} * soffset_a")
        for i in range(1, self.num_loads_b):
            name = f"s_dtl_soff_b{i}"
            ctx.alloc_sgpr_permanent(1, name)
            ctx.s_mul(ctx.sreg(name), ctx.sreg("s_soffset_b"), str(i),
                      comment=f"dtl_soff_b[{i}] = {i} * soffset_b")
        ctx.raw("")

    def emit_first_tile(self, extra_vmcnt: int = 0) -> None:
        self._emit_dtl_loads_a()
        self._emit_dtl_loads_b()
        if extra_vmcnt > 0:
            self.ctx.s_waitcnt(f"vmcnt({extra_vmcnt})",
                               comment=f"wait DTL (leave {extra_vmcnt} in-flight)")
        else:
            self.ctx.s_waitcnt("vmcnt(0)", comment="wait DTL loads")
        self.ctx.s_barrier(comment="sync first tile")

    def emit_loads(self) -> None:
        self._emit_dtl_loads_a()
        self._emit_dtl_loads_b()

    def advance(self) -> None:
        ctx = self.ctx
        for srd in ["s_srd_a", "s_srd_b"]:
            ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                     ctx.sreg(srd, 0, 1), str(self.k_stride),
                     comment=f"{srd} += {self.k_stride}")
            ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                     ctx.sreg(srd, 1, 1), "0", comment="carry")

    def toggle_write(self) -> None:
        ctx = self.ctx
        ctx.inst("s_xor_b32", ctx.sreg("s_lds_wr_a_sg"),
                 ctx.sreg("s_lds_wr_a_sg"), ctx.sreg("s_lds_db_step"),
                 comment="wr_a ^= db")
        ctx.inst("s_xor_b32", ctx.sreg("s_lds_wr_b_sg"),
                 ctx.sreg("s_lds_wr_b_sg"), ctx.sreg("s_lds_db_step"),
                 comment="wr_b ^= db")

    def emit_sync(self) -> None:
        # DTL writes to OTHER buffer; preamble reads from CURRENT.
        pass

    @property
    def num_inflight(self) -> int:
        return self.num_loads_a + self.num_loads_b

    def _emit_dtl_loads_a(self) -> None:
        ctx, tile = self.ctx, self.tile
        elem = self.elem
        tpr = int(tile.unroll_k * elem) // 16
        rpl = tile.block_size // tpr
        lds_data_per_load = int(rpl * tile.unroll_k * elem)
        lds_stride = lds_data_per_load + tile.lds_pad

        ctx.inst("s_mov_b32", "m0", ctx.sreg("s_lds_wr_a_sg"),
                 comment="m0 = LDS base A")
        has_pre = ctx.has("s_dtl_soff_a1") if self.num_loads_a > 1 else False
        if not has_pre:
            ctx.s_mov(ctx.sreg("s_tmp0"), "0", comment="cumulative soffset A")
        for i in range(self.num_loads_a):
            soff = ("0" if i == 0 else ctx.sreg(f"s_dtl_soff_a{i}")) if has_pre else ctx.sreg("s_tmp0")
            ctx.inst("buffer_load_dwordx4",
                     ctx.vreg("v_dtl_off_a"), ctx.sreg("s_srd_a", 0, 4),
                     soff, "offen offset:0, lds", comment=f"DTL A[{i}]")
            if i < self.num_loads_a - 1:
                ctx.inst("s_add_u32", "m0", "m0", str(lds_stride),
                         comment=f"m0 += {lds_stride}")
                if not has_pre:
                    ctx.inst("s_add_u32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                             ctx.sreg("s_soffset_a"), comment="soffset += stride")

    def _emit_dtl_loads_b(self) -> None:
        ctx, tile = self.ctx, self.tile
        elem = self.elem
        tpr = int(tile.unroll_k * elem) // 16
        rpl = tile.block_size // tpr
        lds_data_per_load = int(rpl * tile.unroll_k * elem)
        lds_stride = lds_data_per_load + tile.lds_pad

        ctx.inst("s_mov_b32", "m0", ctx.sreg("s_lds_wr_b_sg"),
                 comment="m0 = LDS base B")
        has_pre = ctx.has("s_dtl_soff_b1") if self.num_loads_b > 1 else False
        if not has_pre:
            ctx.s_mov(ctx.sreg("s_tmp0"), "0", comment="cumulative soffset B")
        for i in range(self.num_loads_b):
            soff = ("0" if i == 0 else ctx.sreg(f"s_dtl_soff_b{i}")) if has_pre else ctx.sreg("s_tmp0")
            ctx.inst("buffer_load_dwordx4",
                     ctx.vreg("v_dtl_off_b"), ctx.sreg("s_srd_b", 0, 4),
                     soff, "offen offset:0, lds", comment=f"DTL B[{i}]")
            if i < self.num_loads_b - 1:
                ctx.inst("s_add_u32", "m0", "m0", str(lds_stride),
                         comment=f"m0 += {lds_stride}")
                if not has_pre:
                    ctx.inst("s_add_u32", ctx.sreg("s_tmp0"), ctx.sreg("s_tmp0"),
                             ctx.sreg("s_soffset_b"), comment="soffset += stride")


class BufferLoader(GlobalLoader):
    """Traditional global_load + ds_write loader.

    Data is loaded into VGPRs via global_load, then written to LDS
    via ds_write. Uses vmcnt for global loads, lgkmcnt for ds_writes.
    """

    def emit_first_tile(self, extra_vmcnt: int = 0) -> None:
        self._emit_global_loads()
        self.ctx.s_waitcnt("vmcnt(0)", comment="wait global loads")
        self._emit_ds_writes()
        self.ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS writes")
        self.ctx.s_barrier(comment="sync first tile")

    def emit_loads(self) -> None:
        self._emit_global_loads()

    def advance(self) -> None:
        """Advance SRD base addresses by unroll_k (same as DTLLoader)."""
        ctx = self.ctx
        for srd in ["s_srd_a", "s_srd_b"]:
            ctx.inst("s_add_u32", ctx.sreg(srd, 0, 1),
                     ctx.sreg(srd, 0, 1), str(self.k_stride),
                     comment=f"{srd} += {self.k_stride}")
            ctx.inst("s_addc_u32", ctx.sreg(srd, 1, 1),
                     ctx.sreg(srd, 1, 1), "0", comment="carry")

    def toggle_write(self) -> None:
        """Toggle LDS write double-buffer offset.

        Uses a scalar register (s_buf_wr_db) that toggles between
        0 and db_step. Added to the per-thread LDS offset in ds_write.
        """
        ctx = self.ctx
        # Also toggle the scalar bases (for DTL compat if mixed)
        ctx.inst("s_xor_b32", ctx.sreg("s_lds_wr_a_sg"),
                 ctx.sreg("s_lds_wr_a_sg"), ctx.sreg("s_lds_db_step"),
                 comment="wr_a ^= db")
        ctx.inst("s_xor_b32", ctx.sreg("s_lds_wr_b_sg"),
                 ctx.sreg("s_lds_wr_b_sg"), ctx.sreg("s_lds_db_step"),
                 comment="wr_b ^= db")
        # Toggle the BUF db offset
        if ctx.has("s_buf_wr_db"):
            ctx.inst("s_xor_b32", ctx.sreg("s_buf_wr_db"),
                     ctx.sreg("s_buf_wr_db"), ctx.sreg("s_lds_db_step"),
                     comment="buf_wr_db ^= db")

    def emit_sync(self) -> None:
        """Wait for LDS writes (issued during emit_produce), then barrier.

        ds_writes are issued inside the load skip check (emit_produce)
        so they only run when new data was loaded. This method just
        waits for them and synchronizes.
        """
        ctx = self.ctx
        ctx.s_waitcnt("lgkmcnt(0)", comment="wait LDS writes")
        ctx.s_barrier(comment="sync")

    @property
    def num_inflight(self) -> int:
        return 0  # all loads drained by emit_sync

    def _emit_global_loads(self) -> None:
        """Emit buffer_load_dwordx4 from global memory to VGPRs.

        Reuses the SRD (s_srd_a/b) and voffset (v_dtl_off_a/b) already
        set up by the DTL setup phase, but loads to VGPRs instead of LDS.
        Multiple loads use soffset for row group separation.
        """
        ctx = self.ctx
        tile = self.tile
        elem = self.elem
        tpr = int(tile.unroll_k * elem) // 16
        rpl = tile.block_size // tpr

        for name, num_loads in [("a", self.num_loads_a), ("b", self.num_loads_b)]:
            gload_name = f"v_gload_{name}"
            total_vgprs = num_loads * 4
            if not ctx.has(gload_name):
                ctx.alloc_vgpr_permanent(total_vgprs, gload_name)

            srd = ctx.sreg(f"s_srd_{name}", 0, 4)
            voff = ctx.vreg(f"v_dtl_off_{name}")
            soff_name = f"s_soffset_{name}"

            for i in range(num_loads):
                dst = ctx.vreg(gload_name, i * 4, 4)
                if i == 0:
                    ctx.inst("buffer_load_dwordx4", dst, voff, srd,
                             "0", "offen",
                             comment=f"global load {name.upper()}[{i}]")
                else:
                    # Use precomputed scalar offset for this load group
                    has_pre = ctx.has(f"s_dtl_soff_{name}{i}")
                    if has_pre:
                        soff = ctx.sreg(f"s_dtl_soff_{name}{i}")
                    else:
                        # Compute: soffset = i * soffset_stride
                        ctx.s_mul(ctx.sreg("s_tmp0"),
                                  ctx.sreg(soff_name), str(i),
                                  comment=f"soff = {i} * soffset_{name}")
                        soff = ctx.sreg("s_tmp0")
                    ctx.inst("buffer_load_dwordx4", dst, voff, srd,
                             soff, "offen",
                             comment=f"global load {name.upper()}[{i}]")

    def _emit_ds_writes(self) -> None:
        """Write loaded VGPRs to LDS.

        When swizzle is active, uses v_lds_wr_swz (pre-computed swizzled
        offset from setup phase). Otherwise uses v_lds_wr_off (LDS offset
        using unroll_k stride, NOT the global K stride in v_dtl_off).
        Both are combined with the scalar LDS base for double-buffering.
        """
        ctx = self.ctx
        tile = self.tile
        elem = self.elem
        tpr = int(tile.unroll_k * elem) // 16
        rpl = tile.block_size // tpr

        swz = tile.resolved_swizzle(elem)
        has_swz = swz is not None and hasattr(swz, 'pair_factor') and ctx.has("v_lds_wr_swz")

        if has_swz:
            pf = swz.pair_factor
            eff_stride = swz.effective_cols * 16
            lds_group_stride = (rpl // pf) * eff_stride
        else:
            row_stride = int(tile.unroll_k * elem)
            lds_group_stride = rpl * row_stride + tile.lds_pad

        for name in ["a", "b"]:
            gload_name = f"v_gload_{name}"
            if not ctx.has(gload_name):
                continue
            load = ctx.get(gload_name)
            num_loads = load.count // 4

            # v_lds_wr_off contains the full per-thread LDS offset
            # (thread_row * lds_stride + col_bytes). We add the
            # double-buffer offset from a dedicated VGPR that tracks
            # which buffer we're writing to.
            #
            # For A: offset = v_lds_wr_off
            # For B: offset = v_lds_wr_off + lds_b_offset
            b_offset = 0
            if name == "b":
                b_offset = int(self.tile.wg_m * self.tile.unroll_k * self.elem)

            # Start with per-thread offset
            if has_swz:
                ctx.v_mov(ctx.vreg("v_tmp0"), ctx.vreg("v_lds_wr_swz"),
                          comment=f"{name.upper()}: swizzled offset")
            elif ctx.has("v_lds_wr_off"):
                ctx.v_mov(ctx.vreg("v_tmp0"), ctx.vreg("v_lds_wr_off"),
                          comment=f"{name.upper()}: lds_wr_off")
            else:
                ctx.v_mov(ctx.vreg("v_tmp0"), ctx.vreg(f"v_dtl_off_{name}"),
                          comment=f"{name.upper()}: dtl_off")

            # Add B region offset
            if b_offset:
                ctx.s_mov(ctx.sreg("s_tmp0"), str(b_offset),
                          comment=f"B_off={b_offset}")
                ctx.v_add(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"),
                          ctx.sreg("s_tmp0"), comment="+ B_off")

            # Add double-buffer offset
            if ctx.has("s_buf_wr_db"):
                ctx.v_add(ctx.vreg("v_tmp0"), ctx.vreg("v_tmp0"),
                          ctx.sreg("s_buf_wr_db"),
                          comment="+ db offset")

            for i in range(num_loads):
                src = ctx.vreg(gload_name, i * 4, 4)
                offset = i * lds_group_stride
                ctx.ds_write(ctx.vreg("v_tmp0"), src, offset=offset,
                             width=4,
                             comment=f"LDS write {name.upper()}[{i}]")
