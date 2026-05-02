"""Swizzle-aware LDS operand reader.

Building block for modular K-loop composition. Handles reading A/B
operands from LDS with swizzle support, double-buffer toggling,
and ping-pong A buffer management.
"""
from __future__ import annotations

from ..schedule.slot_placer import PlacedOp
from ..emit.context import AsmContext
from ..problem import GemmProblem, MfmaConfig, TileConfig

__all__ = ["LDSReader"]


def _a_off(mi: int, ki: int, tile: TileConfig, mfma: MfmaConfig, elem: float) -> int:
    """LDS byte offset for A operand at (mi, ki).

    With swizzle: ki is handled by base-register selection, so
    only the mi row offset is returned.
    """
    row_start = mi * mfma.m
    row_stride = int(tile.unroll_k * elem)
    if getattr(tile, 'lds_swizzle', False) or tile.resolved_swizzle(elem) is not None:
        return int(row_start * row_stride)
    pad_bytes = tile.lds_pad
    tpr = int(tile.unroll_k * elem) // 16
    rpl = (tile.waves_m * tile.waves_n * tile.wave_size) // tpr
    lines_crossed = row_start // rpl
    return int(row_start * row_stride + lines_crossed * pad_bytes + ki * mfma.k * elem)


def _b_off(ni: int, ki: int, tile: TileConfig, mfma: MfmaConfig, elem: float) -> int:
    """LDS byte offset for B operand at (ni, ki)."""
    row_start = ni * mfma.n
    row_stride = int(tile.unroll_k * elem)
    if getattr(tile, 'lds_swizzle', False) or tile.resolved_swizzle(elem) is not None:
        return int(row_start * row_stride)
    pad_bytes = tile.lds_pad
    tpr = int(tile.unroll_k * elem) // 16
    rpl = (tile.waves_m * tile.waves_n * tile.wave_size) // tpr
    lines_crossed = row_start // rpl
    return int(row_start * row_stride + lines_crossed * pad_bytes + ki * mfma.k * elem)


class LDSReader:
    """Swizzle-aware LDS operand reader with ping-pong A buffers.

    B operands are loaded once per K-tile (shared across all mi).
    A operands use ping-pong double-buffering: while mi=N computes
    with buffer X, mi=N+1's data is prefetched into buffer 1-X.

    Args:
        ctx: Assembly emission context.
        tile: TileConfig with mfma, unroll_k, etc.
        problem: GemmProblem (for element_bytes).
        swizzle: Optional Swizzle instance. If None, uses tile.resolved_swizzle().
    """

    def __init__(self, ctx: AsmContext, tile: TileConfig, problem: GemmProblem, swizzle: object = None) -> None:
        self.ctx = ctx
        self.tile = tile
        self.problem = problem
        self.elem = problem.element_bytes
        self.mfma = tile.mfma

        mr = tile.mfma_m_repeat
        nr = tile.mfma_n_repeat
        ki_count = tile.k_iterations
        av = self.mfma.a_vgprs
        bv = self.mfma.b_vgprs

        self.mr = mr
        self.nr = nr
        self.ki_count = ki_count
        self.av = av
        self.bv = bv

        # Resolve swizzle
        if swizzle is not None:
            self._swizzle = swizzle
        else:
            self._swizzle = tile.resolved_swizzle(self.elem)

        # Allocate operand registers
        self.b_names = {}
        for ni in range(nr):
            for ki in range(ki_count):
                name = f"v_b_s{ni}k{ki}"
                if not ctx.has(name):
                    ctx.alloc_vgpr_permanent(bv, name)
                self.b_names[(ni, ki)] = name

        self.a_names = {}
        for buf in range(2):
            for ki in range(ki_count):
                name = f"v_a_b{buf}k{ki}"
                if not ctx.has(name):
                    ctx.alloc_vgpr_permanent(av, name)
                self.a_names[(buf, ki)] = name

    def emit_read_a(self, mi: int, ki: int, buf: int) -> None:
        """Emit ds_read for A operand at (mi, ki) into buffer buf."""
        self._emit_swizzled_ds_read(
            dst=self.ctx.vreg(self.a_names[(buf, ki)], 0, self.av),
            base_reg=self.ctx.vreg("v_lds_rd_a"),
            offset=_a_off(mi, ki, self.tile, self.mfma, self.elem),
            ki=ki, width=self.av,
            comment=f"LR A m{mi}k{ki} b{buf}")

    def emit_read_b(self, ni: int, ki: int) -> None:
        """Emit ds_read for B operand at (ni, ki)."""
        self._emit_swizzled_ds_read(
            dst=self.ctx.vreg(self.b_names[(ni, ki)], 0, self.bv),
            base_reg=self.ctx.vreg("v_lds_rd_b"),
            offset=_b_off(ni, ki, self.tile, self.mfma, self.elem),
            ki=ki, width=self.bv,
            comment=f"LR B n{ni}k{ki}")

    def emit_preamble(self) -> int:
        """Emit preamble reads: B[ki=0], A[m0,k0], B[ki=1], A[m0,k1].

        Returns the number of in-flight lgkm reads after preamble.
        """
        ctx = self.ctx
        nr, ki_count = self.nr, self.ki_count

        ctx.comment("Early B reads (overlap with DTL)")
        for ni in range(nr):
            self.emit_read_b(ni, 0)

        ctx.comment("Preamble: A[m0] + B ki=1")
        self.emit_read_a(mi=0, ki=0, buf=0)
        inflight = nr + 1

        if ki_count > 1:
            for ni in range(nr):
                self.emit_read_b(ni, 1)
            self.emit_read_a(mi=0, ki=1, buf=0)
            inflight += nr + 1

        # Wait for first batch (B[k0] + A[m0,k0])
        first_batch = nr + 1
        remaining = inflight - first_batch
        wait_cnt = min(remaining, 15)
        ctx.s_waitcnt(f"lgkmcnt({wait_cnt})",
                      comment="wait B[ki=0] + A[m0,k0]")
        return inflight

    def emit_recompute_ki_bases(self) -> None:
        """Recompute per-ki LDS read base VGPRs (after toggle or first use)."""
        if self._swizzle is None or self.ki_count <= 1:
            return
        from .swizzle import DataLayout as SwzLayout
        swz_layout = SwzLayout(
            row_stride_bytes=int(self.tile.unroll_k * self.elem),
            mfma_k=self.mfma.k, mfma_m=self.mfma.m,
            elem_bytes=self.elem, wave_size=self.tile.wave_size)
        for ki in range(1, self.ki_count):
            step = ki * swz_layout.k_step
            xor_bytes = step * 16
            for matrix in ["a", "b"]:
                base = self.ctx.vreg(f"v_lds_rd_{matrix}")
                out = self.ctx.vreg(f"v_lds_rd_{matrix}_k{ki}")
                self.ctx.inst("v_xor_b32", out, base, str(xor_bytes),
                              comment=f"rd_{matrix}_k{ki} = rd_{matrix} ^ {xor_bytes}")

    def toggle_read(self) -> None:
        """Toggle LDS read bases for double-buffering + recompute ki bases."""
        ctx = self.ctx
        for matrix in ["a", "b"]:
            base_name = f"v_lds_rd_{matrix}"
            ctx.v_add(ctx.vreg(base_name),
                      ctx.sreg("s_lds_db_step"), ctx.vreg(base_name),
                      comment=f"rd_{matrix} += db")
            if self._swizzle is not None and self.ki_count > 1:
                from .swizzle import DataLayout as SwzLayout
                swz_layout = SwzLayout(
                    row_stride_bytes=int(self.tile.unroll_k * self.elem),
                    mfma_k=self.mfma.k, mfma_m=self.mfma.m,
                    elem_bytes=self.elem, wave_size=self.tile.wave_size)
                for ki in range(1, self.ki_count):
                    step = ki * swz_layout.k_step
                    xor_bytes = step * 16
                    ctx.inst("v_xor_b32",
                             ctx.vreg(f"v_lds_rd_{matrix}_k{ki}"),
                             ctx.vreg(base_name), str(xor_bytes),
                             comment=f"rd_{matrix}_k{ki} = rd_{matrix} ^ {xor_bytes}")

    def make_lr_ops(self, mi: int) -> list:
        """Return PlacedOps for A-prefetch reads for mi+1.

        These are ds_reads that load the NEXT mi's A data into the
        alternate ping-pong buffer while the current mi computes.
        """
        if mi >= self.mr - 1:
            return []
        next_buf = (mi + 1) % 2
        ops = []
        for ki in range(self.ki_count):
            def _mk(mi_: int = mi + 1, ki_: int = ki, buf_: int = next_buf) -> object:
                def emit() -> None:
                    self.emit_read_a(mi_, ki_, buf_)
                return emit
            ops.append(PlacedOp(
                emit_fn=_mk(), op_type="ds_read",
                comment=f"A m{mi+1}k{ki}"))
        return ops

    def make_suffix_ops(self) -> list:
        """Return PlacedOps for LDS toggle at end of iteration."""
        ops = [
            PlacedOp(
                emit_fn=lambda: self.toggle_read(),
                op_type="salu", comment="toggle_rd_a_b"),
            PlacedOp(
                emit_fn=lambda: self.ctx.inst(
                    "s_sub_u32", self.ctx.sreg("s_lds_db_step"),
                    "0", self.ctx.sreg("s_lds_db_step"),
                    comment="negate db"),
                op_type="salu", comment="negate_db"),
        ]
        return ops

    def _emit_swizzled_ds_read(self, dst: str, base_reg: str, offset: int, ki: int, width: int, comment: str) -> None:
        """Emit ds_read using per-ki base VGPR when swizzle is active."""
        ctx = self.ctx
        if self._swizzle is not None and ki > 0:
            if base_reg == ctx.vreg("v_lds_rd_a"):
                swz_reg = ctx.vreg(f"v_lds_rd_a_k{ki}")
            elif base_reg == ctx.vreg("v_lds_rd_b"):
                swz_reg = ctx.vreg(f"v_lds_rd_b_k{ki}")
            else:
                swz_reg = base_reg
            ctx.ds_read(dst, swz_reg, offset=offset,
                        width=width, comment=comment)
        else:
            ctx.ds_read(dst, base_reg, offset=offset,
                        width=width, comment=comment)
