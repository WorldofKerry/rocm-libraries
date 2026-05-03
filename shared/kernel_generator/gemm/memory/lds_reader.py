"""Swizzle-aware LDS operand reader.

Building block for modular K-loop composition. Handles reading A/B
operands from LDS with swizzle support, double-buffer toggling,
and ping-pong A buffer management.
"""
from __future__ import annotations

from ..emit.context import AsmContext
from ..problem import GemmProblem, MfmaConfig, TileConfig

__all__ = ["LDSReader"]


def _a_off(mi: int, ki: int, tile: TileConfig, mfma: MfmaConfig, elem: float) -> int:
    """LDS byte offset for A operand at (mi, ki).

    With paired-row swizzle: returns 0 (mi handled by recomputation).
    With legacy swizzle: ki offset in per-ki base regs, mi row offset only.
    """
    row_start = mi * mfma.m
    row_stride = int(tile.unroll_k * elem)
    swz = tile.resolved_swizzle(elem)
    if swz is not None and hasattr(swz, 'pair_factor'):
        return 0  # mi handled by emit_recompute_a_for_mi
    if getattr(tile, 'lds_swizzle', False) or swz is not None:
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
    swz = tile.resolved_swizzle(elem)
    if swz is not None and hasattr(swz, 'pair_factor'):
        return 0  # ni handled by emit_recompute_b_for_ni
    if getattr(tile, 'lds_swizzle', False) or swz is not None:
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
                # Paired-row swizzle encodes ni in the base register,
                # so recompute before each ni's ki>0 read.
                self.emit_recompute_b_for_ni(ni)
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

    def emit_recompute_a_for_mi(self, mi: int) -> None:
        """Recompute v_lds_rd_a (and ki variants) for a new mi value.

        With paired-row swizzle, the column rotation changes per mi
        because lds_row = m_row // pair_factor changes. This method
        recomputes the read base from scratch for the given mi.

        Must be called before emit_read_a(mi, ...) when mi changes.
        """
        swz = self._swizzle
        if swz is None or not hasattr(swz, 'pair_factor'):
            return  # non-swizzle or non-paired: _a_off handles mi offset

        from .swizzle import DataLayout as SwzLayout, LDS_GFX950
        import math
        ctx = self.ctx
        tile = self.tile
        elem = self.elem
        pf = swz.pair_factor
        ec = swz.effective_cols
        oc = swz.orig_cols
        eff_stride = ec * 16

        swz_layout = SwzLayout(
            row_stride_bytes=int(tile.unroll_k * elem),
            mfma_k=self.mfma.k, mfma_m=self.mfma.m,
            elem_bytes=elem, wave_size=tile.wave_size)

        # m_row_base = mi * mfma_m (offset from the lane's initial m_row)
        # The initial v_lds_rd_a was set up for mi=0.
        # We need to recompute from: m_row = wave_m * m_per_wave + lane_row + mi * mfma_m
        # But we don't have wave_m or lane_row here. Instead, store the
        # initial m_row in a permanent VGPR during setup, and add mi*mfma_m.

        # Use v_tmp regs for recomputation
        m_row_delta = mi * self.mfma.m
        if not ctx.has("v_m_row_base_a"):
            return  # setup didn't allocate this -- fallback to _a_off

        # m_row = v_m_row_base_a + mi * mfma_m
        if m_row_delta > 0:
            if m_row_delta > 64:
                ctx.s_mov(ctx.sreg("s_tmp0"), str(m_row_delta),
                          comment=f"mi_delta={m_row_delta}")
                ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_m_row_base_a"),
                          ctx.sreg("s_tmp0"),
                          comment=f"m_row = m_row_base + {m_row_delta} (mi={mi})")
            else:
                ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_m_row_base_a"),
                          str(m_row_delta),
                          comment=f"m_row = m_row_base + {m_row_delta} (mi={mi})")
        else:
            ctx.v_mov(ctx.vreg("v_tmp2"), ctx.vreg("v_m_row_base_a"),
                      comment="m_row = m_row_base (mi=0)")

        # row_base = (m_row / pf) * eff_stride
        ctx.v_lshr(ctx.vreg("v_tmp3"), ctx.vreg("v_tmp2"),
                   int(math.log2(pf)),
                   comment=f"lds_row = m_row / {pf}")
        ctx.v_mul(ctx.vreg("v_tmp3"), str(eff_stride), ctx.vreg("v_tmp3"),
                  comment=f"row_base = lds_row * {eff_stride}")

        # Recompute swizzled read setup into v_lds_rd_a (and ki variants)
        ki_count = swz_layout.ki_count
        a_out = [ctx.vreg("v_lds_rd_a")]
        for ki in range(1, ki_count):
            a_out.append(ctx.vreg(f"v_lds_rd_a_k{ki}"))

        # Add read-side DB offset (tracks which LDS half to read from)
        if ctx.has("s_rd_db"):
            ctx.v_add(ctx.vreg("v_tmp3"), ctx.vreg("v_tmp3"),
                      ctx.sreg("s_rd_db"), comment="+ rd_db offset")

        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_tmp2"),      # m_row
                            ctx.vreg("v_k_group_a"), # k_group
                            ctx.vreg("v_tmp3"),      # row_base
                            a_out)

    def emit_recompute_b_for_ni(self, ni: int) -> None:
        """Recompute v_lds_rd_b (and ki variants) for a new ni value.

        Same logic as emit_recompute_a_for_mi but for B operand.
        """
        swz = self._swizzle
        if swz is None or not hasattr(swz, 'pair_factor'):
            return

        from .swizzle import DataLayout as SwzLayout, LDS_GFX950
        import math
        ctx = self.ctx
        tile = self.tile
        elem = self.elem
        pf = swz.pair_factor
        ec = swz.effective_cols
        eff_stride = ec * 16

        swz_layout = SwzLayout(
            row_stride_bytes=int(tile.unroll_k * elem),
            mfma_k=self.mfma.k, mfma_m=self.mfma.m,
            elem_bytes=elem, wave_size=tile.wave_size)

        n_row_delta = ni * self.mfma.n
        if not ctx.has("v_n_row_base_b"):
            return

        if n_row_delta > 0:
            if n_row_delta > 64:
                ctx.s_mov(ctx.sreg("s_tmp0"), str(n_row_delta),
                          comment=f"ni_delta={n_row_delta}")
                ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_n_row_base_b"),
                          ctx.sreg("s_tmp0"),
                          comment=f"n_row = n_row_base + {n_row_delta} (ni={ni})")
            else:
                ctx.v_add(ctx.vreg("v_tmp2"), ctx.vreg("v_n_row_base_b"),
                          str(n_row_delta),
                          comment=f"n_row = n_row_base + {n_row_delta} (ni={ni})")
        else:
            ctx.v_mov(ctx.vreg("v_tmp2"), ctx.vreg("v_n_row_base_b"),
                      comment="n_row = n_row_base (ni=0)")

        ctx.v_lshr(ctx.vreg("v_tmp3"), ctx.vreg("v_tmp2"),
                   int(math.log2(pf)),
                   comment=f"lds_row = n_row / {pf}")
        ctx.v_mul(ctx.vreg("v_tmp3"), str(eff_stride), ctx.vreg("v_tmp3"),
                  comment=f"row_base = lds_row * {eff_stride}")

        # Add B LDS offset + read-side DB offset
        lds_b_off = int(tile.wg_m * tile.unroll_k * elem)
        ctx.s_mov(ctx.sreg("s_tmp0"), str(lds_b_off),
                  comment=f"lds_b_off={lds_b_off}")
        ctx.v_add(ctx.vreg("v_tmp3"), ctx.vreg("v_tmp3"),
                  ctx.sreg("s_tmp0"), comment="+ lds_b_offset")
        if ctx.has("s_rd_db"):
            ctx.v_add(ctx.vreg("v_tmp3"), ctx.vreg("v_tmp3"),
                      ctx.sreg("s_rd_db"), comment="+ rd_db offset")

        ki_count = swz_layout.ki_count
        b_out = [ctx.vreg("v_lds_rd_b")]
        for ki in range(1, ki_count):
            b_out.append(ctx.vreg(f"v_lds_rd_b_k{ki}"))

        swz.emit_read_setup(ctx, swz_layout, LDS_GFX950,
                            ctx.vreg("v_tmp2"),
                            ctx.vreg("v_k_group_a"),  # same k_group
                            ctx.vreg("v_tmp3"),
                            b_out)

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
                if xor_bytes > 64:
                    self.ctx.s_mov(self.ctx.sreg("s_tmp0"), str(xor_bytes),
                                  comment=f"xor val {xor_bytes}")
                    self.ctx.inst("v_xor_b32", out, base,
                                 self.ctx.sreg("s_tmp0"),
                                 comment=f"rd_{matrix}_k{ki} = rd_{matrix} ^ {xor_bytes}")
                else:
                    self.ctx.inst("v_xor_b32", out, base, str(xor_bytes),
                                 comment=f"rd_{matrix}_k{ki} = rd_{matrix} ^ {xor_bytes}")

    def toggle_read(self) -> None:
        """Toggle LDS read bases for double-buffering + recompute ki bases.

        For paired-row swizzle: toggle a scalar DB register (s_rd_db)
        instead of XORing v_lds_rd_a/b. The recompute functions
        add s_rd_db to the computed address, so the toggle is preserved
        even when the base register is rewritten from scratch.

        For non-swizzle: XOR the base VGPRs directly (no recompute).
        """
        ctx = self.ctx
        swz = self._swizzle
        if swz is not None and hasattr(swz, 'pair_factor'):
            # Paired-row swizzle: toggle scalar DB state
            ctx.inst("s_xor_b32", ctx.sreg("s_rd_db"),
                     ctx.sreg("s_rd_db"), ctx.sreg("s_lds_db_step"),
                     comment="s_rd_db ^= db_step (toggle read buffer)")
        else:
            for matrix in ["a", "b"]:
                base_name = f"v_lds_rd_{matrix}"
                ctx.inst("v_xor_b32", ctx.vreg(base_name),
                         ctx.sreg("s_lds_db_step"), ctx.vreg(base_name),
                         comment=f"rd_{matrix} ^= db")
                if self._swizzle is not None and self.ki_count > 1:
                    from .swizzle import DataLayout as SwzLayout
                    swz_layout = SwzLayout(
                        row_stride_bytes=int(self.tile.unroll_k * self.elem),
                        mfma_k=self.mfma.k, mfma_m=self.mfma.m,
                        elem_bytes=self.elem, wave_size=self.tile.wave_size)
                    for ki in range(1, self.ki_count):
                        step = ki * swz_layout.k_step
                        xor_bytes = step * 16
                        if xor_bytes > 64:
                            ctx.s_mov(ctx.sreg("s_tmp0"), str(xor_bytes),
                                      comment=f"xor val {xor_bytes}")
                            ctx.inst("v_xor_b32",
                                     ctx.vreg(f"v_lds_rd_{matrix}_k{ki}"),
                                     ctx.vreg(base_name), ctx.sreg("s_tmp0"),
                                     comment=f"rd_{matrix}_k{ki} = rd_{matrix} ^ {xor_bytes}")
                        else:
                            ctx.inst("v_xor_b32",
                                     ctx.vreg(f"v_lds_rd_{matrix}_k{ki}"),
                                     ctx.vreg(base_name), str(xor_bytes),
                                     comment=f"rd_{matrix}_k{ki} = rd_{matrix} ^ {xor_bytes}")

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
