# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Assembly-emitting context: TileContext that produces .s text.

``AsmContext`` wraps ``TileContext`` and provides methods to emit
raw GCN assembly instructions as text strings.  The register allocation,
scoped lifetimes, and named bindings all come from TileContext.
The assembly output is collected in a list of lines.

This bridges the infrastructure (tile tree, transforms, named bindings)
with the working assembly backend -- no stinkytofu dependency needed.
"""
from __future__ import annotations

import math
from typing import List, Optional

from ..tile.context import TileContext

__all__ = ["AsmContext"]


class AsmContext(TileContext):
    """TileContext subclass that emits assembly text lines.

    Usage::

        ctx = AsmContext()
        ctx.alloc_vgpr_permanent(1, "v_tid")
        ctx.alloc_sgpr_permanent(2, "srd_A")

        ctx.comment("Load A pointer")
        ctx.inst("s_load_dwordx2", ctx.sreg("srd_A"),
                 ctx.sreg("s_kernarg"), "0", comment="A ptr")

        # After codegen, get the full .s text
        print(ctx.asm_text())
    """

    def __init__(self) -> None:
        super().__init__(module=None)  # no stinkytofu module
        self._lines: List[str] = []

    # -- Assembly text output -----------------------------------------------

    @property
    def lines(self) -> List[str]:
        return self._lines

    def asm_text(self) -> str:
        """Return the complete assembly text."""
        return "\n".join(self._lines) + "\n"

    # -- Emission helpers ---------------------------------------------------

    def raw(self, text: str) -> None:
        """Emit a raw line of text (directive, blank line, etc.)."""
        self._lines.append(text)

    def comment(self, text: str) -> None:
        """Emit a comment line."""
        self._lines.append(f"    // {text}")

    def label(self, name: str) -> None:
        """Emit a label."""
        self._lines.append(f"{name}:")

    def inst(self, mnemonic: str, *operands: str, comment: str = "") -> None:
        """Emit an instruction with optional comment.

        ``ctx.inst("v_add_u32", "v5", "v3", "v4", comment="a + b")``
        emits: ``    v_add_u32 v5, v3, v4  // a + b``
        """
        ops = ", ".join(str(o) for o in operands)
        text = f"    {mnemonic} {ops}" if ops else f"    {mnemonic}"
        if comment:
            self._lines.append(f"{text:<50s}// {comment}")
        else:
            self._lines.append(text)

    # -- Register name resolution -------------------------------------------

    def vreg(self, name: str, offset: int = 0, count: Optional[int] = None) -> str:
        """Resolve a VGPR binding to assembly syntax.

        ``ctx.vreg("v_tid")``       -> ``"v0"``
        ``ctx.vreg("v_gload_a", 0, 4)`` -> ``"v[16:19]"``
        """
        b = self.get(name)
        start = b.start + offset
        c = count if count is not None else b.count
        if c == 1:
            return f"v{start}"
        return f"v[{start}:{start + c - 1}]"

    def sreg(self, name: str, offset: int = 0, count: Optional[int] = None) -> str:
        """Resolve an SGPR binding to assembly syntax."""
        b = self.get(name)
        start = b.start + offset
        c = count if count is not None else b.count
        if c == 1:
            return f"s{start}"
        return f"s[{start}:{start + c - 1}]"

    def areg(self, name: str, offset: int = 0, count: Optional[int] = None) -> str:
        """Resolve an accumulator binding to assembly syntax."""
        b = self.get(name)
        start = b.start + offset
        c = count if count is not None else b.count
        if c == 1:
            return f"acc{start}"
        return f"acc[{start}:{start + c - 1}]"

    # -- Convenience instruction emitters -----------------------------------

    def v_mov(self, dst: str, src, comment: str = "") -> None:
        self.inst("v_mov_b32", dst, str(src), comment=comment)

    def v_add(self, dst: str, src0, src1, comment: str = "") -> None:
        self.inst("v_add_u32", dst, str(src0), str(src1), comment=comment)

    def v_sub(self, dst: str, src0, src1, comment: str = "") -> None:
        self.inst("v_sub_u32", dst, str(src0), str(src1), comment=comment)

    def v_mul(self, dst: str, src0, src1, comment: str = "") -> None:
        # Use shift for power-of-2 multipliers to avoid literal constant issues
        s0, s1 = str(src0), str(src1)
        try:
            val = int(s0)
            if val > 0 and (val & (val - 1)) == 0:
                import math
                self.v_lshl(dst, s1, int(math.log2(val)), comment=comment)
                return
        except (ValueError, TypeError):
            pass
        try:
            val = int(s1)
            if val > 0 and (val & (val - 1)) == 0:
                import math
                self.v_lshl(dst, s0, int(math.log2(val)), comment=comment)
                return
        except (ValueError, TypeError):
            pass
        s0 = self._ensure_not_literal(s0)
        s1 = self._ensure_not_literal(s1)
        self.inst("v_mul_lo_u32", dst, s0, s1, comment=comment)
    def _ensure_not_literal(self, val_str: str) -> str:
        """If val_str is a large literal, move it to s_tmp0 first.
        gfx950 v_mul_lo_u32 doesn't support literal operands > 64."""
        try:
            v = int(val_str)
            if v > 64 or v < -16:
                self.s_mov(self.sreg("s_tmp0"), val_str,
                           comment=f"literal {v} -> SGPR")
                return self.sreg("s_tmp0")
        except (ValueError, TypeError):
            pass
        return val_str

    def v_lshr(self, dst: str, src, shift: int, comment: str = "") -> None:
        self.inst("v_lshrrev_b32", dst, str(shift), str(src), comment=comment)

    def v_lshl(self, dst: str, src, shift: int, comment: str = "") -> None:
        self.inst("v_lshlrev_b32", dst, str(shift), str(src), comment=comment)

    def v_and(self, dst: str, src, mask: int, comment: str = "") -> None:
        self.inst("v_and_b32", dst, str(mask), str(src), comment=comment)

    def s_mov(self, dst: str, src, comment: str = "") -> None:
        self.inst("s_mov_b32", dst, str(src), comment=comment)

    def s_add(self, dst: str, src0, src1, comment: str = "") -> None:
        self.inst("s_add_u32", dst, str(src0), str(src1), comment=comment)

    def s_sub(self, dst: str, src0, src1, comment: str = "") -> None:
        self.inst("s_sub_u32", dst, str(src0), str(src1), comment=comment)

    def s_mul(self, dst: str, src0, src1, comment: str = "") -> None:
        self.inst("s_mul_i32", dst, str(src0), str(src1), comment=comment)

    def s_lshr(self, dst: str, src, shift: int, comment: str = "") -> None:
        self.inst("s_lshr_b32", dst, str(src), str(shift), comment=comment)

    def s_lshl(self, dst: str, src, shift: int, comment: str = "") -> None:
        self.inst("s_lshl_b32", dst, str(src), str(shift), comment=comment)

    def s_waitcnt(self, what: str, comment: str = "") -> None:
        self.inst("s_waitcnt", what, comment=comment)

    def s_barrier(self, comment: str = "barrier") -> None:
        self.inst("s_barrier", comment=comment)

    def ds_read(self, dst: str, addr: str, offset: int = 0,
                width: int = 1, comment: str = "") -> None:
        """Emit ds_read_b32/b64/b128 depending on width (in dwords)."""
        suffix = {1: "b32", 2: "b64", 4: "b128"}[width]
        if offset:
            self.inst(f"ds_read_{suffix}", dst, addr,
                      f"offset:{offset}", comment=comment)
        else:
            self.inst(f"ds_read_{suffix}", dst, addr, comment=comment)

    def ds_write(self, addr: str, src: str, offset: int = 0,
                 width: int = 1, comment: str = "") -> None:
        """Emit ds_write_b32/b64/b128."""
        suffix = {1: "b32", 2: "b64", 4: "b128"}[width]
        if offset:
            self.inst(f"ds_write_{suffix}", addr, src,
                      f"offset:{offset}", comment=comment)
        else:
            self.inst(f"ds_write_{suffix}", addr, src, comment=comment)

    def flat_load(self, dst: str, addr: str, width: int = 1,
                  comment: str = "") -> None:
        """Emit flat_load_dword/dwordx2/dwordx4."""
        suffix = {1: "dword", 2: "dwordx2", 4: "dwordx4"}[width]
        addr_fmt = f"[{addr}]" if "[" not in addr else addr
        self.inst(f"flat_load_{suffix}", dst, addr_fmt, comment=comment)

    def flat_store(self, addr: str, src: str, width: int = 1,
                   comment: str = "") -> None:
        """Emit flat_store_dword/short/dwordx2/dwordx4."""
        suffix = {1: "dword", 2: "short"}[width]
        addr_fmt = f"[{addr}]" if "[" not in addr else addr
        self.inst(f"flat_store_{suffix}", addr_fmt, src, comment=comment)
