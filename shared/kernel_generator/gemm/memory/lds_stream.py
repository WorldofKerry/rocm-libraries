"""Unified LDS stream interface.

All data that lives in double-buffered LDS (matrix data, MX scales,
future: bias, output scales) implements the LDSStream interface.
Streams are composed into an LDSBufferManager which handles layout,
toggle, and bulk operations.

See DESIGN_STREAMS.md for architecture rationale.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..emit.context import AsmContext
    from ..problem import TileConfig

__all__ = ["LDSStream", "LDSBufferManager"]


class LDSStream(ABC):
    """One data channel occupying a region in double-buffered LDS.

    Lifecycle per K-iteration:
        1. emit_global_loads() -- issue async loads (vmcnt-tracked)
        2. emit_lds_writes()   -- write VGPRs to LDS (after vmcnt drain)
        3. (barrier)           -- managed by LDSBufferManager
        4. emit_read(idx, ki)  -- ds_read from LDS (registered as graph ops)
        5. advance()           -- SRD += k_stride
        6. toggle_write(step)  -- write base += step
        7. toggle_read(step)   -- read base(s) += step

    Streams don't know about each other. The LDSBufferManager
    coordinates timing across all streams.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique stream name, e.g. 'data_a', 'scale_b'."""

    @property
    @abstractmethod
    def region_size(self) -> int:
        """Bytes this stream occupies per LDS buffer."""

    @property
    @abstractmethod
    def num_global_loads(self) -> int:
        """Number of vmcnt-tracked loads per emit_global_loads() call."""

    @property
    @abstractmethod
    def needs_lds_write(self) -> bool:
        """True if global loads go to VGPRs then ds_write (2-step).

        False if loads go directly to LDS (DTL / hardware path).
        """

    @property
    def has_reads(self) -> bool:
        """True if this stream has LDS read ops to register in the graph.

        Default True. Override to False for write-only streams.
        """
        return True

    # -- Setup (called once during kernel preamble) --

    @abstractmethod
    def setup(self, ctx: 'AsmContext', lds_offset: int) -> None:
        """Allocate registers and set up SRDs/offsets.

        Args:
            ctx: Assembly context.
            lds_offset: Byte offset of this stream's region within
                one LDS buffer (assigned by LDSBufferManager).
        """

    # -- Global load phase (async, vmcnt-tracked) --

    @abstractmethod
    def emit_global_loads(self, ctx: 'AsmContext') -> None:
        """Issue async global loads. Tracked by vmcnt."""

    @abstractmethod
    def emit_lds_writes(self, ctx: 'AsmContext') -> None:
        """Write VGPRs to LDS after vmcnt drain.

        No-op for DTL streams (data goes directly to LDS via hardware).
        """

    # -- LDS read phase (registered as graph ops) --

    @abstractmethod
    def read_op_count(self) -> int:
        """Number of distinct read ops this stream registers."""

    # -- K-loop lifecycle --

    @abstractmethod
    def advance(self, ctx: 'AsmContext') -> None:
        """Advance SRD(s) by one K-tile stride."""

    @abstractmethod
    def toggle_write(self, ctx: 'AsmContext') -> None:
        """Toggle LDS write base by the DB step.

        The DB step value is managed by LDSBufferManager and stored
        in s_lds_db_step. Streams use ADD-based toggle.
        """

    @abstractmethod
    def toggle_read(self, ctx: 'AsmContext') -> None:
        """Toggle LDS read base(s) by the DB step."""


class LDSBufferManager:
    """Manages N-way buffered LDS for multiple streams.

    Owns the LDS layout (region offsets), buffer count, DB step,
    and bulk lifecycle operations.

    Toggle mechanism by buffer count:
        N=1: no toggle
        N=2: ADD + negate (s_sub_u32 step, 0, step each iteration)
        N=3: increment + conditional wrap (future)
    """

    def __init__(self, streams: list, num_buffers: int = 2) -> None:
        self.streams = list(streams)
        self.num_buffers = num_buffers
        self._offsets: dict = {}  # stream.name -> offset within buffer
        self._buffer_size = 0

    def compute_layout(self) -> None:
        """Assign region offsets within one buffer and compute totals."""
        offset = 0
        for s in self.streams:
            self._offsets[s.name] = offset
            offset += s.region_size
        self._buffer_size = offset

    @property
    def buffer_size(self) -> int:
        """Total bytes per LDS buffer (all streams combined)."""
        return self._buffer_size

    @property
    def total_lds_bytes(self) -> int:
        """Total LDS allocation (buffer_size * num_buffers)."""
        return self._buffer_size * self.num_buffers

    @property
    def db_step(self) -> int:
        """DB toggle step size (= buffer_size for N=2)."""
        return self._buffer_size

    def stream_offset(self, stream_name: str) -> int:
        """Byte offset of a stream's region within one buffer."""
        return self._offsets[stream_name]

    @property
    def total_global_loads(self) -> int:
        """Total vmcnt-tracked loads across all streams."""
        return sum(s.num_global_loads for s in self.streams)

    # -- Bulk lifecycle operations --

    def setup_all(self, ctx: 'AsmContext') -> None:
        """Set up all streams with their assigned LDS offsets."""
        for s in self.streams:
            s.setup(ctx, self._offsets[s.name])

    def emit_all_loads(self, ctx: 'AsmContext') -> None:
        """Issue global loads for all streams (async)."""
        for s in self.streams:
            s.emit_global_loads(ctx)

    def emit_all_writes(self, ctx: 'AsmContext') -> None:
        """Write all pending VGPRs to LDS (call after vmcnt drain)."""
        for s in self.streams:
            if s.needs_lds_write:
                s.emit_lds_writes(ctx)

    def advance_all(self, ctx: 'AsmContext') -> None:
        """Advance all streams' SRDs."""
        for s in self.streams:
            s.advance(ctx)

    def toggle_all_writes(self, ctx: 'AsmContext') -> None:
        """Toggle all streams' write bases."""
        for s in self.streams:
            s.toggle_write(ctx)

    def toggle_all_reads(self, ctx: 'AsmContext') -> None:
        """Toggle all streams' read bases."""
        for s in self.streams:
            s.toggle_read(ctx)

    def emit_barrier(self, ctx: 'AsmContext') -> None:
        """Shared barrier after all writes land."""
        ctx.s_barrier(comment="sync all LDS streams")

    def emit_negate_step(self, ctx: 'AsmContext') -> None:
        """Negate DB step for ADD-based toggle (N=2 only)."""
        if self.num_buffers == 2:
            ctx.inst("s_sub_u32", ctx.sreg("s_lds_db_step"),
                     "0", ctx.sreg("s_lds_db_step"),
                     comment="negate db_step for next toggle")
