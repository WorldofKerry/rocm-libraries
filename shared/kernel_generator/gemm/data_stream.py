"""DataStream: generalized data loading abstraction for GEMM kernels.

Each DataStream represents a data dependency of MFMA tiles with a defined
sharing pattern across the (mi, ni, ki) index space. The partition plan
uses DataStreams to determine what loads are needed at subtile boundaries,
and the SlotPlacer schedules them alongside MFMA compute.

Examples:
  - A operand:  varies with (mi, ki), shared across ni  -> reload at subtile boundary
  - B operand:  varies with (ni, ki), shared across mi  -> load once in preamble
  - Scale A:    varies with (mi, ki), shared across ni  -> same pattern as A
  - Scale B:    varies with (ni, ki), shared across mi  -> same pattern as B
  - Bias A:     varies with (mi),     shared across ni, ki -> reload at subtile boundary
  - Bias B:     varies with (ni),     shared across mi, ki -> load once in preamble
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple

__all__ = [
    "DataStream", "StreamSource", "StreamBuffering", "StreamSchedule",
]


class StreamSource(Enum):
    """Where the data lives before it reaches VGPRs."""
    LDS = auto()      # ds_read -> VGPR (uses lgkmcnt)
    GLOBAL = auto()   # buffer_load -> VGPR (uses vmcnt)
    SGPR = auto()     # s_buffer_load -> SGPR -> v_mov (uses lgkmcnt)


class StreamBuffering(Enum):
    """How VGPRs are managed across subtiles."""
    NONE = auto()       # single set, overwritten each use
    PING_PONG = auto()  # two sets, alternated at subtile boundaries
    PERSISTENT = auto() # loaded once, never overwritten (e.g., B operands)


@dataclass
class DataStream:
    """A data dependency with defined sharing and load properties.

    The key insight: the sharing pattern determines WHEN loads happen
    relative to the partition schedule, not HOW the data is loaded.
    The source and load_op determine HOW.

    Attributes:
        name: Human-readable identifier ("A", "scale_a", "bias_a", ...).
        varies_with: Set of MFMA indices this stream depends on.
                     Subset of {"mi", "ni", "ki"}.
        source: Where data comes from (LDS, global, SGPR).
        load_width: Bytes per load operation.
        vgprs_per_load: VGPRs consumed per load.
        buffering: VGPR management strategy.
        loads_per_index: How many loads per varying index value.
                         E.g., A needs ki_count loads per mi.
                         Scale needs 1 load per mi (dword = all ki packed).
    """
    name: str
    varies_with: Set[str]
    source: StreamSource
    load_width: int = 4           # bytes per load
    vgprs_per_load: int = 1       # VGPRs consumed
    buffering: StreamBuffering = StreamBuffering.NONE
    loads_per_index: int = 1      # loads per varying index value

    @property
    def varies_with_mi(self) -> bool:
        return "mi" in self.varies_with

    @property
    def varies_with_ni(self) -> bool:
        return "ni" in self.varies_with

    @property
    def varies_with_ki(self) -> bool:
        return "ki" in self.varies_with

    @property
    def waitcnt_type(self) -> str:
        """Which wait counter this stream's loads affect."""
        if self.source == StreamSource.LDS:
            return "lgkmcnt"
        elif self.source == StreamSource.GLOBAL:
            return "vmcnt"
        else:  # SGPR
            return "lgkmcnt"

    @property
    def reload_at_subtile(self) -> bool:
        """Whether this stream needs fresh loads at each subtile boundary.

        Streams that vary with mi need reload when mi changes (subtile
        boundary). Streams that only vary with ni/ki are loaded once.
        """
        return self.varies_with_mi

    @property
    def load_in_preamble(self) -> bool:
        """Whether this stream is loaded once before the MFMA body.

        Streams that vary with ni (but not mi) are loaded in the preamble
        and reused across all subtiles.
        """
        return self.varies_with_ni and not self.varies_with_mi


@dataclass
class StreamSchedule:
    """Per-subtile load schedule derived from DataStreams.

    For each subtile, lists which streams need loads and how many.
    The SlotPlacer converts these into PlacedOp paths.
    """
    # Preamble loads: streams loaded once before first MFMA
    preamble_streams: List[Tuple[DataStream, int]]  # (stream, num_loads)

    # Per-subtile prefetch: streams loaded during subtile N for subtile N+1
    # Key: subtile index. Value: list of (stream, num_loads)
    subtile_prefetch: Dict[int, List[Tuple[DataStream, int]]]

    @staticmethod
    def from_streams(streams: List[DataStream],
                     mr: int, nr: int, ki_count: int,
                     partition_m: int = 2) -> StreamSchedule:
        """Derive load schedule from DataStream definitions.

        Args:
            streams: All DataStreams for this kernel.
            mr: MFMA M repeat count.
            nr: MFMA N repeat count.
            ki_count: K iterations per unroll.
            partition_m: Subtile size (mi values per subtile).
        """
        num_subtiles = mr // partition_m
        preamble = []
        prefetch: Dict[int, List[Tuple[DataStream, int]]] = {}

        for stream in streams:
            if stream.load_in_preamble:
                # Varies with ni: load all ni values once
                # Total loads = nr * ki_count * loads_per_index
                # (or nr * loads_per_index if ki is packed into the load)
                if stream.varies_with_ki:
                    n_loads = nr * ki_count * stream.loads_per_index
                else:
                    n_loads = nr * stream.loads_per_index
                preamble.append((stream, n_loads))

            elif stream.reload_at_subtile:
                # Varies with mi: load partition_m values per subtile
                # Prefetch subtile N+1 during subtile N's MFMAs
                for st in range(num_subtiles):
                    next_st = st + 1
                    if next_st >= num_subtiles:
                        continue  # last subtile doesn't prefetch

                    if stream.varies_with_ki:
                        n_loads = partition_m * ki_count * stream.loads_per_index
                    else:
                        n_loads = partition_m * stream.loads_per_index

                    if st not in prefetch:
                        prefetch[st] = []
                    prefetch[st].append((stream, n_loads))

        return StreamSchedule(
            preamble_streams=preamble,
            subtile_prefetch=prefetch,
        )

    def summary(self) -> str:
        lines = ["StreamSchedule:"]
        lines.append("  Preamble:")
        for stream, n in self.preamble_streams:
            lines.append(f"    {stream.name}: {n} loads ({stream.source.name})")
        for st, loads in sorted(self.subtile_prefetch.items()):
            lines.append(f"  Subtile {st} prefetches:")
            for stream, n in loads:
                lines.append(f"    {stream.name}: {n} loads ({stream.source.name})")
        return "\n".join(lines)


# ── Factory functions for common GEMM streams ────────────────────────

def make_gemm_streams(mfma, ki_count: int,
                      use_scales: bool = False,
                      scale_source: StreamSource = StreamSource.GLOBAL
                      ) -> List[DataStream]:
    """Create DataStreams for a standard GEMM kernel.

    Returns streams for A operand, B operand, and optionally scales.
    """
    streams = [
        DataStream(
            name="A",
            varies_with={"mi", "ki"},
            source=StreamSource.LDS,
            load_width=mfma.a_vgprs * 4,
            vgprs_per_load=mfma.a_vgprs,
            buffering=StreamBuffering.PING_PONG,
            loads_per_index=1,  # one ds_read per (mi, ki)
        ),
        DataStream(
            name="B",
            varies_with={"ni", "ki"},
            source=StreamSource.LDS,
            load_width=mfma.b_vgprs * 4,
            vgprs_per_load=mfma.b_vgprs,
            buffering=StreamBuffering.PERSISTENT,
            loads_per_index=1,
        ),
    ]

    if use_scales and getattr(mfma, 'is_mx', False):
        streams.extend([
            DataStream(
                name="scale_a",
                varies_with={"mi"},  # ki packed into dword (4 bytes = 4 K-blocks)
                source=scale_source,
                load_width=4,       # 1 dword = 4 E8M0 scale bytes
                vgprs_per_load=1,
                buffering=StreamBuffering.NONE,
                loads_per_index=1,  # 1 dword per mi
            ),
            DataStream(
                name="scale_b",
                varies_with={"ni"},  # ki packed, same as scale_a
                source=scale_source,
                load_width=4,
                vgprs_per_load=1,
                buffering=StreamBuffering.PERSISTENT,
                loads_per_index=1,
            ),
        ])

    return streams
