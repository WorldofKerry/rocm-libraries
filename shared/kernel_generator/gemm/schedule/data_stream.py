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

from dataclasses import dataclass
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

def make_gemm_streams(mfma: object, ki_count: int,
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


# ── Cross-iteration software-pipelined prefetch ─────────────────────

@dataclass
class PrefetchOp:
    """A single prefetch load to be placed after a register's last consumer."""
    reg_name: str           # register being prefetched
    emit_fn: Callable       # function to emit the buffer_load
    earliest_slot: int      # slot index after which this op can be placed
    op_type: str = "buffer_load"


def compute_register_last_use(
    mfma_ops: list,
    reg_names: list[str],
) -> dict[str, int]:
    """Find the last MFMA index that reads each register.

    Scans the reads_regs metadata on each PlacedOp to determine
    when each register is last consumed. This drives prefetch
    placement: a register can be overwritten (by a prefetch load
    for the next iteration) only after its last consumer.

    Args:
        mfma_ops: Ordered list of MFMA PlacedOps with reads_regs metadata.
        reg_names: Register names to track.

    Returns:
        Dict mapping reg_name -> last MFMA index that reads it.
        Registers not found in any MFMA get index -1.
    """
    last_use = {name: -1 for name in reg_names}
    for idx, op in enumerate(mfma_ops):
        for reg in op.reads_regs:
            if reg in last_use:
                last_use[reg] = idx
    return last_use


def build_prefetch_path(
    mfma_ops: list,
    prefetch_loads: list[PrefetchOp],
    srd_advance_fn: Optional[Callable] = None,
    module_id: int = 300,
) -> "Path":
    """Build a SlotPlacer Path for cross-iteration prefetch loads.

    Each PrefetchOp has an earliest_slot (derived from register
    last-use analysis). The path places SRD advancement first,
    then each prefetch load in slot order.

    The resulting Path can be passed to SlotPlacer for automatic
    placement alongside MFMAs.

    Args:
        mfma_ops: The MFMA op list (for slot count reference).
        prefetch_loads: PrefetchOps sorted by earliest_slot.
        srd_advance_fn: Optional function to emit SRD advancement
                        instructions. Placed before all loads.
        module_id: Module ID for the path.

    Returns:
        A Path suitable for SlotPlacer placement.
    """
    from .slot_placer import PlacedOp as POp, Path

    ops = []
    if srd_advance_fn:
        ops.append(POp(
            emit_fn=srd_advance_fn, op_type="salu",
            module_id=module_id, comment="advance_scale_srds"))

    # Sort by earliest_slot to maintain placement order
    for pf in sorted(prefetch_loads, key=lambda p: p.earliest_slot):
        ops.append(POp(
            emit_fn=pf.emit_fn, op_type=pf.op_type,
            module_id=module_id, comment=f"prefetch_{pf.reg_name}"))

    return Path(ops=ops, reverse=False, module_id=module_id)


def place_prefetch_path(placer: object, path: object, mfma_ops: list, prefetch_loads: list) -> None:
    """Place prefetch ops in the SlotPlacer, respecting earliest_slot constraints.

    Unlike regular paths where ops are placed sequentially, prefetch ops
    have per-op placement constraints: each op must go into a slot after
    its earliest_slot (which corresponds to the last MFMA that reads the
    register being overwritten).

    Args:
        placer: SlotPlacer instance.
        path: The Path from build_prefetch_path().
        mfma_ops: MFMA op list.
        prefetch_loads: PrefetchOps with earliest_slot constraints.
    """
    # Place SRD advance op(s) first, early in the last subtile
    srd_ops = [op for op in path.ops if op.op_type == "salu"]
    load_ops = [op for op in path.ops if op.op_type != "salu"]

    # Place SRD advance ops as early as possible in the last subtile
    if srd_ops:
        # Find a reasonable start: before the first prefetch load's earliest_slot
        earliest = min(pf.earliest_slot for pf in prefetch_loads) if prefetch_loads else 0
        target_slot = max(0, earliest * 2 - 4)  # a few slots before first load
        for op in srd_ops:
            placed = False
            for s in range(target_slot, len(mfma_ops) * 2):
                if placer._can_place(s, op):
                    placer._slots[s].append(op)
                    if placer._on_place:
                        placer._on_place(placer, s, op)
                    placed = True
                    break
            if not placed:
                placer.leftovers.append(op)

    # Place each prefetch load after its earliest_slot
    pf_iter = iter(sorted(prefetch_loads, key=lambda p: p.earliest_slot))
    for op, pf in zip(load_ops, pf_iter):
        target_slot = pf.earliest_slot * 2 + 1  # slot after the MFMA
        placed = False
        for s in range(target_slot, len(mfma_ops) * 2):
            if placer._can_place(s, op):
                placer._slots[s].append(op)
                if placer._on_place:
                    placer._on_place(placer, s, op)
                placed = True
                break
        if not placed:
            placer.leftovers.append(op)
