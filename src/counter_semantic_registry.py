"""Sourced semantic definitions for Cassini network-frame octet counters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


COUNTER_SEMANTIC_REGISTRY_VERSION = "hpe-cassini-octets-v1"


class CounterScope(str, Enum):
    INTERFACE_WIDE = "interface_wide"


class CounterDirection(str, Enum):
    TX = "tx"
    RX = "rx"


class CounterQuality(str, Enum):
    OK = "ok"
    BAD = "bad"


class CounterResetBehavior(str, Enum):
    NEW_ACCUMULATION_EPOCH = "new_accumulation_epoch"


@dataclass(frozen=True)
class CounterSemanticDefinition:
    name: str
    scope: CounterScope
    direction: CounterDirection
    quality: CounterQuality
    semantics: str = "cumulative"
    unit: str = "bytes"
    measurement: str = "network-frame octets"
    counter_width_bits: int = 56
    wrap_modulus: int = 2**56
    non_destructive: bool = True
    reset_behavior: CounterResetBehavior = CounterResetBehavior.NEW_ACCUMULATION_EPOCH
    caveats: tuple[str, ...] = (
        "FCS/replay accounting is inferred or unresolved.",
        "Register-specific reset domains are unresolved.",
        "Network-frame octets are not application payload or physical-wire occupancy.",
        "Implementation policy: a decrease to zero may mark a reset epoch; all decreases are conservatively excluded, and 56-bit wrap remains unverified without an independent signal.",
    )
    source: str = "User-provided HPE Cassini definition (2026-10-06)"
    registry_version: str = COUNTER_SEMANTIC_REGISTRY_VERSION


def _definition(
    name: str,
    direction: CounterDirection,
    quality: CounterQuality,
) -> CounterSemanticDefinition:
    return CounterSemanticDefinition(
        name=name,
        scope=CounterScope.INTERFACE_WIDE,
        direction=direction,
        quality=quality,
    )


COUNTER_SEMANTIC_REGISTRY: Mapping[str, CounterSemanticDefinition] = MappingProxyType(
    {
        "hni_sts_tx_ok_octets": _definition(
            "hni_sts_tx_ok_octets",
            CounterDirection.TX,
            CounterQuality.OK,
        ),
        "hni_sts_rx_ok_octets": _definition(
            "hni_sts_rx_ok_octets",
            CounterDirection.RX,
            CounterQuality.OK,
        ),
        "hni_sts_tx_bad_octets": _definition(
            "hni_sts_tx_bad_octets",
            CounterDirection.TX,
            CounterQuality.BAD,
        ),
        "hni_sts_rx_bad_octets": _definition(
            "hni_sts_rx_bad_octets",
            CounterDirection.RX,
            CounterQuality.BAD,
        ),
    }
)


def counter_semantic_definition(name: str) -> CounterSemanticDefinition | None:
    return COUNTER_SEMANTIC_REGISTRY.get(name)


def ok_traffic_counter_names() -> tuple[str, ...]:
    return tuple(
        name
        for name, definition in COUNTER_SEMANTIC_REGISTRY.items()
        if definition.quality is CounterQuality.OK
    )
