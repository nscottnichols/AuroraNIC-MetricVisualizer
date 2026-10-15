"""Per-node NIC balance analysis built from full-resolution traffic results."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
import math

from traffic_analysis import (
    AnalysisStatus,
    SeriesQualityFlag,
    TrafficAnalysis,
    TrafficDirection,
)


TRAFFIC_SHARE_FORMULA = "nic_total_bytes / node_total_bytes"
ACTIVE_NIC_RULE = "mean_rate_bytes_per_s > activity_threshold_bytes_per_s"
ACTIVE_NIC_DENOMINATOR = "known NIC inventory"
EFFECTIVE_NIC_COUNT_FORMULA = "1 / sum((nic_total_bytes / node_total_bytes)^2)"
MAX_MIN_RATIO_FORMULA = "max(nic_total_bytes) / min(nic_total_bytes)"
SPATIAL_QUANTILE_INTERPOLATION = (
    "nearest-rank upper order statistic across NIC total bytes; "
    "temporal rate quantiles are never averaged"
)
CROSS_NIC_CV_FORMULA = (
    "population standard deviation of NIC total bytes / mean NIC total bytes"
)
COVERAGE_AGGREGATION = "minimum per-NIC coverage_fraction"


class NodeBalanceStatus(str, Enum):
    OK = "ok"
    ZERO_TRAFFIC = "zero_traffic"
    PARTIAL_DATA = "partial_data"
    MISSING_DATA = "missing_data"
    UNSUPPORTED_DATA = "unsupported_data"


class NodeBalanceState(str, Enum):
    ZERO_TRAFFIC = "zero_traffic"
    MISSING_NIC = "missing_nic"
    UNKNOWN_NIC = "unknown_nic"
    MISSING_OR_INVALID_NIC = "missing_or_invalid_nic"
    PARTIAL_INVALID_DATA = "partial_invalid_data"
    PARTIAL_COVERAGE = "partial_coverage"
    UNSUPPORTED_INPUT = "unsupported_input"
    BASELINE_RELATIVE = "baseline_relative"


class MaxMinRatioDenominator(str, Enum):
    POSITIVE = "positive"
    ZERO = "zero"
    MISSING = "missing"


@dataclass(frozen=True)
class NodeNicBalance:
    """One NIC's contribution to a single-node, single-direction analysis."""

    nic_id: str
    in_known_inventory: bool
    analysis_status: AnalysisStatus
    quality_flags: frozenset[SeriesQualityFlag]
    source_counter: str
    total_bytes: int | float | None
    traffic_share: float | None
    mean_rate_bytes_per_s: float | None
    temporal_p50_rate_bytes_per_s: float | None
    temporal_p99_rate_bytes_per_s: float | None
    is_active: bool | None
    requested_window_ns: tuple[int, int] | None
    represented_window_ns: tuple[int, int] | None
    valid_interval_boundary_signature: tuple[int, ...]
    covered_duration_ns: int
    coverage_fraction: float | None
    valid_interval_count: int
    invalid_interval_count: int
    distribution_interval_count: int
    distribution_excluded_duration_ns: int
    boundary_excluded_interval_count: int


@dataclass(frozen=True)
class NodeNicBalanceResult:
    """Explicit numerical and inventory state for one node and direction."""

    node_id: str
    run_id: str | None
    direction: TrafficDirection | None
    source_counter: str | None
    status: NodeBalanceStatus
    state_flags: frozenset[NodeBalanceState]
    status_reasons: tuple[str, ...]
    common_window_ns: tuple[int, int] | None
    represented_window_ns: tuple[int, int] | None
    valid_interval_boundary_signature: tuple[int, ...] | None
    requested_duration_ns: int
    covered_duration_ns: int
    coverage_fraction: float | None
    known_nic_ids: tuple[str, ...]
    observed_nic_ids: tuple[str, ...]
    missing_nic_ids: tuple[str, ...]
    unknown_observed_nic_ids: tuple[str, ...]
    unavailable_nic_ids: tuple[str, ...]
    partial_nic_ids: tuple[str, ...]
    unsupported_nic_ids: tuple[str, ...]
    known_nic_count: int
    observed_nic_count: int
    active_known_nic_count: int
    active_observed_nic_count: int
    active_nic_fraction: float | None
    activity_threshold_bytes_per_s: float
    node_total_bytes: int | float | None
    per_nic: tuple[NodeNicBalance, ...]
    max_traffic_bytes: int | float | None
    min_traffic_bytes: int | float | None
    max_min_ratio: float | None
    max_min_ratio_denominator: MaxMinRatioDenominator
    effective_nic_count: float | None
    spatial_p50_bytes: int | float | None
    spatial_p99_bytes: int | float | None
    cross_nic_coefficient_of_variation: float | None
    valid_interval_count: int
    invalid_interval_count: int
    distribution_interval_count: int
    distribution_excluded_duration_ns: int
    boundary_excluded_interval_count: int
    baseline_relative_observation_count: int
    traffic_share_formula: str
    effective_nic_count_formula: str
    active_nic_rule: str
    active_nic_denominator: str
    max_min_ratio_formula: str
    spatial_quantile_interpolation: str
    cross_nic_cv_formula: str
    coverage_aggregation: str


def _sum_bytes(values: list[int | float]) -> int | float:
    if any(type(value) is float for value in values):
        return math.fsum(float(value) for value in values)
    return sum(values, start=0)


def _nearest_rank_quantile(
    values: list[int | float], probability: float
) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _cross_nic_coefficient_of_variation(
    values: list[int | float],
) -> float | None:
    if not values:
        return None
    mean = math.fsum(float(value) for value in values) / len(values)
    if mean == 0.0:
        return None
    variance = math.fsum((float(value) - mean) ** 2 for value in values) / len(
        values
    )
    return math.sqrt(variance) / mean


def _nic_result(
    analysis: TrafficAnalysis,
    *,
    node_total_bytes: int | float | None,
    activity_threshold: float,
    known_nic_ids: frozenset[str],
    usable: bool,
) -> NodeNicBalance:
    total_bytes = analysis.total_bytes if usable else None
    share = (
        float(total_bytes / node_total_bytes)
        if usable
        and total_bytes is not None
        and node_total_bytes is not None
        and node_total_bytes > 0
        else None
    )
    mean_rate = analysis.mean_rate_bytes_per_s if usable else None
    is_active = (
        mean_rate is not None and mean_rate > activity_threshold
        if usable
        else None
    )
    return NodeNicBalance(
        nic_id=analysis.identity.nic_id,
        in_known_inventory=analysis.identity.nic_id in known_nic_ids,
        analysis_status=analysis.status,
        quality_flags=analysis.quality_flags,
        source_counter=analysis.source_counter,
        total_bytes=total_bytes,
        traffic_share=share,
        mean_rate_bytes_per_s=mean_rate,
        temporal_p50_rate_bytes_per_s=(
            analysis.p50_rate_bytes_per_s if usable else None
        ),
        temporal_p99_rate_bytes_per_s=(
            analysis.p99_rate_bytes_per_s if usable else None
        ),
        is_active=is_active,
        requested_window_ns=analysis.requested_window_ns,
        represented_window_ns=analysis.represented_window_ns,
        valid_interval_boundary_signature=(
            analysis.valid_interval_boundary_signature
        ),
        covered_duration_ns=analysis.covered_duration_ns,
        coverage_fraction=analysis.coverage_fraction,
        valid_interval_count=analysis.valid_interval_count,
        invalid_interval_count=analysis.invalid_interval_count,
        distribution_interval_count=analysis.distribution_interval_count,
        distribution_excluded_duration_ns=analysis.distribution_excluded_duration_ns,
        boundary_excluded_interval_count=analysis.boundary_excluded_interval_count,
    )


def _unsupported_result(
    node_id: str,
    analyses: Mapping[str, TrafficAnalysis],
    known_nic_ids: tuple[str, ...],
    activity_threshold: float,
    reasons: tuple[str, ...],
) -> NodeNicBalanceResult:
    known_set = frozenset(known_nic_ids)
    per_nic = tuple(
        _nic_result(
            analysis,
            node_total_bytes=None,
            activity_threshold=activity_threshold,
            known_nic_ids=known_set,
            usable=False,
        )
        for _, analysis in sorted(analyses.items())
    )
    return NodeNicBalanceResult(
        node_id=node_id,
        run_id=None,
        direction=None,
        source_counter=None,
        status=NodeBalanceStatus.UNSUPPORTED_DATA,
        state_flags=frozenset({NodeBalanceState.UNSUPPORTED_INPUT}),
        status_reasons=reasons,
        common_window_ns=None,
        represented_window_ns=None,
        valid_interval_boundary_signature=None,
        requested_duration_ns=0,
        covered_duration_ns=0,
        coverage_fraction=None,
        known_nic_ids=known_nic_ids,
        observed_nic_ids=tuple(sorted(analyses)),
        missing_nic_ids=tuple(
            nic_id for nic_id in known_nic_ids if nic_id not in analyses
        ),
        unknown_observed_nic_ids=tuple(
            nic_id for nic_id in sorted(analyses) if nic_id not in known_set
        ),
        unavailable_nic_ids=tuple(sorted(analyses)),
        partial_nic_ids=(),
        unsupported_nic_ids=tuple(sorted(analyses)),
        known_nic_count=len(known_nic_ids),
        observed_nic_count=len(analyses),
        active_known_nic_count=0,
        active_observed_nic_count=0,
        active_nic_fraction=None,
        activity_threshold_bytes_per_s=activity_threshold,
        node_total_bytes=None,
        per_nic=per_nic,
        max_traffic_bytes=None,
        min_traffic_bytes=None,
        max_min_ratio=None,
        max_min_ratio_denominator=MaxMinRatioDenominator.MISSING,
        effective_nic_count=None,
        spatial_p50_bytes=None,
        spatial_p99_bytes=None,
        cross_nic_coefficient_of_variation=None,
        valid_interval_count=sum(
            analysis.valid_interval_count for analysis in analyses.values()
        ),
        invalid_interval_count=sum(
            analysis.invalid_interval_count for analysis in analyses.values()
        ),
        distribution_interval_count=sum(
            analysis.distribution_interval_count for analysis in analyses.values()
        ),
        distribution_excluded_duration_ns=sum(
            analysis.distribution_excluded_duration_ns
            for analysis in analyses.values()
        ),
        boundary_excluded_interval_count=sum(
            analysis.boundary_excluded_interval_count
            for analysis in analyses.values()
        ),
        baseline_relative_observation_count=sum(
            analysis.baseline_relative for analysis in analyses.values()
        ),
        traffic_share_formula=TRAFFIC_SHARE_FORMULA,
        effective_nic_count_formula=EFFECTIVE_NIC_COUNT_FORMULA,
        active_nic_rule=ACTIVE_NIC_RULE,
        active_nic_denominator=ACTIVE_NIC_DENOMINATOR,
        max_min_ratio_formula=MAX_MIN_RATIO_FORMULA,
        spatial_quantile_interpolation=SPATIAL_QUANTILE_INTERPOLATION,
        cross_nic_cv_formula=CROSS_NIC_CV_FORMULA,
        coverage_aggregation=COVERAGE_AGGREGATION,
    )


def analyze_node_nic_balance(
    node_id: str,
    analyses: Mapping[str, TrafficAnalysis],
    known_nic_ids: Collection[str],
    *,
    activity_threshold_bytes_per_s: float | None = None,
) -> NodeNicBalanceResult:
    """Analyze one run, node, traffic direction, and common requested window.

    ``analyses`` must be keyed by NIC identity. Missing known NICs remain
    visible and are never treated as zero-traffic observations. Unknown observed
    NICs are reported separately rather than silently expanding the known
    inventory. All numerical totals come from the supplied full-resolution
    :class:`TrafficAnalysis` results.
    """

    if not isinstance(node_id, str) or not node_id:
        raise ValueError("node_id must be a nonempty string")
    if not isinstance(analyses, Mapping):
        raise ValueError("analyses must be a mapping keyed by NIC identity")
    if isinstance(known_nic_ids, str):
        raise ValueError("known_nic_ids must be a collection, not a string")
    if not isinstance(activity_threshold_bytes_per_s, (int, float)) and (
        activity_threshold_bytes_per_s is not None
    ):
        raise ValueError("activity_threshold_bytes_per_s must be numeric or None")
    if isinstance(activity_threshold_bytes_per_s, bool):
        raise ValueError("activity_threshold_bytes_per_s must be numeric or None")
    if activity_threshold_bytes_per_s is None:
        activity_threshold = 0.0
    else:
        activity_threshold = float(activity_threshold_bytes_per_s)
        if not math.isfinite(activity_threshold) or activity_threshold < 0.0:
            raise ValueError(
                "activity_threshold_bytes_per_s must be finite and nonnegative"
            )

    known_inventory = list(known_nic_ids)
    if any(not isinstance(nic_id, str) or not nic_id for nic_id in known_inventory):
        raise ValueError("every known NIC identity must be a nonempty string")
    if len(set(known_inventory)) != len(known_inventory):
        raise ValueError("known_nic_ids must not contain duplicates")
    known = tuple(sorted(known_inventory))
    known_set = frozenset(known)

    for nic_id, analysis in analyses.items():
        if not isinstance(nic_id, str) or not nic_id:
            raise ValueError("every analysis key must be a nonempty NIC identity")
        if analysis.identity.nic_id != nic_id:
            return _unsupported_result(
                node_id,
                analyses,
                known,
                activity_threshold,
                ("An analysis key does not match its embedded NIC identity.",),
            )

    if not analyses:
        return _missing_result(node_id, known, activity_threshold)

    reference = next(iter(analyses.values()))
    run_id = reference.identity.run_id
    direction = reference.identity.direction
    source_counter = reference.source_counter
    common_window = reference.requested_window_ns
    represented_window = reference.represented_window_ns
    boundary_signature = reference.valid_interval_boundary_signature
    unit = reference.unit
    semantics = reference.semantics
    incompatibility_reasons: list[str] = []
    for analysis in analyses.values():
        identity = analysis.identity
        identity_or_window_incompatible = (
            identity.run_id != run_id
            or identity.node_id != node_id
            or identity.direction != direction
            or analysis.source_counter != source_counter
            or analysis.requested_window_ns != common_window
            or analysis.represented_window_ns != represented_window
            or analysis.unit != unit
            or analysis.semantics != semantics
        )
        boundary_signature_incompatible = (
            analysis.valid_interval_boundary_signature != boundary_signature
        )
        if identity_or_window_incompatible:
            incompatibility_reasons.append(
                "All records must share run, node, direction, counter, "
                "units, semantics, and windows."
            )
        if boundary_signature_incompatible:
            incompatibility_reasons.append(
                "All TrafficAnalysis records must share exact valid "
                "interval boundaries; their boundary signatures differ."
            )
        if identity_or_window_incompatible or boundary_signature_incompatible:
            return _unsupported_result(
                node_id,
                analyses,
                known,
                activity_threshold,
                tuple(incompatibility_reasons),
            )
    if (
        unit != "bytes"
        or semantics not in {"bytes", "network-frame octets"}
        or common_window is None
    ):
        return _unsupported_result(
            node_id,
            analyses,
            known,
            activity_threshold,
            ("Node balance requires cumulative byte analyses and a common window.",),
        )

    ordered_analyses = tuple(
        analyses[nic_id] for nic_id in sorted(analyses)
    )
    usable_analyses = tuple(
        analysis
        for analysis in ordered_analyses
        if analysis.total_bytes is not None
        and analysis.status != AnalysisStatus.UNSUPPORTED_SERIES
    )
    usable_nic_ids = frozenset(
        analysis.identity.nic_id for analysis in usable_analyses
    )
    totals = [analysis.total_bytes for analysis in usable_analyses]
    node_total = _sum_bytes(totals) if totals else None

    per_nic = tuple(
        _nic_result(
            analysis,
            node_total_bytes=node_total,
            activity_threshold=activity_threshold,
            known_nic_ids=known_set,
            usable=analysis.identity.nic_id in usable_nic_ids,
        )
        for analysis in ordered_analyses
    )

    observed = tuple(sorted(analyses))
    missing = tuple(nic_id for nic_id in known if nic_id not in analyses)
    unknown = tuple(
        nic_id for nic_id in observed if nic_id not in known_set
    )
    unavailable = tuple(
        analysis.identity.nic_id
        for analysis in ordered_analyses
        if analysis.status
        in {
            AnalysisStatus.MISSING_DATA,
            AnalysisStatus.INVALID_DATA,
            AnalysisStatus.INSUFFICIENT_DATA,
            AnalysisStatus.INVALID_WINDOW,
        }
    )
    unsupported = tuple(
        analysis.identity.nic_id
        for analysis in ordered_analyses
        if analysis.status == AnalysisStatus.UNSUPPORTED_SERIES
    )
    partial = tuple(
        analysis.identity.nic_id
        for analysis in ordered_analyses
        if analysis.status == AnalysisStatus.PARTIAL_INVALID_DATA
        or analysis.coverage_fraction is None
        or analysis.coverage_fraction < 1.0
    )

    active_known = sum(
        result.is_active is True and result.in_known_inventory
        for result in per_nic
    )
    active_observed = sum(result.is_active is True for result in per_nic)
    active_fraction = (
        active_known / len(known) if known else None
    )

    spatial_values = [
        analysis.total_bytes
        for analysis in usable_analyses
        if analysis.total_bytes is not None
    ]
    max_traffic = max(spatial_values, default=None)
    min_traffic = min(spatial_values, default=None)
    if not spatial_values:
        ratio_denominator = MaxMinRatioDenominator.MISSING
        max_min_ratio = None
    elif min_traffic == 0:
        ratio_denominator = MaxMinRatioDenominator.ZERO
        max_min_ratio = None
    else:
        ratio_denominator = MaxMinRatioDenominator.POSITIVE
        max_min_ratio = max_traffic / min_traffic

    effective_nic_count = (
        1.0 / math.fsum(result.traffic_share**2 for result in per_nic if result.traffic_share is not None)
        if node_total is not None and node_total > 0
        else None
    )

    coverage_values = [
        analysis.coverage_fraction
        for analysis in ordered_analyses
        if analysis.coverage_fraction is not None
    ]
    coverage = min(coverage_values) if coverage_values else None
    requested_duration = common_window[1] - common_window[0]
    covered_duration = min(
        analysis.covered_duration_ns for analysis in ordered_analyses
    )

    states: set[NodeBalanceState] = set()
    if node_total == 0:
        states.add(NodeBalanceState.ZERO_TRAFFIC)
    if missing:
        states.add(NodeBalanceState.MISSING_NIC)
    if unknown:
        states.add(NodeBalanceState.UNKNOWN_NIC)
    if unavailable:
        states.add(NodeBalanceState.MISSING_OR_INVALID_NIC)
    if partial:
        states.add(NodeBalanceState.PARTIAL_INVALID_DATA)
    if coverage is not None and coverage < 1.0:
        states.add(NodeBalanceState.PARTIAL_COVERAGE)
    if unsupported:
        states.add(NodeBalanceState.UNSUPPORTED_INPUT)
    baseline_count = sum(
        analysis.baseline_relative for analysis in ordered_analyses
    )
    if baseline_count:
        states.add(NodeBalanceState.BASELINE_RELATIVE)

    reasons: list[str] = []
    if NodeBalanceState.ZERO_TRAFFIC in states:
        reasons.append(
            "Node total traffic is zero; shares and effective NIC count are undefined."
        )
    if missing:
        reasons.append(
            "Known NICs without analysis records are missing, not zero traffic."
        )
    if unknown:
        reasons.append("Observed NICs absent from the known inventory are unknown.")
    if unavailable:
        reasons.append("Some NIC analyses have missing or invalid source data.")
    if partial:
        reasons.append(
            "Some NIC numerical results exclude invalid or boundary intervals."
        )
    if coverage is not None and coverage < 1.0:
        reasons.append("At least one NIC has incomplete observational coverage.")
    if unsupported:
        reasons.append("Some NIC analyses are unsupported and have no traffic total.")
    if baseline_count:
        reasons.append(
            "Baseline-relative totals cover observed increments only, "
            "not an absolute pre-window baseline."
        )

    if not usable_analyses:
        status = (
            NodeBalanceStatus.UNSUPPORTED_DATA
            if unsupported
            else NodeBalanceStatus.MISSING_DATA
        )
    elif node_total == 0:
        status = NodeBalanceStatus.ZERO_TRAFFIC
    elif states - {NodeBalanceState.ZERO_TRAFFIC}:
        status = NodeBalanceStatus.PARTIAL_DATA
    else:
        status = NodeBalanceStatus.OK

    return NodeNicBalanceResult(
        node_id=node_id,
        run_id=run_id,
        direction=direction,
        source_counter=source_counter,
        status=status,
        state_flags=frozenset(states),
        status_reasons=tuple(reasons),
        common_window_ns=common_window,
        represented_window_ns=represented_window,
        valid_interval_boundary_signature=boundary_signature,
        requested_duration_ns=requested_duration,
        covered_duration_ns=covered_duration,
        coverage_fraction=coverage,
        known_nic_ids=known,
        observed_nic_ids=observed,
        missing_nic_ids=missing,
        unknown_observed_nic_ids=unknown,
        unavailable_nic_ids=unavailable,
        partial_nic_ids=partial,
        unsupported_nic_ids=unsupported,
        known_nic_count=len(known),
        observed_nic_count=len(observed),
        active_known_nic_count=active_known,
        active_observed_nic_count=active_observed,
        active_nic_fraction=active_fraction,
        activity_threshold_bytes_per_s=activity_threshold,
        node_total_bytes=node_total,
        per_nic=per_nic,
        max_traffic_bytes=max_traffic,
        min_traffic_bytes=min_traffic,
        max_min_ratio=max_min_ratio,
        max_min_ratio_denominator=ratio_denominator,
        effective_nic_count=effective_nic_count,
        spatial_p50_bytes=_nearest_rank_quantile(spatial_values, 0.50),
        spatial_p99_bytes=_nearest_rank_quantile(spatial_values, 0.99),
        cross_nic_coefficient_of_variation=_cross_nic_coefficient_of_variation(
            spatial_values
        ),
        valid_interval_count=sum(
            analysis.valid_interval_count for analysis in ordered_analyses
        ),
        invalid_interval_count=sum(
            analysis.invalid_interval_count for analysis in ordered_analyses
        ),
        distribution_interval_count=sum(
            analysis.distribution_interval_count for analysis in ordered_analyses
        ),
        distribution_excluded_duration_ns=sum(
            analysis.distribution_excluded_duration_ns
            for analysis in ordered_analyses
        ),
        boundary_excluded_interval_count=sum(
            analysis.boundary_excluded_interval_count
            for analysis in ordered_analyses
        ),
        baseline_relative_observation_count=baseline_count,
        traffic_share_formula=TRAFFIC_SHARE_FORMULA,
        effective_nic_count_formula=EFFECTIVE_NIC_COUNT_FORMULA,
        active_nic_rule=ACTIVE_NIC_RULE,
        active_nic_denominator=ACTIVE_NIC_DENOMINATOR,
        max_min_ratio_formula=MAX_MIN_RATIO_FORMULA,
        spatial_quantile_interpolation=SPATIAL_QUANTILE_INTERPOLATION,
        cross_nic_cv_formula=CROSS_NIC_CV_FORMULA,
        coverage_aggregation=COVERAGE_AGGREGATION,
    )


def _missing_result(
    node_id: str,
    known_nic_ids: tuple[str, ...],
    activity_threshold: float,
) -> NodeNicBalanceResult:
    return NodeNicBalanceResult(
        node_id=node_id,
        run_id=None,
        direction=None,
        source_counter=None,
        status=NodeBalanceStatus.MISSING_DATA,
        state_flags=frozenset({NodeBalanceState.MISSING_NIC}),
        status_reasons=("No TrafficAnalysis records were supplied.",),
        common_window_ns=None,
        represented_window_ns=None,
        valid_interval_boundary_signature=None,
        requested_duration_ns=0,
        covered_duration_ns=0,
        coverage_fraction=None,
        known_nic_ids=known_nic_ids,
        observed_nic_ids=(),
        missing_nic_ids=known_nic_ids,
        unknown_observed_nic_ids=(),
        unavailable_nic_ids=(),
        partial_nic_ids=(),
        unsupported_nic_ids=(),
        known_nic_count=len(known_nic_ids),
        observed_nic_count=0,
        active_known_nic_count=0,
        active_observed_nic_count=0,
        active_nic_fraction=(
            0.0 / len(known_nic_ids) if known_nic_ids else None
        ),
        activity_threshold_bytes_per_s=activity_threshold,
        node_total_bytes=None,
        per_nic=(),
        max_traffic_bytes=None,
        min_traffic_bytes=None,
        max_min_ratio=None,
        max_min_ratio_denominator=MaxMinRatioDenominator.MISSING,
        effective_nic_count=None,
        spatial_p50_bytes=None,
        spatial_p99_bytes=None,
        cross_nic_coefficient_of_variation=None,
        valid_interval_count=0,
        invalid_interval_count=0,
        distribution_interval_count=0,
        distribution_excluded_duration_ns=0,
        boundary_excluded_interval_count=0,
        baseline_relative_observation_count=0,
        traffic_share_formula=TRAFFIC_SHARE_FORMULA,
        effective_nic_count_formula=EFFECTIVE_NIC_COUNT_FORMULA,
        active_nic_rule=ACTIVE_NIC_RULE,
        active_nic_denominator=ACTIVE_NIC_DENOMINATOR,
        max_min_ratio_formula=MAX_MIN_RATIO_FORMULA,
        spatial_quantile_interpolation=SPATIAL_QUANTILE_INTERPOLATION,
        cross_nic_cv_formula=CROSS_NIC_CV_FORMULA,
        coverage_aggregation=COVERAGE_AGGREGATION,
    )
