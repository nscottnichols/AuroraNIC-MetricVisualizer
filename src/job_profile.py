"""Explainable composition of question evidence into a job profile.

The profile is deliberately descriptive: it reports evidence, configured
comparison rules, and limitations without converting them into an opaque
ranking or diagnosis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


JOB_PROFILE_VERSION = 1

EXPECTED_QUESTION_IDS = (
    "NIC-01",
    "NIC-02",
    "NIC-03",
    "NIC-04",
    "NIC-05",
    "NIC-06",
    "NIC-07",
    "NIC-08",
    "NIC-09",
    "NODE-01",
    "NODE-02",
    "JOB-01",
    "JOB-02",
    "JOB-03",
    "JOB-04",
)


class ProfileStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class DimensionStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class SummaryStatus(str, Enum):
    OBSERVED = "observed"
    OBSERVED_ZERO = "observed_zero"
    PARTIAL = "partial"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"


class SummaryKind(str, Enum):
    TRAFFIC = "traffic"
    JOB = "job"
    NODE = "node"
    PHASE = "phase"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class _DimensionDefinition:
    dimension_id: str
    title: str
    question_ids: tuple[str, ...]
    formula: str
    threshold_rule: str


_DIMENSION_DEFINITIONS = (
    _DimensionDefinition(
        dimension_id="traffic_intensity",
        title="Traffic intensity",
        question_ids=("NIC-01", "JOB-02"),
        formula=(
            "compatible TX and RX byte totals divided by covered duration "
            "in seconds"
        ),
        threshold_rule=(
            "low <= mean bytes/second < high, using configured project "
            "thresholds"
        ),
    ),
    _DimensionDefinition(
        dimension_id="message_size",
        title="Wire-frame size behavior",
        question_ids=("NIC-06",),
        formula="compatible frame-bin counts and byte totals",
        threshold_rule="no numeric size threshold; bins remain wire-frame evidence",
    ),
    _DimensionDefinition(
        dimension_id="temporal_pattern",
        title="Temporal pattern",
        question_ids=("NIC-02", "JOB-03"),
        formula=(
            "adjacent valid observations and explicit manual phase windows; "
            "no sub-sample interpolation"
        ),
        threshold_rule="no burstiness threshold; changes are described by window",
    ),
    _DimensionDefinition(
        dimension_id="host_pressure",
        title="Host-path pressure",
        question_ids=("NIC-04",),
        formula="blocked and request counters with declared compatible populations",
        threshold_rule="no pressure threshold until counter units are verified",
    ),
    _DimensionDefinition(
        dimension_id="translation",
        title="Address-translation pressure",
        question_ids=("NIC-07",),
        formula="translation misses, hits, evictions, and requests by population",
        threshold_rule="request counts are not treated as measured translation time",
    ),
    _DimensionDefinition(
        dimension_id="latency",
        title="Communication latency",
        question_ids=("NIC-08",),
        formula="NIC histogram bins with verified edges, units, and clocks",
        threshold_rule="no latency threshold without sourced bin definitions",
    ),
    _DimensionDefinition(
        dimension_id="protocol",
        title="Protocol and operation evidence",
        question_ids=("NIC-05",),
        formula="command-queue counts by known opcode mapping",
        threshold_rule="unmapped numeric opcodes remain unlabeled",
    ),
    _DimensionDefinition(
        dimension_id="reliability",
        title="Reliability and errors",
        question_ids=("NIC-09",),
        formula="error and reliability observations with explicit populations",
        threshold_rule="observed zeros and missing observations are never merged",
    ),
    _DimensionDefinition(
        dimension_id="nic_node_consistency",
        title="NIC and node consistency",
        question_ids=("NODE-01", "NODE-02"),
        formula="common-window shares, rates, and diagnostic states",
        threshold_rule="comparisons require identical population and units",
    ),
    _DimensionDefinition(
        dimension_id="topology_hotspots",
        title="Topology hotspots",
        question_ids=("JOB-01",),
        formula="node traffic grouped only by a validated topology mapping",
        threshold_rule="hotspot concentration uses the configured project cutoff",
    ),
    _DimensionDefinition(
        dimension_id="spatial_balance",
        title="Spatial balance",
        question_ids=("JOB-02",),
        formula="node traffic shares and effective node count",
        threshold_rule="max/min ratio uses the configured project cutoff",
    ),
    _DimensionDefinition(
        dimension_id="phase_change",
        title="Phase change",
        question_ids=("JOB-03",),
        formula="totals and rates over explicit phase windows",
        threshold_rule="phase labels are user-provided, not inferred labels",
    ),
    _DimensionDefinition(
        dimension_id="job_profile",
        title="Job communication profile",
        question_ids=("JOB-04",),
        formula="transparent composition of capability rows and result summaries",
        threshold_rule="no combined ranking is calculated",
    ),
)

_DIMENSION_IDS = frozenset(
    definition.dimension_id for definition in _DIMENSION_DEFINITIONS
)


@dataclass(frozen=True)
class JobProfileThresholds:
    """Configurable project choices, not catalog-established thresholds."""

    low_intensity_bytes_per_s: float = 1_000_000.0
    high_intensity_bytes_per_s: float = 100_000_000.0
    high_node_max_min_ratio: float = 4.0
    high_hotspot_concentration: float = 0.8

    def __post_init__(self) -> None:
        numeric_fields = (
            self.low_intensity_bytes_per_s,
            self.high_intensity_bytes_per_s,
            self.high_node_max_min_ratio,
            self.high_hotspot_concentration,
        )
        if not all(math.isfinite(float(value)) for value in numeric_fields):
            raise ValueError("threshold values must be finite")
        if self.low_intensity_bytes_per_s < 0:
            raise ValueError("low intensity threshold must be nonnegative")
        if self.high_intensity_bytes_per_s < self.low_intensity_bytes_per_s:
            raise ValueError("high intensity threshold must not be below low threshold")
        if self.high_node_max_min_ratio < 1:
            raise ValueError("max/min threshold must be at least 1")
        if not 0 <= self.high_hotspot_concentration <= 1:
            raise ValueError("hotspot concentration threshold must be in [0, 1]")


@dataclass(frozen=True)
class JobProfileProvenance:
    """Run, capture, source, and clock identity for one profile."""

    run_id: str | None = None
    capture_id: str | None = None
    source: str | None = None
    reader_version: str | None = None
    clock_id: str | None = None
    selected_window_ns: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        for field_name in ("run_id", "capture_id", "source", "reader_version", "clock_id"):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(f"{field_name} must be a nonempty string or None")
        if self.selected_window_ns is not None:
            start_ns, end_ns = self.selected_window_ns
            if start_ns < 0 or end_ns < start_ns:
                raise ValueError("selected window must satisfy 0 <= start <= end")

    @property
    def status(self) -> str:
        if self.run_id and self.capture_id and self.source:
            return "complete"
        if self.run_id or self.capture_id or self.source:
            return "partial"
        return "missing"


@runtime_checkable
class CapabilityRowLike(Protocol):
    """Fields consumed from an immutable question-capability row."""

    contract_version: int
    question_id: str
    status: str
    reasons: Sequence[str]
    scientific_limitations: Sequence[str]


@dataclass(frozen=True)
class ProfileSummary:
    """One bounded, explicit result-summary supplied to the profile."""

    analysis_id: str
    summary_kind: SummaryKind | str
    dimension_id: str
    status: SummaryStatus | str
    run_id: str | None = None
    value: int | float | str | None = None
    unit: str | None = None
    window_ns: tuple[int, int] | None = None
    coverage_fraction: float | None = None
    zero_observation: bool = False
    missing_observation: bool = False
    topology_supported: bool | None = None
    communication_edges_supported: bool | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.analysis_id:
            raise ValueError("analysis_id must be nonempty")
        if self.dimension_id not in _DIMENSION_IDS:
            raise ValueError(f"unknown profile dimension {self.dimension_id!r}")
        summary_kind = _semantic_name(self.summary_kind)
        try:
            summary_kind_value = SummaryKind(summary_kind)
        except ValueError as error:
            raise ValueError(f"unknown summary kind {summary_kind!r}") from error
        status = _semantic_name(self.status)
        try:
            status_value = SummaryStatus(status)
        except ValueError as error:
            raise ValueError(f"unknown summary status {status!r}") from error

        object.__setattr__(self, "summary_kind", summary_kind_value)
        object.__setattr__(self, "status", status_value)
        if self.zero_observation and self.missing_observation:
            raise ValueError("zero and missing observations are mutually exclusive")
        if self.coverage_fraction is not None:
            coverage = float(self.coverage_fraction)
            if not math.isfinite(coverage) or not 0 <= coverage <= 1:
                raise ValueError("coverage_fraction must be in [0, 1]")
            object.__setattr__(self, "coverage_fraction", coverage)
        if self.window_ns is not None:
            start_ns, end_ns = self.window_ns
            if start_ns < 0 or end_ns < start_ns:
                raise ValueError("summary window must satisfy 0 <= start <= end")

        if status_value is SummaryStatus.OBSERVED_ZERO:
            if not self.zero_observation or self.missing_observation:
                raise ValueError("observed_zero requires zero_observation=True")
            if not isinstance(self.value, (int, float)) or self.value != 0:
                raise ValueError("observed_zero requires a numeric value equal to zero")
        elif status_value is SummaryStatus.OBSERVED:
            if self.value is None or self.zero_observation or self.missing_observation:
                raise ValueError("observed status requires a value without zero/missing flags")
        elif status_value is SummaryStatus.MISSING:
            if not self.missing_observation or self.zero_observation:
                raise ValueError("missing status requires missing_observation=True")
            if self.value is not None:
                raise ValueError("missing status cannot also carry a value")


@dataclass(frozen=True)
class ProfileQuestion:
    contract_version: int
    question_id: str
    status: str
    reasons: tuple[str, ...]
    scientific_limitations: tuple[str, ...]


@dataclass(frozen=True)
class ProfileDimension:
    dimension_id: str
    title: str
    status: DimensionStatus
    question_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    observed_evidence_count: int
    zero_observation_count: int
    missing_observation_count: int
    coverage_fraction: float | None
    limitations: tuple[str, ...]
    observation: str


@dataclass(frozen=True)
class ProfileFormula:
    dimension_id: str
    formula: str
    threshold_rule: str


@dataclass(frozen=True)
class JobCommunicationProfile:
    version: int
    status: ProfileStatus
    summary: str
    provenance: JobProfileProvenance
    thresholds: JobProfileThresholds
    dimensions: tuple[ProfileDimension, ...]
    questions: tuple[ProfileQuestion, ...]
    evidence: tuple[ProfileSummary, ...]
    observed_summary_count: int
    zero_observation_count: int
    missing_observation_count: int
    coverage_fraction: float | None
    supported_dimensions: tuple[str, ...]
    partial_dimensions: tuple[str, ...]
    unsupported_dimensions: tuple[str, ...]
    topology_status: str
    topology_reasons: tuple[str, ...]
    communication_edge_status: str
    communication_edge_reasons: tuple[str, ...]
    formulas: tuple[ProfileFormula, ...]
    interpretation_limits: tuple[str, ...]


def _semantic_name(value: Any) -> str:
    return str(getattr(value, "value", value))


def _profile_questions(
    capabilities: Sequence[CapabilityRowLike],
) -> tuple[ProfileQuestion, ...]:
    question_ids = [row.question_id for row in capabilities]
    duplicates = sorted(
        question_id for question_id in set(question_ids) if question_ids.count(question_id) > 1
    )
    if duplicates:
        raise ValueError("duplicate question IDs: " + ", ".join(duplicates))
    missing = [question_id for question_id in EXPECTED_QUESTION_IDS if question_id not in question_ids]
    if missing:
        raise ValueError("missing question IDs: " + ", ".join(missing))
    extra = [question_id for question_id in question_ids if question_id not in EXPECTED_QUESTION_IDS]
    if extra:
        raise ValueError("unexpected question IDs: " + ", ".join(sorted(extra)))

    contract_versions = {row.contract_version for row in capabilities}
    if len(contract_versions) != 1:
        raise ValueError("capability rows must share one contract version")
    if not isinstance(next(iter(contract_versions)), int):
        raise ValueError("capability contract version must be an integer")

    by_id = {row.question_id: row for row in capabilities}
    return tuple(
        ProfileQuestion(
            contract_version=by_id[question_id].contract_version,
            question_id=question_id,
            status=_semantic_name(by_id[question_id].status),
            reasons=tuple(by_id[question_id].reasons),
            scientific_limitations=tuple(
                by_id[question_id].scientific_limitations
            ),
        )
        for question_id in EXPECTED_QUESTION_IDS
    )


def _collect_summaries(
    provenance: JobProfileProvenance,
    traffic_summaries: Sequence[ProfileSummary],
    job_summaries: Sequence[ProfileSummary],
    node_summaries: Sequence[ProfileSummary],
    phase_summaries: Sequence[ProfileSummary],
    diagnostic_summaries: Sequence[ProfileSummary],
) -> tuple[ProfileSummary, ...]:
    summaries = (
        *traffic_summaries,
        *job_summaries,
        *node_summaries,
        *phase_summaries,
        *diagnostic_summaries,
    )
    analysis_ids = [summary.analysis_id for summary in summaries]
    duplicates = sorted(
        analysis_id
        for analysis_id in set(analysis_ids)
        if analysis_ids.count(analysis_id) > 1
    )
    if duplicates:
        raise ValueError("duplicate summary analysis IDs: " + ", ".join(duplicates))
    for summary in summaries:
        if (
            provenance.run_id is not None
            and summary.run_id is not None
            and summary.run_id != provenance.run_id
        ):
            raise ValueError(
                f"summary {summary.analysis_id!r} belongs to run "
                f"{summary.run_id!r}, not {provenance.run_id!r}"
            )
    return tuple(summaries)


def _dimension_limitations(
    definition: _DimensionDefinition,
    questions: MappingProxyType[str, ProfileQuestion],
) -> tuple[str, ...]:
    limitations: list[str] = []
    for question_id in definition.question_ids:
        limitations.extend(questions[question_id].scientific_limitations)
    return tuple(dict.fromkeys(limitations))


def _dimension_status(
    definition: _DimensionDefinition,
    questions: MappingProxyType[str, ProfileQuestion],
    summaries: Sequence[ProfileSummary],
) -> DimensionStatus:
    question_statuses = [questions[question_id].status for question_id in definition.question_ids]
    if not summaries:
        if any(status in ("supported", "partial") for status in question_statuses):
            return DimensionStatus.PARTIAL
        return DimensionStatus.UNSUPPORTED

    summary_statuses = [summary.status for summary in summaries]
    if not any(
        status in (SummaryStatus.OBSERVED, SummaryStatus.OBSERVED_ZERO)
        for status in summary_statuses
    ):
        if any(status is SummaryStatus.PARTIAL for status in summary_statuses):
            return DimensionStatus.PARTIAL
        return DimensionStatus.UNSUPPORTED

    complete_questions = all(status == "supported" for status in question_statuses)
    complete_summaries = all(
        status in (SummaryStatus.OBSERVED, SummaryStatus.OBSERVED_ZERO)
        and summary.coverage_fraction == 1.0
        for status, summary in zip(summary_statuses, summaries, strict=True)
    )
    return (
        DimensionStatus.SUPPORTED
        if complete_questions and complete_summaries
        else DimensionStatus.PARTIAL
    )


def _coverage_fraction(summaries: Sequence[ProfileSummary]) -> float | None:
    values = [summary.coverage_fraction for summary in summaries]
    if not values or any(value is None for value in values):
        return None
    return min(values)


def _dimension_observation(
    definition: _DimensionDefinition,
    summaries: Sequence[ProfileSummary],
) -> str:
    if not summaries:
        return (
            f"No result summary was supplied for {definition.title}. "
            "The profile does not suggest a characterization for this dimension."
        )
    zero_count = sum(summary.status is SummaryStatus.OBSERVED_ZERO for summary in summaries)
    missing_count = sum(summary.status is SummaryStatus.MISSING for summary in summaries)
    return (
        f"Observed {len(summaries)} result summaries for {definition.title}; "
        f"{zero_count} zero-valued and {missing_count} missing states are reported "
        "separately. The configured rule suggests only the comparison stated in "
        "the profile formula."
    )


def _dimension(
    definition: _DimensionDefinition,
    questions: MappingProxyType[str, ProfileQuestion],
    summaries: Sequence[ProfileSummary],
) -> ProfileDimension:
    return ProfileDimension(
        dimension_id=definition.dimension_id,
        title=definition.title,
        status=_dimension_status(definition, questions, summaries),
        question_ids=definition.question_ids,
        evidence_ids=tuple(summary.analysis_id for summary in summaries),
        observed_evidence_count=sum(
            summary.status in (SummaryStatus.OBSERVED, SummaryStatus.OBSERVED_ZERO)
            for summary in summaries
        ),
        zero_observation_count=sum(
            summary.status is SummaryStatus.OBSERVED_ZERO for summary in summaries
        ),
        missing_observation_count=sum(
            summary.status is SummaryStatus.MISSING for summary in summaries
        ),
        coverage_fraction=_coverage_fraction(summaries),
        limitations=_dimension_limitations(definition, questions),
        observation=_dimension_observation(definition, summaries),
    )


def _mapping_status(
    summaries: Sequence[ProfileSummary],
    attribute_name: str,
) -> tuple[str, tuple[str, ...]]:
    values = [
        getattr(summary, attribute_name)
        for summary in summaries
        if getattr(summary, attribute_name) is not None
    ]
    if not values:
        label = "topology" if attribute_name == "topology_supported" else "communication edges"
        return "unsupported", (f"No supplied summary establishes {label} support.",)
    if all(values):
        return "supported", ()
    if any(values):
        return "partial", ("Supplied summaries disagree about mapping support.",)
    return "unsupported", ("The supplied summaries do not establish mapping support.",)


def build_job_communication_profile(
    capabilities: Sequence[CapabilityRowLike],
    *,
    provenance: JobProfileProvenance | None = None,
    thresholds: JobProfileThresholds | None = None,
    traffic_summaries: Sequence[ProfileSummary] = (),
    job_summaries: Sequence[ProfileSummary] = (),
    node_summaries: Sequence[ProfileSummary] = (),
    phase_summaries: Sequence[ProfileSummary] = (),
    diagnostic_summaries: Sequence[ProfileSummary] = (),
) -> JobCommunicationProfile:
    """Compose a pure, versioned, non-ranking job communication profile."""

    selected_provenance = provenance or JobProfileProvenance()
    selected_thresholds = thresholds or JobProfileThresholds()
    questions = _profile_questions(capabilities)
    question_map = MappingProxyType({question.question_id: question for question in questions})
    summaries = _collect_summaries(
        selected_provenance,
        traffic_summaries,
        job_summaries,
        node_summaries,
        phase_summaries,
        diagnostic_summaries,
    )
    summaries_by_dimension = {
        dimension_id: [
            summary for summary in summaries if summary.dimension_id == dimension_id
        ]
        for dimension_id in _DIMENSION_IDS
    }
    dimensions = tuple(
        _dimension(
            definition,
            question_map,
            summaries_by_dimension[definition.dimension_id],
        )
        for definition in _DIMENSION_DEFINITIONS
    )

    if all(dimension.status is DimensionStatus.SUPPORTED for dimension in dimensions):
        status = ProfileStatus.COMPLETE
    elif any(
        dimension.status in (DimensionStatus.SUPPORTED, DimensionStatus.PARTIAL)
        for dimension in dimensions
    ):
        status = ProfileStatus.PARTIAL
    else:
        status = ProfileStatus.UNAVAILABLE

    observed_summary_count = sum(
        summary.status in (SummaryStatus.OBSERVED, SummaryStatus.OBSERVED_ZERO)
        for summary in summaries
    )
    zero_observation_count = sum(
        summary.status is SummaryStatus.OBSERVED_ZERO for summary in summaries
    )
    missing_observation_count = sum(
        summary.status is SummaryStatus.MISSING for summary in summaries
    )
    supported_dimensions = tuple(
        dimension.dimension_id
        for dimension in dimensions
        if dimension.status is DimensionStatus.SUPPORTED
    )
    partial_dimensions = tuple(
        dimension.dimension_id
        for dimension in dimensions
        if dimension.status is DimensionStatus.PARTIAL
    )
    unsupported_dimensions = tuple(
        dimension.dimension_id
        for dimension in dimensions
        if dimension.status is DimensionStatus.UNSUPPORTED
    )
    topology_status, topology_reasons = _mapping_status(
        summaries_by_dimension["topology_hotspots"], "topology_supported"
    )
    communication_edge_status, communication_edge_reasons = _mapping_status(
        summaries_by_dimension["topology_hotspots"], "communication_edges_supported"
    )

    if status is ProfileStatus.COMPLETE:
        summary_text = (
            f"Observed evidence supports all {len(dimensions)} profile dimensions; "
            "this suggests a complete descriptive profile for the selected capture."
        )
    elif status is ProfileStatus.PARTIAL:
        summary_text = (
            f"Observed evidence covers {len(supported_dimensions)} supported and "
            f"{len(partial_dimensions)} partial dimensions; this suggests a partial "
            "descriptive profile, and unavailable dimensions remain visible."
        )
    else:
        summary_text = (
            "No observed result evidence supports a communication characterization; "
            "the unavailable state is explicit."
        )

    return JobCommunicationProfile(
        version=JOB_PROFILE_VERSION,
        status=status,
        summary=summary_text,
        provenance=selected_provenance,
        thresholds=selected_thresholds,
        dimensions=dimensions,
        questions=questions,
        evidence=summaries,
        observed_summary_count=observed_summary_count,
        zero_observation_count=zero_observation_count,
        missing_observation_count=missing_observation_count,
        coverage_fraction=_coverage_fraction(summaries),
        supported_dimensions=supported_dimensions,
        partial_dimensions=partial_dimensions,
        unsupported_dimensions=unsupported_dimensions,
        topology_status=topology_status,
        topology_reasons=topology_reasons,
        communication_edge_status=communication_edge_status,
        communication_edge_reasons=communication_edge_reasons,
        formulas=tuple(
            ProfileFormula(
                dimension_id=definition.dimension_id,
                formula=definition.formula,
                threshold_rule=definition.threshold_rule,
            )
            for definition in _DIMENSION_DEFINITIONS
        ),
        interpretation_limits=(
            "Every classification is descriptive and tied to the displayed formula, window, population, units, and coverage.",
            "Wire-frame evidence is not MPI message-size evidence, and command-queue counts are not wire bytes.",
            "Address-translation request counts are not measured translation time.",
            "Missing evidence and observed zero are distinct states and are counted separately.",
            "Node names alone do not establish topology, communication peers, or distance.",
            "Phase names are user-provided and do not establish an application communication pattern.",
        ),
    )
