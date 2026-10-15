from dataclasses import asdict, dataclass, replace

import pytest

from job_profile import (
    EXPECTED_QUESTION_IDS,
    JOB_PROFILE_VERSION,
    DimensionStatus,
    JobProfileProvenance,
    JobProfileThresholds,
    ProfileStatus,
    ProfileSummary,
    SummaryKind,
    SummaryStatus,
    build_job_communication_profile,
)


@dataclass(frozen=True)
class CapabilityRow:
    contract_version: int
    question_id: str
    status: str
    reasons: tuple[str, ...]
    scientific_limitations: tuple[str, ...]


SUMMARY_KIND_BY_DIMENSION = {
    "traffic_intensity": SummaryKind.TRAFFIC,
    "message_size": SummaryKind.DIAGNOSTIC,
    "temporal_pattern": SummaryKind.PHASE,
    "host_pressure": SummaryKind.DIAGNOSTIC,
    "translation": SummaryKind.DIAGNOSTIC,
    "latency": SummaryKind.DIAGNOSTIC,
    "protocol": SummaryKind.DIAGNOSTIC,
    "reliability": SummaryKind.DIAGNOSTIC,
    "nic_node_consistency": SummaryKind.NODE,
    "topology_hotspots": SummaryKind.JOB,
    "spatial_balance": SummaryKind.JOB,
    "phase_change": SummaryKind.PHASE,
    "job_profile": SummaryKind.JOB,
}


def capability_rows(status: str) -> tuple[CapabilityRow, ...]:
    return tuple(
        CapabilityRow(
            contract_version=1,
            question_id=question_id,
            status=status,
            reasons=(f"{question_id} reason",),
            scientific_limitations=(f"{question_id} limitation",),
        )
        for question_id in EXPECTED_QUESTION_IDS
    )


def summary(
    analysis_id: str,
    dimension_id: str,
    *,
    status: SummaryStatus = SummaryStatus.OBSERVED,
    value: int | float | None = 1,
    coverage_fraction: float | None = 1.0,
    zero_observation: bool = False,
    missing_observation: bool = False,
) -> ProfileSummary:
    return ProfileSummary(
        analysis_id=analysis_id,
        summary_kind=SUMMARY_KIND_BY_DIMENSION[dimension_id],
        dimension_id=dimension_id,
        status=status,
        run_id="run-1",
        value=value,
        unit="bytes" if dimension_id == "traffic_intensity" else "count",
        window_ns=(0, 1_000_000_000),
        coverage_fraction=coverage_fraction,
        zero_observation=zero_observation,
        missing_observation=missing_observation,
        reasons=("summary reason",),
    )


def all_observed_summaries() -> tuple[ProfileSummary, ...]:
    return tuple(
        summary(f"summary-{dimension_id}", dimension_id)
        for dimension_id in SUMMARY_KIND_BY_DIMENSION
    )


def provenance() -> JobProfileProvenance:
    return JobProfileProvenance(
        run_id="run-1",
        capture_id="capture-1",
        source="trace-1",
        reader_version="reader-1",
        clock_id="clock-1",
        selected_window_ns=(0, 1_000_000_000),
    )


def test_complete_evidence_builds_versioned_complete_profile():
    profile = build_job_communication_profile(
        capability_rows("supported"),
        provenance=provenance(),
        traffic_summaries=all_observed_summaries()[:1],
        job_summaries=tuple(
            item
            for item in all_observed_summaries()
            if item.summary_kind is SummaryKind.JOB
        ),
        node_summaries=tuple(
            item
            for item in all_observed_summaries()
            if item.summary_kind is SummaryKind.NODE
        ),
        phase_summaries=tuple(
            item
            for item in all_observed_summaries()
            if item.summary_kind is SummaryKind.PHASE
        ),
        diagnostic_summaries=tuple(
            item
            for item in all_observed_summaries()
            if item.summary_kind is SummaryKind.DIAGNOSTIC
        ),
    )

    assert profile.version == JOB_PROFILE_VERSION
    assert profile.status is ProfileStatus.COMPLETE
    assert profile.provenance.status == "complete"
    assert profile.observed_summary_count == len(SUMMARY_KIND_BY_DIMENSION)
    assert profile.zero_observation_count == 0
    assert profile.missing_observation_count == 0
    assert profile.coverage_fraction == 1.0
    assert profile.supported_dimensions == tuple(SUMMARY_KIND_BY_DIMENSION)
    assert tuple(question.question_id for question in profile.questions) == EXPECTED_QUESTION_IDS
    assert all(question.status == "supported" for question in profile.questions)
    assert all(dimension.status is DimensionStatus.SUPPORTED for dimension in profile.dimensions)
    assert {formula.dimension_id: formula.formula for formula in profile.formulas}[
        "traffic_intensity"
    ].startswith("compatible TX and RX")


def test_partial_evidence_reports_partial_and_unavailable_dimensions():
    supported_ids = {"NIC-01", "JOB-02"}
    rows = tuple(
        replace(row, status="supported")
        if row.question_id in supported_ids
        else row
        for row in capability_rows("unsupported")
    )
    profile = build_job_communication_profile(
        rows,
        provenance=provenance(),
        traffic_summaries=(
            summary("traffic", "traffic_intensity", coverage_fraction=0.5),
        ),
    )

    traffic = next(
        dimension
        for dimension in profile.dimensions
        if dimension.dimension_id == "traffic_intensity"
    )
    assert profile.status is ProfileStatus.PARTIAL
    assert traffic.status is DimensionStatus.PARTIAL
    assert traffic.observed_evidence_count == 1
    assert "traffic_intensity" in profile.partial_dimensions
    assert "message_size" in profile.unsupported_dimensions
    assert profile.missing_observation_count == 0
    assert all(dimension.observation for dimension in profile.dimensions)


def test_unknown_summary_coverage_makes_profile_coverage_unknown():
    profile = build_job_communication_profile(
        capability_rows("supported"),
        provenance=provenance(),
        traffic_summaries=(
            summary("known-coverage", "traffic_intensity", coverage_fraction=1.0),
            summary("unknown-coverage", "traffic_intensity", coverage_fraction=None),
        ),
    )
    traffic = next(
        dimension
        for dimension in profile.dimensions
        if dimension.dimension_id == "traffic_intensity"
    )

    assert profile.status is ProfileStatus.PARTIAL
    assert traffic.status is DimensionStatus.PARTIAL
    assert traffic.coverage_fraction is None
    assert profile.coverage_fraction is None


def test_empty_input_is_unavailable_without_hiding_questions():
    profile = build_job_communication_profile(capability_rows("unsupported"))

    assert profile.status is ProfileStatus.UNAVAILABLE
    assert profile.provenance.status == "missing"
    assert profile.observed_summary_count == 0
    assert profile.coverage_fraction is None
    assert tuple(question.question_id for question in profile.questions) == EXPECTED_QUESTION_IDS
    assert all(question.status == "unsupported" for question in profile.questions)
    assert profile.supported_dimensions == ()
    assert profile.partial_dimensions == ()
    assert profile.unsupported_dimensions == tuple(SUMMARY_KIND_BY_DIMENSION)
    assert profile.topology_status == "unsupported"
    assert profile.communication_edge_status == "unsupported"


def test_missing_or_extra_question_is_rejected():
    rows = capability_rows("partial")

    with pytest.raises(ValueError, match="missing question IDs: JOB-04"):
        build_job_communication_profile(rows[:-1])

    extra_row = CapabilityRow(
        contract_version=1,
        question_id="JOB-05",
        status="partial",
        reasons=(),
        scientific_limitations=(),
    )
    with pytest.raises(ValueError, match="unexpected question IDs: JOB-05"):
        build_job_communication_profile((*rows, extra_row))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"low_intensity_bytes_per_s": -1}, "low intensity threshold"),
        ({"high_intensity_bytes_per_s": 0.5}, "high intensity threshold"),
        ({"high_node_max_min_ratio": 0.5}, "max/min threshold"),
        ({"high_hotspot_concentration": 1.5}, "hotspot concentration threshold"),
    ],
)
def test_thresholds_are_validated(overrides, message):
    with pytest.raises(ValueError, match=message):
        JobProfileThresholds(**overrides)


def test_profile_uses_observed_suggests_language_without_causal_claims():
    profile = build_job_communication_profile(
        capability_rows("supported"),
        provenance=provenance(),
        **{
            "traffic_summaries": all_observed_summaries()[:1],
            "job_summaries": tuple(
                item
                for item in all_observed_summaries()
                if item.summary_kind is SummaryKind.JOB
            ),
            "node_summaries": tuple(
                item
                for item in all_observed_summaries()
                if item.summary_kind is SummaryKind.NODE
            ),
            "phase_summaries": tuple(
                item
                for item in all_observed_summaries()
                if item.summary_kind is SummaryKind.PHASE
            ),
            "diagnostic_summaries": tuple(
                item
                for item in all_observed_summaries()
                if item.summary_kind is SummaryKind.DIAGNOSTIC
            ),
        },
    )
    serialized = repr(asdict(profile)).lower()

    assert "observed" in profile.summary.lower()
    assert "suggests" in profile.summary.lower()
    assert all("observed" in dimension.observation.lower() for dimension in profile.dimensions)
    assert all("suggests" in dimension.observation.lower() for dimension in profile.dimensions)
    assert "score" not in serialized
    assert "diagnosis" not in serialized
    assert "workload" not in serialized
    assert "caused by" not in serialized
    assert "proves" not in serialized


def test_observed_zero_and_missing_are_separate_states():
    profile = build_job_communication_profile(
        capability_rows("partial"),
        provenance=provenance(),
        traffic_summaries=(
            summary(
                "zero-traffic",
                "traffic_intensity",
                status=SummaryStatus.OBSERVED_ZERO,
                value=0,
                zero_observation=True,
            ),
            summary(
                "missing-traffic",
                "traffic_intensity",
                status=SummaryStatus.MISSING,
                value=None,
                coverage_fraction=None,
                missing_observation=True,
            ),
        ),
    )
    traffic = next(
        dimension
        for dimension in profile.dimensions
        if dimension.dimension_id == "traffic_intensity"
    )

    assert profile.zero_observation_count == 1
    assert profile.missing_observation_count == 1
    assert traffic.zero_observation_count == 1
    assert traffic.missing_observation_count == 1
    assert traffic.status is DimensionStatus.PARTIAL
    assert "zero-valued and 1 missing states are reported separately" in traffic.observation
