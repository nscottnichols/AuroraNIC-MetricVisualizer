import pytest

from operation_analysis import (
    DMA_OPCODE_COUNTER_GROUP,
    OpcodeMapping,
    OpcodeMappingEntry,
    OpcodeMappingStatus,
    OperationCounterKind,
    OperationCounterSeries,
    OperationCounterSpec,
    OperationDataset,
    OperationPopulation,
    OperationSeriesIdentity,
    OperationStatus,
    OperationValueKind,
    OperationQualityFlag,
    analyze_operations,
    dma_opcode_spec,
    dma_total_spec,
    idc_total_spec,
)


SECOND = 1_000_000_000
POPULATION = OperationPopulation(
    run_id="run-1",
    node_id="node-1",
    nic_id="nic-1",
    collector_id="collector-1",
)
OPCODE_MAPPING = OpcodeMapping(
    mapping_version="authoritative-fixture-v1",
    evidence="driver documentation fixture",
    counter_group=DMA_OPCODE_COUNTER_GROUP,
    entries=(
        OpcodeMappingEntry(bin_index=0, label="PUT"),
        OpcodeMappingEntry(bin_index=1, label="GET"),
    ),
)


def series(
    series_spec: OperationCounterSpec,
    timestamps,
    values,
    **overrides,
):
    arguments = {
        "spec": series_spec,
        "identity": OperationSeriesIdentity(
            run_id=POPULATION.run_id,
            node_id=POPULATION.node_id,
            nic_id=POPULATION.nic_id,
            collector_id=POPULATION.collector_id,
            source_counter_id=series_spec.counter_id,
        ),
        "timestamps_ns": tuple(timestamps),
        "values": tuple(values),
        "value_kind": OperationValueKind.CUMULATIVE,
        "semantics": "cumulative counter",
        "baseline_relative": False,
    }
    arguments.update(overrides)
    return OperationCounterSeries(**arguments)


def analyze(
    specs,
    series_items,
    window=(0, 2 * SECOND),
    opcode_mapping=OPCODE_MAPPING,
):
    return analyze_operations(
        OperationDataset(
            population=POPULATION,
            requested_counters=tuple(specs),
            series=tuple(series_items),
            window_ns=window,
            opcode_mapping=opcode_mapping,
        )
    )


def result_for(result, counter_id):
    return next(
        item
        for item in (
            *result.dma_opcode_bin_results,
            *result.dma_total_results,
            *result.idc_total_results,
        )
        if item.spec.counter_id == counter_id
    )


def opcode_for(result, counter_id):
    return next(
        item for item in result.opcode_results if item.counter_id == counter_id
    )


def test_exact_nonzero_deltas_rates_and_command_mix():
    specs = (
        dma_opcode_spec(0),
        dma_opcode_spec(1),
        dma_total_spec(0),
        idc_total_spec(0),
    )
    result = analyze(
        specs,
        (
            series(dma_opcode_spec(0), (0, SECOND, 2 * SECOND), (0, 2, 5)),
            series(dma_opcode_spec(1), (0, SECOND, 2 * SECOND), (0, 3, 3)),
            series(dma_total_spec(0), (0, SECOND, 2 * SECOND), (0, 10, 16)),
            series(idc_total_spec(0), (0, SECOND, 2 * SECOND), (0, 1, 4)),
        ),
    )

    put = opcode_for(result, "cq_dma_cmd_counts_0")
    get = opcode_for(result, "cq_dma_cmd_counts_1")
    dma_total = result_for(result, "cq_num_dma_cmds_0")
    idc_total = result_for(result, "cq_num_idc_cmds_0")

    assert [interval.delta for interval in put.counter_result.intervals] == [2, 3]
    assert [interval.delta for interval in get.counter_result.intervals] == [3, 0]
    assert put.total == 5
    assert put.rate_per_second == pytest.approx(2.5)
    assert get.total == 3
    assert get.rate_per_second == pytest.approx(1.5)
    assert put.fraction_of_dma_opcode_commands == pytest.approx(5 / 8)
    assert get.fraction_of_dma_opcode_commands == pytest.approx(3 / 8)
    assert dma_total.total == 16
    assert dma_total.rate_per_second == pytest.approx(8.0)
    assert idc_total.total == 4
    assert idc_total.rate_per_second == pytest.approx(2.0)
    assert result.aggregate_dma_commands.total == 16
    assert result.aggregate_idc_commands.total == 4
    assert result.dma_idc_mix.total_commands == 20
    assert result.dma_idc_mix.dma_fraction == pytest.approx(0.8)
    assert result.dma_idc_mix.idc_fraction == pytest.approx(0.2)
    assert result.dma_idc_mix.coverage_fraction == 1.0
    assert result.opcode_mapping_status is OpcodeMappingStatus.ALL_MAPPED
    assert "not wire bytes" in " ".join(result.limitations)


def test_zero_commands_are_distinct_from_missing_command_counters():
    specs = (
        dma_opcode_spec(0),
        dma_opcode_spec(1),
        dma_total_spec(0),
        idc_total_spec(0),
    )
    result = analyze(
        specs,
        (series(dma_opcode_spec(0), (0, SECOND, 2 * SECOND), (0, 0, 0)),),
    )

    zero_bin = result_for(result, "cq_dma_cmd_counts_0")
    missing_bin = result_for(result, "cq_dma_cmd_counts_1")
    missing_dma_total = result_for(result, "cq_num_dma_cmds_0")
    missing_idc_total = result_for(result, "cq_num_idc_cmds_0")

    assert zero_bin.total == 0
    assert zero_bin.rate_per_second == 0.0
    assert zero_bin.zero_observed is True
    assert zero_bin.status is OperationStatus.OBSERVED_ZERO
    assert missing_bin.total is None
    assert missing_bin.zero_observed is False
    assert missing_bin.status is OperationStatus.MISSING_DATA
    assert missing_dma_total.status is OperationStatus.MISSING_DATA
    assert missing_idc_total.status is OperationStatus.MISSING_DATA
    assert result.aggregate_dma_commands.total is None
    assert result.aggregate_idc_commands.total is None
    assert result.dma_idc_mix.total_commands is None
    assert result.status is OperationStatus.PARTIAL_MISSING_DATA
    assert result.missing_opcode_counter_ids == (
        "cq_dma_cmd_counts_1",
    )


def test_reset_interval_is_excluded_from_total_and_coverage():
    spec = dma_opcode_spec(0)
    result = analyze(
        (spec,),
        (
            series(
                spec,
                (0, SECOND, 2 * SECOND, 3 * SECOND),
                (0, 5, 2, 8),
                quality_flags=(),
            ),
        ),
        window=(0, 3 * SECOND),
    )
    counter = result_for(result, "cq_dma_cmd_counts_0")

    assert [interval.delta for interval in counter.intervals] == [5, -3, 6]
    assert [interval.valid for interval in counter.intervals] == [True, False, True]
    assert counter.total == 11
    assert counter.covered_duration_ns == 2 * SECOND
    assert counter.coverage_fraction == pytest.approx(2 / 3)
    assert counter.rate_per_second == pytest.approx(5.5)
    assert counter.invalid_interval_count == 1
    assert counter.status is OperationStatus.PARTIAL_INVALID_DATA
    assert OperationQualityFlag.DECREASE_OR_RESET in counter.quality_flags


def test_mapped_and_unknown_opcodes_remain_explicit():
    specs = (dma_opcode_spec(0), dma_opcode_spec(1), dma_opcode_spec(2))
    result = analyze(
        specs,
        (
            series(dma_opcode_spec(0), (0, SECOND, 2 * SECOND), (0, 2, 5)),
            series(dma_opcode_spec(1), (0, SECOND, 2 * SECOND), (0, 3, 3)),
            series(dma_opcode_spec(2), (0, SECOND, 2 * SECOND), (0, 1, 4)),
        ),
    )

    put = opcode_for(result, "cq_dma_cmd_counts_0")
    get = opcode_for(result, "cq_dma_cmd_counts_1")
    unknown = opcode_for(result, "cq_dma_cmd_counts_2")

    assert result.opcode_mapping_status is OpcodeMappingStatus.PARTIAL_UNKNOWN_OPCODE
    assert result.opcode_mapping_version == "authoritative-fixture-v1"
    assert put.opcode_label == "PUT"
    assert get.opcode_label == "GET"
    assert unknown.opcode_label is None
    assert unknown.opcode_mapped is False
    assert result.mapped_opcode_total == 8
    assert result.unknown_opcode_total == 4
    assert result.unmapped_opcode_counter_ids == ("cq_dma_cmd_counts_2",)
    assert OperationQualityFlag.UNKNOWN_OPCODE in result.quality_flags


def test_incompatible_opcode_mapping_does_not_label_bins():
    incompatible_mapping = OpcodeMapping(
        mapping_version="wrong-group-v1",
        evidence="fixture for an incompatible counter group",
        counter_group="some_other_counter_group",
        entries=(OpcodeMappingEntry(bin_index=0, label="PUT"),),
    )
    result = analyze(
        (dma_opcode_spec(0),),
        (series(dma_opcode_spec(0), (0, SECOND, 2 * SECOND), (0, 2, 5)),),
        opcode_mapping=incompatible_mapping,
    )
    opcode = opcode_for(result, "cq_dma_cmd_counts_0")

    assert result.opcode_mapping_status is OpcodeMappingStatus.INCOMPATIBLE_MAPPING
    assert opcode.opcode_label is None
    assert opcode.opcode_mapped is False
    assert result.mapped_opcode_total is None
    assert result.unknown_opcode_total is None
    assert OperationQualityFlag.INCOMPATIBLE_OPCODE_MAPPING in result.quality_flags


def test_boundary_intervals_are_excluded_by_exact_window_selection():
    spec = dma_opcode_spec(0)
    result = analyze(
        (spec,),
        (series(spec, (0, SECOND, 2 * SECOND, 3 * SECOND), (0, 3, 8, 15)),),
        window=(500_000_000, 2_500_000_000),
    )
    counter = result_for(result, "cq_dma_cmd_counts_0")

    assert [interval.delta for interval in counter.intervals] == [3, 5, 7]
    assert [interval.inside_window for interval in counter.intervals] == [
        False,
        True,
        False,
    ]
    assert counter.total == 5
    assert counter.covered_duration_ns == SECOND
    assert counter.coverage_fraction == pytest.approx(0.5)
    assert counter.outside_window_interval_count == 2
    assert OperationQualityFlag.BOUNDARY_INTERVAL_EXCLUDED in counter.quality_flags


def test_requested_window_sample_count_is_window_local():
    spec = dma_opcode_spec(0)
    result = analyze(
        (spec,),
        (series(spec, (0, SECOND, 2 * SECOND, 3 * SECOND), (0, 3, 8, 15)),),
        window=(500_000_000, 2_500_000_000),
    )
    counter = result_for(result, "cq_dma_cmd_counts_0")

    assert counter.sample_count == 2


def test_baseline_relative_samples_do_not_claim_exact_deltas():
    spec = dma_opcode_spec(0)
    result = analyze(
        (spec,),
        (
            series(
                spec,
                (0, SECOND, 2 * SECOND),
                (0, 2, 5),
                baseline_relative=True,
            ),
        ),
    )
    counter = result_for(result, "cq_dma_cmd_counts_0")

    assert counter.status is OperationStatus.UNSUPPORTED
    assert counter.total is None
    assert counter.rate_per_second is None
    assert counter.intervals == ()
    assert OperationQualityFlag.BASELINE_RELATIVE in counter.quality_flags


def test_conflicting_duplicate_timestamps_invalidate_adjacent_intervals_and_total():
    spec = dma_opcode_spec(0)
    result = analyze(
        (spec,),
        (series(spec, (0, SECOND, SECOND, 2 * SECOND), (0, 5, 2, 3)),),
    )
    counter = result_for(result, "cq_dma_cmd_counts_0")

    assert counter.status is OperationStatus.UNSUPPORTED
    assert counter.total is None
    assert counter.rate_per_second is None
    assert counter.intervals == ()
    assert counter.valid_interval_count == 0
    assert counter.invalid_interval_count == 0
    assert OperationQualityFlag.CONFLICTING_TIMESTAMP in counter.quality_flags
    assert "adjacent interval deltas unsupported" in counter.status_reasons[0]
