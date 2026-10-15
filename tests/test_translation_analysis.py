from translation_analysis import (
    TrafficDirection,
    TranslationCounterIdentity,
    TranslationCounterRole,
    TranslationCounterSeries,
    TranslationCounterSpec,
    TranslationDataset,
    TranslationMissView,
    TranslationPopulation,
    TranslationQualityFlag,
    TranslationStatus,
    TranslationTrafficTotal,
    ValueKind,
    analyze_translation,
)


SECOND = 1_000_000_000
POPULATION = TranslationPopulation(
    run_id="run-1",
    node_id="node-1",
    nic_id="nic-1",
    collector_id="collector-1",
)


def spec(
    role,
    counter_id,
    unit="events",
    population_id="atu",
    miss_view=TranslationMissView.POOL,
):
    return TranslationCounterSpec(
        counter_id=counter_id,
        role=role,
        unit=unit,
        population_id=population_id,
        miss_view=miss_view if role is TranslationCounterRole.MISS else None,
    )


def series(counter_spec, timestamps, values, **overrides):
    arguments = {
        "spec": counter_spec,
        "identity": TranslationCounterIdentity(
            run_id=POPULATION.run_id,
            node_id=POPULATION.node_id,
            nic_id=POPULATION.nic_id,
            collector_id=POPULATION.collector_id,
            source_counter_id=counter_spec.counter_id,
        ),
        "timestamps_ns": tuple(timestamps),
        "values": tuple(values),
        "value_kind": ValueKind.CUMULATIVE,
        "semantics": "cumulative counter",
        "baseline_relative": False,
    }
    arguments.update(overrides)
    return TranslationCounterSeries(**arguments)


def traffic(total=2, unit="GB", window=(0, 2 * SECOND), population_id="atu"):
    return TranslationTrafficTotal(
        total=total,
        unit=unit,
        direction=TrafficDirection.BOTH,
        population_id=population_id,
        window_ns=window,
        source_counter_ids=("tx_bytes", "rx_bytes"),
    )


def analyze(specs, observations, window=(0, 2 * SECOND), traffic_total=None):
    return analyze_translation(
        TranslationDataset(
            population=POPULATION,
            requested_counters=tuple(specs),
            series=tuple(observations),
            window_ns=window,
            traffic_total=traffic_total,
        )
    )


def base_specs(miss_view=TranslationMissView.POOL):
    request = spec(TranslationCounterRole.REQUEST, "atu_client_req")
    hit = spec(TranslationCounterRole.HIT, "atu_hits")
    miss = spec(
        TranslationCounterRole.MISS,
        "atu_miss_by_pool"
        if miss_view is TranslationMissView.POOL
        else "atu_miss_by_client",
        miss_view=miss_view,
    )
    eviction = spec(TranslationCounterRole.EVICTION, "atu_evictions")
    return request, hit, miss, eviction


def test_exact_nonzero_deltas_ratios_and_evictions():
    request, hit, miss, eviction = base_specs()
    timestamps = (0, SECOND, 2 * SECOND)
    result = analyze(
        [request, hit, miss, eviction],
        [
            series(request, timestamps, (0, 10, 15)),
            series(hit, timestamps, (0, 8, 12)),
            series(miss, timestamps, (0, 2, 3)),
            series(eviction, timestamps, (0, 1, 1)),
        ],
        traffic_total=traffic(total=2),
    )
    view = result.view_results[0]

    assert [interval.delta for interval in result.counter_results[0].intervals] == [10, 5]
    assert view.request_total == 15
    assert view.hit_total == 12
    assert view.miss_total == 3
    assert view.hit_denominator == 15
    assert view.hit_ratio == 12 / 15
    assert view.request_denominator == 15
    assert view.misses_per_request == 3 / 15
    assert view.traffic_total_gb == 2.0
    assert view.misses_per_gb == 1.5
    assert result.eviction_total == 1
    assert view.status is TranslationStatus.OBSERVED_NONZERO
    assert result.coverage_fraction == 1.0
    assert result.hit_ratio_formula == "hit_total / (hit_total + miss_total)"


def test_zero_is_observed_and_zero_denominators_are_unavailable():
    request, hit, miss, eviction = base_specs()
    timestamps = (0, SECOND, 2 * SECOND)
    result = analyze(
        [request, hit, miss, eviction],
        [
            series(request, timestamps, (0, 0, 0)),
            series(hit, timestamps, (0, 0, 0)),
            series(miss, timestamps, (0, 0, 0)),
            series(eviction, timestamps, (0, 0, 0)),
        ],
        traffic_total=traffic(total=0),
    )
    view = result.view_results[0]

    assert view.request_total == 0
    assert view.hit_total == 0
    assert view.miss_total == 0
    assert view.zero_observed is True
    assert view.persistent_zero is True
    assert view.hit_ratio is None
    assert view.hit_denominator == 0
    assert view.misses_per_request is None
    assert view.misses_per_gb is None
    assert view.status is TranslationStatus.OBSERVED_ZERO
    assert TranslationQualityFlag.ZERO_DENOMINATOR in view.quality_flags
    assert result.eviction_total == 0
    assert result.eviction_zero_observed is True


def test_reset_interval_is_excluded_from_scientific_totals():
    request, hit, miss, eviction = base_specs()
    result = analyze(
        [request, hit, miss, eviction],
        [
            series(request, (0, SECOND, 2 * SECOND), (0, 10, 4)),
            series(hit, (0, SECOND, 2 * SECOND), (0, 8, 8)),
            series(miss, (0, SECOND, 2 * SECOND), (0, 2, 2)),
            series(eviction, (0, SECOND, 2 * SECOND), (0, 1, 1)),
        ],
        traffic_total=traffic(total=1, window=(0, SECOND)),
    )
    request_result = result.counter_results[0]
    view = result.view_results[0]

    assert [interval.delta for interval in request_result.intervals] == [10, -6]
    assert [interval.valid for interval in request_result.intervals] == [True, False]
    assert request_result.total == 10
    assert request_result.status is TranslationStatus.PARTIAL_INVALID_DATA
    assert TranslationQualityFlag.DECREASE_OR_RESET in request_result.quality_flags
    assert view.hit_ratio == 8 / 10
    assert view.misses_per_request is None
    assert view.request_denominator is None
    assert TranslationQualityFlag.INCOMPATIBLE_COVERAGE in view.quality_flags
    assert view.covered_duration_ns == SECOND
    assert view.coverage_fraction == 0.5


def test_missing_request_denominator_is_not_zero():
    request, hit, miss, _ = base_specs()
    result = analyze(
        [request, hit, miss],
        [
            series(hit, (0, SECOND, 2 * SECOND), (0, 8, 12)),
            series(miss, (0, SECOND, 2 * SECOND), (0, 2, 3)),
        ],
    )
    view = result.view_results[0]

    assert view.request_total is None
    assert view.request_denominator is None
    assert view.misses_per_request is None
    assert view.hit_ratio == 12 / 15
    assert view.missing_counter_ids == (request.counter_id,)
    assert view.status is TranslationStatus.PARTIAL_MISSING_DATA
    assert result.counter_results[0].zero_observed is False


def test_pool_and_client_misses_stay_separate_without_an_overlap_total():
    request, hit, pool_miss, eviction = base_specs()
    client_miss = spec(
        TranslationCounterRole.MISS,
        "atu_miss_by_client",
        miss_view=TranslationMissView.CLIENT,
    )
    timestamps = (0, SECOND, 2 * SECOND)
    result = analyze(
        [request, hit, pool_miss, client_miss, eviction],
        [
            series(request, timestamps, (0, 10, 15)),
            series(hit, timestamps, (0, 8, 12)),
            series(pool_miss, timestamps, (0, 2, 3)),
            series(client_miss, timestamps, (0, 4, 5)),
            series(eviction, timestamps, (0, 0, 1)),
        ],
    )

    assert not hasattr(result, "miss_total")
    assert [view.miss_view for view in result.view_results] == [
        TranslationMissView.POOL,
        TranslationMissView.CLIENT,
    ]
    assert [view.miss_total for view in result.view_results] == [3, 5]
    assert [view.hit_ratio for view in result.view_results] == [12 / 15, 12 / 17]
    assert [view.misses_per_request for view in result.view_results] == [
        3 / 15,
        5 / 15,
    ]
    assert result.eviction_total == 1


def test_unsupported_series_remains_distinct_from_missing_counter():
    request, hit, miss, _ = base_specs()
    timestamps = (0, SECOND, 2 * SECOND)
    unsupported = series(
        miss,
        timestamps,
        (0, 1.5, 2.5),
        value_kind=ValueKind.UNKNOWN,
        quality_flags=(TranslationQualityFlag.UNSUPPORTED_SERIES,),
    )

    result = analyze(
        [request, hit, miss],
        [
            series(request, timestamps, (0, 10, 15)),
            series(hit, timestamps, (0, 8, 12)),
            unsupported,
        ],
    )
    miss_result = result.counter_results[2]
    view = result.view_results[0]

    assert miss_result.status is TranslationStatus.UNSUPPORTED
    assert miss_result.total is None
    assert miss_result.sample_count == 3
    assert miss_result.intervals == ()
    assert view.miss_total is None
    assert view.missing_counter_ids == ()
    assert TranslationQualityFlag.UNSUPPORTED_SERIES in view.quality_flags


def test_requested_window_sample_count_is_window_local():
    _, _, miss, _ = base_specs()
    result = analyze(
        [miss],
        [
            series(
                miss,
                (0, SECOND, 2 * SECOND, 3 * SECOND),
                (0, 2, 3, 4),
            )
        ],
        window=(500_000_000, 2_500_000_000),
    )

    assert result.counter_results[0].sample_count == 2
    assert [
        (interval.start_ns, interval.end_ns) for interval in result.counter_results[0].intervals
    ] == [(SECOND, 2 * SECOND)]
    assert result.counter_results[0].valid_interval_count == 1
    assert result.counter_results[0].outside_window_interval_count == 2
    assert result.counter_results[0].total == 1
    assert result.counter_results[0].covered_duration_ns == SECOND
    assert result.counter_results[0].coverage_fraction == 0.5


def test_incompatible_traffic_unit_supplies_no_misses_per_gb():
    request, hit, miss, _ = base_specs()
    result = analyze(
        [request, hit, miss],
        [
            series(request, (0, SECOND, 2 * SECOND), (0, 10, 15)),
            series(hit, (0, SECOND, 2 * SECOND), (0, 8, 12)),
            series(miss, (0, SECOND, 2 * SECOND), (0, 2, 3)),
        ],
        traffic_total=traffic(total=100, unit="packets"),
    )
    view = result.view_results[0]

    assert view.misses_per_gb is None
    assert view.traffic_status.value == "incompatible_unit"
    assert TranslationQualityFlag.INCOMPATIBLE_TRAFFIC_UNIT in view.quality_flags
    assert view.hit_ratio == 12 / 15
    assert view.misses_per_request == 3 / 15


def test_boundary_selection_uses_only_complete_intervals_inside_window():
    request, hit, miss, _ = base_specs()
    timestamps = (0, SECOND, 2 * SECOND, 3 * SECOND)
    result = analyze(
        [request, hit, miss],
        [
            series(request, timestamps, (0, 10, 15, 20)),
            series(hit, timestamps, (0, 8, 12, 17)),
            series(miss, timestamps, (0, 2, 3, 3)),
        ],
        window=(500_000_000, 2 * SECOND),
        traffic_total=traffic(
            total=1,
            window=(SECOND, 2 * SECOND),
        ),
    )
    request_result = result.counter_results[0]
    view = result.view_results[0]

    assert [(interval.start_ns, interval.end_ns) for interval in request_result.intervals] == [
        (SECOND, 2 * SECOND)
    ]
    assert request_result.total == 5
    assert view.hit_total == 4
    assert view.miss_total == 1
    assert view.hit_ratio == 4 / 5
    assert view.misses_per_request == 1 / 5
    assert view.misses_per_gb == 1.0
    assert view.covered_duration_ns == SECOND
    assert view.coverage_fraction == 2 / 3
    assert TranslationQualityFlag.BOUNDARY_INTERVAL_EXCLUDED in request_result.quality_flags
