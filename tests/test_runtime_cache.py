import json
import pytest
from unittest import mock

import numpy as np

from app import run_app
from callbacks import MAX_CALCULATED_METRIC_CACHE_ENTRIES
import main


def test_json_cache_round_trip_and_fingerprint_stability(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("initial", encoding="utf-8")
    fingerprint = main.source_fingerprint([source])
    cache_path = tmp_path / "cache.json"
    payload = {"values": np.array([1, 2, 3])}

    main.write_versioned_cache(cache_path, payload, fingerprint)

    assert main.source_fingerprint([source]) == fingerprint
    restored = main.read_versioned_cache(cache_path, fingerprint)
    assert restored == {"values": [1, 2, 3]}

    source.write_text("changed", encoding="utf-8")
    changed_fingerprint = main.source_fingerprint([source])
    assert changed_fingerprint != fingerprint
    try:
        main.read_versioned_cache(cache_path, changed_fingerprint)
    except main.CacheError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("Expected a fingerprint mismatch")


def test_cache_errors_are_actionable(tmp_path):
    missing_path = tmp_path / "missing.json"
    try:
        main.read_versioned_cache(missing_path, "0" * 64)
    except main.CacheError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected a missing-cache error")

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{", encoding="utf-8")
    try:
        main.read_versioned_cache(malformed_path, "0" * 64)
    except main.CacheError as exc:
        assert "malformed" in str(exc)
    else:
        raise AssertionError("Expected a malformed-cache error")

    incompatible_path = tmp_path / "incompatible.json"
    incompatible_path.write_text(
        json.dumps(
            {
                "cache_format": main.CACHE_FORMAT,
                "schema_version": 0,
                "source_fingerprint": "0" * 64,
                "payload": {},
            }
        ),
        encoding="utf-8",
    )
    try:
        main.read_versioned_cache(incompatible_path, "0" * 64)
    except main.CacheError as exc:
        assert "schema version" in str(exc)
    else:
        raise AssertionError("Expected a schema-version error")


@pytest.mark.parametrize("value", [1.9, True, "1", None])
def test_metric_cache_rejects_non_integer_values(value):
    with pytest.raises(main.CacheError, match="int64"):
        main._validate_numeric_tree([[value]], "metric array", integer=True)


@pytest.mark.parametrize("value", [True, "1.0", float("nan"), float("inf")])
def test_benchmark_cache_rejects_invalid_numbers(value):
    with pytest.raises(main.CacheError, match="finite numbers"):
        main._validate_numeric_tree([[value]], "benchmark array", integer=False)


def test_legacy_pickle_is_not_imported():
    assert not hasattr(main, "pickle")


def test_cli_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("AURORA_APP_HOST", raising=False)
    monkeypatch.delenv("AURORA_APP_PORT", raising=False)

    args = main.parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8050


def test_calculated_metric_cache_is_bounded():
    assert 1 <= MAX_CALCULATED_METRIC_CACHE_ENTRIES <= 32


def _minimal_runtime_app():
    metric_array = np.zeros((1, 1, 1, 1, 1), dtype=np.int64)
    bench_array = np.zeros((1, 1, 3, 4), dtype=np.float64)
    return main.create_runtime_app(
        metric_array=metric_array,
        node_counts=[1],
        element_counts=[1],
        metric_names=["metric"],
        node_map={"node-0": 0},
        results_nodes={(1, 1): [0]},
        bench_array=bench_array,
        bench_node_counts=[1],
        bench_elements=[1],
    )


def test_runtime_app_exposes_health():
    app = _minimal_runtime_app()

    response = app.server.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_runner_serves_health_and_stops_cleanly():
    app = _minimal_runtime_app()

    with mock.patch.object(app, "run") as run:
        run_app(app, host="127.0.0.1", port=8050)

    run.assert_called_once_with(host="127.0.0.1", port=8050, debug=False, use_reloader=False)
