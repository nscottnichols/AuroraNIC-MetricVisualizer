import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from dash import Dash

from ctf_adapter import (
    CtfAdapterConfig,
    CtfExtractionStats,
    ctf_events_to_normalized_batch,
)
from trace_cache import (
    TraceCacheCancelled,
    read_trace_store_cache,
    write_trace_store_cache,
)
from src.main import (
    _validate_server_bind,
    TraceImportCancelled,
    TraceImportCancellation,
    create_trace_app,
    load_trace_store,
    main,
    parse_args,
    read_requested_counter_names,
    run_trace_app,
    trace_import_cancellation_context,
)


def event():
    return {
        "name": "cxi",
        "provider": "lttng_ust_cxi_sampling",
        "kind": "raw_sampling",
        "timestamp_ns": 100,
        "hostname": "node-a",
        "collector": {"name": "cxi-sampler", "rank": 0, "process": None},
        "payload": {
            "interface_name": "cxi0",
            "counter": "rx_ok_octets",
            "value": 42,
        },
    }


def trace_args(**overrides):
    values = {
        "trace": "/traces/raw",
        "trace_format": "ctf",
        "trace_run_id": "run-1",
        "trace_source_fingerprint": "sha256:test",
        "trace_reader_version": "reader-1",
        "trace_clock_domain": "ctf-clock",
        "trace_synchronization_evidence": "single clock",
        "trace_requested_counter_file": str(
            Path(__file__).resolve().parents[1] / "config" / "cxi-counters.txt"
        ),
        "trace_cache": None,
        "host": "127.0.0.1",
        "port": 8050,
        "allow_non_loopback": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_trace_mode_infers_provenance_and_preserves_explicit_values(tmp_path):
    args = parse_args(['--trace', str(tmp_path)])
    assert args.trace_format == 'ctf'
    assert args.trace_source_fingerprint == 'auto'
    assert args.trace_reader_version == 'auto'
    explicit = parse_args(['--trace', str(tmp_path), '--trace-run-id', 'run-1',
        '--trace-source-fingerprint', 'sha256:test', '--trace-reader-version', 'reader-1',
        '--trace-clock-domain', 'ctf-clock', '--trace-synchronization-evidence', 'single clock'])
    assert explicit.trace_run_id == 'run-1'
    assert explicit.trace_reader_version == 'reader-1'
    with pytest.raises(SystemExit):
        parse_args(['--trace', str(tmp_path/'missing')])


def test_trace_cache_requires_trace_mode():
    with pytest.raises(SystemExit):
        parse_args(["--trace-cache", "/tmp/trace.npz"])


def test_ctf_mode_uses_extraction_and_preserves_metadata(monkeypatch, tmp_path):
    args = trace_args(trace=str(tmp_path))
    captured = {}

    def fake_extract_with_stats(path):
        captured["path"] = path
        return [event()], CtfExtractionStats(cxi_event_count=1, filtered_event_count=0)

    def fake_adapt(events, config):
        captured["config"] = config
        return ctf_events_to_normalized_batch(events, config)

    monkeypatch.setattr(
        "src.main.extract_babeltrace2_events_with_stats", fake_extract_with_stats
    )
    monkeypatch.setattr("src.main.ctf_events_to_normalized_batch", fake_adapt)

    store = load_trace_store(args)

    assert captured["path"] == tmp_path
    assert isinstance(captured["config"], CtfAdapterConfig)
    assert captured["config"].run_id == "run-1"
    assert captured["config"].source_fingerprint == "sha256:test"
    assert captured["config"].reader_version == "reader-1"
    assert captured["config"].clock_domain == "ctf-clock"
    assert captured["config"].synchronization_evidence == "single clock"
    assert len(captured["config"].requested_counter_names) == 213
    assert store.identity.run_id == "run-1"
    assert store.identity.source_format == "babeltrace2-structured-raw-ctf"

    app = create_trace_app(args)
    assert isinstance(app, Dash)
    assert app.title == "Aurora NIC Metrics"


def test_pftrace_mode_uses_actual_adapter_api(monkeypatch, tmp_path):
    args = trace_args(
        trace=str(tmp_path),
        trace_format="pftrace",
        trace_reader_version="trace-processor-1",
        trace_clock_domain="perfetto-clock",
    )
    captured = {}
    batch = ctf_events_to_normalized_batch(
        [event()],
        CtfAdapterConfig(
            run_id="run-1",
            source_fingerprint="sha256:test",
            reader_version="reader-1",
            clock_domain="ctf-clock",
            synchronization_evidence="single clock",
        ),
    )

    def fake_adapt(source, config, *, should_cancel=None):
        captured["source"] = source
        captured["config"] = config
        return batch

    monkeypatch.setattr("pftrace_native.load_pftrace", fake_adapt)
    store = load_trace_store(args)

    assert captured["source"] == tmp_path
    assert captured["config"].reader_version == "perfetto-protobuf-thapi/1"
    assert captured["config"].clock_domain == "perfetto-clock"
    assert store.row_count == 1


def test_trace_cache_hit_avoids_rereading_the_source(monkeypatch, tmp_path):
    cache_path = tmp_path / "trace-cache.npz"
    args = trace_args(trace=str(tmp_path), trace_cache=str(cache_path))

    def fake_extract_with_stats(path):
        return [event()], CtfExtractionStats(cxi_event_count=1, filtered_event_count=0)

    monkeypatch.setattr(
        "src.main.extract_babeltrace2_events_with_stats", fake_extract_with_stats
    )
    store = load_trace_store(args)
    write_trace_store_cache(cache_path, store, args.trace_source_fingerprint)

    def reject_extract(path):
        raise AssertionError("cache hit must not extract the source trace")

    monkeypatch.setattr(
        "src.main.extract_babeltrace2_events_with_stats", reject_extract
    )
    loaded = load_trace_store(args)
    assert loaded.row_count == store.row_count
    assert loaded.identity == store.identity


def test_trace_import_signal_requests_cancellation():
    with trace_import_cancellation_context() as cancellation:
        os.kill(os.getpid(), signal.SIGINT)
        assert cancellation.should_cancel()
        with pytest.raises(
            TraceImportCancelled, match="cancelled by SIGINT before test boundary"
        ):
            cancellation.check("test boundary")


def test_read_requested_counter_names_validates_and_deduplicates(tmp_path):
    path = tmp_path / "counters.txt"
    path.write_text("# comment\n\nrx_ok_octets\ntx_ok_octets\nrx_ok_octets\n")

    with pytest.raises(RuntimeError, match="contains duplicates"):
        read_requested_counter_names(path)

    path.write_text("# comment\n\nrx_ok_octets\ntx_ok_octets\n")
    assert read_requested_counter_names(path) == ("rx_ok_octets", "tx_ok_octets")


def test_trace_import_signal_handlers_are_restored():
    calls = []

    def sentinel_handler(signal_number, frame):
        calls.append(signal_number)

    original_handler = signal.signal(signal.SIGINT, sentinel_handler)
    original_term_handler = signal.signal(signal.SIGTERM, sentinel_handler)
    try:
        with trace_import_cancellation_context():
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGTERM)
        assert signal.getsignal(signal.SIGINT) is sentinel_handler
        assert signal.getsignal(signal.SIGTERM) is sentinel_handler
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGINT, original_handler)
        signal.signal(signal.SIGTERM, original_term_handler)

    assert set(calls) == {signal.SIGINT, signal.SIGTERM}


def test_trace_import_cancels_after_source_extraction(monkeypatch, tmp_path):
    args = trace_args(trace=str(tmp_path))

    with trace_import_cancellation_context() as cancellation:
        def fake_extract(path):
            result = [event()]
            cancellation.request(signal.SIGINT)
            return result, CtfExtractionStats(
                cxi_event_count=1, filtered_event_count=0
            )

        def reject_adapt(events, config):
            raise AssertionError("adapter must not run after cancellation")

        monkeypatch.setattr(
            "src.main.extract_babeltrace2_events_with_stats", fake_extract
        )
        monkeypatch.setattr(
            "src.main.ctf_events_to_normalized_batch", reject_adapt
        )

        with pytest.raises(
            TraceImportCancelled, match="before CTF source extraction completed"
        ):
            load_trace_store(args, cancellation=cancellation)


def test_trace_cache_cancellation_propagation(monkeypatch, tmp_path):
    args = trace_args(trace=str(tmp_path), trace_cache=str(tmp_path / "cache.npz"))
    read_cancel_callback = None

    def fake_read(path, fingerprint, *, should_cancel):
        nonlocal read_cancel_callback
        read_cancel_callback = should_cancel
        raise TraceCacheCancelled("trace cache operation was cancelled")

    cache_path = tmp_path / "cache.npz"
    cache_path.write_bytes(b"placeholder")
    monkeypatch.setattr("src.main.read_trace_store_cache", fake_read)

    with trace_import_cancellation_context() as cancellation:
        with pytest.raises(
            TraceImportCancelled, match="while reading the trace cache"
        ):
            load_trace_store(args, cancellation=cancellation)

    assert read_cancel_callback == cancellation.should_cancel

    cache_path.unlink()
    write_cancel_callback = None

    def fake_extract(path):
        return [event()], CtfExtractionStats(
            cxi_event_count=1, filtered_event_count=0
        )

    def fake_write(path, store, fingerprint, *, should_cancel):
        nonlocal write_cancel_callback
        write_cancel_callback = should_cancel
        raise TraceCacheCancelled("trace cache operation was cancelled")

    monkeypatch.setattr(
        "src.main.extract_babeltrace2_events_with_stats", fake_extract
    )
    monkeypatch.setattr("src.main.write_trace_store_cache", fake_write)

    with trace_import_cancellation_context() as cancellation:
        with pytest.raises(
            TraceImportCancelled, match="while writing the trace cache"
        ):
            load_trace_store(args, cancellation=cancellation)

    assert write_cancel_callback == cancellation.should_cancel


def test_trace_import_cancellation_reports_atomically_published_cache(
    monkeypatch, tmp_path
):
    args = trace_args(trace=str(tmp_path), trace_cache=str(tmp_path / "cache.npz"))

    def fake_extract(path):
        return [event()], CtfExtractionStats(
            cxi_event_count=1, filtered_event_count=0
        )

    def fake_write(path, store, fingerprint, *, should_cancel):
        cancellation.request(signal.SIGINT)
        return None

    monkeypatch.setattr(
        "src.main.extract_babeltrace2_events_with_stats", fake_extract
    )
    monkeypatch.setattr("src.main.write_trace_store_cache", fake_write)

    with trace_import_cancellation_context() as cancellation:
        with pytest.raises(
            TraceImportCancelled,
            match="while publishing the trace cache",
        ) as excinfo:
            load_trace_store(args, cancellation=cancellation)

    assert excinfo.value.cache_published is True


def test_stale_trace_cache_is_rebuilt_and_atomically_replaced(
    monkeypatch, tmp_path, capsys
):
    cache_path = tmp_path / "trace-cache.npz"
    args = trace_args(trace=str(tmp_path), trace_cache=str(cache_path))

    def fake_extract(path):
        return [event()], CtfExtractionStats(
            cxi_event_count=1, filtered_event_count=0
        )

    monkeypatch.setattr(
        "src.main.extract_babeltrace2_events_with_stats", fake_extract
    )
    store = load_trace_store(args)
    old_args = trace_args(
        trace=str(tmp_path),
        trace_cache=str(cache_path),
        trace_source_fingerprint="sha256:old-source",
    )
    old_store = load_trace_store(old_args)
    write_trace_store_cache(
        cache_path, old_store, old_args.trace_source_fingerprint
    )

    rebuilt = load_trace_store(args)
    assert rebuilt.identity == store.identity
    assert "Trace cache is stale; rebuilding" in capsys.readouterr().err
    assert (
        read_trace_store_cache(cache_path, args.trace_source_fingerprint).identity
        == store.identity
    )


def test_cached_trace_provenance_must_match_requested_metadata(monkeypatch, tmp_path):
    cache_path = tmp_path / "trace-cache.npz"
    args = trace_args(trace=str(tmp_path), trace_cache=str(cache_path))

    def fake_extract(path):
        return [event()], CtfExtractionStats(
            cxi_event_count=1, filtered_event_count=0
        )

    monkeypatch.setattr(
        "src.main.extract_babeltrace2_events_with_stats", fake_extract
    )
    store = load_trace_store(args)
    write_trace_store_cache(cache_path, store, args.trace_source_fingerprint)

    mismatched = trace_args(
        trace=str(tmp_path),
        trace_cache=str(cache_path),
        trace_run_id="different-run",
    )
    with pytest.raises(RuntimeError, match="trace cache provenance"):
        load_trace_store(mismatched)


def test_cached_trace_format_must_match_requested_format(monkeypatch, tmp_path):
    cache_path = tmp_path / "trace-cache.npz"
    args = trace_args(trace=str(tmp_path), trace_cache=str(cache_path))

    def fake_extract(path):
        return [event()], CtfExtractionStats(
            cxi_event_count=1, filtered_event_count=0
        )

    monkeypatch.setattr(
        "src.main.extract_babeltrace2_events_with_stats", fake_extract
    )
    store = load_trace_store(args)
    write_trace_store_cache(cache_path, store, args.trace_source_fingerprint)

    mismatched = trace_args(
        trace=str(tmp_path),
        trace_cache=str(cache_path),
        trace_format="pftrace",
    )
    with pytest.raises(RuntimeError, match="source format"):
        load_trace_store(mismatched)


def test_missing_trace_path_and_adapter_errors_are_actionable(monkeypatch, tmp_path):
    args = trace_args(trace=str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="Trace path does not exist"):
        load_trace_store(args)

    args = trace_args(trace=str(tmp_path))

    def failing_extract(path):
        raise RuntimeError("Babeltrace 2 bindings are not installed")

    monkeypatch.setattr(
        "src.main.extract_babeltrace2_events_with_stats", failing_extract
    )
    with pytest.raises(RuntimeError, match="Babeltrace 2 bindings"):
        load_trace_store(args)


def test_non_loopback_requires_explicit_opt_in_and_warns(capsys):
    args = trace_args(host="0.0.0.0")
    with pytest.raises(RuntimeError, match="Refusing non-loopback bind"):
        _validate_server_bind(args)
    assert "use --allow-non-loopback" in capsys.readouterr().err

    args = trace_args(host="0.0.0.0", allow_non_loopback=True)
    _validate_server_bind(args)
    assert "intentionally exposed" in capsys.readouterr().err


def test_trace_app_uses_existing_safe_server_path(monkeypatch, tmp_path):
    args = trace_args(trace=None, browse_root=str(tmp_path), host="127.0.0.1", port=12345)
    app = object()
    server_calls = []

    monkeypatch.setattr(
        "src.main.create_analysis_app", lambda value, **kwargs: app
    )
    monkeypatch.setattr(
        "src.main.run_app",
        lambda value, host, port: server_calls.append((value, host, port)),
    )

    assert run_trace_app(args) is app
    assert server_calls == [(app, "127.0.0.1", 12345)]


def test_main_selects_trace_mode_and_preserves_legacy_mode(monkeypatch, tmp_path):
    trace_calls = []
    monkeypatch.setattr(
        "src.main.run_trace_app", lambda args: trace_calls.append(args)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--trace",
            str(tmp_path),
            "--trace-format",
            "ctf",
            "--trace-run-id",
            "run-1",
            "--trace-source-fingerprint",
            "sha256:test",
            "--trace-reader-version",
            "reader-1",
            "--trace-clock-domain",
            "ctf-clock",
            "--trace-synchronization-evidence",
            "single clock",
        ],
    )
    main()
    assert trace_calls[-1].trace == str(tmp_path)

    legacy_calls = []
    monkeypatch.setattr(
        "src.main.run_interactive_dash_app",
        lambda *args, **kwargs: legacy_calls.append(args),
    )
    monkeypatch.setattr(
        "src.main._load_metric_data",
        lambda *args: ({}, {}, {}, object(), [], [], []),
    )
    monkeypatch.setattr(
        "src.main._load_benchmark_data",
        lambda *args: (object(), [], []),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--base_dir",
            str(tmp_path),
            "--metric_names_file",
            "names.txt",
            "--benchmark",
            "oneccl",
        ],
    )
    main()
    assert legacy_calls


@pytest.mark.parametrize(
    "signal_number,expected_status",
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_main_reports_actionable_trace_import_cancellation(
    monkeypatch, capsys, signal_number, expected_status
):
    monkeypatch.setattr("src.main.parse_args", lambda: trace_args())

    def cancelled_run(args):
        raise TraceImportCancelled(
            "Trace import cancelled.", signal_number=signal_number
        )

    monkeypatch.setattr("src.main.run_trace_app", cancelled_run)
    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == expected_status
    output = capsys.readouterr()
    signal_name = signal.Signals(signal_number).name
    assert (
        f"Trace import cancelled by {signal_name}; "
        f"exit status {expected_status}."
    ) in output.err
    assert "left unchanged" in output.err
    assert "Traceback" not in output.err


def test_main_reports_published_cache_when_cancellation_follows_publication(
    monkeypatch, capsys
):
    monkeypatch.setattr("src.main.parse_args", lambda: trace_args())
    monkeypatch.setattr(
        "src.main.run_trace_app",
        lambda args: (_ for _ in ()).throw(
            TraceImportCancelled(
                "Trace import cancelled.",
                signal_number=signal.SIGINT,
                cache_published=True,
            )
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 130
    output = capsys.readouterr()
    assert "cache was atomically published before cancellation" in output.err
    assert "existing cache were left unchanged" not in output.err
