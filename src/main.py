import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
import os
import signal
import sys
import uuid
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app import create_app, run_app
from explorer.app import create_explorer_app as create_analysis_app
from ctf_adapter import (
    CtfAdapterConfig,
    RAW_CTF_SOURCE_FORMAT,
    ctf_events_to_normalized_batch,
    extract_babeltrace2_events_with_stats,
)
from metrics_processing import prepare_metric_array, process_all_jobs
from pftrace_adapter import (
    PFTRACE_SOURCE_FORMAT,
    PftraceAdapterConfig,
    adapt_perfetto_trace_processor,
)
from oneccl_processing import (
    prepare_benchmark_array,
    process_oneccl_benchmarks,
)
from osu_processing import process_osu_benchmarks
from trace_cache import (
    TraceCacheCorruptError,
    TraceCacheCancelled,
    TraceCacheStaleError,
    read_trace_store_cache,
    write_trace_store_cache,
)
from trace_store import TraceStore
from dataset_lifecycle import DatasetDescriptor, DatasetLifecycleManager
from performance_metrics import PerformanceMetricsRecorder


CACHE_FORMAT = "aurora-nic-metric-visualizer/cache"
CACHE_SCHEMA_VERSION = 1
DEFAULT_REQUESTED_COUNTER_FILE = (
    Path(__file__).resolve().parents[1] / "config" / "cxi-counters.txt"
)
_CACHE_FILE_NAMES = {
    "cached_data.json",
    "cached_data.pkl",
    "cached_bench_oneccl.json",
    "cached_bench_oneccl.pkl",
    "cached_bench_osu.json",
    "cached_bench_osu.pkl",
}


class CacheError(RuntimeError):
    """Raised when a cache is missing, malformed, or incompatible."""


class TraceImportCancelled(RuntimeError):
    """Raised when cooperative trace import cancellation is requested."""

    def __init__(self, message, signal_number=None, cache_published=False):
        super().__init__(message)
        self.signal_number = signal_number
        self.cache_published = cache_published

    @property
    def exit_status(self):
        if self.signal_number is None:
            return 130
        return 128 + self.signal_number


class TraceImportCancellation:
    def __init__(self):
        self._event = threading.Event()
        self._signal_number = None

    def request(self, signal_number=None):
        self._signal_number = signal_number
        self._event.set()

    def should_cancel(self):
        return self._event.is_set()

    def check(self, stage):
        if self._event.is_set():
            signal_name = (
                signal.Signals(self._signal_number).name
                if self._signal_number is not None
                else "request"
            )
            raise TraceImportCancelled(
                f"Trace import cancelled by {signal_name} before {stage} completed.",
                signal_number=self._signal_number,
            )


@contextmanager
def trace_import_cancellation_context():
    cancellation = TraceImportCancellation()
    installed_handlers = []

    def request_cancellation(signal_number, _frame):
        cancellation.request(signal_number)

    if threading.current_thread() is threading.main_thread():
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handler = signal.signal(
                    signal_number, request_cancellation
                )
            except (OSError, ValueError):
                continue
            installed_handlers.append((signal_number, previous_handler))

    try:
        yield cancellation
    finally:
        for signal_number, previous_handler in reversed(installed_handlers):
            try:
                signal.signal(signal_number, previous_handler)
            except (OSError, ValueError):
                pass


def _is_cache_file(path):
    return path.name in _CACHE_FILE_NAMES


def _iter_source_files(paths):
    seen = set()
    for root in paths:
        root_path = Path(root)
        if root_path.is_file():
            candidates = [root_path]
        elif root_path.is_dir():
            candidates = sorted(root_path.rglob("*"))
        else:
            continue

        for candidate in candidates:
            if not candidate.is_file() or _is_cache_file(candidate):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def source_fingerprint(paths):
    """Return a stable SHA-256 fingerprint for source files and directories."""
    digest = hashlib.sha256()
    digest.update(CACHE_FORMAT.encode("utf-8"))
    digest.update(str(CACHE_SCHEMA_VERSION).encode("ascii"))
    digest.update(b"\0")

    for path in _iter_source_files(paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source_file:
            while True:
                chunk = source_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")

    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_versioned_cache(path, payload, fingerprint):
    """Atomically write a versioned JSON cache document."""
    document = {
        "cache_format": CACHE_FORMAT,
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_fingerprint": fingerprint,
        "payload": _json_ready(payload),
    }
    try:
        serialized = json.dumps(
            document,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise CacheError(f"Cache payload cannot be encoded as JSON: {exc}") from exc

    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as cache_file:
            cache_file.write(serialized)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    except OSError as exc:
        raise CacheError(f"Unable to write cache {path}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def read_versioned_cache(path, expected_fingerprint):
    """Read and validate a versioned JSON cache document."""
    path = Path(path)
    if not path.is_file():
        raise CacheError(f"Cache file is missing: {path}")

    try:
        with path.open("r", encoding="utf-8") as cache_file:
            document = json.load(cache_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheError(f"Cache file is malformed: {path}: {exc}") from exc

    if not isinstance(document, dict):
        raise CacheError(f"Cache document must be a JSON object: {path}")
    if document.get("cache_format") != CACHE_FORMAT:
        raise CacheError(
            f"Cache format is unsupported: {path}: "
            f"expected {CACHE_FORMAT!r}, got {document.get('cache_format')!r}"
        )
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != CACHE_SCHEMA_VERSION
    ):
        raise CacheError(
            f"Cache schema version is unsupported: {path}: "
            f"expected {CACHE_SCHEMA_VERSION}, got {document.get('schema_version')!r}"
        )

    fingerprint = document.get("source_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise CacheError(f"Cache source fingerprint is malformed: {path}")
    if fingerprint != expected_fingerprint:
        raise CacheError(
            f"Cache source fingerprint does not match current inputs: {path}"
        )

    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise CacheError(f"Cache payload is malformed: {path}")
    return payload


def _encode_metric_payload(
    node_map,
    job_results,
    job_results_nodes,
    metric_array,
    node_counts,
    element_counts,
    metric_names,
):
    return _json_ready({
        "node_map": node_map,
        "job_results": [
            {"key": [key[0], key[1]], "values": values}
            for key, values in job_results.items()
        ],
        "job_results_nodes": [
            {"key": [key[0], key[1]], "values": values}
            for key, values in job_results_nodes.items()
        ],
        "metric_array": metric_array,
        "node_counts": node_counts,
        "element_counts": element_counts,
        "metric_names": metric_names,
    })


def _require_int(value, label):
    if type(value) is not int:
        raise CacheError(f"Cache {label} must be an integer")
    return value


def _validate_numeric_tree(value, label, integer):
    if not isinstance(value, list):
        raise CacheError(f"Cache {label} must be a nested list")

    minimum = np.iinfo(np.int64).min if integer else None
    maximum = np.iinfo(np.int64).max if integer else None

    def validate(item):
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if integer:
            if type(item) is not int or not minimum <= item <= maximum:
                raise CacheError(f"Cache {label} must contain int64 integers")
            return
        if item is None:
            return
        if type(item) not in (int, float) or not math.isfinite(item):
            raise CacheError(
                f"Cache {label} must contain finite numbers or null for missing values"
            )

    validate(value)


def _decode_keyed_entries(entries, label):
    if not isinstance(entries, list):
        raise CacheError(f"Cache {label} must be a list")

    decoded = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CacheError(f"Cache {label} entry must be an object")
        key = entry.get("key")
        values = entry.get("values")
        if not isinstance(key, list) or len(key) != 2:
            raise CacheError(f"Cache {label} key must contain two integers")
        node_count = _require_int(key[0], f"{label} node count")
        element_count = _require_int(key[1], f"{label} element count")
        if not isinstance(values, list):
            raise CacheError(f"Cache {label} values must be a list")
        decoded[(node_count, element_count)] = values
    return decoded


def _decode_metric_payload(payload):
    try:
        node_map = payload["node_map"]
        job_results_entries = payload["job_results"]
        job_results_nodes_entries = payload["job_results_nodes"]
        metric_array_value = payload["metric_array"]
        node_counts = payload["node_counts"]
        element_counts = payload["element_counts"]
        metric_names = payload["metric_names"]
    except KeyError as exc:
        raise CacheError(f"Cache metric payload is missing field {exc}") from exc

    if not isinstance(node_map, dict):
        raise CacheError("Cache node map must be an object")
    if any(
        not isinstance(key, str) or type(value) is not int
        for key, value in node_map.items()
    ):
        raise CacheError("Cache node map is malformed")
    if not isinstance(node_counts, list) or not isinstance(element_counts, list):
        raise CacheError("Cache dimensions must be lists")
    if not all(type(value) is int for value in node_counts + element_counts):
        raise CacheError("Cache dimensions must contain integers")
    if not isinstance(metric_names, list) or not all(
        isinstance(value, str) for value in metric_names
    ):
        raise CacheError("Cache metric names must be a list of strings")

    _validate_numeric_tree(metric_array_value, "metric array", integer=True)
    try:
        metric_array = np.asarray(metric_array_value, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CacheError(f"Cache metric array is malformed: {exc}") from exc
    if metric_array.ndim != 5:
        raise CacheError("Cache metric array must have five dimensions")

    return (
        node_map,
        _decode_keyed_entries(job_results_entries, "metric results"),
        _decode_keyed_entries(job_results_nodes_entries, "metric result nodes"),
        metric_array,
        node_counts,
        element_counts,
        metric_names,
    )


def _decode_benchmark_payload(payload):
    try:
        bench_array_value = payload["bench_array"]
        bench_node_counts = payload["bench_node_counts"]
        bench_elements = payload["bench_elements"]
    except KeyError as exc:
        raise CacheError(f"Cache benchmark payload is missing field {exc}") from exc

    if not isinstance(bench_node_counts, list) or not isinstance(bench_elements, list):
        raise CacheError("Cache benchmark dimensions must be lists")
    if not all(type(value) is int for value in bench_node_counts + bench_elements):
        raise CacheError("Cache benchmark dimensions must contain integers")
    _validate_numeric_tree(bench_array_value, "benchmark array", integer=False)
    try:
        bench_array = np.asarray(bench_array_value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CacheError(f"Cache benchmark array is malformed: {exc}") from exc
    if bench_array.ndim != 4:
        raise CacheError("Cache benchmark array must have four dimensions")
    return bench_array, bench_node_counts, bench_elements


def _load_cache_payload(path, fingerprint, label):
    try:
        return read_versioned_cache(path, fingerprint)
    except CacheError as exc:
        print(f"{label} is unavailable; rebuilding it: {exc}", file=sys.stderr)
        return None


def _load_metric_data(base_directory, metric_names_path, num_interfaces):
    base_path = Path(base_directory)
    cache_path = base_path / "cached_data.json"
    fingerprint = source_fingerprint([base_path, metric_names_path])

    payload = _load_cache_payload(cache_path, fingerprint, "Metric cache")
    if payload is not None:
        try:
            return _decode_metric_payload(payload)
        except CacheError as exc:
            print(f"Metric cache is malformed; rebuilding it: {exc}", file=sys.stderr)

    legacy_path = base_path / "cached_data.pkl"
    if legacy_path.is_file():
        print(
            "Ignoring legacy pickle cache; a trusted JSON cache will be rebuilt.",
            file=sys.stderr,
        )

    node_map, job_results, job_results_nodes = process_all_jobs(str(base_path))
    max_nodes = max(
        (len(node_list) for node_list in job_results_nodes.values()),
        default=0,
    )
    metric_array, node_counts, element_counts = prepare_metric_array(
        job_results,
        num_interfaces,
        max_nodes,
        job_results_nodes,
    )
    if metric_array is None:
        raise RuntimeError(f"No metric data found in {base_path}")

    _, metrics_per_interface, _, _, _ = metric_array.shape
    metric_names = []
    if metric_names_path.is_file():
        with metric_names_path.open("r", encoding="utf-8") as names_file:
            metric_names = [line.strip() for line in names_file]
    if len(metric_names) < metrics_per_interface:
        metric_names.extend(
            f"Metric_{index + 1}"
            for index in range(len(metric_names), metrics_per_interface)
        )

    payload = _encode_metric_payload(
        node_map,
        job_results,
        job_results_nodes,
        metric_array,
        node_counts,
        element_counts,
        metric_names,
    )
    write_versioned_cache(cache_path, payload, fingerprint)
    return _decode_metric_payload(payload)


def _load_benchmark_data(base_directory, benchmark_type):
    base_path = Path(base_directory)
    cache_path = base_path / f"cached_bench_{benchmark_type}.json"
    fingerprint = source_fingerprint([base_path])

    payload = _load_cache_payload(cache_path, fingerprint, "Benchmark cache")
    if payload is not None:
        try:
            return _decode_benchmark_payload(payload)
        except CacheError as exc:
            print(
                f"Benchmark cache is malformed; rebuilding it: {exc}",
                file=sys.stderr,
            )

    legacy_path = base_path / f"cached_bench_{benchmark_type}.pkl"
    if legacy_path.is_file():
        print(
            "Ignoring legacy pickle cache; a trusted JSON cache will be rebuilt.",
            file=sys.stderr,
        )

    if benchmark_type == "oneccl":
        bench_results = process_oneccl_benchmarks(str(base_path))
    else:
        bench_results = process_osu_benchmarks(str(base_path))
    bench_array, bench_node_counts, bench_elements = prepare_benchmark_array(
        bench_results
    )
    payload = _json_ready({
        "bench_array": bench_array,
        "bench_node_counts": bench_node_counts,
        "bench_elements": bench_elements,
    })
    write_versioned_cache(cache_path, payload, fingerprint)
    return _decode_benchmark_payload(payload)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Process metrics and benchmarks for the interactive Dash app."
    )
    parser.add_argument(
        "--base_dir",
        default="examples/jobs/oneccl",
        help="Base directory containing job and benchmark data.",
    )
    parser.add_argument(
        "--num_interfaces",
        type=int,
        default=8,
        help="Number of NIC interfaces.",
    )
    parser.add_argument(
        "--metric_names_file",
        default="metric_names.txt",
        help="File containing one metric name per line.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["oneccl", "osu"],
        default="oneccl",
        help="Benchmark data to process.",
    )
    parser.add_argument("--browse-root", help="Allowed server-side file browser root (default: working directory, or parent of an initial trace).")
    parser.add_argument("--analysis-config", help="Optional explicit topology and latency-bin calibration JSON.")
    parser.add_argument("--ctf-reader", dest="trace_ctf_reader", choices=["auto","babeltrace2","portable"], default="auto", help="CTF decoder; auto uses bt2 when installed, otherwise the guarded THAPI/LTTng decoder.")
    parser.add_argument("--legacy-inputs", action="store_true", help="Explicitly select the historical JSON benchmark workflow.")
    parser.add_argument("--demo", action="store_true", help="Open an explicitly synthetic, local UI demo; no Sunspot connection.")
    parser.add_argument("--trace-ui", choices=["explorer", "legacy"], default="explorer",
                        help="Trace interface (default: explorer). Legacy report retained for regression/reference only.")
    parser.add_argument(
        "--trace",
        help="Path to a raw CTF trace or binary pftrace file.",
    )
    parser.add_argument(
        "--trace-format",
        choices=["ctf", "pftrace"],
        help="Trace format; inferred from a CTF directory or pftrace suffix when omitted.",
    )
    parser.add_argument("--trace-run-id", help="Explicit trace run ID.")
    parser.add_argument(
        "--trace-source-fingerprint",
        help="Explicit source fingerprint recorded by the caller.",
    )
    parser.add_argument(
        "--trace-reader-version",
        help="Version of the extraction reader used by the caller.",
    )
    parser.add_argument(
        "--trace-clock-domain",
        help="Clock-domain provenance for trace timestamps.",
    )
    parser.add_argument(
        "--trace-synchronization-evidence",
        help="Evidence used to synchronize or relate trace clocks.",
    )
    parser.add_argument(
        "--trace-requested-counter-file",
        default=None,
        help="Explicit run sampling inventory. Omit when unknown; the hardware catalog is not an acquisition manifest.",
    )
    parser.add_argument(
        "--trace-cache",
        help="Optional atomic NPZ cache path for the requested trace.",
    )
    parser.add_argument(
        "--dataset-manifest",
        help=(
            "JSON manifest containing an approved dataset root, explicit dataset "
            "descriptors, and the initial dataset ID."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("AURORA_APP_HOST", "127.0.0.1"),
        help="Bind address (default: 127.0.0.1; env: AURORA_APP_HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AURORA_APP_PORT", "8050")),
        help="Bind port (default: 8050; env: AURORA_APP_PORT).",
    )
    parser.add_argument(
        "--server",
        choices=["development", "waitress"],
        default=os.environ.get("AURORA_APP_SERVER", "development"),
        help=(
            "WSGI server implementation (default: development; "
            "env: AURORA_APP_SERVER). Waitress is an optional production server."
        ),
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        default=os.environ.get("AURORA_APP_ALLOW_NON_LOOPBACK", "").lower()
        in {"1", "true", "yes"},
        help="Explicitly permit a non-loopback bind (default: rejected).",
    )
    args = parser.parse_args(argv)
    effective_argv = argv if argv is not None else sys.argv[1:]
    if any(option in effective_argv for option in ('--base_dir', '--benchmark', '--metric_names_file', '--num_interfaces')) and not args.trace and not args.dataset_manifest:
        args.legacy_inputs = True
    if args.demo and (args.trace or args.dataset_manifest):
        parser.error("--demo cannot be combined with a real trace or dataset manifest")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.trace is not None:
        from trace_io import trace_arguments
        # Provenance is derived in the worker; explicit capture metadata still wins.
        try:
            trace_path = Path(args.trace).expanduser().resolve(strict=True)
            root = Path(args.browse_root).expanduser().resolve(strict=True) if args.browse_root else trace_path.parent
            defaults = trace_arguments(trace_path, root=root, format=args.trace_format or 'auto')
        except (OSError, ValueError) as error:
            parser.error(str(error))
        for key, value in vars(defaults).items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
        args.trace = defaults.trace
    elif not args.dataset_manifest and any(getattr(args, key) is not None for key in (
            'trace_format','trace_cache','trace_run_id','trace_source_fingerprint','trace_reader_version',
            'trace_clock_domain','trace_synchronization_evidence')):
        parser.error('trace metadata options require --trace or --dataset-manifest')
    return args


def create_runtime_app(
    metric_array,
    node_counts,
    element_counts,
    metric_names,
    node_map,
    results_nodes,
    bench_array,
    bench_node_counts,
    bench_elements,
):
    return create_app(
        metric_array=metric_array,
        node_counts=node_counts,
        element_counts=element_counts,
        metric_names=metric_names,
        node_map=node_map,
        results_nodes=results_nodes,
        bench_array=bench_array,
        bench_node_counts=bench_node_counts,
        bench_elements=bench_elements,
    )


def read_requested_counter_names(path):
    if not path:
        return ()  # No run sampling inventory was supplied.
    counter_path = Path(path)
    if not counter_path.is_file():
        raise RuntimeError(f"Requested counter file does not exist: {counter_path}")
    names = tuple(
        line.strip()
        for line in counter_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not names:
        raise RuntimeError(f"Requested counter file is empty: {counter_path}")
    if len(names) != len(set(names)):
        raise RuntimeError(f"Requested counter file contains duplicates: {counter_path}")
    return names


def run_interactive_dash_app(
    metric_array,
    node_counts,
    element_counts,
    metric_names,
    node_map,
    results_nodes,
    bench_array,
    bench_node_counts,
    bench_elements,
    host="127.0.0.1",
    port=8050,
    server="development",
):
    app = create_runtime_app(
        metric_array,
        node_counts,
        element_counts,
        metric_names,
        node_map,
        results_nodes,
        bench_array,
        bench_node_counts,
        bench_elements,
    )
    run_wsgi_app(app, host=host, port=port, server=server)
    return app


def run_wsgi_app(app, host, port, server="development"):
    if server == "waitress":
        run_waitress_server(app, host=host, port=port)
        return
    if server != "development":
        raise RuntimeError(f"Unsupported WSGI server {server!r}")
    run_app(app, host=host, port=port)


def run_waitress_server(app, host, port):
    try:
        import waitress
    except ImportError as error:
        raise RuntimeError(
            "The --server waitress option requires the optional Waitress WSGI "
            "server. Install it with `python -m pip install -e '.[waitress]'`."
        ) from error

    server = waitress.create_server(
        app.server,
        host=host,
        port=port,
        clear_untrusted_proxy_headers=True,
        ident="AuroraNIC-MetricVisualizer",
    )
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def request_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, original_sigterm_handler)


def _cache_cancellation_error(stage, cancellation, cache_published=False):
    signal_name = (
        signal.Signals(cancellation._signal_number).name
        if cancellation._signal_number is not None
        else "request"
    )
    return TraceImportCancelled(
        f"Trace import cancelled by {signal_name} while {stage}.",
        signal_number=cancellation._signal_number,
        cache_published=cache_published,
    )


def load_trace_store(args, cancellation=None):
    if cancellation is None:
        cancellation = TraceImportCancellation()

    args = copy.copy(args)
    from trace_io import PIPELINE_VERSION, trace_fingerprint
    if args.trace_source_fingerprint == "auto":
        args.trace_source_fingerprint = trace_fingerprint(args.trace, cancellation)
    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise RuntimeError(f"Trace path does not exist: {trace_path}")

    cache_path = Path(args.trace_cache) if args.trace_cache is not None else None
    if cache_path is not None and cache_path.is_file():
        print("Trace import: cache hit; reading trace cache...", file=sys.stderr)
        cancellation.check("trace cache read")
        try:
            store = read_trace_store_cache(
                cache_path,
                args.trace_source_fingerprint,
                should_cancel=cancellation.should_cancel,
            )
            cancellation.check("trace cache read")
            if PIPELINE_VERSION not in store.coverage.notes:
                raise TraceCacheStaleError("Cache input schema is outdated; rebuild process/thread identity and counter semantics from the source trace.")
            _validate_cached_trace_provenance(store, args)
            return store
        except TraceCacheStaleError as error:
            print(
                f"Trace cache is stale; rebuilding from the trace: {error}",
                file=sys.stderr,
            )
            print("Trace import: cache rebuild required; loading source...", file=sys.stderr)
        except TraceCacheCorruptError as error:
            print(
                f"Trace cache is corrupt; rebuilding from the trace: {error}",
                file=sys.stderr,
            )
            print("Trace import: cache rebuild required; loading source...", file=sys.stderr)
        except TraceCacheCancelled as error:
            raise _cache_cancellation_error("reading the trace cache", cancellation) from error

    common_metadata = {
        "run_id": args.trace_run_id,
        "source_fingerprint": args.trace_source_fingerprint,
        "reader_version": args.trace_reader_version,
        "clock_domain": args.trace_clock_domain,
        "synchronization_evidence": args.trace_synchronization_evidence,
    }
    if args.trace_format == "ctf":
        print("Trace import: source load; extracting CTF events...", file=sys.stderr)
        cancellation.check("CTF source extraction")
        from ctf_adapter import Bt2UnavailableError, CtfExtractionStats
        reader = getattr(args, 'trace_ctf_reader', 'auto')
        try:
            if reader == 'portable':
                raise Bt2UnavailableError('Portable reader explicitly selected.')
            events, extraction_stats = extract_babeltrace2_events_with_stats(trace_path)
            if common_metadata['reader_version'] == 'auto':
                import bt2
                common_metadata['reader_version'] = 'babeltrace2-' + str(getattr(bt2, '__version__', 'installed'))
        except Bt2UnavailableError:
            if reader == 'babeltrace2':
                raise
            from portable_ctf import extract_portable_ctf, VERSION
            try:
                events, filtered = extract_portable_ctf(trace_path, should_cancel=cancellation.should_cancel)
            except ValueError:
                cancellation.check('CTF source extraction')
                raise
            extraction_stats = CtfExtractionStats(len(events), filtered)
            common_metadata['reader_version'] = VERSION
            common_metadata['source_format'] = 'portable-structured-raw-ctf'
            print('Trace import: validated portable LTTng CTF reader (no bt2 binding required).', file=sys.stderr)
        cancellation.check("CTF source extraction")
        requested_counter_names = read_requested_counter_names(
            args.trace_requested_counter_file
        )
        print(
            "Trace import: CTF extraction statistics: "
            f"CXI events={extraction_stats.cxi_event_count}; "
            f"filtered non-CXI event messages={extraction_stats.filtered_event_count}; "
            f"requested counters={len(requested_counter_names)}.",
            file=sys.stderr,
        )
        batch = ctf_events_to_normalized_batch(
            events,
            CtfAdapterConfig(
                **common_metadata,
                requested_counter_names=requested_counter_names,
            ),
        )
        cancellation.check("CTF adapter conversion")
    else:
        print("Trace import: source load; reading pftrace...", file=sys.stderr)
        cancellation.check("pftrace source read")
        from pftrace_native import load_pftrace, VERSION
        common_metadata['reader_version'] = VERSION
        try:
            batch = load_pftrace(trace_path, PftraceAdapterConfig(**common_metadata,
                requested_counter_names=read_requested_counter_names(args.trace_requested_counter_file)),
                should_cancel=cancellation.should_cancel)
        except ValueError:
            cancellation.check('pftrace source read')
            raise
        cancellation.check("pftrace adapter conversion")
    batch = replace(batch, coverage=replace(batch.coverage, notes=(*batch.coverage.notes, PIPELINE_VERSION)))
    print("Trace import: store build; indexing normalized batches...", file=sys.stderr)
    cancellation.check("trace store build")
    store = TraceStore.from_batches((batch,))
    cancellation.check("trace store build")
    if cache_path is not None:
        print("Trace import: cache write; publishing trace cache...", file=sys.stderr)
        cancellation.check("trace cache write")
        try:
            write_trace_store_cache(
                cache_path,
                store,
                args.trace_source_fingerprint,
                should_cancel=cancellation.should_cancel,
            )
        except TraceCacheCancelled as error:
            raise _cache_cancellation_error(
                "writing the trace cache", cancellation
            ) from error
        if cancellation.should_cancel():
            raise _cache_cancellation_error(
                "publishing the trace cache",
                cancellation,
                cache_published=True,
            )
    return store


def _validate_cached_trace_provenance(store, args):
    mismatches = []
    expected_source_format = (
        RAW_CTF_SOURCE_FORMAT
        if args.trace_format == "ctf"
        else PFTRACE_SOURCE_FORMAT
    )
    requested = {
        "source format": (store.identity.source_format, expected_source_format),
        "run ID": (store.identity.run_id, args.trace_run_id),
        "reader version": (store.identity.reader_version, args.trace_reader_version),
        "clock domain": (store.identity.clock_domain, args.trace_clock_domain),
        "synchronization evidence": (
            store.identity.synchronization_evidence,
            args.trace_synchronization_evidence,
        ),
    }
    for name, (cached, requested_value) in requested.items():
        if requested_value == 'auto':
            continue
        if name == 'source format' and args.trace_format == 'ctf' and cached == 'portable-structured-raw-ctf':
            continue
        if name == 'reader version' and cached in {'thapi-lttng-ctf/1','perfetto-protobuf-thapi/1'}:
            continue  # The actual selected decoder is recorded, not the caller's historical label.
        if cached != requested_value:
            mismatches.append(f"{name} {cached!r} != {requested_value!r}")
    if mismatches:
        raise RuntimeError(
            "trace cache provenance does not match the requested trace: "
            + "; ".join(mismatches)
        )


def create_trace_app(args, cancellation=None):
    return _trace_ui_factory(args)(load_trace_store(args, cancellation=cancellation))


def _load_dataset_manifest(args, manifest_path: Path):
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unable to read dataset manifest {manifest_path}: {error}") from error

    if not isinstance(document, dict):
        raise RuntimeError("dataset manifest must be a JSON object")
    root_value = document.get("approved_root")
    datasets = document.get("datasets")
    initial_dataset_id = document.get("initial_dataset_id")
    if not isinstance(root_value, str) or not root_value.strip():
        raise RuntimeError("dataset manifest requires a nonempty approved_root")
    if not isinstance(datasets, list) or not datasets:
        raise RuntimeError("dataset manifest requires at least one dataset")
    if not isinstance(initial_dataset_id, str) or not initial_dataset_id.strip():
        raise RuntimeError("dataset manifest requires a nonempty initial_dataset_id")

    approved_root = Path(root_value).expanduser().resolve(strict=True)
    descriptors = []
    dataset_ids = set()
    for index, entry in enumerate(datasets):
        if not isinstance(entry, dict):
            raise RuntimeError(f"dataset manifest entry {index} must be an object")
        dataset_id = entry.get("dataset_id")
        trace_value = entry.get("trace_path")
        cache_value = entry.get("cache_path")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise RuntimeError(f"dataset manifest entry {index} requires dataset_id")
        if dataset_id in dataset_ids:
            raise RuntimeError(f"duplicate dataset_id in manifest: {dataset_id}")
        if not isinstance(trace_value, str) or not trace_value.strip():
            raise RuntimeError(f"dataset {dataset_id} requires trace_path")
        dataset_ids.add(dataset_id)

        trace_path = (approved_root / trace_value).resolve(strict=False)
        cache_path = (
            (approved_root / cache_value).resolve(strict=False)
            if isinstance(cache_value, str) and cache_value.strip()
            else None
        )
        trace_format = entry.get("trace_format", args.trace_format)
        if trace_format not in {"ctf", "pftrace"}:
            raise RuntimeError(f"dataset {dataset_id} requires trace_format ctf or pftrace")
        dataset_args = SimpleNamespace(
            trace=str(trace_path),
            trace_format=trace_format,
            trace_run_id=entry.get("trace_run_id", args.trace_run_id) or trace_path.parent.name,
            trace_source_fingerprint=entry.get(
                "trace_source_fingerprint", args.trace_source_fingerprint
            ) or 'auto',
            trace_reader_version=entry.get(
                "trace_reader_version", args.trace_reader_version
            ) or 'auto',
            trace_clock_domain=entry.get(
                "trace_clock_domain", args.trace_clock_domain
            ) or ('CTF metadata clock' if trace_format == 'ctf' else 'THAPI Perfetto timestamp'),
            trace_synchronization_evidence=entry.get(
                "trace_synchronization_evidence",
                args.trace_synchronization_evidence,
            ) or 'Cross-host synchronization not independently established.',
            trace_requested_counter_file=entry.get(
                "trace_requested_counter_file",
                args.trace_requested_counter_file,
            ),
            trace_cache=str(cache_path) if cache_path is not None else None,
        )
        descriptors.append(
            DatasetDescriptor(
                dataset_id=dataset_id,
                trace_path=trace_path,
                cache_path=cache_path,
                args=dataset_args,
            )
        )
    if initial_dataset_id not in dataset_ids:
        raise RuntimeError(f"initial dataset is not approved: {initial_dataset_id}")
    return approved_root, descriptors, initial_dataset_id


def _validate_server_bind(args):
    if args.host in {"127.0.0.1", "localhost", "::1"}:
        return
    if not args.allow_non_loopback:
        print(
            f"Refusing non-loopback bind {args.host!r}; use --allow-non-loopback "
            "only for an intentionally exposed listener.",
            file=sys.stderr,
        )
        raise RuntimeError(
            f"Refusing non-loopback bind {args.host!r}; use --allow-non-loopback "
            "only for an intentionally exposed listener."
        )
    print(
        f"Warning: binding a non-loopback address ({args.host}); "
        "ensure the listener is intentionally exposed.",
        file=sys.stderr,
    )


def _trace_ui_factory(args):
    if getattr(args, "trace_ui", "explorer") == "legacy":
        from analysis_app import create_analysis_app as legacy_factory
        return legacy_factory
    return create_analysis_app


def run_trace_app(args):
    """Start a single-user, root-confined explorer, with or without an initial trace."""
    from explorer.analysis_config import load_analysis_config
    config = load_analysis_config(getattr(args, 'analysis_config', None))
    recorder = PerformanceMetricsRecorder()
    initial_id = None
    if getattr(args, 'dataset_manifest', None):
        approved_root, descriptors, initial_id = _load_dataset_manifest(args, Path(args.dataset_manifest))
    else:
        initial = Path(args.trace).resolve() if getattr(args, 'trace', None) else None
        approved_root = Path(args.browse_root).expanduser().resolve(strict=True) if getattr(args, 'browse_root', None) else (initial.parent if initial else Path.cwd())
        descriptors = []
        if initial is not None:
            initial_id = 'initial'
            descriptors.append(DatasetDescriptor(initial_id, initial,
                Path(args.trace_cache) if args.trace_cache else None, args))
    manager = DatasetLifecycleManager(approved_root, descriptors, performance_recorder=recorder)
    try:
        if initial_id is not None:
            request = manager.open(initial_id)
            with trace_import_cancellation_context() as cancellation:
                while True:
                    try:
                        status = manager.wait(request, timeout=0.2)
                        break
                    except TimeoutError:
                        if cancellation.should_cancel():
                            manager.cancel(request)
                            cancellation.check('initial trace import')
            if status.state.value != 'ready':
                # The path picker remains accessible for a corrected path or format.
                print(f'Initial import failed: {status.error}. Open another trace in the browser.', file=sys.stderr)
        current = manager.current()
        kwargs = dict(dataset_manager=manager, performance_recorder=recorder)
        if getattr(args, 'trace_ui', 'explorer') != 'legacy':
            kwargs.update(trace_browser=True, analysis_config=config)
        elif current is None:
            raise RuntimeError('The legacy report requires a successfully loaded trace; use the default explorer to choose a path.')
        app = _trace_ui_factory(args)(current.store if current else None, **kwargs)
        print(f'File browser root: {approved_root}', file=sys.stderr)
        run_wsgi_app(app, host=args.host, port=args.port, server=getattr(args, 'server', 'development'))
        return app
    finally:
        manager.close()


def main():
    args = parse_args()
    _validate_server_bind(args)
    if getattr(args, "demo", False):
        from explorer.demo import make_demo_store
        app = create_analysis_app(make_demo_store())
        run_wsgi_app(app, host=args.host, port=args.port, server=args.server)
        return
    if args.trace is not None or getattr(args, "dataset_manifest", None) is not None or not getattr(args, "legacy_inputs", False):
        try:
            run_trace_app(args)
            return
        except TraceImportCancelled as error:
            signal_name = (
                signal.Signals(error.signal_number).name
                if error.signal_number is not None
                else "request"
            )
            print(
                f"Trace import cancelled by {signal_name}; exit status "
                f"{error.exit_status}. "
                + (
                    "The source trace was left unchanged; the cache was "
                    "atomically published before cancellation."
                    if error.cache_published
                    else "The source trace and existing cache were left unchanged."
                )
                + " Re-run the command when ready.",
                file=sys.stderr,
            )
            raise SystemExit(error.exit_status) from None
        except KeyboardInterrupt:
            print(
                "Trace import interrupted; exit status 130. The source trace and "
                "existing cache were left unchanged. Re-run the command when ready.",
                file=sys.stderr,
            )
            raise SystemExit(130) from None

    base_directory = Path(args.base_dir)
    if not base_directory.is_dir():
        raise RuntimeError(f"Base directory does not exist: {base_directory}")

    (
        node_map,
        job_results,
        job_results_nodes,
        metric_array,
        node_counts,
        element_counts,
        metric_names,
    ) = _load_metric_data(
        base_directory,
        Path(args.metric_names_file),
        args.num_interfaces,
    )
    bench_array, bench_node_counts, bench_elements = _load_benchmark_data(
        base_directory,
        args.benchmark,
    )

    run_interactive_dash_app(
        metric_array,
        node_counts,
        element_counts,
        metric_names,
        node_map,
        job_results_nodes,
        bench_array,
        bench_node_counts,
        bench_elements,
        host=args.host,
        port=args.port,
        server=args.server,
    )


if __name__ == "__main__":
    main()
