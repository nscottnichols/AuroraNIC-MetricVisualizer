# Architecture

The application separates source readers, normalized storage, numerical
analysis and the Dash interface. The default entry point is
[`src/main.py`](../src/main.py).

## Source loading and storage

[`trace_io.py`](../src/trace_io.py) resolves server-side paths, detects source
formats, derives provenance defaults and computes source fingerprints.
[`dataset_lifecycle.py`](../src/dataset_lifecycle.py) serializes imports on a
worker, reports progress, supports cancellation and publishes a new dataset only
after loading succeeds. The previous usable dataset survives a failed request.
All source/cache paths must remain inside the configured browser root.

[`ctf_adapter.py`](../src/ctf_adapter.py) normalizes structured THAPI CXI events
from Babeltrace 2. [`portable_ctf.py`](../src/portable_ctf.py) provides the guarded
THAPI/LTTng fallback. [`pftrace_native.py`](../src/pftrace_native.py) reads binary
THAPI Perfetto protobuf data, then uses
[`pftrace_adapter.py`](../src/pftrace_adapter.py) to normalize track/counter rows.
These readers reject unsupported representations instead of guessing layouts
or clocks.

[`trace_model.py`](../src/trace_model.py) defines identities, counter metadata,
coverage and normalized batches. [`trace_store.py`](../src/trace_store.py) builds
an immutable NumPy column store for one run, indexed by node/NIC/counter and
collector. Exact unsigned counters and floating-point converted observations
occupy separate value columns. Multiple collectors remain separate streams;
ambiguous selections require a collector choice.

[`trace_cache.py`](../src/trace_cache.py) validates and atomically publishes
typed NPZ caches without pickle deserialization. Source fingerprints, schema,
reader pipeline and semantic registry metadata determine compatibility. Keep
these checks when changing ingestion behavior; incompatible cached results
must be rebuilt.

## Numerical analysis

[`traffic_analysis.py`](../src/traffic_analysis.py) computes observed byte totals,
covered-time rates and duration-weighted statistics from validated intervals.
[`trace_analysis_bridge.py`](../src/trace_analysis_bridge.py) adapts normalized
series to the analysis interfaces.

[`explorer/diagnostics.py`](../src/explorer/diagnostics.py) defines derived
metrics and aligns their inputs on common valid support.
[`cassini_metrics.py`](../src/cassini_metrics.py) and
[`counter_semantic_registry.py`](../src/counter_semantic_registry.py) supply
sourced counter definitions. Specialist modules such as `latency_analysis.py`,
`translation_analysis.py`, `frame_analysis.py` and `reliability_analysis.py`
provide reusable analysis interfaces.

Scientific summaries use full-resolution observations; chart bins are display
approximations. Counter resets, missing support, unknown semantics and source
precision limits remain explicit. See [metrics](METRICS.md) for formulas and
[configuration](CONFIGURATION.md) for calibration and topology metadata.

## Explorer interface

[`explorer/model.py`](../src/explorer/model.py) owns immutable `Query` selections,
the `ExplorerEngine`, a four-selection result cache, bounded display series and
full-selection exports. A query identifies the exact time window, physical
devices, metric and collector choices. Nanosecond timestamps cross the browser
boundary as strings to preserve integer precision.

[`explorer/app.py`](../src/explorer/app.py) builds the Dash layout and callbacks.
[`explorer/overview.py`](../src/explorer/overview.py) produces paged heatmaps with
shared scales and physical identity maps for row clicks. `charts.py` and
`diagnostic_ui.py` render focused plots and details. `metadata.py` discovers
metrics from encountered counters; reference catalog labels alone cannot
authorize derived measurements.

[`explorer/workspace.py`](../src/explorer/workspace.py) handles bounded selection
history, saved views, table paging and CSV export. Browser stores contain
selection/display metadata, while raw trace arrays remain on the server.
Display search and paging do not change numerical scope or exports.
`trace_browser.py` implements the server-side path dialog.

The server exposes `/healthz` for process health and `/readyz` for dataset
readiness. An empty server is healthy but not ready until a trace is open.
`main.py` chooses the development server or Waitress and enforces loopback
binding unless explicitly allowed otherwise. The active dataset is shared by
all tabs connected to that server.

## Compatibility paths and tests

`--trace-ui legacy` selects [`analysis_app.py`](../src/analysis_app.py) and its
`src/ui/` views. `--legacy-inputs` selects the text-counter/benchmark application
using `app.py`, `layout.py`, `callbacks.py` and `metrics_processing.py`. Changes to
these paths do not validate the default explorer.

`tests/` contains reader, cache, exact-integer numerical, selection/export, CLI,
HTTP and optional Playwright browser checks. Generated artifacts and capture
data stay outside the tracked source tree. Use the commands in the
[README](../README.md) to install test dependencies and run the suite.
