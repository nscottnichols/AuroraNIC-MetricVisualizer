# Aurora NIC Explorer

A Dash/Plotly application for exploring THAPI CXI counters from raw CTF traces
and binary `.pftrace` files. Select physical nodes/NICs and a time range, compare
traffic or derived metrics, and export the selected measurements.

## Install and launch

Use Python 3.11 or later from this application directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[waitress]'
python src/main.py --browse-root /path/to/runs --server waitress
```

Open `http://127.0.0.1:8050/`. The file picker browses paths on the application
server. Choose a CTF directory or `.pftrace` file, then select **Open selected
trace**. Opening runs asynchronously with cancellation and visible errors; a
failed import preserves the previous dataset. Extract trace archives before
opening them.

To open a trace at startup:

```bash
python src/main.py --trace /path/to/run/ctf --server waitress
python src/main.py --trace /path/to/run/timeline.pftrace --server waitress
```

`--browse-root` restricts browsing and imports to one directory tree. Its default
is the working directory, or the parent of an initial trace. Run
`python src/main.py --help` for all options. `requirements.txt` provides optional
pins for direct runtime dependencies; it is not a complete transitive lockfile.

The CTF reader uses installed Babeltrace 2 Python bindings when available, and
otherwise a built-in reader for supported THAPI/LTTng layouts. Choose
`--ctf-reader portable` or `--ctf-reader babeltrace2` explicitly when needed.
Unsupported layouts fail with an error. Binary pftrace uses the `perfetto`
Python schema without downloading a Trace Processor binary.

`--demo` opens synthetic data. `--trace-ui legacy` opens the compatibility report,
and `--legacy-inputs` selects the text-counter/benchmark workflow.

## Explore a trace

**Auto** shows a population heatmap or a single-NIC trend. Select a node/device
row, brush a time interval, and use **Back** to restore the previous selection.
Populations above 64 NICs group by node automatically; grouping and paging also
support explicit physical-device inspection. Pages share a color scale. Gray
cells indicate unavailable values; observed zero remains distinct.

Search metric families, readable names or hardware IDs, and save favorites.
Raw-counter selectors contain only counters encountered in the loaded trace.
The bundled catalog is a reference superset; it does not establish what a run
requested or sampled. Supply `--trace-requested-counter-file` only for a known
run inventory. Name-derived labels do not establish counter units or semantics.

Expand **Metric definition & calculation** for formulas, inputs, units and
caveats. Save A/B intervals to compare the current device/metric scope.
**Saved views** stores up to 12 named selections and display settings in the
browser tab; downloaded views can be imported for the same trace fingerprint.
JSON exports include formulas and measurement evidence. CSV exports contain
the full selected population. Display row search and paging do not alter that
population; use the node/NIC filters to change scientific scope.

The **Open trace** dialog supports keyboard navigation and Escape to close.
Press **?** for interaction help. See [metrics](docs/METRICS.md) for numerical
definitions and [configuration](docs/CONFIGURATION.md) for topology, calibration,
cache behavior and input limits.

The server uses one shared active dataset across browser tabs. It binds to
loopback by default; non-loopback binding requires explicit opt-in. For remote
use, run one server on the analysis host and connect through an SSH local
forward. This single-user application has no public authentication service.

## Development

```bash
python -m pip install -e '.[waitress,test,browser]'
python -m pytest -q
```

Browser tests need Playwright and a Chromium installation:

```bash
python -m playwright install chromium
python -m pytest -q tests/test_explorer_browser.py
```

Browser tests accept `NICVIZ_CHROMIUM=/path/to/chromium` for an existing browser.
Tests requiring external captures skip when the source data is unavailable.
The real-capture and HTTP checks are available as
`scripts/validate_real_capture.py` and `scripts/validate_runtime.py`; use each
script's `--help` for its required inputs. See [architecture](docs/ARCHITECTURE.md)
for the reader, storage, analysis and interface modules.
