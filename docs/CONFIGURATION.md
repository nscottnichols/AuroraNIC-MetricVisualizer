# Capture-specific analysis configuration

The app works without an analysis configuration. To supply verified histogram
calibration or explicit topology:

```bash
python src/main.py --browse-root /approved/runs --analysis-config config/analysis.json --server waitress
```

Copy `config/analysis.example.json` as a starting point. Its empty objects are
intentional: there is no fabricated calibration. The configuration is validated
once at startup and applies to subsequent source switches under the restrictions
below. The file is JSON, at most 1 MiB, with only `latency_histograms` and `topology`
at the top level.

## Latency histograms

`latency_histograms` maps `pct_req_rsp_latency` (32 bins) and/or
`pct_host_access_latency` (16 bins) to objects containing:

- `run_id`: exact run ID shown in the app/export; a different loaded run does not
  receive this calibration.
- `source`: a nonempty explanation of the captured device configuration or
  authoritative metadata from which these edges were obtained.
- `edges_ns`: 33 or 17 strictly increasing, nonnegative numeric edges in
  nanoseconds. The last edge may be `null` for an open-ended upper tail.

This schema deliberately supplies no illustrative numerical bin values that
might be mistaken for the capture's calibration. Capture the actual settings
from the site/device metadata. Configuration currently assumes the same verified
bin scheme for all selected NICs in the named run. Do not use it when NICs have
heterogeneous bin configurations; split the dataset or extend per-NIC calibration
first. Changing configuration requires a server restart.

Without edges, the distribution and P99 **bucket index** are available only
where all required bin observations exist. With edges, the single-NIC view/export
adds a P99 interval in nanoseconds, mean lower/upper bounds, and a labeled bin-
midpoint estimate when the tail is bounded. Binned data cannot provide an exact
within-bin P99 or exact mean. The chart remains explicitly labeled by bucket;
physical bounds appear in the detail, never by silently relabeling the index.

## Explicit topology

`topology` maps exact node labels to any of the string fields `cabinet`, `rack`,
`switch`, and `region`. For example, the **schema shape** is:

```json
{
  "latency_histograms": {},
  "topology": {
    "replace-with-real-node-name": {
      "rack": "replace-with-verified-rack",
      "region": "replace-with-verified-region"
    }
  }
}
```

No topology is parsed from hostname spelling or invented for unmapped nodes.
Traffic groups show observed volume and population. Derived-metric groups show
per-NIC minimum/median/maximum values, not averages masquerading as pooled ratios.
This is static physical-node grouping: verify it is still valid for later traces
opened in the same server. A link map, communication peers, routes and distances
are separate metadata and are not reconstructed from these device counters.

## Input, cache, and resource behavior

Use a narrow `--browse-root`; absolute paths outside it, relative `..` escapes,
and symlink escapes are rejected. This is a single-user convenience boundary,
not a sandbox for hostile local users who can concurrently mutate files. The
UI handles local server paths, not archives, uploads, URL fetching, or file edits.
CTF `metadata` selections normalize to their containing directory.

Source content, reader pipeline and registry version participate in cache
validation. Default generated caches live in `.nicviz-cache` under the approved
root. Caches with incompatible identity, reader or semantic metadata are rebuilt
from the source. Explicit `--trace-cache` paths remain supported. The raw trace
is never deleted. For read-only sources, use a writable approved parent root and,
when needed, an explicit writable `--trace-cache` path inside that root.

The query-result LRU keeps four selections. Timelines use at most 240 display
bins, diagnostic trends show at most 12 devices, and the A/B comparison table
shows at most 200 NICs. The device table pages through the full selected
population with 12 rows per page by default. Population heatmaps show 24 rows per
page, with automatic node grouping
above 64 physical NICs and a shared color scale across pages. Display search and
paging do not change selection or export scope. Saved views retain at most 12
entries, and an imported view file is limited to 128,000 bytes. Export retains all
selected per-NIC summaries. Scientific summaries use full-resolution valid
intervals, not the display bins. Directory lists cap work at 10,000 entries
and show at most 300 choices; the typed path can select another known entry.
Imports are serialized with cancellation at safe boundaries. Descriptor/request
history is bounded, and the last good dataset survives failed imports. Generated
disk caches are not globally evicted: periodically remove only unneeded cache
files according to local storage policy; never remove original traces as cleanup.

Portable CTF validates a restricted byte-aligned little-endian LTTng-UST CTF 1.8
schema, one 1 GHz metadata clock, packet identity/sizes and loss indicators. It
rejects unsupported layouts rather than guessing; site Babeltrace 2 is the
alternative. Native pftrace supports the THAPI CXI track hierarchy with absolute
timestamps and counters; unsupported clock conversions/incremental/scaled
encodings fail explicitly. Neither reader is a general arbitrary-trace viewer.
