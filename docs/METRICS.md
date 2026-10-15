# Metrics and numerical behavior

The explorer's derived metric definitions are implemented in
[`src/explorer/diagnostics.py`](../src/explorer/diagnostics.py). Counter meanings,
units and source references are recorded in
[`src/cassini_metrics.py`](../src/cassini_metrics.py) and
[`src/counter_semantic_registry.py`](../src/counter_semantic_registry.py).
The bundled catalog also contains name-derived labels; those labels alone do
not establish semantics. Only encountered counters appear in raw selectors,
and each derived metric requires its source observations.

## Shared numerical contract

Use one run, physical node/NIC, chosen process/thread collector, and exact time
window. Complete valid sample intervals only: boundary-crossing intervals,
resets/decreases, conflicting timestamps and flagged gaps are excluded; wraps
are not guessed. CTF diagnostic intervals exceeding five times the median
sampling cadence are excluded and reported. Missing observations are not zero.
For exact counters, subtract integers before converting to chart floats.

For multiple counters, integrate on the intersection of valid sample supports.
Within each observed interval, assume constant rate (linear cumulative
interpolation) to align asynchronous reads. This is an **estimate with an explicit
assumption**, not fabricated additional sampling precision. The ratio is
integrated numerator / integrated denominator on that same support, not the
mean of per-sample ratios. Zero denominator is undefined. Source exact whole-
interval deltas and aligned estimates are both included in exported evidence.
Cross-host clock synchronization requires independent evidence for the run;
joint phase/coincidence interpretation depends on that evidence. Per-NIC values
do not establish cross-host synchronization.

### Host pushback

Posted = Δ`parbs_tarb_pi_posted_blocked_cnt` /
Δ`parbs_tarb_pi_posted_pkts`.
Non-posted = Δ`parbs_tarb_pi_non_posted_blocked_cnt` /
Δ`parbs_tarb_pi_non_posted_pkts`.

Units are **cycles per packet**, not percentage, nanoseconds or a count of
unique blocked requests. HPE suggests these ratios as host-pressure indicators;
no universal red/green threshold or causal bottleneck verdict is imposed.

## Implemented derived metric definitions

In the definitions below, “delta” means the common-support aligned count.
Rate = sum of selected source deltas / covered seconds. Maximum-rate views use
the largest individual counter rate, not the sum of overlapping stall causes.
Fractions use the displayed scale; no division by an assumed run duration.

### Host / PCIe — Posted pushback

Key: `posted_pushback`. Unit: **cycles/packet**. Operation: ratio of summed counts; scale 1.

Numerator/bin sources: `parbs_tarb_pi_posted_blocked_cnt`.

Denominator sources: `parbs_tarb_pi_posted_pkts`.

Blocked posted-path cycles per posted PCIe packet; an indicator of host-side pressure, not proof of a PCIe bottleneck.

### Host / PCIe — Non-posted pushback

Key: `nonposted_pushback`. Unit: **cycles/packet**. Operation: ratio of summed counts; scale 1.

Numerator/bin sources: `parbs_tarb_pi_non_posted_blocked_cnt`.

Denominator sources: `parbs_tarb_pi_non_posted_pkts`.

Blocked non-posted-path cycles per non-posted PCIe packet. Reads and writes remain separate.

### Host / PCIe — Posted packet rate

Key: `posted_packet_rate`. Unit: **packets/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `parbs_tarb_pi_posted_pkts`.

### Host / PCIe — Non-posted packet rate

Key: `nonposted_packet_rate`. Unit: **packets/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `parbs_tarb_pi_non_posted_pkts`.

### Host / PCIe — Host write stalls · busiest path

Key: `host_stalls`. Unit: **cycles/s**. Operation: maximum individual counter rate; scale 1.

Numerator/bin sources: `ixe_dmawr_stall_np_cdt`, `ixe_dmawr_stall_p_cdt`.

Posted and non-posted credit-stall counters remain separate; the displayed summary is the largest path rate.

### Flow control — TX paused time · busiest class

Key: `tx_pause`. Unit: **%**. Operation: maximum individual counter rate; scale 1e-07.

Numerator/bin sources: `hni_tx_paused_0`, `hni_tx_paused_1`, `hni_tx_paused_2`, `hni_tx_paused_3`, `hni_tx_paused_4`, `hni_tx_paused_5`, `hni_tx_paused_6`, `hni_tx_paused_7`.

Per-traffic-class paused cycles / (1 GHz × covered seconds). Maximum class shown; overlapping classes are not summed into wall time.

### Flow control — RX paused time · busiest class

Key: `rx_pause`. Unit: **%**. Operation: maximum individual counter rate; scale 1e-07.

Numerator/bin sources: `hni_rx_paused_0`, `hni_rx_paused_1`, `hni_rx_paused_2`, `hni_rx_paused_3`, `hni_rx_paused_4`, `hni_rx_paused_5`, `hni_rx_paused_6`, `hni_rx_paused_7`.

Per-class received pause time under the sourced 1 GHz counter clock. This does not measure application stall time.

### Flow control — TX XOFF events

Key: `tx_xoff_rate`. Unit: **events/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `hni_pause_xoff_sent_0`, `hni_pause_xoff_sent_1`, `hni_pause_xoff_sent_2`, `hni_pause_xoff_sent_3`, `hni_pause_xoff_sent_4`, `hni_pause_xoff_sent_5`, `hni_pause_xoff_sent_6`, `hni_pause_xoff_sent_7`.

### Flow control — Received pause events

Key: `rx_pause_rate`. Unit: **events/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `hni_pause_recv_0`, `hni_pause_recv_1`, `hni_pause_recv_2`, `hni_pause_recv_3`, `hni_pause_recv_4`, `hni_pause_recv_5`, `hni_pause_recv_6`, `hni_pause_recv_7`.

### Flow control — Fabric XOFF events

Key: `fgfc_xoff`. Unit: **events/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `hni_fgfc_event_xoff`.

### Flow control — TX buffer stalls · busiest counter

Key: `tx_stalls`. Unit: **cycles/s**. Operation: maximum individual counter rate; scale 1.

Numerator/bin sources: `oxe_stall_pbuf_bc_0`, `oxe_stall_pbuf_bc_1`, `oxe_stall_pbuf_bc_2`, `oxe_stall_pbuf_bc_3`, `oxe_stall_pbuf_bc_4`, `oxe_stall_pbuf_bc_5`, `oxe_stall_pbuf_bc_6`, `oxe_stall_pbuf_bc_7`, `oxe_stall_pbuf_bc_8`, `oxe_stall_pbuf_bc_9`.

Maximum individual TX buffer stall rate; counters are not summed into a wall-clock percentage.

### Host / PCIe — Receive dispatch stalls · busiest cause

Key: `dispatch_stalls`. Unit: **cycles/s**. Operation: maximum individual counter rate; scale 1.

Numerator/bin sources: `ixe_disp_stall_atu_cdt`, `ixe_disp_stall_atu_fifo`, `ixe_disp_stall_conflict`, `ixe_disp_stall_mdid_cdts`, `ixe_disp_stall_put_resp`, `ixe_disp_stall_resp_fifo`.

Largest per-cause receive-dispatch stall rate; overlapping causes are not summed into wall time.

### Flow control — Pause frames sent

Key: `pause_sent_rate`. Unit: **events/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `hni_pause_sent`.

### Flow control — Pause refresh events

Key: `pause_refresh_rate`. Unit: **events/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `hni_pause_refresh`.

### Flow control — Fabric flow-control discards

Key: `fabric_discard_rate`. Unit: **events/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `hni_fgfc_discard`.

### Matching — Immediate priority matches

Key: `priority_fraction`. Unit: **%**. Operation: ratio of summed counts; scale 100.

Numerator/bin sources: `lpe_net_match_priority_0`, `lpe_net_match_priority_1`, `lpe_net_match_priority_2`, `lpe_net_match_priority_3`.

Denominator sources: `lpe_net_match_priority_0`, `lpe_net_match_priority_1`, `lpe_net_match_priority_2`, `lpe_net_match_priority_3`, `lpe_net_match_overflow_0`, `lpe_net_match_overflow_1`, `lpe_net_match_overflow_2`, `lpe_net_match_overflow_3`.

Priority / (priority + overflow). Zero total matches makes this undefined.

### Matching — Overflow matches

Key: `overflow_fraction`. Unit: **%**. Operation: ratio of summed counts; scale 100.

Numerator/bin sources: `lpe_net_match_overflow_0`, `lpe_net_match_overflow_1`, `lpe_net_match_overflow_2`, `lpe_net_match_overflow_3`.

Denominator sources: `lpe_net_match_priority_0`, `lpe_net_match_priority_1`, `lpe_net_match_priority_2`, `lpe_net_match_priority_3`, `lpe_net_match_overflow_0`, `lpe_net_match_overflow_1`, `lpe_net_match_overflow_2`, `lpe_net_match_overflow_3`.

Overflow / (priority + overflow); matching behavior is not an MPI protocol classification.

### Frames — Mean transmitted frame size

Key: `tx_frame_bytes`. Unit: **B/frame**. Operation: ratio of summed counts; scale 1.

Numerator/bin sources: `hni_sts_tx_ok_octets`.

Denominator sources: `hni_tx_ok_27`, `hni_tx_ok_35`, `hni_tx_ok_36_to_63`, `hni_tx_ok_64`, `hni_tx_ok_65_to_127`, `hni_tx_ok_128_to_255`, `hni_tx_ok_256_to_511`, `hni_tx_ok_512_to_1023`, `hni_tx_ok_1024_to_2047`, `hni_tx_ok_2048_to_4095`, `hni_tx_ok_4096_to_8191`, `hni_tx_ok_8192_to_max`.

OK network-frame octets / OK wire-frame count. This is not MPI message size or application payload size.

### Frames — Mean received frame size

Key: `rx_frame_bytes`. Unit: **B/frame**. Operation: ratio of summed counts; scale 1.

Numerator/bin sources: `hni_sts_rx_ok_octets`.

Denominator sources: `hni_rx_ok_27`, `hni_rx_ok_35`, `hni_rx_ok_36_to_63`, `hni_rx_ok_64`, `hni_rx_ok_65_to_127`, `hni_rx_ok_128_to_255`, `hni_rx_ok_256_to_511`, `hni_rx_ok_512_to_1023`, `hni_rx_ok_1024_to_2047`, `hni_rx_ok_2048_to_4095`, `hni_rx_ok_4096_to_8191`, `hni_rx_ok_8192_to_max`.

OK received octets / OK wire frames, not MPI messages.

### Frames — Small transmitted frames (<256 B)

Key: `tx_small_frames`. Unit: **%**. Operation: ratio of summed counts; scale 100.

Numerator/bin sources: `hni_tx_ok_27`, `hni_tx_ok_35`, `hni_tx_ok_36_to_63`, `hni_tx_ok_64`, `hni_tx_ok_65_to_127`, `hni_tx_ok_128_to_255`.

Denominator sources: `hni_tx_ok_27`, `hni_tx_ok_35`, `hni_tx_ok_36_to_63`, `hni_tx_ok_64`, `hni_tx_ok_65_to_127`, `hni_tx_ok_128_to_255`, `hni_tx_ok_256_to_511`, `hni_tx_ok_512_to_1023`, `hni_tx_ok_1024_to_2047`, `hni_tx_ok_2048_to_4095`, `hni_tx_ok_4096_to_8191`, `hni_tx_ok_8192_to_max`.

### Frames — Large transmitted frames (≥4096 B)

Key: `tx_large_frames`. Unit: **%**. Operation: ratio of summed counts; scale 100.

Numerator/bin sources: `hni_tx_ok_4096_to_8191`, `hni_tx_ok_8192_to_max`.

Denominator sources: `hni_tx_ok_27`, `hni_tx_ok_35`, `hni_tx_ok_36_to_63`, `hni_tx_ok_64`, `hni_tx_ok_65_to_127`, `hni_tx_ok_128_to_255`, `hni_tx_ok_256_to_511`, `hni_tx_ok_512_to_1023`, `hni_tx_ok_1024_to_2047`, `hni_tx_ok_2048_to_4095`, `hni_tx_ok_4096_to_8191`, `hni_tx_ok_8192_to_max`.

### Translation — Cache misses per translation request

Key: `atu_misses_per_request`. Unit: **misses/request**. Operation: ratio of summed counts; scale 1.

Numerator/bin sources: `atu_cache_miss_ee`, `atu_cache_miss_ixe`, `atu_cache_miss_oxe`.

Denominator sources: `atu_client_req_ee`, `atu_client_req_ixe`, `atu_client_req_oxe`.

Client-counted misses / client translation requests. Pool and client miss populations are not added together; this ratio is not asserted to be a probability.

### Translation — Translation cache hit fraction

Key: `atu_hit_fraction`. Unit: **%**. Operation: ratio of summed counts; scale 100.

Numerator/bin sources: `atu_cache_hit_base_page_size_0`, `atu_cache_hit_base_page_size_1`, `atu_cache_hit_base_page_size_2`, `atu_cache_hit_base_page_size_3`, `atu_cache_hit_derivative1_page_size_0`, `atu_cache_hit_derivative1_page_size_1`, `atu_cache_hit_derivative1_page_size_2`, `atu_cache_hit_derivative1_page_size_3`, `atu_cache_hit_derivative2_page_size_0`, `atu_cache_hit_derivative2_page_size_1`, `atu_cache_hit_derivative2_page_size_2`, `atu_cache_hit_derivative2_page_size_3`.

Denominator sources: `atu_cache_hit_base_page_size_0`, `atu_cache_hit_base_page_size_1`, `atu_cache_hit_base_page_size_2`, `atu_cache_hit_base_page_size_3`, `atu_cache_hit_derivative1_page_size_0`, `atu_cache_hit_derivative1_page_size_1`, `atu_cache_hit_derivative1_page_size_2`, `atu_cache_hit_derivative1_page_size_3`, `atu_cache_hit_derivative2_page_size_0`, `atu_cache_hit_derivative2_page_size_1`, `atu_cache_hit_derivative2_page_size_2`, `atu_cache_hit_derivative2_page_size_3`, `atu_cache_miss_0`, `atu_cache_miss_1`, `atu_cache_miss_2`, `atu_cache_miss_3`.

### Translation — Translation misses per observed GB

Key: `atu_misses_per_GB`. Unit: **misses/GB**. Operation: ratio of summed counts; scale 1e+09.

Numerator/bin sources: `atu_cache_miss_0`, `atu_cache_miss_1`, `atu_cache_miss_2`, `atu_cache_miss_3`.

Denominator sources: `hni_sts_tx_ok_octets`, `hni_sts_rx_ok_octets`.

Pool-counted misses normalized by local TX + RX network-frame octets (decimal GB); not unique job payload.

### Translation — Translation request rate

Key: `atu_request_rate`. Unit: **requests/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `atu_client_req_ee`, `atu_client_req_ixe`, `atu_client_req_oxe`.

### Translation — Translation cache evictions

Key: `atu_eviction_rate`. Unit: **events/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `atu_cache_evictions`.

### Operations — DMA share of commands

Key: `dma_share`. Unit: **%**. Operation: ratio of summed counts; scale 100.

Numerator/bin sources: `cq_num_dma_cmds_0`, `cq_num_dma_cmds_1`, `cq_num_dma_cmds_2`, `cq_num_dma_cmds_3`.

Denominator sources: `cq_num_dma_cmds_0`, `cq_num_dma_cmds_1`, `cq_num_dma_cmds_2`, `cq_num_dma_cmds_3`, `cq_num_idc_cmds_0`, `cq_num_idc_cmds_1`, `cq_num_idc_cmds_2`, `cq_num_idc_cmds_3`.

Control-path command mix only. DMA / IDC does not establish eager / rendezvous MPI protocol.

### Operations — PUT command share

Key: `put_share`. Unit: **%**. Operation: ratio of summed counts; scale 100.

Numerator/bin sources: `cq_dma_cmd_counts_1`.

Denominator sources: `cq_dma_cmd_counts_0`, `cq_dma_cmd_counts_1`, `cq_dma_cmd_counts_2`, `cq_dma_cmd_counts_3`, `cq_dma_cmd_counts_4`, `cq_dma_cmd_counts_5`, `cq_dma_cmd_counts_6`, `cq_dma_cmd_counts_7`, `cq_dma_cmd_counts_8`, `cq_dma_cmd_counts_9`, `cq_dma_cmd_counts_10`, `cq_dma_cmd_counts_11`, `cq_dma_cmd_counts_12`, `cq_dma_cmd_counts_13`, `cq_dma_cmd_counts_14`, `cq_dma_cmd_counts_15`.

### Operations — GET command share

Key: `get_share`. Unit: **%**. Operation: ratio of summed counts; scale 100.

Numerator/bin sources: `cq_dma_cmd_counts_2`.

Denominator sources: `cq_dma_cmd_counts_0`, `cq_dma_cmd_counts_1`, `cq_dma_cmd_counts_2`, `cq_dma_cmd_counts_3`, `cq_dma_cmd_counts_4`, `cq_dma_cmd_counts_5`, `cq_dma_cmd_counts_6`, `cq_dma_cmd_counts_7`, `cq_dma_cmd_counts_8`, `cq_dma_cmd_counts_9`, `cq_dma_cmd_counts_10`, `cq_dma_cmd_counts_11`, `cq_dma_cmd_counts_12`, `cq_dma_cmd_counts_13`, `cq_dma_cmd_counts_14`, `cq_dma_cmd_counts_15`.

### Operations — Rendezvous PUT command share

Key: `rendezvous_share`. Unit: **%**. Operation: ratio of summed counts; scale 100.

Numerator/bin sources: `cq_dma_cmd_counts_3`.

Denominator sources: `cq_dma_cmd_counts_0`, `cq_dma_cmd_counts_1`, `cq_dma_cmd_counts_2`, `cq_dma_cmd_counts_3`, `cq_dma_cmd_counts_4`, `cq_dma_cmd_counts_5`, `cq_dma_cmd_counts_6`, `cq_dma_cmd_counts_7`, `cq_dma_cmd_counts_8`, `cq_dma_cmd_counts_9`, `cq_dma_cmd_counts_10`, `cq_dma_cmd_counts_11`, `cq_dma_cmd_counts_12`, `cq_dma_cmd_counts_13`, `cq_dma_cmd_counts_14`, `cq_dma_cmd_counts_15`.

Hardware command opcode mix, not a reconstruction of software protocol or MPI message sizes.

### Operations — Small-message command share

Key: `small_command_share`. Unit: **%**. Operation: ratio of summed counts; scale 100.

Numerator/bin sources: `cq_dma_cmd_counts_7`.

Denominator sources: `cq_dma_cmd_counts_0`, `cq_dma_cmd_counts_1`, `cq_dma_cmd_counts_2`, `cq_dma_cmd_counts_3`, `cq_dma_cmd_counts_4`, `cq_dma_cmd_counts_5`, `cq_dma_cmd_counts_6`, `cq_dma_cmd_counts_7`, `cq_dma_cmd_counts_8`, `cq_dma_cmd_counts_9`, `cq_dma_cmd_counts_10`, `cq_dma_cmd_counts_11`, `cq_dma_cmd_counts_12`, `cq_dma_cmd_counts_13`, `cq_dma_cmd_counts_14`, `cq_dma_cmd_counts_15`.

### Latency — Request / response latency distribution

Key: `request_latency`. Unit: **bin index**. Operation: P99 bucket from aligned histogram counts; scale 1.

Numerator/bin sources: `pct_req_rsp_latency_0`, `pct_req_rsp_latency_1`, `pct_req_rsp_latency_2`, `pct_req_rsp_latency_3`, `pct_req_rsp_latency_4`, `pct_req_rsp_latency_5`, `pct_req_rsp_latency_6`, `pct_req_rsp_latency_7`, `pct_req_rsp_latency_8`, `pct_req_rsp_latency_9`, `pct_req_rsp_latency_10`, `pct_req_rsp_latency_11`, `pct_req_rsp_latency_12`, `pct_req_rsp_latency_13`, `pct_req_rsp_latency_14`, `pct_req_rsp_latency_15`, `pct_req_rsp_latency_16`, `pct_req_rsp_latency_17`, `pct_req_rsp_latency_18`, `pct_req_rsp_latency_19`, `pct_req_rsp_latency_20`, `pct_req_rsp_latency_21`, `pct_req_rsp_latency_22`, `pct_req_rsp_latency_23`, `pct_req_rsp_latency_24`, `pct_req_rsp_latency_25`, `pct_req_rsp_latency_26`, `pct_req_rsp_latency_27`, `pct_req_rsp_latency_28`, `pct_req_rsp_latency_29`, `pct_req_rsp_latency_30`, `pct_req_rsp_latency_31`.

Cumulative histogram bin deltas. Without capture-specific bin edges, P99 is a bucket index, not nanoseconds.

### Latency — Host-access latency distribution

Key: `host_latency`. Unit: **bin index**. Operation: P99 bucket from aligned histogram counts; scale 1.

Numerator/bin sources: `pct_host_access_latency_0`, `pct_host_access_latency_1`, `pct_host_access_latency_2`, `pct_host_access_latency_3`, `pct_host_access_latency_4`, `pct_host_access_latency_5`, `pct_host_access_latency_6`, `pct_host_access_latency_7`, `pct_host_access_latency_8`, `pct_host_access_latency_9`, `pct_host_access_latency_10`, `pct_host_access_latency_11`, `pct_host_access_latency_12`, `pct_host_access_latency_13`, `pct_host_access_latency_14`, `pct_host_access_latency_15`.

Requires capture-specific bin edges for physical latency bounds; no example/default calibration is assumed.

### Reliability — Reliability event increments

Key: `reliability_events`. Unit: **increments/s**. Operation: sum of counter rates; scale 1.

Numerator/bin sources: `pct_no_mst_nacks`, `pct_no_tct_nacks`, `pct_no_trs_nacks`, `pct_retry_srb_requests`, `pct_sct_timeouts`, `pct_spt_timeouts`, `pct_trs_rsp_nack_drops`, `hni_llr_rx_replay_event`, `hni_llr_tx_replay_event`, `hni_pcs_corrected_cw`, `hni_pcs_uncorrected_cw`, `hni_discard_cntr_0`, `hni_discard_cntr_1`, `hni_discard_cntr_2`, `hni_discard_cntr_3`, `hni_discard_cntr_4`, `hni_discard_cntr_5`, `hni_discard_cntr_6`, `hni_discard_cntr_7`, `ixe_rx_pkt_drop_ixe_parser`, `ixe_rx_pkt_drop_pct`, `ixe_rx_pkt_drop_rmu_norsp`, `ixe_rx_pkt_drop_rmu_wrsp`.

Sum of named event-counter increments, not unique failures. Good codewords are context, never errors; corrected and uncorrected events remain individually visible.

## Traffic, balance, phases, and interpretation

Traffic retains exact observed TX/RX byte totals, covered-time mean/median/P99
bandwidth, variability/CV, active fraction and bounded timeline/heatmap displays.
A saved interval is a manual phase hypothesis, not an automatically detected
MPI phase. A/B comparisons use the same current metric/device/collector scope.
Window-local statistics and export metadata move together.

Traffic share is within the selected population (and per node). Effective NICs
= 1 / sum(share²). Active means observed nonzero traffic; max/min is infinite only
when the observed population includes measured zero alongside positive traffic.
Missing/ambiguous NICs do not become idle zeros. Population P50/P99 volume is a
percentile **across NICs**, not a bandwidth/time percentile or an average of P99s.
Unequal coverage limits volume-comparison conclusions; inspect coverage first.

Wire frames are not MPI messages. DMA/IDC and opcode shares are hardware command
mix, not proof of eager/rendezvous software protocol. Device counters can include
background or unrelated traffic during the collection window. Reliability event
increments need not count unique failures; good codewords are not errors. Bad-
octet invariants refer to **window increments**, not the counter's lifetime value.

Change-only, baseline-relative pftrace does not preserve the initial observation,
all unchanged samples, missing all-zero tracks, or a uniform sampling cadence.
The app does not infer idle fraction/CV/native-cadence traffic percentiles from
that representation. Derived ratios require every input; missing histogram bins
make latency unavailable rather than zero-filled. Double-precision source values
cannot promise arbitrary uint64 fidelity.

## Data-limited analyses

Device counters alone do not supply communication peers, routes, distances,
cross-host clock synchronization or physical latency-bin calibration. The app
accepts explicit topology groups and run-bound latency calibration through the
[analysis configuration](CONFIGURATION.md). Translation latency needs its own
observations, and pause/XOFF is not relabeled as ECN. Correlated plots can
support investigation; they do not establish causation or a bottleneck.
