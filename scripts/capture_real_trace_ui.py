#!/usr/bin/env python3
"""Capture browser evidence for a live real-trace legacy analysis UI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:18050"
DEFAULT_1440_OUTPUT = ROOT / "artifacts" / "real-trace-1440x900.png"
DEFAULT_1024_OUTPUT = ROOT / "artifacts" / "real-trace-1024x768.png"
OVERVIEW_SECTION_SELECTORS = (
    "#overview-heading",
    "#node-heading",
    "#analysis-node-table",
    "#job-profile-heading",
    "#analysis-job-profile-table",
)
NIC_DETAIL_SECTION_SELECTORS = (
    "#nic-detail-heading",
    "#analysis-nic-graph",
    "#nic-02-flow-control-heading",
    "#nic-03-matching-heading",
    "#nic-04-host-pressure-heading",
    "#nic-05-operations-heading",
    "#nic-06-frames-heading",
    "#nic-07-translation-heading",
    "#nic-08-latency-heading",
    "#nic-09-reliability-heading",
)
DIAGNOSTIC_EXPORT_SECTIONS = (
    "flow_control",
    "matching",
    "host_pressure",
    "operations",
    "frames",
    "translation",
    "latency",
    "reliability",
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Exercise and screenshot a live real-trace analysis UI."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-1440", type=Path, default=DEFAULT_1440_OUTPUT)
    parser.add_argument("--output-1024", type=Path, default=DEFAULT_1024_OUTPUT)
    parser.add_argument("--expected-run-id", default="m1-allreduce-20261006T0040Z")
    parser.add_argument("--expected-nic-detail", default="x1921c6s3b0n0 / cxi0")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.output_1440.resolve() == args.output_1024.resolve():
        parser.error("the two screenshot destinations must be different")
    return args


def format_timing_line(viewport_width, viewport_height, navigation_ms, first_useful_view_ms, total_ms):
    return (
        f"timing viewport={viewport_width}x{viewport_height} "
        f"navigation_to_dom_content_loaded_ms={navigation_ms:.3f} "
        f"dom_content_loaded_to_first_useful_view_ms={first_useful_view_ms:.3f} "
        f"total_capture_ms={total_ms:.3f}"
    )


def capture_page(
    browser,
    url,
    timeout,
    output,
    width,
    height,
    expected_run_id="m1-allreduce-20261006T0040Z",
    expected_nic_detail="x1921c6s3b0n0 / cxi0",
):
    page = browser.new_page(viewport={"width": width, "height": height})
    capture_started = time.perf_counter()
    page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
    dom_content_loaded = time.perf_counter()
    page.wait_for_selector("#analysis-overview", state="visible", timeout=timeout * 1000)
    first_useful_view = time.perf_counter()

    for selector in OVERVIEW_SECTION_SELECTORS:
        page.wait_for_selector(selector, state="visible", timeout=timeout * 1000)

    assert page.title() == "Aurora NIC Metrics"
    overview_text = page.locator("#analysis-overview").first.inner_text()
    assert "Job overview" in overview_text
    assert "QUESTION COVERAGE" in overview_text

    page.locator(".tab", has_text="NIC detail").click()
    for selector in NIC_DETAIL_SECTION_SELECTORS:
        page.wait_for_selector(selector, state="visible", timeout=timeout * 1000)

    nic_detail_text = page.locator("#analysis-nic-detail").first.inner_text()
    assert "NIC detail" in nic_detail_text
    assert expected_nic_detail in nic_detail_text

    page.screenshot(path=str(output), full_page=True)
    export_path = output.with_name(
        f"real-trace-export-{width}x{height}.json"
    )
    with page.expect_download(timeout=timeout * 1000) as download_info:
        page.locator("#analysis-export-button").click()
    download = download_info.value
    download.save_as(str(export_path))
    export = json.loads(export_path.read_text(encoding="utf-8"))

    assert "schema_version" in export
    assert "run" in export
    assert "nic_detail" in export
    assert "methodology" in export
    assert "formulas" in export["methodology"]
    assert "export" in export
    assert export["run"]["run_id"] == expected_run_id

    for section_name in DIAGNOSTIC_EXPORT_SECTIONS:
        section = export["nic_detail"][section_name]
        assert section["state"]
        assert section["status"]
        assert section.get("limitations") or section.get("status_reasons")

    questions = export["question_capabilities"]
    assert len(questions) == 15
    assert len({question["id"] for question in questions}) == 15
    for question in questions:
        assert question["state"]
        assert question["status"]
        assert question["reason"]

    assert export["node_balance"]["rows"]
    assert all(row["state"] and row["status"] for row in export["node_balance"]["rows"])
    assert export["job_balance"]["directions"]
    assert all(
        direction["state"] and direction["status"]
        for direction in export["job_balance"]["directions"]
    )
    assert export["job_profile"]["status"]
    assert len(export["job_profile"]["questions"]) == 15

    forbidden_keys = {
        "uint64_values",
        "float64_values",
        "timestamps_ns",
        "display_values",
        "display_timestamps_ns",
        "raw_store",
    }

    def key_names(value):
        if isinstance(value, dict):
            yield from value.keys()
            for child in value.values():
                yield from key_names(child)
        elif isinstance(value, list):
            for child in value:
                yield from key_names(child)

    assert not forbidden_keys.intersection(key_names(export))

    page.close()
    capture_completed = time.perf_counter()
    print(
        format_timing_line(
            width,
            height,
            (dom_content_loaded - capture_started) * 1000,
            (first_useful_view - dom_content_loaded) * 1000,
            (capture_completed - capture_started) * 1000,
        ),
        flush=True,
    )
    return export_path


def main(argv=None):
    args = parse_arguments(argv)
    args.output_1440.parent.mkdir(parents=True, exist_ok=True)
    args.output_1024.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        export_1440 = capture_page(
            browser,
            args.url,
            args.timeout,
            args.output_1440,
            1440,
            900,
            args.expected_run_id,
            args.expected_nic_detail,
        )
        export_1024 = capture_page(
            browser,
            args.url,
            args.timeout,
            args.output_1024,
            1024,
            768,
            args.expected_run_id,
            args.expected_nic_detail,
        )
        browser.close()

    print(f"URL: {args.url}")
    print(f"Screenshot: {args.output_1440}")
    print(f"Screenshot: {args.output_1024}")
    print(f"Export: {export_1440}")
    print(f"Export: {export_1024}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
