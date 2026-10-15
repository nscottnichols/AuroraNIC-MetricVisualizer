from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


APP_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = APP_ROOT / "scripts" / "capture_real_trace_ui.py"


def _load_capture_module():
    spec = importlib.util.spec_from_file_location(
        "capture_real_trace_ui_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeLocator:
    def __init__(self, text=""):
        self.text = text

    @property
    def first(self):
        return self

    def click(self):
        return None

    def inner_text(self):
        return self.text


class FakeDownload:
    def __init__(self, document):
        self.document = document

    def save_as(self, path):
        Path(path).write_text(json.dumps(self.document), encoding="utf-8")


class FakePage:
    def __init__(self, export_document):
        self.export_document = export_document
        self.waited_selectors = []
        self.clicked_tabs = []

    def goto(self, *_args, **_kwargs):
        return None

    def wait_for_selector(self, *_args, **_kwargs):
        self.waited_selectors.append(_args[0])
        return None

    def title(self):
        return "Aurora NIC Metrics"

    def locator(self, selector, **_kwargs):
        texts = {
            "#analysis-overview": "Job overview\nQUESTION COVERAGE",
            "#analysis-nic-detail": "NIC detail\nx1921c6s3b0n0 / cxi0",
        }
        return FakeLocator(texts.get(selector, ""))

    def screenshot(self, *_args, **_kwargs):
        return None

    @contextmanager
    def expect_download(self, *_args, **_kwargs):
        yield SimpleNamespace(value=FakeDownload(self.export_document))

    def close(self):
        return None


def test_capture_page_reports_deterministic_timing_fields(monkeypatch, tmp_path, capsys):
    module = _load_capture_module()
    timestamps = iter((0.0, 0.1, 0.2, 0.4))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(timestamps))
    export_document = {
        "schema_version": 1,
        "run": {"run_id": "m1-allreduce-20261006T0040Z"},
        "nic_detail": {
            section_name: {
                "state": "unsupported",
                "status": "Unsupported",
                "limitations": ["Sourced metadata is required."],
            }
            for section_name in module.DIAGNOSTIC_EXPORT_SECTIONS
        },
        "question_capabilities": [
            {
                "id": f"QUESTION-{index:02d}",
                "state": "partial",
                "status": "Pending",
                "reason": "Required metadata is incomplete.",
            }
            for index in range(15)
        ],
        "node_balance": {
            "rows": [{"state": "missing", "status": "Missing Data"}]
        },
        "job_balance": {
            "directions": [{"state": "missing", "status": "Missing Data"}]
        },
        "job_profile": {
            "status": "partial",
            "questions": [f"QUESTION-{index:02d}" for index in range(15)],
        },
        "methodology": {"formulas": {}},
        "export": {},
    }
    output = tmp_path / "real-trace.png"
    page = FakePage(export_document)

    export_path = module.capture_page(
        SimpleNamespace(new_page=lambda **_kwargs: page),
        "http://127.0.0.1:18050",
        timeout=1,
        output=output,
        width=1440,
        height=900,
    )

    assert export_path == tmp_path / "real-trace-export-1440x900.json"
    expected_selectors = (
        ("#analysis-overview",)
        + module.OVERVIEW_SECTION_SELECTORS
        + module.NIC_DETAIL_SECTION_SELECTORS
    )
    assert page.waited_selectors == list(expected_selectors)
    output_line = capsys.readouterr().out.strip()
    assert output_line == (
        "timing viewport=1440x900 "
        "navigation_to_dom_content_loaded_ms=100.000 "
        "dom_content_loaded_to_first_useful_view_ms=100.000 "
        "total_capture_ms=400.000"
    )
