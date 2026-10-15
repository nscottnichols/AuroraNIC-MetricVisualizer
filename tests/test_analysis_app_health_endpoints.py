from analysis_app import create_analysis_app
from analysis_fixture import make_store


def test_trace_app_health_returns_json():
    app = create_analysis_app(make_store())

    response = app.server.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.content_type == "application/json"
    assert response.get_json() == {"status": "ok"}


def test_trace_app_ready_returns_meaningful_json_when_ready():
    app = create_analysis_app(make_store(sample_count=12))

    response = app.server.test_client().get("/readyz")

    assert response.status_code == 200
    assert response.content_type == "application/json"
    report = response.get_json()
    assert report["status"] == "ready"
    assert report["checks"] == {
        "store_present": True,
        "store_schema_supported": True,
        "indexed_rows_present": True,
        "trace_dimensions_present": True,
        "observed_counters_present": True,
        "layout_initialized": True,
        "callbacks_registered": True,
    }


def test_trace_app_ready_returns_unavailable_when_initialization_is_incomplete():
    app = create_analysis_app(make_store(sample_count=12))
    app.callback_map.clear()

    response = app.server.test_client().get("/readyz")

    assert response.status_code == 503
    assert response.content_type == "application/json"
    report = response.get_json()
    assert report["status"] == "not_ready"
    assert report["checks"]["callbacks_registered"] is False
    assert report["checks"]["store_present"] is True
