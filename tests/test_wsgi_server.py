import signal
import sys
from types import SimpleNamespace

import pytest

from src.main import (
    _validate_server_bind,
    main,
    parse_args,
    run_interactive_dash_app,
    run_trace_app,
    run_wsgi_app,
    run_waitress_server,
)


def test_server_option_parsing_and_default(monkeypatch):
    monkeypatch.delenv("AURORA_APP_SERVER", raising=False)
    assert parse_args([]).server == "development"

    assert parse_args(["--server", "waitress"]).server == "waitress"
    monkeypatch.setenv("AURORA_APP_SERVER", "waitress")
    assert parse_args([]).server == "waitress"

    with pytest.raises(SystemExit):
        parse_args(["--server", "unsupported"])


def test_server_selection_preserves_development_path(monkeypatch):
    app = object()
    calls = []
    monkeypatch.setattr(
        "src.main.run_app",
        lambda value, host, port: calls.append((value, host, port)),
    )

    run_wsgi_app(app, "127.0.0.1", 12345, "development")
    assert calls == [(app, "127.0.0.1", 12345)]


def test_server_selection_uses_waitress(monkeypatch):
    app = object()
    calls = []
    monkeypatch.setattr(
        "src.main.run_waitress_server",
        lambda value, host, port: calls.append((value, host, port)),
    )

    run_wsgi_app(app, "127.0.0.1", 12345, "waitress")
    assert calls == [(app, "127.0.0.1", 12345)]


def test_trace_and_legacy_modes_receive_server_choice(monkeypatch):
    trace_app = object()
    legacy_app = object()
    trace_calls = []
    legacy_calls = []

    monkeypatch.setattr(
        "src.main.create_analysis_app", lambda store, **kwargs: trace_app
    )
    monkeypatch.setattr(
        "src.main.run_wsgi_app",
        lambda app, host, port, server: trace_calls.append(
            (app, host, port, server)
        ),
    )
    trace_args = SimpleNamespace(
        host="127.0.0.1", port=12345, server="waitress"
    )
    assert run_trace_app(trace_args) is trace_app
    assert trace_calls == [(trace_app, "127.0.0.1", 12345, "waitress")]

    monkeypatch.setattr(
        "src.main.create_runtime_app", lambda *args: legacy_app
    )
    monkeypatch.setattr(
        "src.main.run_wsgi_app",
        lambda app, host, port, server: legacy_calls.append(
            (app, host, port, server)
        ),
    )
    run_interactive_dash_app(
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        host="127.0.0.1",
        port=12346,
        server="waitress",
    )
    assert legacy_calls == [(legacy_app, "127.0.0.1", 12346, "waitress")]


def test_waitress_non_loopback_is_rejected_before_server_startup(monkeypatch):
    args = SimpleNamespace(
        host="0.0.0.0",
        port=12345,
        server="waitress",
        allow_non_loopback=False,
    )

    with pytest.raises(RuntimeError, match="Refusing non-loopback bind"):
        _validate_server_bind(args)

    monkeypatch.setattr("src.main.parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="Refusing non-loopback bind"):
        main()


def test_missing_waitress_dependency_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "waitress", None)

    with pytest.raises(RuntimeError, match="Install it with"):
        run_waitress_server(object(), host="127.0.0.1", port=12345)


def test_waitress_startup_and_clean_shutdown(monkeypatch):
    wsgi_app = object()
    dash_app = SimpleNamespace(server=wsgi_app)
    created = {}
    run_calls = []
    original_sigterm = signal.getsignal(signal.SIGTERM)
    captured_handler = None

    class FakeWaitressServer:
        def run(self):
            run_calls.append("run")
            captured_handler(signal.SIGTERM, None)

    def fake_create_server(application, *, host, port, **kwargs):
        created.update(
            {
                "application": application,
                "host": host,
                "port": port,
                "kwargs": kwargs,
            }
        )
        return FakeWaitressServer()

    def fake_signal(signal_number, handler):
        nonlocal captured_handler
        assert signal_number == signal.SIGTERM
        captured_handler = handler
        return original_sigterm

    fake_waitress = SimpleNamespace(create_server=fake_create_server)
    monkeypatch.setitem(sys.modules, "waitress", fake_waitress)
    monkeypatch.setattr(signal, "signal", fake_signal)

    run_waitress_server(dash_app, host="127.0.0.1", port=12345)

    assert created["application"] is wsgi_app
    assert created["host"] == "127.0.0.1"
    assert created["port"] == 12345
    assert created["kwargs"]["clear_untrusted_proxy_headers"] is True
    assert run_calls == ["run"]
    assert signal.getsignal(signal.SIGTERM) is original_sigterm
