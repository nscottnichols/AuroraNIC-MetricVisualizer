#!/usr/bin/env python3
"""Real-source import timings and a real loopback Waitress/Dash path-opening smoke.

Never contacts Sunspot. Reads an extracted run containing ctf/ and timeline.pftrace.
Run: python scripts/validate_runtime.py /path/to/run --output artifacts/runtime
"""
from pathlib import Path
import argparse
import copy
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / 'src'))
from main import load_trace_store
from trace_io import trace_arguments
from explorer.diagnostics import METRICS
from explorer.model import ExplorerEngine, Query


def request(base, path, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=payload,
        headers={'Content-Type': 'application/json'} if payload else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def components(node):
    found = {}
    if isinstance(node, dict):
        props = node.get('props', {})
        if 'id' in props:
            found[props['id']] = props
        for value in node.values():
            found.update(components(value))
    elif isinstance(node, list):
        for value in node:
            found.update(components(value))
    return found


def outputs(key):
    parts = key[2:-2].split('...') if key.startswith('..') else [key]
    result = [{'id': part.rsplit('.', 1)[0], 'property': part.rsplit('.', 1)[1]} for part in parts]
    return result if key.startswith('..') else result[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve(strict=True)
    dest = args.output.resolve(); dest.mkdir(parents=True, exist_ok=True)
    report = {'environment': {'python': sys.version}, 'metric_count': len(METRICS), 'imports': {}, 'http': []}
    import importlib.metadata as metadata
    for name in ('numpy', 'dash', 'plotly', 'perfetto', 'protobuf', 'waitress', 'pytest', 'playwright'):
        try: report['environment'][name] = metadata.version(name)
        except metadata.PackageNotFoundError: pass
    with tempfile.TemporaryDirectory(prefix='nicviz-runtime-cache-') as temporary:
        for source in ('ctf', 'timeline.pftrace'):
            arguments = trace_arguments(root / source, root=root, cache=False)
            start = time.perf_counter(); store = load_trace_store(arguments)
            cold = time.perf_counter() - start
            arguments.trace_cache = str(Path(temporary) / (source.replace('.', '-') + '.npz'))
            start = time.perf_counter(); load_trace_store(arguments)
            cold_cache = time.perf_counter() - start
            start = time.perf_counter(); cached = load_trace_store(arguments)
            warm = time.perf_counter() - start
            assert cached.row_count == store.row_count
            engine = ExplorerEngine(store)
            start = time.perf_counter()
            for metric in ('traffic', *METRICS): engine.query(Query(engine.origin, engine.end, metric=metric))
            first_queries = time.perf_counter() - start
            start = time.perf_counter()
            for metric in ('traffic', *METRICS): engine.query(Query(engine.origin, engine.end, metric=metric))
            repeat_queries = time.perf_counter() - start
            repeated=Query(engine.origin,engine.end,metric='posted_pushback')
            engine.query(repeated)
            start=time.perf_counter()
            for _ in range(100):engine.query(repeated)
            same_query=(time.perf_counter()-start)/100
            report['imports'][source] = {'samples': store.row_count, 'cold_no_app_cache_seconds': cold,
                'cold_build_cache_seconds': cold_cache, 'warm_app_cache_seconds': warm,
                'first_all_metric_queries_seconds': first_queries, 'repeated_all_metric_queries_seconds': repeat_queries,
                'mean_same_query_cache_hit_seconds_over_100_calls':same_query,
                'reader': store.identity.reader_version, 'fingerprint': store.identity.source_fingerprint,
                'note': 'One local run; OS filesystem cache was not flushed; query timing excludes browser rendering. Scanning 35 metrics exceeds the four-selection LRU; only same-query timing measures hits.'}
        previous = root / 'trace-cache-hpe-cassini-octets-v1.npz'
        if previous.is_file():
            cache = Path(temporary) / 'copied-legacy-cache.npz'
            shutil.copy2(previous, cache)
            arguments = trace_arguments(root / 'ctf', root=root, cache=False)
            arguments.trace_cache = str(cache)
            rebuilt = load_trace_store(arguments)
            assert rebuilt.row_count == 93720 and len(rebuilt.collectors) == 2
            report['stale_cache_rebuild'] = {'copied_old_cache_rebuilt': True, 'collectors': 2,
                'original_cache_modified': False}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    log = (dest / 'waitress-smoke.log').open('w')
    process = subprocess.Popen([sys.executable, str(APP/'src/main.py'), '--browse-root', str(root),
        '--host', '127.0.0.1', '--port', str(port), '--server', 'waitress'], cwd=APP,
        stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 30
        while True:
            if process.poll() is not None:
                raise RuntimeError('Server exited: ' + (dest/'waitress-smoke.log').read_text())
            try:
                status, _ = request(base, '/healthz')
                if status == 200: break
            except (OSError, urllib.error.URLError): pass
            if time.monotonic() > deadline: raise TimeoutError('Waitress startup timed out.')
            time.sleep(.1)
        assert request(base, '/readyz')[0] == 503
        layout = json.loads(request(base, '/_dash-layout')[1])
        assert 'ex-file-path' in components(layout)
        deps = json.loads(request(base, '/_dash-dependencies')[1])
        keys = [item['output'] for item in deps]
        action = next(k for k in keys if k.startswith('..ex-file-request.data'))
        poll = next(k for k in keys if k.startswith('..ex-file-status.children'))
        report['http'].append({'stage': 'empty picker', 'healthz': 200, 'readyz': 503, 'actual_socket_http': True})
        for index, source in enumerate(('ctf', 'timeline.pftrace'), 1):
            code, content = request(base, '/_dash-update-component', {
                'output': action, 'outputs': outputs(action),
                'inputs': [{'id':'ex-file-load','property':'n_clicks','value':index},
                    {'id':'ex-file-cancel','property':'n_clicks','value':0}],
                'state': [{'id':'ex-file-path','property':'value','value':str(root/source)},
                    {'id':'ex-file-format','property':'value','value':'auto'},
                    {'id':'ex-file-request','property':'data','value':None}],
                'changedPropIds': ['ex-file-load.n_clicks']})
            assert code == 200, content
            token = json.loads(content)['response']['ex-file-request']['data']
            deadline = time.monotonic() + 30
            while True:
                code, content = request(base, '/_dash-update-component', {
                    'output':poll,'outputs':outputs(poll),
                    'inputs':[{'id':'ex-file-poll','property':'n_intervals','value':1},
                        {'id':'ex-file-request','property':'data','value':token}],
                    'state':[{'id':'ex-file-current','property':'data','value':None},
                        {'id':'ex-file-landing','property':'data','value':True}],
                    'changedPropIds':['ex-file-poll.n_intervals']})
                assert code == 200, content
                data = json.loads(content)['response']
                text = data['ex-file-status']['children']
                if 'Trace ready' in text: break
                if 'failed' in text.lower(): raise AssertionError(data)
                if time.monotonic() > deadline: raise TimeoutError(text)
                time.sleep(.1)
            assert data['ex-file-location']['href'] == '/'
            assert request(base, '/readyz')[0] == 200
            code, content = request(base, '/_dash-layout')
            loaded = components(json.loads(content)); assert 'ex-state' in loaded
            state = loaded['ex-state']['data']; state['query']['metric'] = 'posted_pushback'
            render = next(k for k in keys if k.startswith('..ex-cards.children'))
            code, content = request(base, '/_dash-update-component', {'output':render,'outputs':outputs(render),
                'inputs':[{'id':'ex-state','property':'data','value':state}], 'state':[], 'changedPropIds':['ex-state.data']})
            assert code == 200, content
            assert 'cycles/packet' in json.dumps(json.loads(content)), content[:500]
            report['http'].append({'stage':source,'load_action':200,'readyz':200,'loaded_layout':200,
                'posted_pushback_render':200,'render_response_bytes':len(content), 'actual_socket_http':True})
    finally:
        process.send_signal(signal.SIGTERM)
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(); report['forced_shutdown'] = True
        log.close()
    report['server_exit_code'] = process.returncode
    (dest/'runtime-validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
