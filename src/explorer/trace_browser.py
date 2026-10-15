"""Root-confined server file picker and asynchronous, cancellable trace opening.

No file bytes cross the browser. A single-user server has one active dataset;
requests are serialized by DatasetLifecycleManager and the latest request wins.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

from dash import Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from dataset_lifecycle import DatasetDescriptor, DatasetRequestToken
from trace_io import TRACE_SUFFIXES, resolve_trace_path, trace_arguments


class TracePathControls:
    current_dataset_store_id = 'ex-file-current'

    def __init__(self, manager):
        self.manager = manager
        self.root = manager.approved_root

    def listing(self, value):
        path = resolve_trace_path(self.root, value)
        folder = path if path.is_dir() else path.parent
        entries = []
        # Cap directory work and refuse symlink escapes. No recursive listing.
        for index, child in enumerate(folder.iterdir()):
            if index >= 10_000:
                break
            if child.name.startswith('.'):
                continue
            try:
                resolved = child.resolve(strict=True)
                if not resolved.is_relative_to(self.root):
                    continue
                if child.is_dir():
                    entries.append((0, child.name, str(resolved)))
                elif child.is_file() and (child.suffix.lower() in TRACE_SUFFIXES or child.name == 'metadata'):
                    entries.append((1, child.name, str(resolved)))
            except OSError:
                continue
        entries.sort(key=lambda item: (item[0], item[1].lower()))
        options = [{'label': ('Directory · ' if kind == 0 else 'Trace · ') + name, 'value': value}
                   for kind, name, value in entries[:300]]
        return options, str(folder), f'{len(options)} entries shown' + (' (listing capped)' if len(entries) > 300 else '')

    def open_path(self, value, format='auto'):
        path = resolve_trace_path(self.root, value)
        args = trace_arguments(path, root=self.root, format=format)
        dataset_id = 'path-' + hashlib.sha256((args.trace + args.trace_format).encode()).hexdigest()[:24]
        self.manager.register(DatasetDescriptor(dataset_id, Path(args.trace), Path(args.trace_cache) if args.trace_cache else None, args))
        return self.manager.open(dataset_id)

    def render(self, *, landing=False):
        current = self.manager.current()
        initial = str(current.descriptor.trace_path) if current else str(self.root)
        options, folder, message = self.listing(initial)
        return html.Div(className='trace-picker', children=[
            dcc.Store(id='ex-file-request'), dcc.Store(id=self.current_dataset_store_id),
            dcc.Store(id='ex-file-landing', data=landing),
            dcc.Interval(id='ex-file-poll', interval=600, n_intervals=0),
            dcc.Location(id='ex-file-location', refresh=True),
            html.P('Open a CTF directory or a THAPI .pftrace file on the application server. Nothing is uploaded through the tunnel.', className='muted'),
            html.Label('Server path', htmlFor='ex-file-path'),
            html.Div(className='path-row', children=[
                dcc.Input(id='ex-file-path', value=initial, type='text', debounce=True, placeholder='Path relative to browse root, or an absolute path'),
                html.Button('Browse', id='ex-file-browse', className='subtle-button'),
                html.Button('Parent directory', id='ex-file-up', className='subtle-button')]),
            html.Div(id='ex-file-folder', children=folder, className='tiny'),
            dcc.Dropdown(id='ex-file-entry', options=options, placeholder='Choose a directory or trace file…', className='file-list', clearable=True),
            html.Div(id='ex-file-list-status', children=message, className='tiny'),
            html.Div(className='path-row', children=[
                dcc.Dropdown(id='ex-file-format', options=[{'label':'Detect format','value':'auto'},{'label':'CTF directory','value':'ctf'},{'label':'Perfetto trace','value':'pftrace'}], value='auto', clearable=False, style={'minWidth':'180px'}),
                html.Button('Open selected trace', id='ex-file-load', className='primary-button'),
                html.Button('Cancel import', id='ex-file-cancel', className='subtle-button')]),
            html.Div(id='ex-file-action-error', role='alert', className='input-error'),
            html.Div(id='ex-file-status', role='status', children='Ready to open a trace.'),
            html.Div(id='ex-file-load-error', role='alert', className='input-error'),
            html.P(f'Browse root: {self.root}. Start with --browse-root PATH to allow a different directory. The last working trace remains usable if an import fails.', className='tiny'),
        ])

    def register_callbacks(self, app):
        @app.callback(Output('ex-file-path','value'), Output('ex-file-entry','options'),
                      Output('ex-file-entry','value'), Output('ex-file-folder','children'),
                      Output('ex-file-list-status','children'),
                      Input('ex-file-browse','n_clicks'), Input('ex-file-up','n_clicks'), Input('ex-file-entry','value'),
                      State('ex-file-path','value'), prevent_initial_call=True)
        def browse(_browse, _up, entry, value):
            try:
                if ctx.triggered_id == 'ex-file-entry':
                    if not entry: raise PreventUpdate
                    value = entry
                path = resolve_trace_path(self.root, value)
                if ctx.triggered_id == 'ex-file-up':
                    path = path.parent if path != self.root else self.root
                options, folder, message = self.listing(str(path))
                return str(path), options, None, folder, message
            except PreventUpdate:
                raise
            except (OSError, ValueError) as exc:
                return no_update, no_update, no_update, no_update, str(exc)

        @app.callback(Output('ex-file-request','data'), Output('ex-file-action-error','children'),
                      Input('ex-file-load','n_clicks'), Input('ex-file-cancel','n_clicks'),
                      State('ex-file-path','value'), State('ex-file-format','value'), State('ex-file-request','data'),
                      prevent_initial_call=True)
        def action(_open, _cancel, value, format, previous):
            try:
                if ctx.triggered_id == 'ex-file-cancel':
                    if previous: self.manager.cancel(DatasetRequestToken(**previous))
                    return no_update, ''
                token = self.open_path(value, format)
                return {'request_id':token.request_id,'dataset_id':token.dataset_id,'generation':token.generation}, ''
            except (OSError, ValueError, RuntimeError) as exc:
                return no_update, str(exc)

        @app.callback(Output('ex-file-status','children'), Output('ex-file-load-error','children'),
                      Output(self.current_dataset_store_id,'data'), Output('ex-file-location','href'),
                      Input('ex-file-poll','n_intervals'), Input('ex-file-request','data'),
                      State(self.current_dataset_store_id,'data'), State('ex-file-landing','data'))
        def poll(_n, token, previous, landing):
            if not token: raise PreventUpdate
            try:
                snapshot = self.manager.status(DatasetRequestToken(**token))
            except RuntimeError as exc:
                return 'Request is no longer active.', str(exc), no_update, no_update
            state = snapshot.state.value
            elapsed = max(0., (snapshot.ended_at or time.monotonic()) - (snapshot.started_at or snapshot.created_at))
            if state == 'ready':
                current = {'request_id':token['request_id'], 'dataset_id':token['dataset_id']}
                return f'Trace ready · {elapsed:.2f} s', '', current if current != previous else no_update, app.config.requests_pathname_prefix if landing else no_update
            message = {'queued':'Queued', 'loading':'Reading and indexing trace', 'cancelling':'Cancelling at the next safe boundary',
                       'cancelled':'Import cancelled', 'superseded':'A newer import replaced this request', 'failed':'Import failed; the previous trace is unchanged'}.get(state,state)
            return f'{message} · {elapsed:.1f} s', snapshot.error or '', no_update, no_update
