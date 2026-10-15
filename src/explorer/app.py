"""Selection-driven Dash interface. No question catalog is mounted or queried."""
from __future__ import annotations
from dataclasses import replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import threading
import time

from dash import Dash, Input, Output, State, ctx, dcc, html, dash_table, no_update
from dash.exceptions import PreventUpdate

from . import charts
from . import diagnostic_ui, overview, metadata, workspace
from .diagnostics import METRICS, metric_options
from .model import ExplorerEngine, Query, human, MAX_HEATMAP_ROWS, MAX_TABLE_ROWS


def token(engine):
    return hashlib.sha256((engine.store.identity.run_id+engine.store.identity.source_fingerprint).encode()).hexdigest()[:24]


def source_label(engine):
    source=engine.store.identity.source_format.lower()
    kind='Synthetic demo' if 'synthetic' in source else 'CTF' if 'ctf' in source else 'pftrace' if 'perfetto' in source or 'pftrace' in source else engine.store.identity.source_format
    return f'{kind} · {len(engine.counters):,} counters'


def initial_state(engine):
    options=metric_options(engine)
    default=options[0]["value"] if options else "traffic"
    return {"dataset":token(engine),"query":Query(engine.origin,engine.end,metric=default).to_dict(),"history":[]}


def _seconds_ns(value):
    try:
        number=Decimal(str(value))
        if not number.is_finite():
            raise ValueError("Time must be finite.")
        return int(number*Decimal(1_000_000_000))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError("Enter a numeric time in seconds.") from None


def normalize_query(engine, query):
    nodes=tuple(v for v in query.node_ids if v in engine.nodes)
    nics=tuple(v for v in query.nic_ids if v in engine.nics and
               (not nodes or engine.nics[v].node_id in nodes))
    start=max(engine.origin,min(engine.end-1,query.start_ns))
    end=min(engine.end,max(start+1,query.end_ns))
    choices=tuple((n,c) for n,c in query.collectors if n in engine.nics)
    available={o["value"] for o in metric_options(engine)}
    metric=query.metric if query.metric in available else next(iter(metric_options(engine)), {"value":"traffic"})["value"]
    return replace(query,node_ids=nodes,nic_ids=nics,start_ns=start,end_ns=end,
                   metric=metric,collectors=choices,mode=query.mode if query.mode in {"rate","volume"} else "rate",
                   direction=query.direction if query.direction in {"TX","RX"} else "TX")


def card(label,value,note="",accent=""):
    return html.Div([html.Span(label,className="metric-label"),html.Strong(value,className="metric-value"),
                     html.Span(note,className="metric-note")],className="metric-card "+accent)


def cards(view):
    if "diagnostic" in view:
        return diagnostic_ui.cards(view)
    if view["query"].metric == "octet_errors":
        observed=sum(r["observed_counter_count"] for r in view["records"])
        zero=sum(r["zero_counter_count"] for r in view["records"])
        nonzero=sum(r["nonzero_counter_count"] for r in view["records"])
        return [card("NO INCREMENTS",f"{zero} / {observed}","Does not imply lifetime zero","error-card"),
                card("COUNTERS WITH INCREMENTS",str(nonzero),"Measured in this window","diagnostic-card"),
                card("SCOPE",f"{view['node_count']} {'node' if view['node_count']==1 else 'nodes'} · {view['nic_count']} {'NIC' if view['nic_count']==1 else 'NICs'}","Selected physical devices"),
                card("TIME WINDOW",f"{view['duration_s']:.4g} s","Observed endpoint intervals only")]
    if view["query"].metric != "traffic":
        ready=[r for r in view["records"] if r["sample_count"]]
        return [card("COUNTER",metadata.describe_counter(view['counter'])['label'],"Source-native observations","counter-card"),
                card("OBSERVED NICS",f"{len(ready)} / {view['nic_count']}","No absent values filled in"),
                card("WINDOW",f"{view['duration_s']:.4g} s","Relative to trace start")]
    tx,rx=view["aggregate"]["TX"],view["aggregate"]["RX"]
    total=view["nic_count"]
    return [card("TRANSMITTED",human(tx["total"]),f"Observed on {tx['usable']} of {total} NICs","tx"),
            card("RECEIVED",human(rx["total"]),f"Observed on {rx['usable']} of {total} NICs","rx"),
            card("SCOPE",f"{view['node_count']} {'node' if view['node_count']==1 else 'nodes'} · {total} {'NIC' if total==1 else 'NICs'}","Selected physical devices"),
            card("TIME WINDOW",f"{view['duration_s']:.4g} s","Observed endpoint intervals only")]


def selection_label(engine,q):
    if len(q.nic_ids)==1:
        nic=engine.nics[q.nic_ids[0]]
        return f"{engine.nodes[nic.node_id].label} / {nic.label}"
    if q.nic_ids:
        return f"{len(q.nic_ids)} selected NICs"
    if len(q.node_ids)==1:
        return f"{engine.nodes[q.node_ids[0]].label} / All NICs"
    return f"{len(q.node_ids)} selected nodes / All NICs" if q.node_ids else "All nodes / All NICs"


def nic_options(engine,nodes):
    return [{"label":f"{engine.nodes[n.node_id].label} / {n.label}","value":n.nic_id}
            for n in engine.store.nics if not nodes or n.node_id in nodes]


def collector_control(engine,q):
    if len(q.nic_ids)!=1:
        return [{"label":"Select one NIC to inspect collectors","value":"auto"}],"auto"
    options=[{"label":"Use unique stream only","value":"auto"},*engine.collector_options(q.nic_ids[0])]
    current=dict(q.collectors).get(q.nic_ids[0])
    return options,str(current) if current is not None else "auto"


def coverage_notice(view):
    if 'diagnostic' in view:
        missing=sum(r['total'] is None for r in view['records'])
        return (view['diagnostic']['note'] or 'Full-resolution derived metric on common valid counter support.') + f" {missing} of {view['nic_count']} NIC values unavailable; absent data is not zero."
    if view['query'].metric == 'octet_errors':
        records=view['records']; ambiguous=sum(r['state']=='Choose collector' for r in records)
        missing=sum(r['observed_counter_count']==0 for r in records)
        messages=["Only sourced TX/RX bad-octet counters are included here. Zero means no measured increment in this window, not a zero lifetime register. Use Reliability event increments for link events, retries, and drops."]
        if ambiguous:
            messages.append("Multiple collectors on some NICs: select a NIC and choose an observation stream in the sidebar.")
        if missing:
            messages.append(f"{missing} of {len(records)} reliability summaries are unavailable and excluded, not treated as zero.")
        if any(r['coverage'] < .999999 for r in records if r['total'] is not None):
            messages.append("Some intervals have partial coverage; gaps are not interpolated.")
        return " ".join(messages)
    if view['query'].metric != 'traffic':
        return "Raw counter observations. No bandwidth, event rate, latency, or selected-window total is inferred from these values."
    records=view['records']; ambiguous=sum(r['state']=='Choose collector' for r in records)
    missing=sum(r['total'] is None for r in records)
    messages=[]
    if ambiguous:
        messages.append("Multiple collectors on some NICs: select a NIC and choose an observation stream in the sidebar.")
    if missing:
        messages.append(f"{missing} of {len(records)} TX/RX series are unavailable and excluded from totals, not treated as zero.")
    if any(r['state']=='Partial' for r in records):
        messages.append("Some intervals have partial coverage; gaps are not interpolated.")
    if not messages:
        messages.append("TX and RX are separate. Rates are interval averages; drag across the timeline to select a shared time window.")
    return " ".join(messages)


def view_notice(engine, view):
    message = coverage_notice(view)
    evidence = engine.store.identity.synchronization_evidence.lower()
    if view['node_count'] > 1 and any(word in evidence for word in ('not','unverified','unknown')):
        message += ' Cross-host clock synchronization is unverified; aligned plots do not prove simultaneity.'
    if 'pftrace' in engine.store.identity.source_format:
        message += ' Change-only pftrace: missing counters are not zero; rates average emitted intervals. Unchanged counter events can be suppressed; this alone does not indicate lost data.'
    return message


def details(engine,view):
    store=engine.store; q=view['query']
    prefix = ([html.H4('Derived measurement'),html.P(view['formula']),html.P(view['method']),
               html.P(view['diagnostic']['note']),html.A('Counter definitions (HPE)',href=view['source_reference'],target='_blank',rel='noopener noreferrer')]
              if 'diagnostic' in view else [])
    return [*prefix,html.Dl(className="provenance",children=[
        html.Dt("Source"),html.Dd(store.identity.source_format),
        html.Dt("Run"),html.Dd(store.identity.run_id),
        html.Dt("Fingerprint"),html.Dd(store.identity.source_fingerprint),
        html.Dt("Clock"),html.Dd(store.identity.clock_domain),
        html.Dt("Synchronization evidence"),html.Dd(store.identity.synchronization_evidence),
        html.Dt("Attribution"),html.Dd("Device-wide NIC counters, not process-exclusive MPI payload. Background device activity can contribute; nonzero traffic does not prove participation in this benchmark."),
        html.Dt("Exact window (ns)"),html.Dd(f"{q.start_ns} – {q.end_ns}"),
        html.Dt("Trace records"),html.Dd(f"{store.row_count:,}"),
        html.Dt("Missing counter names"),html.Dd(", ".join(store.coverage.missing_counter_names) or "None recorded"),
        html.Dt("Display"),html.Dd("240 shared time bins. Constant interval-rate integration, not instantaneous peak reconstruction. Incomplete bins are gaps. Scientific statistics use full-resolution samples."),
        html.Dt("Measurement"),html.Dd("Network-frame octets, not application payload or physical-wire occupancy. Packet observations do not reconstruct message sizes, communication peers, or network topology."),
        html.Dt("Collector policy"),html.Dd("Ambiguous physical observations are excluded until one collector is explicitly selected. This selection is recorded in exports."),
    ]),html.H4("Selection evidence"),html.Ul([html.Li(
        f"{r['node']} / {r['nic']} · {r['direction']} · collector {r['collector_id'] if r['collector_id'] is not None else 'unresolved'}: "
        f"{r['state']}. {r['reason']}" + (f" Flags: {', '.join(r['flags'])}." if r['flags'] else "")
    ) for r in view['records'][:40]]),html.P("Evidence is capped at 40 displayed series. Export contains every selected summary.")]


def table_columns(mode="traffic"):
    if mode == "reliability":
        labels=[("Node","node"),("NIC","nic"),("Zero counters","zero_counter_count"),("Nonzero counters","nonzero_counter_count"),("Coverage","coverage"),("State","state"),("Collector","collector")]
        return [{"name":a,"id":b,**({"type":"numeric","format":{"specifier":".1%" if b=="coverage" else "","nully":"—"}} if b in {"zero_counter_count","nonzero_counter_count","coverage"} else {})} for a,b in labels]
    traffic = mode == "traffic"
    labels = [("Node","node"),("NIC","nic"),("Volume (B)","total"),("Mean (B/s)","mean"),("P99 (B/s)","p99"),("CV","cv"),("Traffic share","share"),("Active time","active"),("Coverage","coverage"),("State","state"),("Collector","collector")] if traffic else [("Node","node"),("NIC","nic"),("Last value (native units)","total"),("Samples","samples"),("State","state"),("Collector","collector")]
    return [{"name":a,"id":b,**({"type":"numeric","format":{"specifier":".1%" if b=='coverage' else '.3s',"nully":"—"}} if b in {'total','mean','p99','coverage','samples','cv','share','active'} else {})} for a,b in labels]


def reliability_table_rows(view):
    return [{
        "id":row["id"],"node":row["node"],"nic":row["nic"],
        "zero_counter_count":row["zero_counter_count"],
        "nonzero_counter_count":row["nonzero_counter_count"],
        "coverage":row["coverage"],"state":row["state"],
        "collector":str(row["collector_id"]) if row["collector_id"] is not None else "—"
    } for row in view["ranked"]]




def create_explorer_app(store=None,*,path="/",dataset_manager=None,performance_recorder=None,trace_browser=False,analysis_config=None):
    """Create the default UI; use the same CTF/pftrace loaders and hosting path."""
    lock=threading.RLock(); holder=[None,None]
    def engine():
        current=dataset_manager.current() if dataset_manager is not None else None
        current_store=current.store if current is not None else store
        if current_store is None:
            return None
        with lock:
            if holder[0] is not current_store:
                holder[:]=[current_store,ExplorerEngine(current_store, analysis_config=analysis_config)]
            return holder[1]
    app=Dash(__name__,url_base_pathname=path,assets_folder=str(Path(__file__).parent/'assets'),title="Aurora NIC Metrics",suppress_callback_exceptions=True)
    app.server.config['AURORA_PERFORMANCE_RECORDER']=performance_recorder
    app.server.config['AURORA_UI']='explorer'
    @app.server.get('/healthz')
    def healthz():
        return {"status":"ok","ui":"explorer"}
    @app.server.get('/readyz')
    def readyz():
        e=engine()
        if e is None: return {'status':'not_ready','ui':'explorer','reason':'Open a trace to begin.'},503
        ready=e.store.row_count>0 and bool(e.store.nics)
        return {"status":"ready" if ready else "not_ready","ui":"explorer",
                "run_id":e.store.identity.run_id,"indexed_rows":e.store.row_count},200 if ready else 503

    @app.server.before_request
    def reject_cross_site_mutations():
        from flask import request, abort
        if request.method == 'POST' and request.headers.get('Sec-Fetch-Site') == 'cross-site':
            abort(403, description='Use the local application origin, not a cross-site request.')

    @app.server.after_request
    def browser_security_headers(response):
        response.headers['X-Frame-Options']='SAMEORIGIN'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        return response

    dataset_controls=None
    if dataset_manager is not None:
        if trace_browser:
            from .trace_browser import TracePathControls
            dataset_controls=TracePathControls(dataset_manager)
            dataset_controls.register_callbacks(app)
        else:
            from ui.dataset_controls import DatasetControls
            dataset_controls=DatasetControls(dataset_manager.dataset_ids,id_prefix="explorer-dataset")
            dataset_controls.register_callbacks(app,dataset_manager)

    def layout():
        e=engine()
        if e is None:
            return html.Main(className='trace-landing', children=[
                html.P('AURORA · NIC EXPLORER', className='eyebrow'),
                html.H1('Start with your trace'),
                html.P('Open a server-side CTF directory or pftrace. Compare NICs, narrow the time window, and inspect the evidence.',className='muted'),
                html.Section(className='panel', children=dataset_controls.render(landing=True) if dataset_controls else 'No file browser is configured.')])
        state=initial_state(e); q=Query.from_dict(state['query']); view=e.query(q)
        opts,col=collector_control(e,q)
        source=source_label(e)
        figure,mapnote,_,_=overview.build_heatmap(view)
        families=sorted({m.family for m in METRICS.values()})
        return html.Div(className='explorer',children=[
            dcc.Store(id='ex-state',data=state),dcc.Store(id='ex-saved',data=[]),dcc.Download(id='ex-download'),
            dcc.Store(id='ex-page',data=0),dcc.Store(id='ex-restore'),
            dcc.Store(id='ex-favorites',data=[],storage_type='local'),
            dcc.Store(id='ex-bookmarks',data=[],storage_type='session'),dcc.Download(id='ex-view-download'),
            html.A('Skip to analysis',href='#ex-main',className='skip-link'),
            html.Aside(className='sidebar',children=[
                html.Div([html.Span('A',className='brand-mark'),html.Div([html.Strong('AURORA'),html.Span('NIC Explorer')])],className='brand'),
                html.Div([html.Span(source,id='ex-source-format',className='source-tag'),html.P(e.store.identity.run_id,id='ex-run-name',title=e.store.identity.run_id)],className='source-card'),
                html.Div([html.Span('DEVICES',className='side-heading'),html.Button('Clear scope',id='ex-reset-scope',className='text-button')],className='side-heading-row'),
                html.Label('Nodes',htmlFor='ex-nodes'),dcc.Dropdown(id='ex-nodes',options=[{'label':n.label,'value':n.node_id} for n in e.store.nodes],value=[],multi=True,placeholder='All nodes',className='side-dropdown'),
                html.Label('NICs',htmlFor='ex-nics'),dcc.Dropdown(id='ex-nics',options=nic_options(e,()),value=[],multi=True,placeholder='All NICs in scope',className='side-dropdown'),
                html.P('Click a heatmap row to drill down. Back restores the previous selection.',className='side-hint'),
                html.Div([html.Label('METRIC',htmlFor='ex-metric',className='side-heading'),html.Button('☆ Favorite',id='ex-favorite',className='text-button',title='Save this metric in your browser favorites')],className='side-heading-row spaced'),
                html.Label('Metric family',htmlFor='ex-family',className='sr-only'),
                dcc.Dropdown(id='ex-family',options=[{'label':'All available metrics','value':'all'},{'label':'Favorites','value':'favorites'},*({'label':f,'value':f} for f in ['Traffic',*families,'Raw counters'])],value='all',clearable=False,className='side-dropdown'),
                dcc.Dropdown(id='ex-metric',options=metric_options(e),value=q.metric,clearable=False,searchable=True,className='side-dropdown',placeholder='Find a metric or raw counter'),
                html.P('Search readable names or hardware IDs. Only encountered counters and available metric inputs are listed.',className='side-hint'),
                html.Button('Definition & calculation',id='ex-definition-link',className='text-button definition-link'),
                html.Details(className='collector-details',children=[html.Summary('Observation source'),
                    html.Label('Collector for selected NIC',htmlFor='ex-collector'),dcc.Dropdown(id='ex-collector',options=opts,value=col,clearable=False),
                    html.P('Duplicate observers are never added. This choice applies to one physical NIC.',className='side-hint')]),
                html.Div([html.Span('TRACE-LOCAL ANALYSIS'),html.P('Data stays on the server. Press ? for interaction help.')],className='sidebar-foot')]),
            html.Main(id='ex-main',className='main',tabIndex=-1,children=[
                html.Header(className='topbar',children=[html.Div([html.P('NETWORK ANALYSIS',className='eyebrow'),html.H1('Traffic explorer',id='ex-title')]),
                    html.Div([html.Button('Help',id='ex-help-button',className='subtle-button'),html.Button('Open trace',id='ex-open',className='subtle-button'),
                        html.Label('Export format',htmlFor='ex-export-format',className='sr-only'),
                        dcc.Dropdown(id='ex-export-format',options=[{'label':'JSON + evidence','value':'json'},{'label':'CSV summaries','value':'csv'}],value='json',clearable=False,className='export-format'),
                        html.Button('Export',id='ex-export',className='primary-button',title='Export the full selected population, not only the visible heatmap page')],className='export-action')]),
                html.Div(className='scope-row',children=[html.Div([html.Button('← Back',id='ex-back',className='subtle-button',disabled=True),html.Strong(selection_label(e,q),id='ex-scope')],className='scope-breadcrumb'),
                    html.Span(id='ex-window-label',children=f"0.000 – {(e.end-e.origin)/1e9:.3f} s")]),
                html.Div(id='ex-cards',className='metrics',children=cards(view)),
                html.Div(id='ex-notice',className='notice',role='status',children=view_notice(e,view)),
                html.Div(id='ex-input-error',className='input-error',role='alert'),
                html.Section(className='panel primary-chart',children=[
                    html.Div(className='panel-heading',children=[html.Div([html.H2('Traffic over time',id='ex-timeline-title'),
                        html.P('Click a row to inspect · drag horizontally to select time · double-click to reset',className='muted')]),
                        dcc.RadioItems(id='ex-view',options=[{'label':'Auto','value':'auto'},{'label':'Heatmap','value':'map'},{'label':'Trends','value':'trends'}],value='auto',inline=True,className='segmented')]),
                    html.Div(className='heat-controls',id='ex-heat-controls',children=[
                        dcc.RadioItems(id='ex-direction',options=[{'label':'TX','value':'TX'},{'label':'RX','value':'RX'}],value='TX',inline=True,className='segmented'),
                        html.Label('Rows',htmlFor='ex-row-group'),dcc.Dropdown(id='ex-row-group',options=[{'label':'Auto grouping','value':'auto'},{'label':'Physical NICs','value':'nic'},{'label':'Nodes','value':'node'}],value='auto',clearable=False),
                        html.Label('Sort',htmlFor='ex-row-sort'),dcc.Dropdown(id='ex-row-sort',options=[{'label':'Node / NIC','value':'identity'},{'label':'Highest value','value':'value'},{'label':'Lowest coverage','value':'coverage'}],value='identity',clearable=False),
                        html.Label('Color',htmlFor='ex-color'),dcc.Dropdown(id='ex-color',options=[{'label':'Linear','value':'linear'},{'label':'Logarithmic','value':'log'}],value='linear',clearable=False),
                        html.Label('Find devices',htmlFor='ex-row-search',className='sr-only'),dcc.Input(id='ex-row-search',type='search',placeholder='Find a node or NIC…',value='',debounce=.35)]),
                    html.Div(id='ex-primary-map',children=[dcc.Loading(type='dot',delay_show=300,children=dcc.Graph(id='ex-heatmap',figure=figure,config=charts.CONFIG)),
                        html.Div(className='map-footer',children=[html.P(mapnote,id='ex-page-note'),html.Button('Previous',id='ex-prev-page',className='subtle-button'),html.Button('Next',id='ex-next-page',className='subtle-button')])]),
                    html.Div(id='ex-trend-wrap',style={'display':'none'},children=[
                        html.Div(dcc.RadioItems(id='ex-mode',options=[{'label':'Rate','value':'rate'},{'label':'Accumulated','value':'volume'}],value='rate',inline=True,className='segmented'),className='trend-options'),
                        dcc.Loading(type='dot',delay_show=300,children=dcc.Graph(id='ex-timeline',figure=charts.timeline(view) if q.metric not in METRICS else diagnostic_ui.timeline(view),config=charts.CONFIG))]),
                    html.Div(className='time-toolbar',children=[html.Label('Window (s)',htmlFor='ex-from'),dcc.Input(id='ex-from',type='number',value=0,min=0,step='any',debounce=True),
                        html.Label('to',htmlFor='ex-to'),dcc.Input(id='ex-to',type='number',value=(e.end-e.origin)/1e9,min=0,step='any',debounce=True),
                        html.Button('Full trace',id='ex-reset-time',className='subtle-button'),html.Span(className='toolbar-spacer'),
                        html.Button('Save as A',id='ex-save-a',className='subtle-button'),html.Button('Save as B',id='ex-save-b',className='subtle-button')])]),
                html.Details(id='ex-definition',className='panel drawer',children=[html.Summary('Metric definition & calculation'),html.Div(id='ex-definition-content',children=metadata.definition_panel(e,q.metric))]),
                html.Section(id='ex-nic-detail',className='panel nic-detail',style={'display':'none'}),
                html.Details(id='ex-population',className='panel drawer',children=[html.Summary('Device table & ranking'),
                    html.Div(id='ex-comparison-heading',className='comparison-heading',children=html.H2('Device comparison')),
                    html.Div(id='ex-comparison-grid',className='comparison-grid',children=[
                        html.Section(className='panel activity-panel',children=[
                            html.Div(className='panel-heading',children=[html.H3('Device table',id='ex-heatmap-title'),dcc.RadioItems(id='ex-display',options=[{'label':'Table','value':'table'}],value='table',inline=True,className='segmented small')]),
                            html.Div(id='ex-map-wrap',style={'display':'none'}),
                            html.Div(id='ex-table-wrap',children=dash_table.DataTable(id='ex-table',data=charts.table_rows(view)[:12],columns=table_columns(),
                                page_current=0,page_size=12,page_action='custom',sort_action='custom',sort_by=[],cell_selectable=True,
                                style_table={'overflowX':'auto'},style_cell={'fontFamily':'Arial, sans-serif','fontSize':12,'padding':'10px','textAlign':'left','border':'none','borderBottom':'1px solid #eef1f5'},
                                style_header={'fontWeight':'bold','backgroundColor':'#f6f8fa','whiteSpace':'normal'})),
                            html.P(id='ex-map-note',className='panel-note')]),
                        html.Section(className='panel',children=[html.Div(className='panel-heading',children=html.H3('Traffic distribution',id='ex-rank-title')),dcc.Graph(id='ex-ranking',figure=charts.ranking(view),config={**charts.CONFIG,'displayModeBar':False}),html.P('Highest observed values. The paged table contains all matching devices. Missing is not zero.',className='panel-note')])])]),
                html.Details(className='panel drawer',children=[html.Summary('Population balance'),html.Section(id='ex-balance',className='balance-panel',children=diagnostic_ui.balance_panel(view,e))]),
                html.Details(id='ex-interval-comparison',className='panel drawer',children=[html.Summary('Compare intervals A / B'),html.Div(id='ex-compare-content')]),
                html.Details(id='ex-workspace',className='panel drawer',children=[html.Summary('Saved views'),
                    html.P('Save the current devices, interval, metric, collector choices and display settings. Views are tied to this trace fingerprint; raw data and server paths are not included.'),
                    html.Div(className='workspace-row',children=[html.Label('Name',htmlFor='ex-view-name'),dcc.Input(id='ex-view-name',value='',placeholder='e.g. Posted pressure during burst',maxLength=80),html.Button('Save view',id='ex-save-view',className='subtle-button'),
                        html.Label('Saved view',htmlFor='ex-bookmark',className='sr-only'),dcc.Dropdown(id='ex-bookmark',options=[],placeholder='Choose saved view',className='bookmark-select'),
                        html.Button('Restore',id='ex-restore-view',className='subtle-button'),html.Button('Delete',id='ex-delete-view',className='subtle-button'),
                        html.Button('Download view',id='ex-download-view',className='subtle-button'),
                        dcc.Upload(id='ex-upload-view',children=html.Button('Import view',className='subtle-button'),multiple=False,accept='.json',max_size=128000)]),
                    html.Div(id='ex-workspace-status',role='status'),html.P('Up to 12 views per browser tab, retained across refreshes. Download a view to keep it after closing the tab.',className='tiny')]),
                html.Details(className='panel drawer',children=[html.Summary('Trace details & measurement notes'),html.Div(id='ex-evidence',children=details(e,view))]),
                html.Details(id='ex-open-trace',className='trace-modal',role='dialog',**{'aria-label':'Open trace','aria-modal':'true'},children=[html.Summary('Open trace'),
                    html.Div(className='modal-inner',children=[html.Div([html.H2('Open trace'),html.Button('Close',id='ex-close-trace',className='subtle-button')],className='modal-heading'),
                        dataset_controls.render() if dataset_controls else html.P('Start with --browse-root PATH to open server-side traces.')])]),
                html.Details(id='ex-help',className='panel drawer',children=[html.Summary('Interaction guide'),
                    html.P('Heatmap: one device or node per row; color shows the selected metric. Auto groups by node above 64 NICs. Click a row to drill down; Back returns to your previous scope and interval.'),
                    html.P('Drag horizontally to select a common interval, or type exact relative seconds. Full trace resets only the interval. Clear scope resets only devices. Search filters displayed rows, not the analysis population or export.'),
                    html.P('Linear colors support magnitude comparison. Log color reveals small signals without changing values or hover units. Gray means no supported value, never measured zero. Scale is shared across pages but recomputed when the metric, interval or row filter changes.'),
                    html.P('Keys outside text fields: / finds a metric; F finds a device; B goes back; R resets time; O opens a trace; ? opens this guide. Escape closes the trace picker. Table navigation is an alternative to chart clicking.'),
                    html.P('Save A/B for temporal comparisons. Save view preserves an investigation; JSON exports include formulas and provenance, CSV contains full-selection summaries.')]),
                html.Div(id='ex-query-status',className='query-status',role='status')])])
    app.layout=layout

    event_inputs=[Input("ex-nodes","value"),Input("ex-nics","value"),Input("ex-from","value"),Input("ex-to","value"),
        Input("ex-reset-scope","n_clicks"),Input("ex-reset-time","n_clicks"),Input("ex-timeline","selectedData"),Input("ex-timeline","relayoutData"),
        Input("ex-heatmap","clickData"),Input("ex-ranking","clickData"),Input("ex-table","active_cell"),
        Input("ex-metric","value"),Input("ex-mode","value"),Input("ex-direction","value"),Input("ex-collector","value"),
        Input("ex-back","n_clicks"),Input("ex-restore","data"),Input("ex-heatmap","selectedData"),Input("ex-heatmap","relayoutData")]
    if dataset_controls is not None:
        event_inputs.append(Input(dataset_controls.current_dataset_store_id,"data"))

    @app.callback(Output("ex-state","data"),Output("ex-nodes","value"),Output("ex-nodes","options"),
        Output("ex-nics","value"),Output("ex-nics","options"),Output("ex-from","value"),Output("ex-to","value"),
        Output("ex-collector","options"),Output("ex-collector","value"),Output("ex-input-error","children"),
        Output("ex-metric","value"),Output("ex-mode","value"),Output("ex-direction","value"),
        *event_inputs,State("ex-state","data"),State("ex-table","derived_viewport_data"),State("ex-heatmap","figure"),prevent_initial_call=True)
    def navigate(*args):
        nodes,nics,start_s,end_s,_,_,selected,relayout,heat,rank,cell,metric,mode,direction,collector=args[:15]
        back,restored,map_selected,map_relayout=args[15:19]
        state,table,map_figure=args[-3:]; e=engine(); trigger=ctx.triggered_id
        if e is None or not state: raise PreventUpdate
        q=Query.from_dict(state['query']); original=q
        error=""; dataset_changed=state.get('dataset')!=token(e)
        if not dataset_changed:
            # Read the current controls together. Never write a stale server value
            # back into a control the user has already changed again.
            q=replace(q,node_ids=tuple(nodes or ()),nic_ids=tuple(nics or ()),
                      metric=metric or 'traffic',mode=mode or 'rate',direction=direction or 'TX')
            try:
                current_start=e.origin+_seconds_ns(start_s); current_end=e.origin+_seconds_ns(end_s)
                if e.origin<=current_start<current_end<=e.end:
                    q=replace(q,start_ns=current_start,end_ns=current_end)
            except ValueError:
                pass
        if dataset_changed:
            q=Query.from_dict(initial_state(e)['query'])
        elif dataset_controls is not None and trigger==dataset_controls.current_dataset_store_id:
            raise PreventUpdate
        else:
            if trigger=="ex-back":
                history=state.get('history',[])
                if not history: raise PreventUpdate
                q=Query.from_dict(history[-1])
            elif trigger=="ex-restore":
                if not restored or restored.get('dataset')!=token(e): raise PreventUpdate
                q=Query.from_dict(restored['query'])
            elif trigger=="ex-nodes":
                q=replace(q,node_ids=tuple(nodes or ()))
            elif trigger=="ex-nics":
                q=replace(q,nic_ids=tuple(nics or ()))
            elif trigger=="ex-reset-scope":
                q=replace(q,node_ids=(),nic_ids=())
            elif trigger=="ex-reset-time":
                q=replace(q,start_ns=e.origin,end_ns=e.end)
            elif trigger in {"ex-from","ex-to"}:
                try:
                    start=e.origin+_seconds_ns(start_s); end=e.origin+_seconds_ns(end_s)
                    if not e.origin<=start<end<=e.end:
                        raise ValueError("Use a non-empty time range inside this trace.")
                    q=replace(q,start_ns=start,end_ns=end)
                except ValueError as exc:
                    error=str(exc)
            elif trigger=="ex-timeline" or (trigger=="ex-heatmap" and not ctx.triggered[0]['prop_id'].endswith('clickData')):
                if trigger=='ex-heatmap': selected,relayout=map_selected,map_relayout
                prop=ctx.triggered[0]['prop_id']
                axis=None
                if prop.endswith('selectedData') and selected and 'range' in selected:
                    axis=selected['range'].get('x')
                elif prop.endswith('relayoutData') and relayout:
                    if relayout.get('xaxis.autorange'):
                        q=replace(q,start_ns=e.origin,end_ns=e.end)
                    elif 'xaxis.range' in relayout:
                        axis=relayout['xaxis.range']
                    elif 'xaxis.range[0]' in relayout:
                        axis=[relayout['xaxis.range[0]'],relayout['xaxis.range[1]']]
                if axis and len(axis)==2:
                    a,b=sorted(e.origin+_seconds_ns(v) for v in axis)
                    if a<b:
                        q=replace(q,start_ns=a,end_ns=b)
            elif trigger in {"ex-heatmap","ex-ranking","ex-table"}:
                nic_id=None
                event=heat if trigger=='ex-heatmap' else rank
                if trigger=='ex-table' and cell:
                    nic_id=cell.get('row_id')
                    if nic_id is None and table and cell.get('row',len(table))<len(table):
                        nic_id=table[cell['row']]['id']
                elif trigger=='ex-heatmap':
                    target=overview.target_from_click(event,map_figure)
                    if target:
                        kind,identity=target.split(':');identity=int(identity)
                        if kind=='node' and identity in e.nodes:q=replace(q,node_ids=(identity,),nic_ids=())
                        elif kind=='nic':nic_id=identity
                elif event and event.get('points'):
                    nic_id=event['points'][0].get('customdata')
                if nic_id is not None and int(nic_id) in e.nics:
                    n=e.nics[int(nic_id)]
                    q=replace(q,node_ids=(n.node_id,),nic_ids=(n.nic_id,))
            elif trigger=="ex-metric":
                q=replace(q,metric=metric or 'traffic')
            elif trigger=="ex-mode":
                q=replace(q,mode=mode)
            elif trigger=="ex-direction":
                q=replace(q,direction=direction)
            elif trigger=="ex-collector" and len(q.nic_ids)==1:
                choices=dict(q.collectors)
                if collector=='auto':
                    choices.pop(q.nic_ids[0],None)
                elif collector is not None:
                    allowed={v['value'] for v in e.collector_options(q.nic_ids[0])}
                    if str(collector) not in allowed:
                        raise PreventUpdate
                    choices[q.nic_ids[0]]=int(collector)
                q=replace(q,collectors=tuple(sorted(choices.items())))
        q=normalize_query(e,q)
        opts,col=collector_control(e,q)
        programmatic_scope=dataset_changed or trigger in {'ex-reset-scope','ex-heatmap','ex-ranking','ex-table','ex-back','ex-restore'}
        scope_changed=programmatic_scope or trigger in {'ex-nodes','ex-nics'}
        programmatic_time=dataset_changed or trigger in {'ex-reset-time','ex-timeline','ex-heatmap','ex-back','ex-restore'}
        normalized_nics=list(q.nic_ids)!=list(nics or [])
        if q==original and not error and not dataset_changed and not normalized_nics:
            # Plotly autosize/selection-clear events must not trigger render loops.
            raise PreventUpdate
        newstate=workspace.push_history(state,q,token(e))
        if trigger=='ex-back':newstate['history']=state.get('history',[])[:-1]
        return (newstate if q!=original or dataset_changed else no_update,
            list(q.node_ids) if programmatic_scope else no_update,
            [{"label":n.label,"value":n.node_id} for n in e.store.nodes] if dataset_changed else no_update,
            list(q.nic_ids) if programmatic_scope or normalized_nics else no_update,
            nic_options(e,q.node_ids) if scope_changed else no_update,
            (q.start_ns-e.origin)/1e9 if programmatic_time else no_update,
            (q.end_ns-e.origin)/1e9 if programmatic_time else no_update,
            opts if scope_changed else no_update,col if scope_changed else no_update,error,
            q.metric if dataset_changed or trigger in {'ex-back','ex-restore'} else no_update,
            q.mode if dataset_changed or trigger in {'ex-back','ex-restore'} else no_update,
            q.direction if dataset_changed or trigger in {'ex-back','ex-restore'} else no_update)


    @app.callback(Output("ex-cards","children"),Output("ex-scope","children"),Output("ex-window-label","children"),
        Output("ex-notice","children"),Output("ex-timeline","figure"),Output("ex-heatmap","figure"),
        Output("ex-ranking","figure"),Output("ex-table","data"),Output("ex-evidence","children"),
        Output("ex-title","children"),Output("ex-timeline-title","children"),Output("ex-heatmap-title","children"),
        Output("ex-rank-title","children"),Output("ex-map-note","children"),Output("ex-query-status","children"),
        Output("ex-run-name","children"),Output("ex-mode","style"),Output("ex-direction","style"),
        Output("ex-table","columns"),Output("ex-display","style"),Output("ex-nic-detail","children"),
        Output("ex-nic-detail","style"),Output("ex-comparison-grid","style"),Output("ex-comparison-heading","style"),
        Output("ex-source-format","children"),Output("ex-balance","children"),Output("ex-balance","style"),
        Output('ex-page-note','children'),Output('ex-prev-page','disabled'),Output('ex-next-page','disabled'),
        Output('ex-definition-content','children'),Output('ex-table','page_count'),
        Input("ex-state","data"),Input('ex-row-group','value'),Input('ex-row-sort','value'),Input('ex-color','value'),
        Input('ex-row-search','value'),Input('ex-page','data'),Input('ex-table','page_current'),Input('ex-table','page_size'),Input('ex-table','sort_by'))
    def render(state,grouping='auto',order='identity',color='linear',search='',page=0,table_page=0,table_size=12,table_sort=None):
        e=engine()
        if e is None or state['dataset']!=token(e):
            raise PreventUpdate
        started=time.perf_counter(); q=normalize_query(e,Query.from_dict(state['query'])); view=e.query(q)
        traffic=q.metric=='traffic'; reliability_metric=q.metric=='octet_errors'; diagnostic=q.metric in METRICS
        one_nic = (traffic or reliability_metric or diagnostic) and view['nic_count']==1
        stats=[]
        if diagnostic:
            spec=view['diagnostic']
            figures=(diagnostic_ui.timeline(view),charts.empty(charts.base(height=305),'Select a NIC for component counters.'),diagnostic_ui.ranking(view))
            rows=diagnostic_ui.table_rows(view); columns=diagnostic_ui.table_columns(view)
            title=spec['family']+' analysis'; plot_title=spec['label']; map_title='Device measurements'; rank_title=spec['unit']+' by NIC'
            note='Same physical devices and time window. Ratios use common-support integrated counts. Missing and zero-denominator values remain undefined. Timeline shows up to 12 NICs.'
            if one_nic:stats=diagnostic_ui.stats(view)
        elif reliability_metric:
            figures=(charts.reliability(view),charts.empty(charts.base(height=305),'Bad-octet measurements are in the table.'),charts.empty(charts.base(height=315),'Select a NIC for counter details.'))
            rows=reliability_table_rows(view);columns=table_columns('reliability')
            title='Error explorer';plot_title='Bad octets by device';map_title='Bad-octet detail';rank_title='Observed bad octets'
            note='Bad-octet totals use complete intervals; use Reliability event increments for retries, link counters, and drops.'
            if one_nic:
                stats=[html.Div(className='panel-heading',children=html.H2('Bad-octet statistics')),
                    html.Table([html.Thead(html.Tr([html.Th(v) for v in ['Counter','Total','Rate','Coverage','Samples','State']])),
                      html.Tbody([html.Tr([html.Td(r['counter']),html.Td(human(r['total'])),html.Td(human(r['rate_per_second'],f"{r['unit']}/s")),html.Td(f"{r['coverage']:.1%}"),html.Td(r['sample_count']),html.Td(r['state'])]) for r in view['records'][0]['counter_results']])],className='compare-table'),
                    html.P(view['records'][0]['reason'],className='panel-note')]
        else:
            figures=(charts.timeline(view),charts.heatmap(view),charts.ranking(view));rows=charts.table_rows(view);columns=table_columns('traffic' if traffic else 'counter')
            title='Traffic explorer' if traffic else 'Counter explorer';plot_title='Traffic over time' if traffic else metadata.describe_counter(q.metric,e.counters.get(q.metric))['label']
            map_title=f'NIC activity · {q.direction}' if traffic else 'Source observations';rank_title=f'{q.direction} traffic distribution' if traffic else 'Last observed value'
            note=f'{q.direction} mean rate · top {min(MAX_HEATMAP_ROWS,view["nic_count"])} of {view["nic_count"]} NICs. Gaps are unavailable data, not zero.' if traffic else 'Raw observations on up to 12 NICs. Absolute counter values are not selected-window totals.'
            if one_nic:
                stats=[html.Div(className='panel-heading',children=html.H2('NIC traffic statistics')),
                    html.Table([html.Thead(html.Tr([html.Th(v) for v in ['Direction','Observed bytes','Mean rate','Median rate','P99 rate','CV','Active time','Coverage','State']])),
                      html.Tbody([html.Tr([html.Td(r['direction']),html.Td(human(r['total'])),html.Td(human(r['mean'],'B/s')),html.Td(human(r['p50'],'B/s')),html.Td(human(r['p99'],'B/s')),html.Td(f"{r['cv']:.3f}" if r['cv'] is not None else '—'),html.Td(f"{r['active_fraction']:.1%}" if r.get('active_fraction') is not None else '—'),html.Td(f"{r['coverage']:.1%}"),html.Td(r['state'])]) for r in view['records']])],className='compare-table'),
                    html.P('P50/P99 are duration-weighted rate statistics, not latency. Active time uses nonzero observed interval deltas; unobserved time is excluded. Native-cadence percentiles and idle time are unavailable for change-only pftrace.',className='panel-note')]
        heatmap,map_note,first_page,last_page=overview.build_heatmap(view,grouping=grouping,order=order,color=color,search=search,page=page)
        figures=(figures[0],heatmap,figures[2])
        rows,page_count=workspace.page_table(rows,table_page,table_size,table_sort,search)
        note='Table is paged across the selected population; device search filters displayed rows only. Exports include the full selection. Missing is not zero.'
        notice=view_notice(e,view)
        elapsed=(time.perf_counter()-started)*1000
        source=source_label(e)
        return (cards(view),selection_label(e,q),f'{(q.start_ns-e.origin)/1e9:.4f} – {(q.end_ns-e.origin)/1e9:.4f} s',
            notice,*figures,rows,details(e,view),title,plot_title,map_title,rank_title,note,
            f'Query + figures: {elapsed:.0f} ms · {e.store.row_count:,} source records remain server-side · SI units',
            e.store.identity.run_id,{} if traffic else {'display':'none'},{} if traffic else {'display':'none'},
            columns,{} if traffic else {'display':'none'},stats,{} if one_nic else {'display':'none'},
            {'display':'none'} if one_nic else {},{'display':'none'} if one_nic else {},source,
            diagnostic_ui.balance_panel(view,e),{} if traffic or (diagnostic and e.analysis_config.get('topology')) else {'display':'none'},
            map_note,first_page,last_page,metadata.definition_panel(e,q.metric),page_count)

    @app.callback(Output("ex-map-wrap","style"),Output("ex-table-wrap","style"),Input("ex-display","value"),Input("ex-state","data"))
    def change_display(value,state):
        if state['query']['metric'] != 'traffic':
            value='table'
        return ({'display':'none'} if value=='table' else {},{} if value=='table' else {'display':'none'})

    @app.callback(Output("ex-download","data"),Input("ex-export","n_clicks"),State("ex-state","data"),State("ex-saved","data"),State('ex-export-format','value'),prevent_initial_call=True)
    def export(clicks,state,saved=None,format='json'):
        if not clicks:
            raise PreventUpdate
        e=engine()
        if state.get('dataset')!=token(e):
            raise PreventUpdate
        query=Query.from_dict(state['query'])
        doc=e.export(query)
        doc['metric_definition']=metadata.definition(e,query.metric)
        if format=='csv':return dcc.send_string(workspace.csv_export(doc),'nic-explorer-selection.csv',type='text/csv')
        doc['saved_intervals']=[]
        if query.metric=='traffic' or query.metric in METRICS:
            for interval in (saved or []):
                if interval.get('dataset') != state['dataset']:continue
                selection=replace(query,start_ns=int(interval['start_ns']),end_ns=int(interval['end_ns']))
                comparison=e.query(selection)
                doc['saved_intervals'].append({'label':interval['label'],'selection':selection.to_dict(),
                    'values':[{'node':r['node'],'nic':r['nic'],'collector_id':r['collector_id'],
                        'value':r['mean'],'unit':'B/s' if query.metric=='traffic' else METRICS[query.metric].unit,
                        'coverage':r['coverage']} for r in comparison['ranked']]})
        return dcc.send_string(json.dumps(doc,indent=2,allow_nan=False),"nic-explorer-selection.json")

    @app.callback(Output("ex-saved","data"),Input("ex-save-a","n_clicks"),Input("ex-save-b","n_clicks"),State("ex-state","data"),State("ex-saved","data"),prevent_initial_call=True)
    def save_interval(a,b,state,saved):
        label='A' if ctx.triggered_id=='ex-save-a' else 'B'
        saved=[s for s in (saved or []) if s.get('dataset')==state['dataset'] and s['label']!=label]
        q=state['query']
        return sorted([*saved,{'label':label,'dataset':state['dataset'],'start_ns':q['start_ns'],'end_ns':q['end_ns']}],key=lambda s:s['label'])

    @app.callback(Output("ex-compare-content","children"),Input("ex-saved","data"),Input("ex-state","data"))
    def compare_intervals(saved,state):
        saved=[s for s in (saved or []) if s['dataset']==state['dataset']]
        if not saved:
            return "Save two timeline selections with ‘Save as A’ and ‘Save as B’. Comparisons use the current device scope and collector choices."
        e=engine()
        if state['dataset']!=token(e):
            raise PreventUpdate
        q=Query.from_dict(state['query'])
        if q.metric!='traffic' and q.metric not in METRICS:
            return 'Choose traffic or a derived metric for interval comparison. Raw cumulative observations are not window totals.'
        windows={s['label']:e.query(replace(q,start_ns=int(s['start_ns']),end_ns=int(s['end_ns']))) for s in saved}
        captions=[html.Span(f"{label}: {(v['query'].start_ns-e.origin)/1e9:.4f}–{(v['query'].end_ns-e.origin)/1e9:.4f} s",className='interval-chip') for label,v in windows.items()]
        if len(windows)<2:
            return [html.Div(captions),html.P("Save the other interval to compare mean rates.")]
        a={r['nic_id']:r for r in windows['A']['ranked']}; b={r['nic_id']:r for r in windows['B']['ranked']}
        rows=[]
        for nic in list(a)[:MAX_TABLE_ROWS]:
            ar,br=a[nic],b[nic]; av,bv=ar['mean'],br['mean']
            delta=f"{(bv/av-1)*100:+.1f}%" if av is not None and av>0 and bv is not None else '—'
            unit='B/s' if q.metric=='traffic' else METRICS[q.metric].unit
            rows.append(html.Tr([html.Td(f"{ar['node']} / {ar['nic']}"),html.Td(human(av,unit)),html.Td(human(bv,unit)),html.Td(delta),html.Td(f"{ar['coverage']:.1%} / {br['coverage']:.1%}")]))
        return [html.Div(captions),html.P("Same devices and collector choices. Coverage A / B is shown; different or partial coverage limits comparability. Derived ratios are recomputed from each window, not averaged."),
                html.Table([html.Thead(html.Tr([html.Th(v) for v in ['NIC','A mean','B mean','Change','Coverage A / B']])),html.Tbody(rows)],className='compare-table')]

    app.clientside_callback(
        """function(n,c){var id=window.dash_clientside.callback_context.triggered_id;
        var open=id==='ex-open';setTimeout(function(){
          var modal=document.getElementById('ex-open-trace'),active=document.activeElement;
          if(!modal)return;
          if(open&&modal.open&&!modal.contains(active)){var path=document.getElementById('ex-file-path');if(path)path.focus();}
          if(!open&&!modal.open&&(!active||active===document.body||modal.contains(active))){var button=document.getElementById('ex-open');if(button)button.focus();}
        },60);return open;}""",
        Output('ex-open-trace','open'),Input('ex-open','n_clicks'),Input('ex-close-trace','n_clicks'),prevent_initial_call=True)
    app.clientside_callback(
        """function(n){if(!n)return window.dash_clientside.no_update;setTimeout(function(){if(document.activeElement===document.getElementById('ex-definition-link'))document.getElementById('ex-definition').scrollIntoView({behavior:'instant',block:'start'});},70);return true;}""",
        Output('ex-definition','open'),Input('ex-definition-link','n_clicks'),prevent_initial_call=True)
    app.clientside_callback(
        """function(n){if(!n)return window.dash_clientside.no_update;setTimeout(function(){var a=document.activeElement;if(a===document.getElementById('ex-help-button')||a===document.getElementById('ex-main'))document.getElementById('ex-help').scrollIntoView({behavior:'instant',block:'start'});},70);return true;}""",
        Output('ex-help','open'),Input('ex-help-button','n_clicks'),prevent_initial_call=True)
    # Dash/Plotly may switch SVG height to 100% during a width-only resize.
    # An auto-height Loading wrapper then collapses to zero. Keep a definite
    # graph-container height from the bounded figure, while width stays fluid.
    for graph_id in ('ex-heatmap','ex-timeline','ex-ranking'):
        app.clientside_callback(
            "function(f){var h=f&&f.layout&&f.layout.height;return {height:Math.max(260,Math.min(1000,Number(h)||450))+'px'};}",
            Output(graph_id,'style'),Input(graph_id,'figure'))
    register_workspace_callbacks(app,engine)

    # Testable handles, no global mutable selection and no source arrays sent to Dash stores.
    app.explorer_engine=engine
    return app


def register_workspace_callbacks(app,engine):
    """Display state is separate from numerical Query and never alters raw data."""
    @app.callback(Output('ex-page','data'),Input('ex-prev-page','n_clicks'),Input('ex-next-page','n_clicks'),
        Input('ex-state','data'),Input('ex-row-group','value'),Input('ex-row-sort','value'),Input('ex-row-search','value'),
        State('ex-page','data'),State('ex-heatmap','figure'),prevent_initial_call=True)
    def page(_prev,_next,_state,_group,_sort,_search,current,figure):
        pages=(figure or {}).get('layout',{}).get('meta',{}).get('pages',1)
        if ctx.triggered_id=='ex-prev-page':return max(0,int(current or 0)-1)
        if ctx.triggered_id=='ex-next-page':return min(pages-1,int(current or 0)+1)
        return 0

    @app.callback(Output('ex-primary-map','style'),Output('ex-trend-wrap','style'),Output('ex-heat-controls','style'),
                  Output('ex-back','disabled'),Input('ex-view','value'),Input('ex-state','data'))
    def presentation(display,state):
        e=engine()
        if e is None or not state or state.get('dataset')!=token(e):raise PreventUpdate
        q=Query.from_dict(state['query'])
        count=len([n for n in e.store.nics if (not q.node_ids or n.node_id in q.node_ids) and (not q.nic_ids or n.nic_id in q.nic_ids)])
        use_map=q.metric!='octet_errors' and (display=='map' or display=='auto' and count>1)
        return ({} if use_map else {'display':'none'}, {'display':'none'} if use_map else {},
                {} if use_map else {'display':'none'},not bool(state.get('history')))

    @app.callback(Output('ex-favorites','data'),Input('ex-favorite','n_clicks'),State('ex-state','data'),State('ex-favorites','data'),prevent_initial_call=True)
    def favorite(clicks,state,favorites):
        favorites=list(favorites or []);key=state['query']['metric']
        if key in favorites:favorites.remove(key)
        else:favorites=(favorites+[key])[-100:]
        return favorites

    @app.callback(Output('ex-metric','options'),Output('ex-favorite','children'),Output('ex-family','options'),Output('ex-family','value'),Input('ex-state','data'),Input('ex-family','value'),Input('ex-favorites','data'))
    def options(state,family,favorites):
        e=engine()
        if e is None or not state or state.get('dataset')!=token(e):raise PreventUpdate
        current=state['query']['metric'];all_options=metadata.available_options(e,favorites)
        available_families=sorted({m.family for m in METRICS.values() if set(m.numerator+m.denominator)<=set(e.counters)})
        families=['all','favorites',*(['Traffic'] if any(o['value']=='traffic' for o in all_options) else []),*available_families,'Raw counters']
        normalized_family=family if family in families else 'all'
        family_options=[{'label': 'All available metrics' if f=='all' else 'Favorites' if f=='favorites' else f,'value':f} for f in families]
        filtered=metadata.available_options(e,favorites,normalized_family)
        # Retain the active, encountered metric so a category change cannot
        # silently clear it or change the numerical selection.
        if current not in {o['value'] for o in filtered}:
            filtered=[*[o for o in all_options if o['value']==current],*filtered]
        return filtered,('★ Favorited' if current in (favorites or []) else '☆ Favorite'),family_options,(normalized_family if normalized_family!=family else no_update)

    @app.callback(Output('ex-bookmarks','data'),Output('ex-workspace-status','children'),Output('ex-restore','data'),Output('ex-view-download','data'),Output('ex-upload-view','contents'),
        Input('ex-save-view','n_clicks'),Input('ex-restore-view','n_clicks'),Input('ex-delete-view','n_clicks'),Input('ex-download-view','n_clicks'),Input('ex-upload-view','contents'),
        State('ex-state','data'),State('ex-bookmarks','data'),State('ex-bookmark','value'),State('ex-view-name','value'),
        State('ex-view','value'),State('ex-row-group','value'),State('ex-row-sort','value'),State('ex-color','value'),State('ex-row-search','value'),prevent_initial_call=True)
    def views(_save,_restore,_delete,_download,upload,state,bookmarks,chosen,name,display,group,order,color,search):
        bookmarks=list(bookmarks or []);trigger=ctx.triggered_id
        settings={'display':display,'grouping':group,'order':order,'color':color,'search':search}
        active=workspace.snapshot(state,name or f'View {len(bookmarks)+1}',settings)
        if trigger=='ex-upload-view':
            if not upload:raise PreventUpdate
            try:restored=workspace.decode_view(upload,state['dataset'])
            except ValueError as exc:return no_update,str(exc),no_update,no_update,None
            restored['nonce']=time.time_ns()
            return no_update,'Restored imported view: '+restored['name'],restored,no_update,None
        if trigger=='ex-save-view':
            bookmarks=[b for b in bookmarks if not (b['dataset']==state['dataset'] and b['name']==active['name'])]
            bookmarks=(bookmarks+[active])[-workspace.MAX_VIEWS:]
            return bookmarks,'Saved '+active['name']+' in this browser tab.',no_update,no_update,no_update
        if trigger=='ex-download-view':
            return no_update,'Downloaded the current view; no raw data or server paths are included.',no_update,dcc.send_string(json.dumps(active,indent=2),'nic-explorer-view.json'),no_update
        matches=[b for b in bookmarks if b['dataset']==state['dataset'] and b['name']==chosen]
        if not matches:return no_update,'Choose a saved view for this trace first.',no_update,no_update,no_update
        selected=dict(matches[0])
        if trigger=='ex-delete-view':
            return [b for b in bookmarks if b is not matches[0]],'Deleted '+selected['name'],no_update,no_update,no_update
        selected['nonce']=time.time_ns()
        return no_update,'Restored '+selected['name'],selected,no_update,no_update

    @app.callback(Output('ex-bookmark','options'),Input('ex-bookmarks','data'),Input('ex-state','data'))
    def bookmark_options(bookmarks,state):
        return [{'label':b['name'],'value':b['name']} for b in (bookmarks or []) if b.get('dataset')==state.get('dataset')]

    @app.callback(Output('ex-view','value'),Output('ex-row-group','value'),Output('ex-row-sort','value'),Output('ex-color','value'),Output('ex-row-search','value'),
                  Input('ex-restore','data'),prevent_initial_call=True)
    def restore_settings(data):
        if not data or data.get('dataset')!=token(engine()):raise PreventUpdate
        s=data.get('settings',{})
        def choice(key,choices,default):return s.get(key) if s.get(key) in choices else default
        return (choice('display',{'auto','map','trends'},'auto'),choice('grouping',{'auto','nic','node'},'auto'),
                choice('order',{'identity','value','coverage'},'identity'),choice('color',{'linear','log'},'linear'),str(s.get('search',''))[:200])

    app.clientside_callback(
        "function(status){var el=document.querySelector('#ex-upload-view input[type=file]');if(el)el.value='';return 'view-upload';}",
        Output('ex-upload-view','className'),Input('ex-workspace-status','children'))
