"""Focused diagnostic views. Requirements remain in documentation, not navigation."""
import math
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc
from . import charts
from .model import human


def value_text(value,unit):
    if value is None:return '—'
    return f'{value:,.3g} {unit}'


def cards(view):
    from .app import card
    spec=view['diagnostic']; rows=[r for r in view['records'] if r['total'] is not None]
    top=max(rows,key=lambda r:r['total']) if rows else None
    label='P99 BUCKET' if spec['mode']=='histogram' else 'SELECTED NIC' if view['nic_count']==1 else 'HIGHEST NIC'
    return [card(label,value_text(top['total'],spec['unit']) if top else '—',f"{top['node']} / {top['nic']}" if top else 'No supported observation', 'diagnostic-card'),
            card('OBSERVED NICS',f"{len(rows)} / {view['nic_count']}",'Missing is not zero'),
            card('MINIMUM COVERAGE',f"{min(r['coverage'] for r in rows):.1%}" if rows else '—','Common support of contributing counters'),
            card('TIME WINDOW',f"{view['duration_s']:.4g} s",'Shared device and time selection')]


def timeline(view):
    fig=charts.base(height=278);q=view['query'];spec=view['diagnostic']
    rows=[r for r in view['ranked'] if r['raw_x']][:12]
    for r in rows:
        fig.add_trace(go.Scatter(x=r['raw_x'],y=r['raw_y'],mode='lines',connectgaps=False,
            name=f"{r['node']} / {r['nic']}",line={'width':2,'shape':'hv'},
            hovertemplate='%{x:.4f} s<br>%{y:,.4g} '+spec['unit']+'<extra>%{fullData.name}</extra>'))
    if not rows or not any(any(y is not None for y in r['raw_y']) for r in rows):charts.empty(fig,'No supported intervals in this selection.<br>Inspect missing counters below or widen the time window.')
    fig.update_yaxes(title_text=spec['unit']);fig.update_xaxes(title_text='Time from trace start (s)',range=[(q.start_ns-view['origin_ns'])/1e9,(q.end_ns-view['origin_ns'])/1e9])
    fig.update_layout(dragmode='select',selectdirection='h',uirevision=f'{q.metric}/{q.start_ns}/{q.end_ns}/{q.nic_ids}')
    return fig


def ranking(view):
    rows=[r for r in view['ranked'] if r['total'] is not None][:24];fig=charts.base(height=315)
    if not rows:return charts.empty(fig,'No supported values to rank; unavailable is not zero.')
    fig.add_trace(go.Bar(x=[r['total'] for r in rows],y=[f"{r['node']} / {r['nic']}" for r in rows],orientation='h',customdata=[r['nic_id'] for r in rows],marker_color=charts.TX_COLOR,
        hovertemplate='%{y}<br>%{x:,.4g} '+view['diagnostic']['unit']+'<extra></extra>'))
    fig.update_yaxes(autorange='reversed',rangemode='normal',tickfont_size=10);fig.update_xaxes(title_text=view['diagnostic']['unit']);fig.update_layout(margin_l=170,hovermode='closest')
    return fig


def table_rows(view):
    return [{'id':r['id'],'nic_id':r['nic_id'],'node':r['node'],'nic':r['nic'],'value':r['total'],'coverage':r['coverage'],'state':r['state'],'collector':str(r['collector_id']) if r['collector_id'] is not None else '—'} for r in view['ranked']]


def table_columns(view):
    unit=view['diagnostic']['unit']
    return [{'name':name,'id':key,**({'type':'numeric','format':{'specifier':'.1%' if key=='coverage' else '.4g','nully':'—'}} if key in {'value','coverage'} else {})} for name,key in [('Node','node'),('NIC','nic'),(unit,'value'),('Coverage','coverage'),('State','state'),('Collector','collector')]]


def distribution(view,row):
    fig=charts.base(height=270);items=row.get('distribution',[])
    if not items:return charts.empty(fig,'This metric has no histogram or categorical distribution.')
    histogram=view['diagnostic']['mode']=='histogram'
    fig.add_trace(go.Bar(x=[x['label'] for x in items],y=[x['delta'] for x in items],marker_color=charts.TX_COLOR,
        hovertemplate='%{x}<br>%{y:,.5g} increments<extra></extra>'))
    fig.update_layout(showlegend=False,margin_b=85)
    fig.update_yaxes(title_text='Aligned bin increments' if histogram else 'Counter increments (see support notes)')
    if not histogram:fig.update_xaxes(tickangle=-25)
    return fig


def stats(view):
    row=view['records'][0];spec=view['diagnostic']
    children=[html.Div(className='panel-heading',children=[html.H2(spec['label']),html.Span(row['state'],className='muted')]),html.P(row['reason'],className='panel-note')]
    if spec['mode']=='histogram':
        latency=row.get('latency',{})
        if latency.get('calibrated'):
            hi=latency['p99_upper_ns'];mean=latency.get('mean_midpoint_estimate_ns')
            text=f"P99 lies in [{latency['p99_lower_ns']:g}, {hi if hi is not None else 'unbounded'}) ns. "
            text+=f"Mean midpoint estimate: {mean:.4g} ns." if mean is not None else 'Mean is unbounded by the open tail bin.'
            children.append(html.P(text))
        else:
            if latency.get('calibration_warning'):children.append(html.P(latency['calibration_warning']))
            children.append(html.P(f"P99 bucket: {latency.get('p99_bin','unavailable')}. Bin positions are not nanoseconds. Supply --analysis-config with capture-specific bin edges to obtain latency bounds."))
    elif row.get('denominator') is not None:
        children.append(html.P(f"Aligned numerator: {row['numerator']:,.6g} · denominator: {row['denominator']:,.6g} · ratio: {value_text(row['total'],spec['unit'])}",className='panel-note'))
    if row.get('distribution'):children.append(dcc.Graph(figure=distribution(view,row),config=charts.CONFIG))
    children.append(html.Details(children=[html.Summary('Source counter deltas and coverage'),
        html.Table([html.Thead(html.Tr([html.Th(x) for x in ['Counter','Complete-interval delta','Aligned delta (estimate)','Unit','Samples']])),
        html.Tbody([html.Tr([html.Td(s['name']),html.Td(s['total'] if s['total'] is not None else '—'),html.Td(f"{s['aligned_delta']:,.6g}" if s.get('aligned_delta') is not None else '—'),html.Td(s['unit']),html.Td(s['samples'])]) for s in row['sources']])],className='compare-table')]))
    return children


def balance_rows(records):
    usable=[r for r in records if r['total'] is not None]
    values=np.array([r['total'] for r in usable],dtype=float)
    total=float(values.sum()) if len(values) else 0
    active=int((values>0).sum());shares=values/total if total>0 else np.zeros(len(values))
    return {'selected':len(records),'observed':len(usable),'active':active,'total':total if usable else None,
        'effective':float(1/np.sum(shares**2)) if total>0 else None,
        'largest_share':float(shares.max()) if total>0 else None,
        'max_min':float(values.max()/values.min()) if len(values) and values.min()>0 else '∞' if active and active<len(values) else None,
        'volume_p50':float(np.quantile(values,.5)) if len(values) else None,
        'volume_p99':float(np.quantile(values,.99)) if len(values) else None,
        'cv':float(values.std()/values.mean()) if total>0 else None}


def balance_panel(view,engine):
    if view['query'].metric!='traffic':return topology_panel(view,engine)
    records=view['ranked'];b=balance_rows(records)
    labels=[('Observed active NICs',f"{b['active']} / {b['observed']} observed"),('Effective NIC count',f"{b['effective']:.2f}" if b['effective'] is not None else '—'),('Largest traffic share',f"{b['largest_share']:.1%}" if b['largest_share'] is not None else '—'),('Max/min volume',str(b['max_min']) if b['max_min'] is not None else '—')]
    content=[html.Div([html.Div([html.Span(label,className='metric-label'),html.Strong(value)]) for label,value in labels],className='balance-metrics')]
    nodes=sorted({r['node'] for r in records});rows=[]
    for node in nodes:
        n=balance_rows([r for r in records if r['node']==node]); rows.append(html.Tr([html.Td(node),html.Td(f"{n['active']} / {n['observed']} of {n['selected']}"),html.Td(human(n['total'])),html.Td(f"{n['effective']:.2f}" if n['effective'] is not None else '—'),html.Td(human(n['volume_p50'])),html.Td(human(n['volume_p99']))]))
    content.append(html.Details(children=[html.Summary('Node balance and distribution'),html.Table([html.Thead(html.Tr([html.Th(x) for x in ['Node','Active / observed','Observed volume','Effective NICs','P50 NIC volume','P99 NIC volume']])),html.Tbody(rows)],className='compare-table')]))
    topo=engine.analysis_config.get('topology',{})
    if topo:
        groups={}
        for r in records:
            for level,label in topo.get(r['node'],{}).items():groups.setdefault((level,label),[]).append(r)
        content.append(html.Details(children=[html.Summary('Explicit topology groups'),html.Table([html.Tr([html.Th('Grouping'),html.Th('Group'),html.Th('Observed volume'),html.Th('Observed / selected NICs')]),*[html.Tr([html.Td(level),html.Td(label),html.Td(human(balance_rows(rs)['total'])),html.Td(f"{balance_rows(rs)['observed']} / {len(rs)}")]) for (level,label),rs in sorted(groups.items())]],className='compare-table')]))
    content.append(html.P('Effective NICs = 1 / Σ(traffic share²). Distribution percentiles are across NIC volumes, not time. Active means observed nonzero traffic; missing devices and partial coverage limit comparison.',className='panel-note'))
    return content


def topology_distribution(view, engine):
    """Per-device distributions in explicitly supplied groups; no pooled ratios."""
    topology = engine.analysis_config.get('topology', {})
    groups = {}
    for row in view['ranked']:
        for level, label in topology.get(row['node'], {}).items():
            groups.setdefault((level, label), []).append(row)
    result = []
    for (level, label), rows in sorted(groups.items()):
        values = [float(row['total']) for row in rows if row['total'] is not None]
        result.append({'level': level, 'group': label, 'selected_nics': len(rows),
            'observed_nics': len(values), 'minimum': min(values) if values else None,
            'median': float(np.median(values)) if values else None,
            'maximum': max(values) if values else None})
    return result


def topology_panel(view, engine):
    if 'diagnostic' not in view:
        return []
    groups = topology_distribution(view, engine)
    if not groups:
        return []
    unit = view['diagnostic']['unit']
    return [html.Details(children=[html.Summary('Compare explicit topology groups'),
        html.Table([html.Thead(html.Tr([html.Th(x) for x in ['Grouping','Group','Observed / selected','Minimum','Median NIC','Maximum']])),
            html.Tbody([html.Tr([html.Td(row['level']),html.Td(row['group']),html.Td(f"{row['observed_nics']} / {row['selected_nics']}"),
                html.Td(value_text(row['minimum'],unit)),html.Td(value_text(row['median'],unit)),html.Td(value_text(row['maximum'],unit))]) for row in groups[:200]])],className='compare-table'),
        html.P('These are distributions of individual NIC values within supplied groups, not pooled ratios. Unmapped nodes are not assigned an invented rack or switch.',className='panel-note')])]
