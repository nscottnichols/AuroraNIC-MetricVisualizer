"""Paged, identity-safe population heatmaps for traffic, metrics and observations.

Display choices never change scientific summaries. Scale is computed before
paging. A synthetic row-position mapping travels in figure metadata, rather
than trusting Plotly's version-dependent heatmap customdata events.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import re
import numpy as np
import plotly.graph_objects as go
from . import charts

PAGE_ROWS=24

def natural(text):
    return tuple((0,int(x)) if x.isdigit() else (1,x.lower()) for x in re.split(r'(\d+)',str(text)))


def _grid(view):
    q=view['query']; origin=view['origin_ns']
    count=min(240,q.end_ns-q.start_ns)
    return np.linspace((q.start_ns-origin)/1e9,(q.end_ns-origin)/1e9,count+1)


def matrix_rows(view):
    edges=_grid(view); count=len(edges)-1;rows=[]
    metric=view['query'].metric
    for row in view['ranked']:
        if metric=='traffic':
            y=np.asarray(row['rates'],dtype=float)
        elif 'diagnostic' in view:
            y=np.asarray([np.nan if x is None else x for x in row.get('raw_y',[])],dtype=float)
            if len(y)!=count:y=np.full(count,np.nan)
        else:
            y=np.full(count,np.nan)
            # Actual full-resolution observed values, not the extrema envelope.
            if row.get('heat_y') is not None:
                y=np.asarray(row['heat_y'],dtype=float)
            else:
                for x,value in zip(row.get('raw_x',[]),row.get('raw_y',[])):
                    i=min(count-1,int(np.searchsorted(edges,x,side='right')-1))
                    if 0<=i<count:y[i]=value
        rows.append({'key':f"nic:{row['nic_id']}",'label':f"{row['node']} / {row['nic']}",
            'node':row['node'],'node_id':row['node_id'],'nic_id':row['nic_id'],
            'y':y,'value':row.get('total'),'state':row['state'],
            'coverage':row.get('coverage',0),'members':1,'contributors':np.isfinite(y).astype(int)})
    return edges, rows


def group_rows(rows, metric):
    grouped={}
    for row in rows:grouped.setdefault(row['node_id'],[]).append(row)
    output=[]
    for nid,members in grouped.items():
        values=np.array([r['y'] for r in members],dtype=float)
        finite=np.isfinite(values); contributors=finite.sum(axis=0)
        if metric=='traffic':
            y=np.where(finite,values,0).sum(axis=0)
            y[contributors<len(members)]=np.nan
            totals=[r['value'] for r in members]
            value=sum(totals) if all(v is not None for v in totals) else None
        else:
            y=np.max(np.where(finite,values,-np.inf),axis=0)
            y[contributors==0]=np.nan
            usable=[r['value'] for r in members if r['value'] is not None]
            value=max(usable) if usable else None
        output.append({'key':f'node:{nid}','label':members[0]['node']+f" · {len(members)} NICs",
          'node':members[0]['node'],'node_id':nid,'nic_id':None,'y':y,'value':value,
          'state':'Observed' if np.any(np.isfinite(y)) else 'Unavailable',
          'coverage':min(r['coverage'] for r in members),'members':len(members),'contributors':contributors})
    return output


def build_heatmap(view, *, grouping='auto', order='identity', color='linear', search='', page=0):
    edges,rows=matrix_rows(view)
    metric=view['query'].metric
    grouped=grouping=='node' or (grouping=='auto' and len(rows)>64)
    if grouped:rows=group_rows(rows,metric)
    population=len(rows)
    terms=(search or '').lower().split()
    rows=[r for r in rows if all(t in r['label'].lower() for t in terms)]
    if order=='value':rows.sort(key=lambda r:(r['value'] is None,-float(r['value'] or 0),natural(r['label'])))
    elif order=='coverage':rows.sort(key=lambda r:(r['coverage'],natural(r['label'])))
    else:rows.sort(key=lambda r:natural(r['label']))
    matrix=np.asarray([r['y'] for r in rows],dtype=float)
    unit=('B/s' if metric=='traffic' else view['diagnostic']['unit'] if 'diagnostic' in view
          else next((r.get('unit') for r in view['records'] if r.get('unit')),'native units'))
    factor,unit=charts.scale(matrix,unit) if metric=='traffic' else (1.,unit)
    # All rows in the display filter share one color range across every page.
    finite=matrix[np.isfinite(matrix)]/factor if matrix.size else np.array([])
    zmin=min(0.,float(finite.min())) if len(finite) else 0.
    zmax=float(finite.max()) if len(finite) else 1.
    zmax=max(zmin+1e-12,zmax)
    log=color=='log' and zmin>=0
    count=len(rows);pages=max(1,math.ceil(count/PAGE_ROWS));page=max(0,min(pages-1,int(page or 0)))
    subset=rows[page*PAGE_ROWS:(page+1)*PAGE_ROWS]
    positions=list(range(len(subset))); centers=((edges[:-1]+edges[1:])/2).tolist()
    z=[];custom=[]
    for row in subset:
        values=row['y']/factor
        z.append([float(np.log10(1+v) if log else v) if np.isfinite(v) else None for v in values])
        custom.append([[row['key'],row['label'],float(v) if np.isfinite(v) else None,
                        int(row['contributors'][i]),row['members']] for i,v in enumerate(values)])
    fig=charts.base(height=max(305,min(700,100+len(subset)*22)))
    ticks=np.linspace(0,zmax,5).tolist() if zmin==0 else np.linspace(zmin,zmax,5).tolist()
    bar={'title':{'text':unit+('<br>log color' if log else '')},'thickness':11,'len':.85,
         'tickvals':[math.log10(1+x) if log else x for x in ticks],
         'ticktext':[f'{x:.3g}' for x in ticks]}
    if subset:
        # Discrete neutral underlay distinguishes unavailable from observed zero,
        # and supplies tooltips even when the main heatmap has a gap.
        fig.add_trace(go.Heatmap(x=centers,y=positions,z=[[0 if v is None else None for v in line] for line in z],
            customdata=custom,zmin=0,zmax=1,colorscale=[[0,'#e5e9ef'],[1,'#e5e9ef']],showscale=False,
            hoverongaps=False,hovertemplate='%{customdata[1]}<br>%{x:.5g} s · no supported value<br>Contributors: %{customdata[3]}/%{customdata[4]}<extra></extra>'))
        fig.add_trace(go.Heatmap(x=centers,y=positions,z=z,customdata=custom,
            colorscale='Viridis',zmin=math.log10(1+zmin) if log else zmin,
            zmax=math.log10(1+zmax) if log else zmax,colorbar=bar,hoverongaps=False,
            hovertemplate='%{customdata[1]}<br>%{x:.5g} s · %{customdata[2]:,.5g} '+unit+
                '<br>Contributors: %{customdata[3]}/%{customdata[4]}<extra></extra>'))
    else:charts.empty(fig,'No rows match this display search. Clear the search to see the population.')
    fig.update_layout(margin={'l':185,'r':30,'t':12,'b':48},hovermode='closest',dragmode='select',selectdirection='h',
        meta={'row_keys':[r['key'] for r in subset],'page':page,'pages':pages,'rows':count,
              'population':population,'grouped':grouped,'scale':[zmin,zmax],'unit':unit,'color':color,
              'aggregation':'sum of complete NIC rates' if grouped and metric=='traffic' else 'maximum available NIC value' if grouped else 'physical NIC'},
        uirevision=f'{metric}/{view["query"].start_ns}/{view["query"].end_ns}/{page}/{grouping}/{order}/{search}')
    fig.update_yaxes(tickmode='array',tickvals=positions,ticktext=[r['label'] for r in subset],
                     autorange='reversed',range=None,rangemode='normal',tickfont_size=10,showgrid=False)
    fig.update_xaxes(title_text='Time from trace start (s)',range=[edges[0],edges[-1]],showgrid=False)
    aggregation=('Node rows: sum of complete NIC rates.' if metric=='traffic' else 'Node rows: maximum available NIC value, not a summed or averaged ratio.') if grouped else 'One row per physical NIC.'
    note=f'Rows {page*PAGE_ROWS+1 if count else 0}–{min(count,(page+1)*PAGE_ROWS)} of {count}'+(f' matching / {population} in scope' if terms else '')+'. '+aggregation+' Shared scale across pages. Gray = unsupported; zero uses the color-scale minimum.'
    if metric not in {'traffic','octet_errors'} and 'diagnostic' not in view:note+=' Raw values are observations within bins, not rates or totals.'
    return fig,note,page==0,page==pages-1


def target_from_click(event, figure):
    """Resolve a click using the exact figure's row map, never a display label."""
    if not isinstance(event,dict) or not event.get('points'):return None
    point=event['points'][0];meta=figure.get('layout',{}).get('meta',{}) if isinstance(figure,dict) else {}
    keys=meta.get('row_keys',[])
    custom=point.get('customdata')
    if isinstance(custom,list) and custom and custom[0] in keys:return custom[0]
    try:
        y=point['y'];index=int(y)
        if index==float(y) and 0<=index<len(keys):return keys[index]
    except (KeyError,TypeError,ValueError,OverflowError):pass
    return None
