"""Small, testable UI state/export policies; no raw trace data in browser storage."""
from __future__ import annotations
import base64
import csv
import io
import json
from dataclasses import replace
from .model import Query

MAX_VIEWS=12
SCHEMA='aurora-nic-view/1'


def push_history(state, query, dataset):
    history=list(state.get('history',[])) if state.get('dataset')==dataset else []
    current=state.get('query')
    if current and current!=query.to_dict() and state.get('dataset')==dataset:
        history=(history+[current])[-32:]
    return {'dataset':dataset,'query':query.to_dict(),'history':history}


def snapshot(state,name,settings=None):
    return {'schema':SCHEMA,'dataset':state['dataset'],'name':str(name or 'Saved view').strip()[:80] or 'Saved view',
            'query':json.loads(json.dumps(state['query'])), 'settings':settings or {}}


def decode_view(contents,dataset):
    if not isinstance(contents,str) or len(contents)>180_000:
        raise ValueError('Choose a small saved-view JSON file (at most 128 KB), not a trace or analysis export.')
    try:
        _, encoded=contents.split(',',1)
        raw=base64.b64decode(encoded,validate=True)
        if len(raw)>128_000:raise ValueError()
        doc=json.loads(raw)
        if not isinstance(doc,dict) or doc.get('schema')!=SCHEMA:raise ValueError()
        q=doc['query']
        if not isinstance(q,dict) or any(len(q.get(k,[]))>10000 for k in ('node_ids','nic_ids','collectors')):raise ValueError()
        parsed=Query.from_dict(q)
        if not all(isinstance(q.get(k,default),str) for k,default in [('metric','traffic'),('mode','rate'),('direction','TX')]):raise ValueError()
        if not isinstance(doc.get('settings',{}),dict):raise ValueError()
        if not all(isinstance(v,str) for v in doc.get('settings',{}).values()):raise ValueError()
    except (ValueError,TypeError,KeyError,AttributeError,OverflowError,RecursionError):
        raise ValueError('Not a valid NIC Explorer saved-view file.') from None
    if doc.get('dataset')!=dataset:
        raise ValueError('This view belongs to a different trace fingerprint. Open the matching trace before restoring it.')
    return snapshot(doc,doc.get('name'),doc.get('settings'))


def page_table(rows, page=0, size=12, sort_by=None, search=''):
    terms=(search or '').lower().split()
    filtered=[r for r in rows if all(t in f"{r.get('node','')} {r.get('nic','')}".lower() for t in terms)]
    for spec in reversed(sort_by or []):
        key=spec.get('column_id'); descending=spec.get('direction')=='desc'
        good=[r for r in filtered if r.get(key) is not None]; missing=[r for r in filtered if r.get(key) is None]
        try:good.sort(key=lambda r:r[key],reverse=descending)
        except (TypeError,KeyError):pass
        filtered=good+missing
    size=max(1,min(100,int(size or 12)));count=max(1,(len(filtered)+size-1)//size)
    page=max(0,min(count-1,int(page or 0)))
    return filtered[page*size:(page+1)*size],count


def csv_export(doc):
    """Export full selection, not just displayed rows. Guard spreadsheet formulas."""
    out=io.StringIO(newline='')
    fields=['node','nic','metric','direction','collector_id','start_ns','end_ns','value','unit','mean','coverage','state','reason']
    writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
    q=doc['selection']
    def safe(value):
        if isinstance(value,str) and value[:1] in {'=','+','-','@','\t','\r','\n'}:return "'"+value
        return value
    for row in doc['records']:
        writer.writerow({k:safe(v) for k,v in dict(node=row['node'],nic=row['nic'],metric=q['metric'],
            direction=row.get('direction',''),collector_id=row.get('collector_id'),start_ns=q['start_ns'],end_ns=q['end_ns'],
            value=row.get('total'),unit=row.get('unit',''),mean=row.get('mean'),coverage=row.get('coverage'),state=row.get('state'),reason=row.get('reason')).items()})
    return out.getvalue()
