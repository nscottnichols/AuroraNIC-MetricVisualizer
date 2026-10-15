#!/usr/bin/env python3
"""Validate a preserved THAPI allreduce run; never contacts a remote host.

Run from the application checkout:
  python scripts/validate_real_capture.py /path/to/run --output artifacts/real-capture
The optional original reference NPZ is read with allow_pickle=False and never modified.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path,help='Extracted directory containing ctf/ and timeline.pftrace')
parser.add_argument('--output',type=Path,required=True,help='Destination for compact JSON evidence')
parsed=parser.parse_args()
from collections import Counter
from dataclasses import replace
import hashlib,json,time
import numpy as np
from main import load_trace_store
from trace_io import trace_arguments
from trace_cache import read_trace_store_cache
from explorer.model import ExplorerEngine,Query
from explorer.app import create_explorer_app,token
from explorer.diagnostics import METRICS
BASE=parsed.run.resolve(strict=True)
EVID=parsed.output.resolve()
EVID.mkdir(parents=True,exist_ok=True)
oldpath=BASE/'trace-cache-hpe-cassini-octets-v1.npz'
old=None
if oldpath.is_file():
    with np.load(oldpath,allow_pickle=False) as archive:
        fingerprint=json.loads(archive['metadata_json'].tobytes())['source_fingerprint']
    old=read_trace_store_cache(oldpath,fingerprint)

def rows(store):
    nodes={x.node_id:x.label for x in store.nodes}; nics={x.nic_id:x.label for x in store.nics}; counters={x.counter_id:x.name for x in store.counters}
    return Counter((nodes[int(n)],nics[int(ni)],counters[int(c)],int(t),int(v)) for n,ni,c,t,v in zip(store.node_ids,store.nic_ids,store.counter_ids,store.timestamps_ns,store.uint64_values))

report={}
for filename in ('ctf','timeline.pftrace'):
    args=trace_arguments(BASE/filename,root=BASE)
    start=time.perf_counter();store=load_trace_store(args);loaded=time.perf_counter()-start
    e=ExplorerEngine(store);app=create_explorer_app(store);client=app.server.test_client()
    q=Query(e.origin,e.end)
    metrics={};http=[]
    key=next(key for key in app.callback_map if key.startswith('..ex-cards.children'))
    outputs=[{'id':v.component_id,'property':v.component_property} for v in app.callback_map[key]['output']]
    for metric in ('traffic','octet_errors',*METRICS):
        for nic_ids in ((),(0,)):
            selection=replace(q,metric=metric,nic_ids=nic_ids)
            state={'dataset':token(e),'query':selection.to_dict()}
            response=client.post('/_dash-update-component',json={'output':key,'outputs':outputs,'inputs':[{'id':'ex-state','property':'data','value':state}],'state':[],'changedPropIds':['ex-state.data']})
            assert response.status_code==200,(filename,metric,nic_ids,response.data[:500])
            http.append({'metric':metric,'nic_ids':nic_ids,'status':response.status_code,'payload_bytes':len(response.data)})
            json.dumps(e.export(selection),allow_nan=False)
        view=e.query(replace(q,metric=metric))
        metrics[metric]={'finite_summary_records':sum(r['total'] is not None for r in view['records']), 'selected_records':len(view['records']), 'observed_source_counters':sum(r.get('observed_counter_count',len(r.get('sources',[]))) for r in view['records'])}
    for metric in ('posted_pushback','nonposted_pushback','traffic','priority_fraction','request_latency','octet_errors'):
        document=e.export(replace(q,metric=metric))
        (EVID/f'{filename.replace(".","-")}-{metric}.json').write_text(json.dumps(document,allow_nan=False,indent=2))
    item={'rows':store.row_count,'collectors':[vars(c) for c in store.collectors],'nics':len(store.nics),'counter_names':len(store.counters),'duration_ns':str(e.end-e.origin),'warm_load_seconds':loaded,'metrics':metrics,'http_render_checks':http}
    if filename=='ctf' and old is not None:
        expected=rows(old);actual=rows(store)
        item['raw_sample_equivalence']={'historical_cache_rows':sum(expected.values()),'new_rows':sum(actual.values()),'equal':expected==actual,'missing':sum((expected-actual).values()),'extra':sum((actual-expected).values()),'comparison_fields':['hostname','nic','counter','timestamp_ns','exact_uint64_value']}
        assert expected==actual
        # Independent endpoint calculation; no diagnostic/alignment code used here.
        checks=[]
        for nic in store.nics:
            if nic.label!='cxi0':continue
            node=next(n.label for n in store.nodes if n.node_id==nic.node_id)
            for key,prefix in [('posted_pushback','posted'),('nonposted_pushback','non_posted')]:
                blocked=f'parbs_tarb_pi_{prefix}_blocked_cnt';packets=f'parbs_tarb_pi_{prefix}_pkts'
                b=store.query_series(nic.node_id,nic.nic_id,blocked).unique();p=store.query_series(nic.node_id,nic.nic_id,packets).unique()
                independent=(int(b.values[-1])-int(b.values[0]))/(int(p.values[-1])-int(p.values[0]))
                row=e.query(replace(q,metric=key,nic_ids=(nic.nic_id,)))['records'][0]
                # Only identical supports should exactly equal raw endpoint ratios.
                same=np.array_equal(b.timestamps_ns,p.timestamps_ns)
                if same:assert abs(row['total']-independent)<1e-10
                checks.append({'node':node,'nic':'cxi0','metric':key,'blocked_delta':str(int(b.values[-1])-int(b.values[0])),'packet_delta':str(int(p.values[-1])-int(p.values[0])),'independent_endpoint_ratio':independent,'application_common_support_ratio':row['total'],'counter_timestamps_equal':bool(same)})
        item['independent_pushback_checks']=checks
    report[filename]=item
# Old cache can still be inspected safely, though CLI deliberately rebuilds it.
if old is not None:
    app=create_explorer_app(old);e=app.explorer_engine();view=e.query(Query(e.origin,e.end,metric='octet_errors'))
    from explorer.app import cards
    cards(view)
    report['old_cache_error_regression']={'cards_rendered':True,'records':len(view['records']),'ambiguous_records':sum(r['state']=='Choose collector' for r in view['records'])}
(EVID/'real-validation.json').write_text(json.dumps(report,indent=2,default=str))
print('Real trace validation passed.',[(k,v.get('rows')) for k,v in report.items()])
