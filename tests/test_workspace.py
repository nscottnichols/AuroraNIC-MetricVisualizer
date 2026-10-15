"""Workspace contracts: identity-safe charts, complete paging, and portable view state."""
import base64
from dataclasses import replace
import json
import numpy as np
import pytest
from explorer.app import initial_state,normalize_query,token,create_explorer_app
from explorer.demo import make_demo_store
from explorer.model import ExplorerEngine,Query
from explorer.metadata import catalog,available_options,definition,describe_counter
from explorer.overview import build_heatmap,target_from_click,PAGE_ROWS
from explorer.workspace import push_history,snapshot,decode_view,csv_export,page_table
from main import parse_args,read_requested_counter_names

@pytest.fixture
def engine():return ExplorerEngine(make_demo_store())

def test_catalog_is_reference_not_requested_inventory(engine):
    assert len(catalog())==1890
    assert describe_counter('future_tx_dma_pkts')['label']=='Future TX DMA packets'
    assert read_requested_counter_names(None)==()
    assert parse_args([]).trace_requested_counter_file is None
    values={o['value'] for o in available_options(engine)}
    assert 'traffic' in values
    assert 'posted_pushback' not in values
    assert not (set(catalog())-set(engine.counters)) & values
    unknown=describe_counter('new_block_future_register')
    assert 'unverified' in unknown['provenance'] and unknown['unit']=='unverified native units'


def test_every_derived_formula_has_named_inputs():
    from explorer.diagnostics import METRICS
    for key,m in METRICS.items():
        doc=definition(None,key)
        assert set(m.numerator+m.denominator)==set(doc['inputs'])
        assert doc['unit']==m.unit and doc['method'] and doc['limits']
        if m.mode=='ratio':assert all(n in doc['formula'] for n in doc['inputs'])


def test_heatmap_click_uses_figure_row_identity_even_without_customdata(engine):
    view=engine.query(Query(engine.origin,engine.end))
    fig,_,_,_=build_heatmap(view,order='value')
    doc=fig.to_plotly_json();keys=doc['layout']['meta']['row_keys']
    assert target_from_click({'points':[{'y':0}]},doc)==keys[0]
    assert target_from_click({'points':[{'y':'node-a / cxi0'}]},doc) is None
    assert target_from_click({'points':[{'y':float('nan')}]},doc) is None
    assert target_from_click({'points':[{'y':99999}]},doc) is None
    assert target_from_click({'points':[{'y':0,'customdata':['evil:9']}]},doc)==keys[0]


def test_heatmap_missing_is_not_zero_and_log_preserves_hover(engine):
    view=engine.query(Query(engine.origin,engine.end))
    fig,*_=build_heatmap(view,color='log')
    doc=fig.to_plotly_json();missing,value=doc['data']
    assert any(v is not None for row in missing['z'] for v in row)
    for zs,cs in zip(value['z'],value['customdata']):
        for z,c in zip(zs,cs):
            if z is not None:assert z==pytest.approx(np.log10(1+c[2]))
    assert doc['layout']['meta']['color']=='log'


def test_heatmap_paging_does_not_discard_devices_or_rescale(engine):
    view=engine.query(Query(engine.origin,engine.end));base=view['ranked'][0]
    view['ranked']=[{**base,'nic_id':i,'node_id':i//4,'node':f'node-{i//4}','nic':f'cxi{i%4}','total':i,
                    'rates':np.full(240,i,dtype=float)} for i in range(4000)]
    first,*_=build_heatmap(view,grouping='nic',page=0)
    last,*_=build_heatmap(view,grouping='nic',page=99999)
    a,b=first.layout.meta,last.layout.meta
    assert a['rows']==4000 and b['page']==166 and b['rows']==4000
    assert a['scale']==b['scale']
    assert len(first.data[1].z)==PAGE_ROWS and len(last.data[1].z)==16
    auto,*_=build_heatmap(view)
    assert auto.layout.meta['rows']==1000 and auto.layout.meta['grouped']


def test_node_derived_values_are_explicit_max_not_average_of_ratios(engine):
    view=engine.query(Query(engine.origin,engine.end))
    view={**view,'query':replace(view['query'],metric='posted_pushback'),
          'diagnostic':{'unit':'cycles/packet'},'ranked':[{**r,'raw_y':[float(r['nic_id'])]*240} for r in view['ranked']]}
    fig,note,*_=build_heatmap(view,grouping='node')
    assert 'maximum available NIC value' in note
    assert fig.data[1].customdata[0][0][2]==3


def test_history_is_bounded_and_reset_on_dataset_switch(engine):
    state=initial_state(engine)
    for i in range(50):
        q=replace(Query.from_dict(state['query']),start_ns=engine.origin+i+1)
        state=push_history(state,q,token(engine))
    assert len(state['history'])==32
    new=push_history(state,q,'new-fingerprint');assert new['history']==[]


def test_snapshot_roundtrip_rejects_other_traces_and_invalid_files(engine):
    state=initial_state(engine);doc=snapshot(state,'Phase A',{'color':'log'})
    contents='data:application/json;base64,'+base64.b64encode(json.dumps(doc).encode()).decode()
    assert decode_view(contents,token(engine))==doc
    with pytest.raises(ValueError,match='different trace'):decode_view(contents,'not-this-trace')
    with pytest.raises(ValueError):decode_view('data:;base64,bad!',token(engine))
    with pytest.raises(ValueError):decode_view('x'*200000,token(engine))
    assert 'path' not in json.dumps(doc).lower()


def test_csv_all_selected_records_exact_timestamps_and_formula_guard(engine):
    query=Query(engine.origin,engine.end);doc=engine.export(query)
    doc['records'][0]['node']='=HYPERLINK("evil")'
    text=csv_export(doc)
    import csv,io
    rows=list(csv.DictReader(io.StringIO(text)))
    assert len(rows)==32
    assert rows[0]['node'].startswith("'=")
    assert rows[0]['start_ns']==str(engine.origin)
    assert rows[0]['end_ns']==str(engine.end)


def test_table_sorts_full_population_before_slicing_missing_last():
    rows=[{'id':str(i),'node':f'n{i}','nic':'cxi0','value':i} for i in range(800)]
    rows.append({'id':'missing','node':'missing','nic':'cxi0','value':None})
    page,count=page_table(rows,0,12,[{'column_id':'value','direction':'desc'}])
    assert count==67 and page[0]['value']==799
    page,_=page_table(rows,66,12,[{'column_id':'value','direction':'desc'}])
    assert page[-1]['value'] is None


def test_layout_has_unique_ids_and_no_raw_browser_stores(engine):
    app=create_explorer_app(engine.store)
    data=app.server.test_client().get('/_dash-layout').json
    ids=[];stores=[]
    def walk(x):
        if isinstance(x,dict):
            if 'props' in x:
                if 'id' in x['props']:ids.append(str(x['props']['id']))
                if x.get('type')=='Store':stores.append(x['props'])
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(data)
    assert len(ids)==len(set(ids))
    assert all(len(json.dumps(s))<10000 for s in stores)
    assert all(not any(k in json.dumps(s) for k in ('raw_y','timestamps_ns','rates"')) for s in stores)
