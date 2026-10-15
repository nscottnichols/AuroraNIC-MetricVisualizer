"""Executable acceptance for the replacement explorer, not the historical report."""
from dataclasses import replace
import json
import numpy as np
import pytest
from explorer.model import ExplorerEngine, Query, traffic_bins, human, extrema_indices
from explorer.app import create_explorer_app, normalize_query, _seconds_ns
from trace_model import (RunIdentity,NodeIdentity,NicIdentity,CollectorIdentity,CounterMetadata,
                        CounterSemantics,ValueKind,Coverage,NormalizedBatch)
from trace_store import TraceStore


def make_store(specs, origin=0):
    """spec = (node/nic id, counter id [TX=0,RX=1], collector, times, values)."""
    ids=sorted({s[0] for s in specs})
    nodes=tuple(NodeIdentity(i,f'node-{i}',f'node-{i}') for i in ids)
    nics=tuple(NicIdentity(i,i,'cxi0','cxi0') for i in ids)
    counters=tuple(CounterMetadata(i,name,ValueKind.EXACT_UINT64,CounterSemantics.CUMULATIVE,'bytes',False,'test')
                   for i,name in enumerate(('hni_sts_tx_ok_octets','hni_sts_rx_ok_octets',
                                            'hni_sts_tx_bad_octets','hni_sts_rx_bad_octets')))
    collectors=tuple(CollectorIdentity(i,f'collector-{i}',rank=i) for i in sorted({s[2] for s in specs}))
    rows=[(n,c,co,origin+t,v) for n,c,co,ts,vs in specs for t,v in zip(ts,vs)]
    ts=[r[3] for r in rows]; names=tuple(c.name for c in counters)
    batch=NormalizedBatch(identity=RunIdentity('test','sha256:test','synthetic-ctf','test','direct','test-clock','single clock'),nodes=nodes,nics=nics,counters=counters,collectors=collectors,
        coverage=Coverage(requested_counter_names=names,available_counter_names=names,readable_counter_names=names,observed_counter_names=names,acquisition_start_ns=min(ts),acquisition_end_ns=max(ts)),
        node_ids=np.asarray([r[0] for r in rows],dtype=np.int64),nic_ids=np.asarray([r[0] for r in rows],dtype=np.int64),counter_ids=np.asarray([r[1] for r in rows],dtype=np.int64),collector_ids=np.asarray([r[2] for r in rows],dtype=np.int64),timestamps_ns=np.asarray(ts,dtype=np.int64),uint64_values=np.asarray([r[4] for r in rows],dtype=np.uint64),float64_values=np.full(len(rows),np.nan),quality_flags=np.zeros(len(rows),dtype=np.int64),record_ids=np.asarray([f'r{i}' for i in range(len(rows))]))
    return TraceStore.from_batches([batch])


@pytest.fixture
def basic():
    return ExplorerEngine(make_store([(0,0,0,[0,1_000_000_000,2_000_000_000],[0,10,30]),
                                     (0,1,0,[0,1_000_000_000,2_000_000_000],[0,0,0]),
                                     (1,0,0,[0,1_000_000_000,2_000_000_000],[0,100,300])]))


def test_full_selection_uses_every_physical_nic(basic):
    v=basic.query(Query(basic.origin,basic.end))
    assert v['aggregate']['TX']['total']==330
    assert np.allclose(v['aggregate']['TX']['rates'][:120],110)
    assert np.allclose(v['aggregate']['TX']['rates'][120:],220)
    assert {(r['node_id'],r['nic_id']) for r in v['records']}=={(0,0),(1,1)}


def test_missing_rx_is_not_idle_or_numeric_zero(basic):
    v=basic.query(Query(basic.origin,basic.end))
    rows=[r for r in v['records'] if r['direction']=='RX']
    assert rows[0]['total']==0 and rows[0]['state']=='Idle'
    assert np.all(rows[0]['rates']==0)
    assert rows[1]['total'] is None and np.all(np.isnan(rows[1]['rates']))
    assert v['aggregate']['RX']['usable']==1
    assert v['aggregate']['RX']['complete'] is False


def test_repeated_nic_label_cannot_leak_between_nodes(basic):
    v=basic.query(Query(basic.origin,basic.end,nic_ids=(0,)))
    assert {r['node_id'] for r in v['records']}=={0}
    assert v['aggregate']['TX']['total']==30


def test_ambiguous_collectors_require_explicit_selection():
    e=ExplorerEngine(make_store([(0,0,0,[0,1_000_000_000],[0,100]),(0,0,1,[0,1_000_000_000],[0,900])]))
    q=Query(e.origin,e.end)
    assert e.query(q)['records'][0]['state']=='Choose collector'
    assert e.query(q)['aggregate']['TX']['total'] is None
    chosen=e.query(replace(q,collectors=((0,1),)))
    assert chosen['aggregate']['TX']['total']==900
    assert chosen['records'][0]['collector_id']==1
    assert e.export(replace(q,collectors=((0,1),)))['selection']['collectors']==((0,1),)


def test_wrong_collector_does_not_fall_back():
    e=ExplorerEngine(make_store([(0,0,0,[0,1_000_000_000],[0,100])]))
    assert e.query(Query(e.origin,e.end,collectors=((0,99),)))['aggregate']['TX']['total'] is None


def test_octet_error_metric_reports_observed_zero_and_counter_details():
    timestamps=[0,1_000_000_000,2_000_000_000]
    e=ExplorerEngine(make_store([
        (0,0,0,timestamps,[0,100,300]),
        (0,2,0,timestamps,[0,0,0]),
        (0,3,0,timestamps,[0,0,0]),
    ]))
    view=e.query(Query(e.origin,e.end,metric='octet_errors'))
    record=view['records'][0]
    assert record['state']=='Observed Zero'
    assert record['total']==0
    assert record['zero_counter_count']==2
    assert record['nonzero_counter_count']==0
    assert {counter['counter'] for counter in record['counter_results']}=={
        'hni_sts_tx_bad_octets','hni_sts_rx_bad_octets'
    }
    assert all(counter['total']==0 for counter in record['counter_results'])
    exported=e.export(Query(e.origin,e.end,metric='octet_errors'))
    assert exported['traffic_totals']=={}
    assert exported['records'][0]['zero_counter_count']==2


def test_reordered_and_duplicate_selections_reuse_cached_query():
    e=ExplorerEngine(make_store([
        (0,0,0,[0,1_000_000_000],[0,100]),
        (1,0,0,[0,1_000_000_000],[0,200]),
    ]))
    first=e.query(Query(e.origin,e.end,node_ids=(0,1),nic_ids=(0,1),collectors=((0,0),)))
    second=e.query(Query(e.origin,e.end,node_ids=(1,0,1),nic_ids=(1,0,1),collectors=((0,0),)))
    third=e.query(Query(e.origin,e.end,node_ids=(1,0,0,1),nic_ids=(1,0,1,0),collectors=((0,0),(0,0))))

    assert first["records"][0]["total"] == second["records"][0]["total"] == third["records"][0]["total"]
    assert e.query_executions==1


def test_unaligned_series_are_integrated_on_common_grid():
    e=ExplorerEngine(make_store([(0,0,0,[0,10_000_000_000],[0,100]),(1,0,0,[1_000_000_000,9_000_000_000],[0,160])]))
    rates=e.query(Query(e.origin,e.end))['aggregate']['TX']['rates']
    assert np.allclose(rates[24:216],30)
    assert np.all(np.isnan(rates[:24])) and np.all(np.isnan(rates[216:]))


def test_reset_is_a_gap_not_a_negative_rate():
    ts=[0,1_000_000_000,2_000_000_000,3_000_000_000]
    e=ExplorerEngine(make_store([(0,0,0,ts,[100,150,0,30])]))
    row=e.query(Query(e.origin,e.end))['records'][0]
    assert row['total']==80
    assert np.all(np.isnan(row['rates'][80:160]))
    assert np.all(row['rates'][:80]==50) and np.all(row['rates'][160:]==30)


def test_exact_large_uint64_deltas_and_export_strings():
    e=ExplorerEngine(make_store([(0,0,0,[0,1_000_000_000,2_000_000_000],[2**63,2**63+1,2**63+3])]))
    q=Query(e.origin,e.end); doc=e.export(q)
    assert doc['records'][0]['total']=='3'
    assert e.query(q)['aggregate']['TX']['total']==3
    assert np.allclose(e.query(q)['records'][0]['rates'][:120],1)
    json.dumps(doc,allow_nan=False)


def test_epoch_round_trip_preserves_nanoseconds():
    origin=1_700_000_000_123_456_789
    q=Query(origin+3,origin+2000000007)
    assert isinstance(q.to_dict()['start_ns'],str)
    assert Query.from_dict(json.loads(json.dumps(q.to_dict())))==q
    assert origin+_seconds_ns('0.000000003')==origin+3


def test_partial_boundary_transitions_are_not_prorated():
    t=[i*1_000_000_000 for i in range(11)]
    e=ExplorerEngine(make_store([(0,0,0,t,list(range(11)))]))
    q=Query(2_200_000_000,5_500_000_000)
    row=e.query(q)['records'][0]
    assert row['total']==2
    assert row['coverage']==pytest.approx(2/3.3)
    assert np.sum(row['mass'])==pytest.approx(2)


def test_binned_display_mass_matches_exact_summary(basic):
    v=basic.query(Query(basic.origin,basic.end))
    for r in v['records']:
        if r['total'] is not None:
            assert sum(r['mass'])==pytest.approx(r['total'])


def test_display_toggles_do_not_query_sources_again(basic):
    q=Query(basic.origin,basic.end)
    basic.query(q); basic.query(replace(q,mode='volume')); basic.query(replace(q,direction='RX'))
    assert basic.query_executions==1


def test_query_cache_is_bounded(basic):
    for i in range(8):
        basic.query(Query(i,basic.end))
    assert len(basic._cache)==4


def test_node_change_normalizes_incompatible_nic(basic):
    q=normalize_query(basic,Query(basic.origin,basic.end,node_ids=(1,),nic_ids=(0,)))
    assert q.nic_ids==()
    assert {r['node_id'] for r in basic.query(q)['records']}=={1}


def test_raw_counter_does_not_masquerade_as_bandwidth(basic):
    q=Query(basic.origin,basic.end,metric='hni_sts_tx_ok_octets')
    r=basic.query(q)['records'][0]
    assert r['raw_y']==[0.,10.,30.]
    assert r['mean'] is None and r['p99'] is None
    assert 'not selected-window totals' in r['reason']
    assert basic.query(q)['aggregate']=={}


def test_minmax_display_retains_a_narrow_peak():
    v=np.zeros(10000); v[1555]=1000
    indices=extrema_indices(v,240)
    assert 1555 in indices and len(indices)<=240


def test_no_requirement_questions_in_default_layout_or_dependencies(basic):
    a=create_explorer_app(basic.store); c=a.server.test_client()
    layout=c.get('/_dash-layout').data.decode()
    deps=c.get('/_dash-dependencies').data.decode()
    for forbidden in ('Question index','NIC-01','JOB-01','analysis-question','question-select'):
        assert forbidden not in layout+deps
    assert c.get('/readyz').json['ui']=='explorer'


def test_browser_stores_contain_only_selection_and_saved_ranges(basic):
    a=create_explorer_app(basic.store); root=a.server.test_client().get('/_dash-layout').json
    def walk(node):
        if isinstance(node,dict):
            if node.get('type')=='Store':
                yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node,list):
            for value in node:
                yield from walk(value)
    stores=list(walk(root))
    assert {s['props']['id'] for s in stores}=={'ex-state','ex-saved','ex-page','ex-restore','ex-favorites','ex-bookmarks'}
    assert len(json.dumps(stores))<2000


def test_component_ids_are_unique(basic):
    a=create_explorer_app(basic.store); root=a.server.test_client().get('/_dash-layout').json
    ids=[]
    def walk(v):
        if isinstance(v,dict):
            if 'props' in v and 'id' in v['props']:
                ids.append(str(v['props']['id']))
            for child in v.values():walk(child)
        elif isinstance(v,list):
            for child in v:walk(child)
    walk(root)
    assert len(ids)==len(set(ids))


def test_default_cli_factory_is_replacement_not_report():
    import main
    assert main._trace_ui_factory(main.parse_args(['--demo'])).__module__=='explorer.app'
    assert main._trace_ui_factory(main.parse_args(['--trace-ui','legacy'])).__module__=='analysis_app'


@pytest.mark.parametrize('value',['nan','inf','abc',None])
def test_bad_times_rejected(value):
    with pytest.raises(ValueError):_seconds_ns(value)


def test_unknown_identity_rejected(basic):
    with pytest.raises(ValueError):basic.query(Query(basic.origin,basic.end,nic_ids=(999,)))


def test_manifest_only_cli_dispatches_to_trace_runtime(monkeypatch):
    """An approved manifest must not fall through to legacy example data."""
    import main as entry
    args = entry.parse_args(["--dataset-manifest", "/approved/datasets.json"])
    seen = []
    monkeypatch.setattr(entry, "parse_args", lambda: args)
    monkeypatch.setattr(entry, "run_trace_app", lambda value: seen.append(value))
    monkeypatch.setattr(entry, "_load_metric_data", lambda *a, **k: pytest.fail("legacy path selected"))
    entry.main()
    assert seen == [args]
