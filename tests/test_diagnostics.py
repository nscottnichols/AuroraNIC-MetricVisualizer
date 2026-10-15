"""Numerical regressions independent of UI snapshots and historical golden files."""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
from ctf_adapter import CtfAdapterConfig, ctf_events_to_normalized_batch, _event_collector_process
from trace_store import TraceStore
from explorer.model import ExplorerEngine, Query
from explorer.app import cards
from explorer.diagnostics import METRICS, signal
from explorer.diagnostic_ui import balance_rows
from explorer.analysis_config import load_analysis_config

POST='parbs_tarb_pi_posted_pkts'
BLOCK='parbs_tarb_pi_posted_blocked_cnt'

def store_for(series, *, collector_names=('sampler',)):
    events=[]
    for collector in collector_names:
        for name,(timestamps,values) in series.items():
            for timestamp,value in zip(timestamps,values):
                events.append({'name':'cxi','provider':'lttng_ust_cxi_sampling','kind':'raw_sampling',
                    'timestamp_ns':int(timestamp),'hostname':'node-a','collector':{'name':collector,'rank':None,'process':collector},
                    'payload':{'interface_name':'cxi0','counter':name,'value':int(value)}})
    config=CtfAdapterConfig(run_id='numeric-test',source_fingerprint='numeric-test',reader_version='fixture',clock_domain='fixture',synchronization_evidence='one clock')
    return TraceStore.from_batches((ctf_events_to_normalized_batch(events,config),))


def diagnostic(series, key='posted_pushback', window=None):
    e=ExplorerEngine(store_for(series));q=Query(*(window or (e.origin,e.end)),metric=key)
    return e.query(q)['records'][0]


def test_posted_ratio_uses_exact_deltas_above_double_precision():
    b=2**60
    row=diagnostic({BLOCK:([0,10,20],[b,b+100,b+300]),POST:([0,10,20],[b,b+10,b+110])})
    assert row['total']==pytest.approx(300/110)
    assert row['numerator']==pytest.approx(300)
    assert row['denominator']==pytest.approx(110)
    assert row['total']!=pytest.approx((10+2)/2)  # Not mean of interval ratios.
    assert next(x for x in row['sources'] if x['name']==BLOCK)['total']=='300'


def test_zero_denominator_is_undefined_even_when_blocked_cycles_exist():
    row=diagnostic({BLOCK:([0,10],[0,100]),POST:([0,10],[5,5])})
    assert row['total'] is None
    assert row['state']=='Undefined'
    assert 'Zero denominator' in row['reason']
    assert all(y is None for y in row['raw_y'])


def test_missing_denominator_is_not_filled_with_zero():
    row=diagnostic({BLOCK:([0,10],[0,100])})
    assert row['total'] is None
    assert 'missing' in row['reason'].lower()


def test_async_counters_share_intersection_not_independent_endpoints():
    row=diagnostic({BLOCK:([0,10,20],[0,100,200]),POST:([5,15,25],[0,10,110])})
    assert row['numerator']==pytest.approx(150)
    assert row['denominator']==pytest.approx(60)
    assert row['total']==pytest.approx(2.5)
    assert row['coverage']==pytest.approx(15/25)


def test_reset_interval_is_excluded_from_numerator_and_denominator():
    row=diagnostic({BLOCK:([0,10,20,30],[0,100,0,40]),POST:([0,10,20,30],[0,10,20,30])})
    assert row['total']==pytest.approx(7.)
    assert row['coverage']==pytest.approx(2/3)
    assert any('decreases' in flag for flag in row['flags'])


def test_selection_does_not_assign_boundary_straddling_counts():
    data={BLOCK:([0,10,20],[0,100,300]),POST:([0,10,20],[0,10,110])}
    row=diagnostic(data,window=(5,20))
    assert row['total']==2.
    assert row['numerator']==200
    assert row['coverage']==pytest.approx(10/15)
    empty=diagnostic(data,window=(1,9))
    assert empty['total'] is None


def test_reordered_input_does_not_invalidate_recovered_monotonic_series():
    row=diagnostic({BLOCK:([20,0,10],[300,0,100]),POST:([20,0,10],[110,0,10])})
    assert row['total']==pytest.approx(300/110)


def test_pause_classes_are_not_summed_into_over_100_percent():
    from cassini_metrics import GROUPS
    data={name:([0,1_000_000_000],[0,500_000_000]) for name in GROUPS['tx_paused_cycles']}
    row=diagnostic(data,key='tx_pause')
    assert row['total']==pytest.approx(50.)  # Not 400% for eight overlapping classes.
    assert len(row['distribution'])==8


def test_latency_is_bucket_index_without_capture_calibration():
    from cassini_metrics import GROUPS
    data={name:([0,10],[0,100 if i==10 else 0]) for i,name in enumerate(GROUPS['req_rsp_latency'])}
    row=diagnostic(data,key='request_latency')
    assert row['total']==10
    assert row['latency']=={'p99_bin':10,'calibrated':False}
    assert 'Physical latency unavailable' in row['reason']


def test_latency_calibration_reports_bounds_not_false_exact_p99(tmp_path):
    from cassini_metrics import GROUPS
    data={name:([0,10],[0,100 if i==10 else 0]) for i,name in enumerate(GROUPS['req_rsp_latency'])}
    path=tmp_path/'calibration.json'
    path.write_text(json.dumps({'latency_histograms':{'pct_req_rsp_latency':{'run_id':'numeric-test','edges_ns':list(range(0,320,10))+[None],'source':'Explicit synthetic calibration used only in this test'}}}))
    e=ExplorerEngine(store_for(data),load_analysis_config(path))
    row=e.query(Query(e.origin,e.end,metric='request_latency'))['records'][0]
    assert row['latency']['p99_lower_ns']==100
    assert row['latency']['p99_upper_ns']==110
    assert row['latency']['mean_midpoint_estimate_ns']==105


def test_bad_octet_ambiguous_record_has_complete_card_schema():
    store=store_for({'hni_sts_rx_bad_octets':([0,10],[0,0]),'hni_sts_tx_bad_octets':([0,10],[0,0])},collector_names=('first','second'))
    e=ExplorerEngine(store);view=e.query(Query(e.origin,e.end,metric='octet_errors'))
    assert view['records'][0]['state']=='Choose collector'
    assert view['records'][0]['observed_counter_count']==0
    assert len(cards(view))==4


def test_cpu_migration_does_not_create_collectors_and_threads_stay_separate():
    trace=SimpleNamespace(uuid='capture-uuid',name=None)
    a=SimpleNamespace(id=1,name='cpu1',trace=trace)
    b=SimpleNamespace(id=105,name='cpu105',trace=trace)
    event=SimpleNamespace(common_context_field={'vpid':10,'vtid':12})
    assert _event_collector_process(event,a)==_event_collector_process(event,b)
    assert _event_collector_process(SimpleNamespace(common_context_field={'_vpid':10,'_vtid':12}),a)==_event_collector_process(event,a)
    assert _event_collector_process(SimpleNamespace(common_context_field={'vpid':10,'vtid':13}),a)!=_event_collector_process(event,a)


def test_all_derived_metrics_on_missing_capture_have_explicit_states():
    e=ExplorerEngine(store_for({'hni_sts_tx_ok_octets':([0,10],[0,100])}))
    for metric in METRICS:
        q=Query(e.origin,e.end,metric=metric)
        view=e.query(q)
        assert view['records'][0]['total'] is None
        assert len(cards(view))==4
        json.dumps(e.export(q),allow_nan=False)


def test_balance_excludes_missing_not_observed_idle():
    row=balance_rows([{'total':100},{'total':100},{'total':0},{'total':None}])
    assert row['active']==2 and row['observed']==3 and row['selected']==4
    assert row['effective']==2
    assert row['max_min']=='∞'
    assert row['largest_share']==.5


def test_latency_calibration_cannot_leak_to_a_different_run():
    from cassini_metrics import GROUPS
    data={name:([0,10],[0,100 if i==10 else 0]) for i,name in enumerate(GROUPS['req_rsp_latency'])}
    config={'latency_histograms':{'pct_req_rsp_latency':{'run_id':'another-run','edges_ns':list(range(0,320,10))+[None],'source':'Other capture'}}}
    e=ExplorerEngine(store_for(data),config)
    row=e.query(Query(e.origin,e.end,metric='request_latency'))['records'][0]
    assert not row['latency']['calibrated']
    assert 'does not match' in row['latency']['calibration_warning']


def test_explicit_topology_groups_do_not_average_into_fake_pooled_ratio():
    from explorer.diagnostic_ui import topology_distribution
    engine=SimpleNamespace(analysis_config={'topology':{'node-a':{'rack':'r1'},'node-b':{'rack':'r1'}}})
    rows=[{'node':'node-a','total':1.0},{'node':'node-b','total':9.0},{'node':'unmapped','total':99.0}]
    result=topology_distribution({'ranked':rows},engine)
    assert result==[{'level':'rack','group':'r1','selected_nics':2,'observed_nics':2,'minimum':1.0,'median':5.0,'maximum':9.0}]
