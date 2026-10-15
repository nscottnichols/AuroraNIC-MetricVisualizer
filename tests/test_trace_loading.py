"""Path confinement, real directory loading, binary pftrace, and empty-start tests."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from dataset_lifecycle import DatasetLifecycleManager, DatasetDescriptor
from explorer.trace_browser import TracePathControls
from explorer.app import create_explorer_app
from explorer.demo import make_demo_store
from trace_io import trace_arguments, resolve_trace_path, trace_fingerprint
from trace_cache import read_trace_store_cache, write_trace_store_cache
from main import load_trace_store, parse_args


def write_pftrace(path):
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace
    trace=Trace()
    for uid,name,parent,counter in [(1,'Hostname node-a',0,False),(2,'Sampling',1,False),(3,'NIC',2,False),(4,'Interface cxi0',3,False),
            (5,'parbs_tarb_pi_posted_pkts',4,True),(6,'parbs_tarb_pi_posted_blocked_cnt',4,True)]:
        packet=trace.packet.add();packet.trusted_packet_sequence_id=7
        td=packet.track_descriptor;td.uuid=uid;td.name=name
        if parent:td.parent_uuid=parent
        if counter:td.counter.SetInParent()
    for tid,values in [(5,[10,20,30]),(6,[100,200,300])]:
        for time,value in zip((100,200,300),values):
            packet=trace.packet.add();packet.trusted_packet_sequence_id=7;packet.timestamp=time
            te=packet.track_event;te.type=4;te.track_uuid=tid;te.double_counter_value=value
    path.write_bytes(trace.SerializeToString());return path


def test_native_pftrace_needs_no_trace_processor_and_groups_counter_sources(tmp_path,monkeypatch):
    from explorer.model import ExplorerEngine,Query
    import pftrace_adapter
    monkeypatch.setattr(pftrace_adapter,'adapt_perfetto_trace_processor',lambda *a,**k:pytest.fail('No external Trace Processor may run'))
    source=write_pftrace(tmp_path/'timeline.pftrace')
    store=load_trace_store(trace_arguments(source,root=tmp_path))
    assert store.row_count==6 and len(store.collectors)==1
    assert all(c.baseline_relative for c in store.counters)
    e=ExplorerEngine(store);row=e.query(Query(e.origin,e.end,metric='posted_pushback'))['records'][0]
    assert row['total']==10
    assert store.identity.reader_version=='perfetto-protobuf-thapi/1'
    # Cold and warm imports share the validated process/counter identity.
    second=load_trace_store(trace_arguments(source,root=tmp_path))
    assert second.identity==store.identity


def test_outdated_pipeline_cache_is_rebuilt_from_source(tmp_path,capsys):
    from trace_io import PIPELINE_VERSION
    source=write_pftrace(tmp_path/'timeline.pftrace')
    args=trace_arguments(source,root=tmp_path)
    store=load_trace_store(args)
    outdated=replace(store,coverage=replace(store.coverage,notes=('nicviz-input/0',)))
    write_trace_store_cache(args.trace_cache,outdated,store.identity.source_fingerprint)
    capsys.readouterr()

    rebuilt=load_trace_store(args)

    assert rebuilt.identity==store.identity and rebuilt.row_count==store.row_count
    assert PIPELINE_VERSION in rebuilt.coverage.notes
    assert 'Trace cache is stale; rebuilding' in capsys.readouterr().err
    published=read_trace_store_cache(args.trace_cache,store.identity.source_fingerprint)
    assert published.coverage==rebuilt.coverage


def test_path_picker_opens_file_and_retains_last_good_on_failure(tmp_path):
    source=write_pftrace(tmp_path/'timeline.pftrace')
    with DatasetLifecycleManager(tmp_path,[]) as manager:
        picker=TracePathControls(manager)
        token=picker.open_path(str(source));assert manager.wait(token,10).state.value=='ready'
        original=manager.current()
        bad=tmp_path/'broken.pftrace';bad.write_bytes(b'not protobuf')
        failure=picker.open_path(str(bad));assert manager.wait(failure,10).state.value=='failed'
        assert manager.current() is original


def test_lifecycle_accepts_ctf_directory_not_only_files(tmp_path):
    ctf=tmp_path/'ctf';ctf.mkdir();(ctf/'metadata').write_text('fixture schema')
    args=trace_arguments(ctf,root=tmp_path,cache=False)
    expected=object()
    with DatasetLifecycleManager(tmp_path,[DatasetDescriptor('ctf',ctf,None,args)],loader=lambda args,cancel:expected) as manager:
        token=manager.open('ctf');assert manager.wait(token,5).state.value=='ready'
        assert manager.current().store is expected


def test_browser_cannot_escape_root_via_absolute_relative_or_symlink(tmp_path):
    root=tmp_path/'approved';root.mkdir();outside=tmp_path/'outside';outside.mkdir()
    (root/'escape').symlink_to(outside,target_is_directory=True)
    for value in ['../outside',str(outside),'escape']:
        with pytest.raises(ValueError,match='outside'):
            resolve_trace_path(root,value)
    with DatasetLifecycleManager(root,[]) as manager:
        options,folder,_=TracePathControls(manager).listing(str(root))
        assert options==[]
        assert folder==str(root)


def test_fingerprint_detects_changed_contents_and_rejects_internal_symlinks(tmp_path):
    root=tmp_path/'ctf';root.mkdir();source=root/'metadata';source.write_text('a')
    before=trace_fingerprint(root);source.write_text('b');assert trace_fingerprint(root)!=before
    target=tmp_path/'external';target.write_text('external')
    (root/'stream').symlink_to(target)
    with pytest.raises(ValueError,match='symlinks'):trace_fingerprint(root)


def test_empty_start_is_a_file_picker_not_example_data(tmp_path):
    with DatasetLifecycleManager(tmp_path,[]) as manager:
        app=create_explorer_app(None,dataset_manager=manager,trace_browser=True)
        client=app.server.test_client()
        assert client.get('/healthz').status_code==200
        assert client.get('/readyz').status_code==503
        layout=client.get('/_dash-layout').get_data(as_text=True)
        assert 'Start with your trace' in layout
        assert 'ex-file-path' in layout and 'Open selected trace' in layout
        assert 'synthetic' not in layout.lower()
        assert client.post('/_dash-update-component',headers={'Sec-Fetch-Site':'cross-site'},json={}).status_code==403


def test_minimal_trace_cli_infers_provenance_and_keeps_explicit_values(tmp_path):
    source=write_pftrace(tmp_path/'timeline.pftrace')
    args=parse_args(['--trace',str(source)])
    assert args.trace_format=='pftrace'
    assert args.trace_source_fingerprint=='auto'
    assert args.trace_reader_version=='auto'
    args=parse_args(['--trace',str(source),'--trace-run-id','specified-run','--trace-synchronization-evidence','explicit capture evidence'])
    assert args.trace_run_id=='specified-run'
    assert args.trace_synchronization_evidence=='explicit capture evidence'
