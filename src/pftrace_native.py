"""Streaming THAPI Perfetto protobuf reader; no trace-processor binary or download.

Uses Perfetto's maintained generated protobuf schema from its Python package.
Only THAPI CXI counter tracks with explicit absolute timestamps are accepted.
Other track families are deliberately ignored. Unsupported counter encodings fail
with a useful error instead of guessing clock/track defaults.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace

VERSION = 'perfetto-protobuf-thapi/1'
MAX_PACKET_BYTES = 64 * 1024 * 1024
MAX_COUNTER_ROWS = 10_000_000


def _varint(source):
    result=0
    for shift in range(0,70,7):
        byte=source.read(1)
        if not byte:
            if shift==0:return None
            raise ValueError('Truncated Perfetto varint.')
        n=byte[0];result|=(n&127)<<shift
        if not n&128:return result
    raise ValueError('Invalid Perfetto varint.')


def packets(path, should_cancel=None):
    try:
        from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TracePacket
    except ImportError as error:
        raise ImportError("Install the pftrace extra: python -m pip install -e '.[pftrace]'. No Perfetto server is required.") from error
    with Path(path).open('rb') as source:
        count=0
        while (tag:=_varint(source)) is not None:
            if tag!=10:raise ValueError('Expected binary Perfetto Trace packets, not JSON or a compressed archive.')
            size=_varint(source)
            if size is None or size>MAX_PACKET_BYTES:raise ValueError('Invalid/oversized Perfetto packet.')
            payload=source.read(size)
            if len(payload)!=size:raise ValueError('Truncated Perfetto packet.')
            packet=TracePacket();packet.ParseFromString(payload)
            yield packet
            count+=1
            if count%1024==0 and should_cancel and should_cancel():
                raise ValueError('pftrace import cancelled.')


def load_pftrace(path, config, *, should_cancel=None):
    from pftrace_adapter import pftrace_rows_to_normalized_batch
    descriptors={}; rows=[]
    for packet in packets(path,should_cancel):
        sequence=int(packet.trusted_packet_sequence_id)
        if packet.HasField('track_descriptor'):
            td=packet.track_descriptor
            if td.HasField('counter') and (td.counter.is_incremental or td.counter.unit_multiplier not in (0,1)):
                raise ValueError('Incremental or scaled Perfetto counters require a Trace Processor import.')
            signature=(td.name,int(td.parent_uuid),td.HasField('counter'))
            if td.uuid in descriptors and descriptors[td.uuid][0]!=signature:
                raise ValueError('Conflicting Perfetto track descriptors.')
            descriptors[int(td.uuid)]=(signature,sequence)
        if packet.HasField('track_event') and packet.track_event.type==4:
            te=packet.track_event
            if not packet.HasField('timestamp') or not te.HasField('track_uuid'):
                raise ValueError('Counter track/timestamp defaults are not supported by the THAPI reader.')
            if packet.HasField('timestamp_clock_id') and packet.timestamp_clock_id not in (0,6):
                raise ValueError('This Perfetto clock requires explicit clock-snapshot conversion.')
            if te.HasField('double_counter_value'):value=float(te.double_counter_value)
            elif te.HasField('counter_value'):value=float(te.counter_value)
            else:raise ValueError('Counter sample has no numeric value.')
            rows.append({'ts':int(packet.timestamp),'track_id':int(te.track_uuid),'value':value})
            if len(rows)>MAX_COUNTER_ROWS:raise ValueError('Trace exceeds the ten-million-counter-sample limit.')
    tracks={};accepted=set()
    for tid,((name,parent,is_counter),sequence) in descriptors.items():
        if not is_counter:continue
        chain=[]; current=parent;seen={tid}
        while current:
            if current in seen or current not in descriptors:raise ValueError('Invalid Perfetto track ancestry.')
            seen.add(current);chain.append(current);current=descriptors[current][0][1]
        if not chain:continue
        nic_id=chain[0];nic_name=descriptors[nic_id][0][0]
        if not nic_name.startswith('Interface ') or not any(descriptors[x][0][0] in {'CXI','NIC'} for x in chain):continue
        node_id=chain[-1];node_name=descriptors[node_id][0][0]
        if not node_name.startswith('Hostname '):raise ValueError('THAPI CXI track has no hostname ancestor.')
        tracks[node_id]={'id':node_id,'name':node_name.removeprefix('Hostname '),'parent_id':None,'type':'node'}
        tracks[nic_id]={'id':nic_id,'name':nic_name.removeprefix('Interface '),'parent_id':node_id,'type':'nic'}
        tracks[tid]={'id':tid,'name':name,'parent_id':nic_id,'type':'counter',
                     'collector':{'name':'THAPI timeline writer','rank':None,'process':f'sequence={sequence};node={node_id}'}}
        accepted.add(tid)
    selected=[row for row in rows if row['track_id'] in accepted]
    if not selected:raise ValueError('No THAPI CXI counter samples found in this pftrace.')
    return pftrace_rows_to_normalized_batch({'tracks':list(tracks.values()),'counter_rows':selected},
        replace(config,group_collectors_by_source=True))
