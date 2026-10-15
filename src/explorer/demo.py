"""Deterministic synthetic demo. This is not a Sunspot capture or benchmark result."""
from __future__ import annotations
import numpy as np
from trace_model import (CollectorIdentity, CounterMetadata, CounterSemantics, Coverage,
                         NicIdentity, NodeIdentity, NormalizedBatch, RunIdentity, ValueKind)
from trace_store import TraceStore
from counter_semantic_registry import COUNTER_SEMANTIC_REGISTRY_VERSION


def make_demo_store():
    origin=1_700_000_000_123_456_789
    times=np.arange(1201,dtype=np.int64)*10_000_000
    seconds=times/1e9
    nodes=tuple(NodeIdentity(i,f"node-{chr(97+i)}",f"node-{chr(97+i)}") for i in range(4))
    nics=tuple(NicIdentity(i*4+j,i,f"cxi{j}",f"cxi{j}") for i in range(4) for j in range(4))
    counters=(CounterMetadata(0,"hni_sts_tx_ok_octets",ValueKind.EXACT_UINT64,CounterSemantics.CUMULATIVE,"bytes",False,COUNTER_SEMANTIC_REGISTRY_VERSION),
              CounterMetadata(1,"hni_sts_rx_ok_octets",ValueKind.EXACT_UINT64,CounterSemantics.CUMULATIVE,"bytes",False,COUNTER_SEMANTIC_REGISTRY_VERSION),
              CounterMetadata(2,"hni_tx_paused_0",ValueKind.EXACT_UINT64,CounterSemantics.CUMULATIVE,"cycles",False,"synthetic-demo"),
              CounterMetadata(3,"hni_sts_tx_bad_octets",ValueKind.EXACT_UINT64,CounterSemantics.CUMULATIVE,"bytes",False,COUNTER_SEMANTIC_REGISTRY_VERSION),
              CounterMetadata(4,"hni_sts_rx_bad_octets",ValueKind.EXACT_UINT64,CounterSemantics.CUMULATIVE,"bytes",False,COUNTER_SEMANTIC_REGISTRY_VERSION))
    collectors=(CollectorIdentity(0,"synthetic-observer",rank=0),CollectorIdentity(1,"synthetic-duplicate-observer",rank=1))
    columns={k:[] for k in ('node_ids','nic_ids','counter_ids','collector_ids','timestamps_ns','uint64_values')}
    def append(nic,counter,collector,values):
        for key,val in [('node_ids',nic.node_id),('nic_ids',nic.nic_id),('counter_ids',counter.counter_id),('collector_ids',collector)]:
            columns[key].append(np.full(len(times),val,dtype=np.int64))
        columns['timestamps_ns'].append(origin+times)
        columns['uint64_values'].append(values.astype(np.uint64))
    for nic in nics:
        i=nic.nic_id
        pulse=np.exp(-((seconds-2.5)/.85)**2)+1.4*np.exp(-((seconds-6.0)/1.4)**2)+.65*np.exp(-((seconds-9.6)/.85)**2)
        ripple=1+.10*np.sin(seconds*8+i*.7)
        rate=(.12+1.4*pulse*ripple)*(1+.06*i)*1e9
        if i==9:
            rate*=2.3
        if i==14:
            rate[:]=0
        tx=np.r_[0,np.cumsum(np.rint(rate[1:]*.01).astype(np.uint64))]
        rx=np.r_[0,np.cumsum(np.rint(rate[1:]*(.68+.1*np.cos(seconds[1:]*.5+i))*.01).astype(np.uint64))]
        pause=np.r_[0,np.cumsum(np.rint(np.maximum(rate[1:]/1e9-2,0)*500).astype(np.uint64))]
        tx_bad=np.zeros_like(tx)
        rx_bad=np.zeros_like(rx)
        if i==9:
            tx_bad[1:]=np.cumsum(np.full(len(times)-1,11,dtype=np.uint64))
            rx_bad[1:]=np.cumsum(np.full(len(times)-1,7,dtype=np.uint64))
        append(nic,counters[0],0,tx)
        if i!=15:
            append(nic,counters[1],0,rx)
        append(nic,counters[2],0,pause)
        append(nic,counters[3],0,tx_bad)
        append(nic,counters[4],0,rx_bad)
        if i==15:
            append(nic,counters[0],1,(tx//2).astype(np.uint64))
    columns={k:np.concatenate(v) for k,v in columns.items()}; count=len(columns['timestamps_ns'])
    names=tuple(c.name for c in counters)
    batch=NormalizedBatch(identity=RunIdentity("synthetic-allreduce-demo","sha256:synthetic-allreduce-demo","synthetic-ctf-demo","demo-generator","direct","synthetic-shared-clock","generated single clock; not hardware evidence"),
        nodes=nodes,nics=nics,counters=counters,collectors=collectors,
        coverage=Coverage(requested_counter_names=names,available_counter_names=names,readable_counter_names=names,observed_counter_names=names,acquisition_start_ns=origin,acquisition_end_ns=origin+int(times[-1])),
        **columns,record_ids=np.asarray([f'demo-{i}' for i in range(count)]),float64_values=np.full(count,np.nan),quality_flags=np.zeros(count,dtype=np.int64))
    return TraceStore.from_batches([batch])
