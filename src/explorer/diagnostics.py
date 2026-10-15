"""Window-consistent Cassini diagnostics over physical NIC observation streams.

Single-counter differences are exact before conversion to display floats.
Multi-counter diagnostics use the intersection of valid sample intervals and
piecewise-linear cumulative interpolation (constant interval rates). Ratios are
ratios of integrated counts, never unweighted averages of ratios. Sampling skew
is therefore an explicit estimation assumption, not hidden endpoint mismatch.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from cassini_metrics import GROUPS, OPCODES, DEFINITIONS, HPE_SOURCE
from trace_model import CounterSemantics, QualityFlag, ValueKind
from trace_store import UnknownSeriesError

BINS=240

@dataclass(frozen=True)
class Metric:
    key: str
    family: str
    label: str
    unit: str
    numerator: tuple[str,...]
    denominator: tuple[str,...]=()
    mode: str='ratio'
    scale: float=1.
    distribution: tuple[str,...]=()
    note: str=''


def g(name):return GROUPS[name]

_METRICS=[
 Metric('posted_pushback','Host / PCIe','Posted pushback','cycles/packet',g('pi_posted_blocked'),g('pi_posted_pkts'),note='Blocked posted-path cycles per posted PCIe packet; an indicator of host-side pressure, not proof of a PCIe bottleneck.'),
 Metric('nonposted_pushback','Host / PCIe','Non-posted pushback','cycles/packet',g('pi_nonposted_blocked'),g('pi_nonposted_pkts'),note='Blocked non-posted-path cycles per non-posted PCIe packet. Reads and writes remain separate.'),
 Metric('posted_packet_rate','Host / PCIe','Posted packet rate','packets/s',g('pi_posted_pkts'),mode='rate'),
 Metric('nonposted_packet_rate','Host / PCIe','Non-posted packet rate','packets/s',g('pi_nonposted_pkts'),mode='rate'),
 Metric('host_stalls','Host / PCIe','Host write stalls · busiest path','cycles/s',g('ixe_host_stall'),mode='max_rate',distribution=g('ixe_host_stall'),note='Posted and non-posted credit-stall counters remain separate; the displayed summary is the largest path rate.'),
 Metric('tx_pause','Flow control','TX paused time · busiest class','%',g('tx_paused_cycles'),mode='max_rate',scale=100/1e9,distribution=g('tx_paused_cycles'),note='Per-traffic-class paused cycles / (1 GHz × covered seconds). Maximum class shown; overlapping classes are not summed into wall time.'),
 Metric('rx_pause','Flow control','RX paused time · busiest class','%',g('rx_paused_cycles'),mode='max_rate',scale=100/1e9,distribution=g('rx_paused_cycles'),note='Per-class received pause time under the sourced 1 GHz counter clock. This does not measure application stall time.'),
 Metric('tx_xoff_rate','Flow control','TX XOFF events','events/s',g('pause_xoff_sent'),mode='rate',distribution=g('pause_xoff_sent')),
 Metric('rx_pause_rate','Flow control','Received pause events','events/s',g('pause_recv'),mode='rate',distribution=g('pause_recv')),
 Metric('fgfc_xoff','Flow control','Fabric XOFF events','events/s',('hni_fgfc_event_xoff',),mode='rate'),
 Metric('tx_stalls','Flow control','TX buffer stalls · busiest counter','cycles/s',g('oxe_tx_stall'),mode='max_rate',distribution=g('oxe_tx_stall'),note='Maximum individual TX buffer stall rate; counters are not summed into a wall-clock percentage.'),
 Metric('dispatch_stalls','Host / PCIe','Receive dispatch stalls · busiest cause','cycles/s',g('ixe_disp_stall'),mode='max_rate',distribution=g('ixe_disp_stall'),note='Largest per-cause receive-dispatch stall rate; overlapping causes are not summed into wall time.'),
 Metric('pause_sent_rate','Flow control','Pause frames sent','events/s',('hni_pause_sent',),mode='rate'),
 Metric('pause_refresh_rate','Flow control','Pause refresh events','events/s',('hni_pause_refresh',),mode='rate'),
 Metric('fabric_discard_rate','Flow control','Fabric flow-control discards','events/s',('hni_fgfc_discard',),mode='rate'),
 Metric('priority_fraction','Matching','Immediate priority matches','%',g('priority_match'),g('priority_match')+g('overflow_match'),scale=100,distribution=g('priority_match')+g('overflow_match'),note='Priority / (priority + overflow). Zero total matches makes this undefined.'),
 Metric('overflow_fraction','Matching','Overflow matches','%',g('overflow_match'),g('priority_match')+g('overflow_match'),scale=100,distribution=g('priority_match')+g('overflow_match'),note='Overflow / (priority + overflow); matching behavior is not an MPI protocol classification.'),
 Metric('tx_frame_bytes','Frames','Mean transmitted frame size','B/frame',g('bytes_transferred'),g('tx_frame_sizes'),distribution=g('tx_frame_sizes'),note='OK network-frame octets / OK wire-frame count. This is not MPI message size or application payload size.'),
 Metric('rx_frame_bytes','Frames','Mean received frame size','B/frame',g('bytes_received'),g('rx_frame_sizes'),distribution=g('rx_frame_sizes'),note='OK received octets / OK wire frames, not MPI messages.'),
 Metric('tx_small_frames','Frames','Small transmitted frames (<256 B)','%',g('tx_frame_sizes')[:6],g('tx_frame_sizes'),scale=100,distribution=g('tx_frame_sizes')),
 Metric('tx_large_frames','Frames','Large transmitted frames (≥4096 B)','%',g('tx_frame_sizes')[-2:],g('tx_frame_sizes'),scale=100,distribution=g('tx_frame_sizes')),
 Metric('atu_misses_per_request','Translation','Cache misses per translation request','misses/request',g('atu_miss_by_client'),g('atu_client_req'),distribution=g('atu_miss_by_client'),note='Client-counted misses / client translation requests. Pool and client miss populations are not added together; this ratio is not asserted to be a probability.'),
 Metric('atu_hit_fraction','Translation','Translation cache hit fraction','%',g('atu_hits'),g('atu_hits')+g('atu_miss_by_pool'),scale=100,distribution=g('atu_miss_by_pool')),
 Metric('atu_misses_per_GB','Translation','Translation misses per observed GB','misses/GB',g('atu_miss_by_pool'),g('bytes_transferred')+g('bytes_received'),scale=1e9,distribution=g('atu_miss_by_pool'),note='Pool-counted misses normalized by local TX + RX network-frame octets (decimal GB); not unique job payload.'),
 Metric('atu_request_rate','Translation','Translation request rate','requests/s',g('atu_client_req'),mode='rate',distribution=g('atu_client_req')),
 Metric('atu_eviction_rate','Translation','Translation cache evictions','events/s',g('atu_evictions'),mode='rate'),
 Metric('dma_share','Operations','DMA share of commands','%',g('cq_num_dma_cmds'),g('cq_num_dma_cmds')+g('cq_num_idc_cmds'),scale=100,distribution=g('cq_num_dma_cmds')+g('cq_num_idc_cmds'),note='Control-path command mix only. DMA / IDC does not establish eager / rendezvous MPI protocol.'),
 Metric('put_share','Operations','PUT command share','%',('cq_dma_cmd_counts_1',),g('cq_dma_cmd_counts'),scale=100,distribution=g('cq_dma_cmd_counts')),
 Metric('get_share','Operations','GET command share','%',('cq_dma_cmd_counts_2',),g('cq_dma_cmd_counts'),scale=100,distribution=g('cq_dma_cmd_counts')),
 Metric('rendezvous_share','Operations','Rendezvous PUT command share','%',('cq_dma_cmd_counts_3',),g('cq_dma_cmd_counts'),scale=100,distribution=g('cq_dma_cmd_counts'),note='Hardware command opcode mix, not a reconstruction of software protocol or MPI message sizes.'),
 Metric('small_command_share','Operations','Small-message command share','%',('cq_dma_cmd_counts_7',),g('cq_dma_cmd_counts'),scale=100,distribution=g('cq_dma_cmd_counts')),
 Metric('request_latency','Latency','Request / response latency distribution','bin index',g('req_rsp_latency'),mode='histogram',distribution=g('req_rsp_latency'),note='Cumulative histogram bin deltas. Without capture-specific bin edges, P99 is a bucket index, not nanoseconds.'),
 Metric('host_latency','Latency','Host-access latency distribution','bin index',g('host_access_latency'),mode='histogram',distribution=g('host_access_latency'),note='Requires capture-specific bin edges for physical latency bounds; no example/default calibration is assumed.'),
 Metric('reliability_events','Reliability','Reliability event increments','increments/s',g('pct_errors')+tuple(x for x in g('link_reliability') if x!='hni_pcs_good_cw')+g('discards'),mode='rate',distribution=g('pct_errors')+g('link_reliability')+g('discards'),note='Sum of named event-counter increments, not unique failures. Good codewords are context, never errors; corrected and uncorrected events remain individually visible.'),
]
METRICS={x.key:x for x in _METRICS}


def metric_options(engine):
    from .metadata import available_options
    return available_options(engine)


def _finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def signal(result, start, end):
    """Full-resolution valid complete intervals; preserves integer differences."""
    t=result.timestamps_ns; v=result.values
    mask=(t>=start)&(t<=end); count=int(mask.sum())
    out={'name':result.counter.name,'unit':result.counter.unit,'samples':count,
         'total':None,'start_ns':None,'end_ns':None,'baseline_relative':result.counter.baseline_relative,
         'flags':[], 'starts':np.array([],dtype=np.int64),'ends':np.array([],dtype=np.int64),'amounts':np.array([],dtype=float)}
    if result.counter.semantics is not CounterSemantics.CUMULATIVE:
        out['flags'].append('Unknown counter semantics');return out
    if len(t)<2:return out
    if result.counter.value_kind is ValueKind.FLOAT64 and np.any(np.abs(v)>2**53):
        out['flags'].append('Double precision cannot guarantee unit increments');return out
    dt=t[1:]-t[:-1]
    valid=(dt>0)&(v[1:]>=v[:-1])&np.isfinite(v[1:])&np.isfinite(v[:-1])&(v[:-1]>=0)
    badmask=int(QualityFlag.GAP)
    valid &= ((result.quality_flags[1:]|result.quality_flags[:-1])&badmask)==0
    conflict=np.flatnonzero((dt==0)&(v[1:]!=v[:-1]))
    for i in conflict:valid[max(0,i-1):min(len(valid),i+2)]=False
    if np.any(v[1:]<v[:-1]):out['flags'].append('Counter decreases excluded (reset/wrap not guessed)')
    if len(conflict):out['flags'].append('Conflicting duplicate timestamps excluded')
    if not result.counter.baseline_relative and np.any(dt>0):
        cadence=float(np.median(dt[dt>0])); gaps=dt>5*cadence
        if np.any(gaps):out['flags'].append('Intervals >5× median sampling cadence excluded')
        valid &= ~gaps
    valid &= (t[:-1]>=start)&(t[1:]<=end)
    # uint64 subtraction happens before conversion; large absolute baselines stay exact.
    delta=(v[1:]-v[:-1])[valid]
    if len(delta):
        out['total']=str(sum(int(x) for x in delta)) if result.counter.value_kind is ValueKind.EXACT_UINT64 else format(float(delta.sum()),'.17g')
        out.update(starts=t[:-1][valid]-start,ends=t[1:][valid]-start,amounts=delta.astype(float),
                   start_ns=str(int(t[:-1][valid][0])),end_ns=str(int(t[1:][valid][-1])))
    return out


def aligned(signals, duration):
    """Integrate all counters on exactly the same intersection of valid supports."""
    if not signals or any(not len(s['starts']) for s in signals):return None
    edges=np.unique(np.concatenate([np.array([0,duration],dtype=np.int64),*[s[k] for s in signals for k in ('starts','ends')]]))
    left,right=edges[:-1],edges[1:];mids=left+(right-left)//2
    valid=np.ones(len(left),dtype=bool); rates=[]
    for s in signals:
        ids=np.searchsorted(s['starts'],mids,side='right')-1
        safe=np.clip(ids,0,len(s['starts'])-1)
        ok=(ids>=0)&(mids<s['ends'][safe])
        valid &= ok
        rates.append(s['amounts'][safe]/(s['ends'][safe]-s['starts'][safe]))  # counts/ns
    return left[valid],right[valid],np.asarray(rates)[:,valid]


def _evaluate(spec, counts, duration_ns, names):
    """Operate on columns of integrated counts with one shared support duration."""
    num=counts[[names.index(n) for n in spec.numerator]]
    with np.errstate(divide='ignore',invalid='ignore'):
        if spec.mode=='histogram':
            total=num.sum(axis=0); cdf=np.cumsum(num,axis=0)
            return np.where(total>0,np.argmax(cdf>=.99*total,axis=0),np.nan)
        if spec.mode in {'rate','max_rate'}:
            mass=np.max(num,axis=0) if spec.mode=='max_rate' else np.sum(num,axis=0)
            return np.where(duration_ns>0,mass/(duration_ns/1e9)*spec.scale,np.nan)
        den=counts[[names.index(n) for n in spec.denominator]].sum(axis=0)
        return np.where(den>0,num.sum(axis=0)/den*spec.scale,np.nan)


def _distribution_label(name):
    if name.startswith('cq_dma_cmd_counts_'):
        code=int(name.rsplit('_',1)[1]);return f'{code} · {OPCODES.get(code,"Unmapped opcode")}'
    for prefix in ('hni_tx_ok_','hni_rx_ok_'):
        if name.startswith(prefix):return name[len(prefix):].replace('_to_','–')+' B'
    if name.startswith(('pct_req_rsp_latency_','pct_host_access_latency_')):return 'Bin '+name.rsplit('_',1)[1]
    return name


def _latency_bounds(spec, distribution, config, run_id):
    group='pct_req_rsp_latency' if spec.key=='request_latency' else 'pct_host_access_latency'
    calibration=config.get('latency_histograms',{}).get(group)
    values=np.asarray([x['delta'] or 0 for x in distribution],dtype=float)
    if len(values)!=len(spec.numerator) or any(x['delta'] is None for x in distribution) or values.sum()<=0:return {}
    p99=int(np.searchsorted(np.cumsum(values),values.sum()*.99,side='left'))
    output={'p99_bin':p99,'calibrated':False}
    if not calibration:return output
    if calibration['run_id'] != run_id:
        output['calibration_warning']='Calibration run_id does not match this trace; physical latency is not computed.'
        return output
    edges=calibration['edges_ns']; output.update(calibrated=True,calibration_source=calibration['source'],p99_lower_ns=edges[p99],p99_upper_ns=edges[p99+1])
    lows=np.array(edges[:-1],dtype=float); highs=np.array([np.inf if x is None else x for x in edges[1:]])
    output['mean_lower_ns']=float(np.sum(values*lows)/values.sum())
    if np.any((values>0)&~np.isfinite(highs)):
        output['mean_upper_ns']=None;output['mean_midpoint_estimate_ns']=None
    else:
        highs=np.where(values>0,highs,0)
        output['mean_upper_ns']=float(np.sum(values*highs)/values.sum())
        output['mean_midpoint_estimate_ns']=(output['mean_lower_ns']+output['mean_upper_ns'])/2
    return output


def _record(engine, nic, spec, query):
    row={'id':str(nic.nic_id),'nic_id':nic.nic_id,'node_id':nic.node_id,
         'node':engine.nodes[nic.node_id].label,'nic':nic.label,'direction':'Counter','counter':spec.key,
         'state':'Unavailable','reason':'','unit':spec.unit,'total':None,'mean':None,'p50':None,'p99':None,'cv':None,
         'coverage':0.,'collector_id':None,'sample_count':0,'flags':[],
         'raw_x':[],'raw_y':[],'sources':[],'distribution':[],'numerator':None,'denominator':None}
    options=engine.collector_options(nic.nic_id);collector=dict(query.collectors).get(nic.nic_id)
    if collector is None:
        if len(options)!=1:
            row.update(state='Choose collector',reason='Choose one process/thread observation source; streams are not summed.');return row
        collector=int(options[0]['value'])
    row['collector_id']=collector
    names=tuple(dict.fromkeys(spec.numerator+spec.denominator));signals={}; missing=[]
    for name in dict.fromkeys(names+spec.distribution):
        if name not in engine.counters:missing.append(name);continue
        try:
            results=engine.store.query_series(nic.node_id,nic.nic_id,name).results
        except UnknownSeriesError:
            missing.append(name);continue
        result=next((r for r in results if r.collector.collector_id==collector),None)
        if result is None:missing.append(name);continue
        s=signal(result,query.start_ns,query.end_ns);signals[name]=s
        row['sources'].append({k:v for k,v in s.items() if k not in {'starts','ends','amounts'}})
        row['sample_count']+=s['samples'];row['flags'].extend(s['flags'])
    row['flags']=sorted(set(row['flags']))
    for name in spec.distribution:
        s=signals.get(name)
        row['distribution'].append({'counter':name,'label':_distribution_label(name),
           'delta':float(s['total']) if s and s['total'] is not None else None,
           'exact_delta':s['total'] if s else None,'unit':DEFINITIONS[name].unit,
           'state':'Missing' if s is None else 'No valid intervals' if s['total'] is None else 'Observed zero' if float(s['total'])==0 else 'Observed nonzero'})
    absent=[name for name in names if name not in signals]
    if absent:
        row['reason']='Required observations missing: '+', '.join(absent[:4])+(' …' if len(absent)>4 else '')
        if 'pftrace' in engine.store.identity.source_format:row['reason']+='; change-only absence is not proof of zero.'
        return row
    common=aligned([signals[n] for n in names],query.end_ns-query.start_ns)
    if common is None or not len(common[0]):
        row['reason']='No common valid complete intervals. Widen the window or inspect missing/reset counters.';return row
    starts,ends,rates=common; widths=ends-starts
    mass=rates*widths;total_counts=mass.sum(axis=1);covered=int(widths.sum())
    value=_finite(_evaluate(spec,total_counts,covered,names))
    row.update(total=value,mean=value,coverage=covered/(query.end_ns-query.start_ns),
               state='Observed' if value is not None else 'Undefined',
               reason='Ratios use integrated counts on shared valid support; asynchronous samples are linearly aligned.',
               aligned_support_ns=covered,alignment='piecewise-linear cumulative interpolation on intersection of valid intervals')
    for source in row['sources']:
        if source['name'] in names:
            source['aligned_delta']=float(total_counts[names.index(source['name'])])
            source['common_support_ns']=covered
    for item in row['distribution']:
        if item['counter'] in names:
            item['delta']=float(total_counts[names.index(item['counter'])])
            item['alignment']='shared support estimate'
    row['numerator']=float(total_counts[[names.index(n) for n in spec.numerator]].sum())
    if spec.denominator:
        row['denominator']=float(total_counts[[names.index(n) for n in spec.denominator]].sum())
        if row['denominator']==0:row['reason']='Zero denominator on the common support; the ratio is undefined, not zero.'
    if row['coverage']<.999 and value is not None:row['state']='Partial'
    if any(s['baseline_relative'] for s in signals.values()):
        row['reason']+=' Change-only pftrace: interval averages are not native-cadence activity; hidden initial samples and idle tails are unavailable.'
    # Reuse the validated integral primitive; all time arithmetic is relative int64.
    from .model import _integral_at
    edges=np.linspace(0,query.end_ns-query.start_ns,min(BINS,query.end_ns-query.start_ns)+1,dtype=np.int64)
    display_counts=np.asarray([np.diff(_integral_at(edges,starts,ends,amount)) for amount in mass])
    display_covered=np.diff(_integral_at(edges,starts,ends,widths))
    y=np.asarray(_evaluate(spec,display_counts,display_covered,names),dtype=float)
    y[display_covered<np.diff(edges)-.5]=np.nan
    row['raw_x']=((edges[:-1]+query.start_ns-engine.origin)/1e9).tolist()
    row['raw_y']=[_finite(x) for x in y]
    if spec.mode=='histogram':
        # Use aligned bin deltas, not disjoint support totals, for distribution statistics.
        for item in row['distribution']:
            if item['counter'] in names:item['delta']=float(total_counts[names.index(item['counter'])]);item['alignment']='shared support estimate'
        row['latency']=_latency_bounds(spec,row['distribution'],getattr(engine,'analysis_config',{}),engine.store.identity.run_id)
        if not row['latency'].get('calibrated'):row['reason']+=' Physical latency unavailable until bin calibration is provided.'
    return row


def build_view(engine, query):
    spec=METRICS[query.metric]
    nics=[n for n in engine.store.nics if (not query.node_ids or n.node_id in query.node_ids) and (not query.nic_ids or n.nic_id in query.nic_ids)]
    records=[_record(engine,n,spec,query) for n in nics]
    return {'query':query,'duration_s':(query.end_ns-query.start_ns)/1e9,'records':records,
       'ranked':sorted(records, key=lambda r: (r['total'] is None, -(r['total'] or 0), r['node'], r['nic'])),'node_count':len({n.node_id for n in nics}),'nic_count':len(nics),
       'origin_ns':engine.origin,'run':asdict(engine.store.identity),'counter':query.metric,
       'source_rows':engine.store.row_count,'diagnostic':asdict(spec),
       'formula':('max(counter deltas / common seconds)' if spec.mode=='max_rate' else 'counter delta sum / common seconds' if spec.mode=='rate' else 'P99 bucket from aligned histogram counts' if spec.mode=='histogram' else 'sum(aligned numerator deltas) / sum(aligned denominator deltas)')+f' × {spec.scale:g}',
       'method':'Complete valid sample intervals; common-support linear cumulative alignment. No reset bridging or zero filling.',
       'source_reference':HPE_SOURCE}
