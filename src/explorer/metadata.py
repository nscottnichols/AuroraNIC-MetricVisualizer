"""Display metadata, not a counter-semantic or acquisition registry.

The supplied catalog is a superset. Selectors only expose encountered series.
Unverified descriptions are explicitly name-derived; they never authorize math.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import re
from dash import html
from cassini_metrics import DEFINITIONS, GROUPS, HPE_SOURCE

FAMILIES = {'atu':'Address translation', 'cq':'Command queues', 'ee':'Event engine',
 'hni':'Network interface', 'ixe':'Inbound engine', 'lpe':'List processing',
 'mb':'MB hardware block', 'mst':'Message tracking', 'oxe':'Outbound engine',
 'parbs':'PCIe arbitration', 'pct':'Packet tracking', 'pi':'Processor interface',
 'rmu':'Resource management', 'tou':'Triggered operations'}
WORDS = {'atu':'ATU','cq':'CQ','hni':'HNI','ixe':'IXE','lpe':'LPE','oxe':'OXE','pct':'PCT',
 'pi':'PI','ee':'EE','rmu':'RMU','tou':'TOU','dma':'DMA','idc':'IDC','pcie':'PCIe',
 'tx':'TX','rx':'RX','ats':'ATS','odp':'ODP','cntr':'counter','cnt':'count','pkts':'packets',
 'req':'requests','reqs':'requests','rsp':'responses','resps':'responses','cdt':'credits',
 'cdts':'credits','err':'error','errs':'errors','sts':'status','num':'number','ftch':'fetch',
 'np':'non-posted','p':'posted','cxi':'CXI','llr':'link retry','bcast':'broadcast',
 'ucor':'uncorrectable','cor':'correctable','inval':'invalidation','ptl':'Portals'}
GROUP_TEXT = {
 'bytes_transferred':('Transmitted OK frame octets','Counts transmitted OK network-frame octets. An octet is one byte; this is not application payload.'),
 'bytes_received':('Received OK frame octets','Counts received OK network-frame octets; not MPI payload or message size.'),
 'pi_posted_pkts':('Posted PCIe packets','Packets on the posted host-interface path; compare interval increments with posted blocked cycles.'),
 'pi_nonposted_pkts':('Non-posted PCIe packets','Packets on the non-posted host-interface path; compare with its own blocked-cycle counter.'),
 'pi_posted_blocked':('Posted path blocked cycles','Cycles blocked on the posted host path. Divide the aligned increment by posted packets to obtain cycles per packet, not a percentage.'),
 'pi_nonposted_blocked':('Non-posted path blocked cycles','Cycles blocked on the non-posted host path. Use the corresponding packet counter; do not mix the two paths.'),
 'priority_match':('Priority-list matches','Messages matched on the priority list. Compare with overflow-list matches on the same window and population.'),
 'overflow_match':('Overflow-list matches','Messages delivered via the overflow list rather than a priority match; can indicate receive-posting pressure, not a definitive root cause.'),
 'tx_paused_cycles':('TX paused cycles','Transmit pause duration by traffic class. Classes may overlap and must not be added into wall time.'),
 'rx_paused_cycles':('RX paused cycles','Receive pause duration by traffic class. This is not application stall time.'),
 'tx_frame_sizes':('TX frame-size bin','Count of OK transmitted wire frames in this size bin. Wire-frame sizes are not MPI message sizes.'),
 'rx_frame_sizes':('RX frame-size bin','Count of OK received wire frames in this size bin. Do not infer software message size.'),
 'req_rsp_latency':('Request/response latency bin','Cumulative histogram bin observations. Capture-specific bin edges are required for physical latency.'),
 'host_access_latency':('Host-access latency bin','Cumulative host-latency histogram. A bin index is not a duration without calibration.'),
 'atu_miss_by_pool':('Translation misses by pool','Translation-cache misses partitioned by pool. Do not also sum the client view of misses.'),
 'atu_miss_by_client':('Translation misses by client','Client view of translation misses. Not additive with the pool view.'),
 'atu_hits':('Translation-cache hits','Cache-hit observations partitioned by page-size category and pool.'),
 'atu_evictions':('Translation-cache evictions','Translation entries evicted. Compare with requests, misses, and traffic; correlation is not causation.'),
 'atu_client_req':('Translation requests','Requests to the translation unit by client. Useful as a denominator for the matching miss population.'),
 'cq_dma_cmd_counts':('DMA command opcode','Count of a hardware DMA command opcode. Control-path commands do not directly determine bytes or MPI protocols.'),
 'cq_num_dma_cmds':('DMA commands','Control-path DMA command count. Compare with IDC command counts, not data-path bytes.'),
 'cq_num_idc_cmds':('IDC commands','Control-path immediate-data command count. Does not alone establish eager/rendezvous software protocol.'),
 'bad_octets':('Bad frame octets','Observed bad TX/RX octets. Use increments in the selected window, not lifetime register values.'),
}

@lru_cache(maxsize=1)
def catalog():
    path=Path(__file__).with_name('data')/'counter-catalog.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def readable(name):
    tokens=name.split('_')
    expanded=' '.join(WORDS.get(t,t) for t in tokens).strip()
    return expanded[:1].upper()+expanded[1:]


def describe_counter(name, observed=None):
    definition=DEFINITIONS.get(name)
    group=definition.group if definition else None
    label, description=GROUP_TEXT.get(group, (readable(name),
        f'Hardware observation named {name}. The readable label is expanded from its identifier; exact behavior requires the hardware documentation.'))
    if group in {'tx_frame_sizes','rx_frame_sizes'}:
        label=('TX' if group=='tx_frame_sizes' else 'RX')+' frame size · '+name.split('_ok_',1)[1].replace('_to_','–').replace('_',' ')+' B'
    elif group=='atu_hits':label+=' · '+name.split('atu_cache_hit_',1)[1].replace('_',' ')
    elif group=='bad_octets':label=('TX' if '_tx_' in name else 'RX')+' bad frame octets'
    elif group in GROUP_TEXT and len(GROUPS.get(group,()))>1:
        suffix=name.split('_')[-1]
        label+=' · '+suffix
    result={'name':name,'label':label,'family':FAMILIES.get(name.split('_')[0],'Other counters'),
        'description':description,'unit':definition.unit if definition else 'unverified native units',
        'provenance':'Curated family description' if group in GROUP_TEXT else 'Name-derived label; semantics unverified',
        'source':definition.source if definition else 'User-supplied available-cxi-counters.txt (name only)',
        'diagnostic':'Compare changes on the same physical NIC and interval. Do not infer root cause, thresholds, rates, or percentages from the name alone.',
        'known_name':name in catalog()}
    # Curators can improve the shipped map without changing numerical semantics.
    stored=catalog().get(name,{})
    for key in ('label','family','description','unit','provenance','source','diagnostic'):
        if isinstance(stored.get(key),str):result[key]=stored[key]
    if observed is not None:
        result['unit']=observed.unit or result['unit']
        result['semantics']=observed.semantics.value
        result['baseline_relative']=observed.baseline_relative
    return result


def available_options(engine, favorites=(), family='all'):
    from .diagnostics import METRICS
    seen=set(engine.counters)
    opts=[]
    if seen & {'hni_sts_tx_ok_octets','hni_sts_rx_ok_octets'}:
        opts.append({'label':'Traffic · TX & RX','value':'traffic','family':'Traffic','search':'bandwidth transmitted received bytes'})
    for m in METRICS.values():
        if set(m.numerator+m.denominator)<=seen:
            opts.append({'label':f'{m.family} · {m.label}','value':m.key,'family':m.family,
                         'search':' '.join(m.numerator+m.denominator)+' '+m.note})
    if seen & {'hni_sts_tx_bad_octets','hni_sts_rx_bad_octets'}:
        opts.append({'label':'Reliability · Bad octets','value':'octet_errors','family':'Reliability'})
    for name,counter in sorted(engine.counters.items()):
        info=describe_counter(name,counter)
        opts.append({'label':f"Raw · {info['label']}",'value':name,'family':'Raw counters',
                     'search':name+' '+info['family']+' '+info['description'],'title':name})
    favored=set(favorites or ())
    if family=='favorites': opts=[o for o in opts if o['value'] in favored]
    elif family!='all': opts=[o for o in opts if o['family']==family]
    opts.sort(key=lambda o: (o['value'] not in favored,))
    return [{k:v for k,v in o.items() if k!='family'} for o in opts]


def definition(engine, metric):
    from .diagnostics import METRICS
    if metric in METRICS:
        m=METRICS[metric]
        def delta(names):return ' + '.join('Δ'+n for n in names)
        num=delta(m.numerator)
        if m.mode=='ratio':formula=f'({num}) / ({delta(m.denominator)})'
        elif m.mode=='max_rate':formula='max('+', '.join('Δ'+n+'/covered_seconds' for n in m.numerator)+')'
        elif m.mode=='histogram':formula='First bin whose cumulative aligned increments reach 99% of all aligned histogram increments'
        else:formula='('+num+') / covered_seconds'
        if m.scale!=1:formula+=f' × {m.scale:g}'
        return {'label':m.label,'family':m.family,'unit':m.unit,'description':m.note or f'{m.label} over the selected interval and devices.',
                'formula':formula,'inputs':list(dict.fromkeys(m.numerator+m.denominator)),
                'method':'Differences of valid cumulative observations, aligned on common time support. Ratios use integrated counts, never an average of per-sample ratios. A zero denominator is undefined.',
                'limits':'No interpolation across resets or invalid intervals. Sparse change events do not imply missing samples; the original cadence and unsupported trace boundaries cannot be recovered. Node rows show the maximum NIC value, not an aggregate ratio.',
                'source':HPE_SOURCE,'provenance':'Derived metric with explicit source-counter formula'}
    if metric=='traffic':
        return {'label':'Traffic · TX & RX','family':'Traffic','unit':'B/s (rate) or B (volume)',
                'description':'Network-frame octets over time for the selected physical NICs. TX and RX stay separate.',
                'formula':'Interval rate = (counter[t₂] − counter[t₁]) / ((t₂ − t₁) / 1e9). Observed volume = sum of valid interval increments.',
                'inputs':['hni_sts_tx_ok_octets','hni_sts_rx_ok_octets'],
                'method':'Complete observed endpoint intervals only. Scientific summaries use full-resolution data; display bins integrate those interval-average rates.',
                'limits':'Not application payload. Change-only sources do not expose native sampling cadence or idle tails; absent tracks are not zero. Node heatmaps sum supported NIC rates; partial membership is explicitly marked.',
                'source':HPE_SOURCE,'provenance':'Curated traffic analysis'}
    if metric=='octet_errors':
        return {'label':'Bad frame octets','family':'Reliability','unit':'B','description':'Bad-octet increments on the selected NICs.',
                'formula':'Sum of valid endpoint differences, separately for TX and RX.',
                'inputs':['hni_sts_tx_bad_octets','hni_sts_rx_bad_octets'],'method':'Complete intervals only; measured zero is distinct from absence.',
                'limits':'Bad octets are not a count of unique packet errors.','source':HPE_SOURCE,'provenance':'Curated reliability analysis'}
    info=describe_counter(metric,engine.counters.get(metric))
    return {**info,'formula':'Source-native recorded value; no automatic rate or percentage conversion.',
            'inputs':[metric],'method':'Heatmap cells show the last recorded observation within a display bin, not a bin mean. Empty bins are not filled.',
            'limits':'Raw cumulative values are not selected-window totals. No gauge/counter semantics are inferred from identifier text.'}


def definition_panel(engine, metric):
    data=definition(engine,metric)
    return [html.Div([html.H3(data['label']),html.Span(data['unit'],className='unit-badge')],className='definition-title'),
       html.P(data['description']),html.H4('Calculation'),html.Pre(data['formula'],className='formula'),
       html.P(data['method']),html.P(data['limits'],className='muted'),
       html.Details([html.Summary(f"Source counters ({len(data['inputs'])})"),html.Ul([
          html.Li([html.Code(n),html.Span(' — '+describe_counter(n,engine.counters.get(n))['label']),
                   html.Span(' · not encountered' if n not in engine.counters else '',className='muted')]) for n in data['inputs']])]),
       html.P('Metadata: '+data['provenance'],className='tiny'),
       html.A('Source documentation',href=data['source'],target='_blank',rel='noopener noreferrer')
         if data['source'].startswith('https://') else html.P(data['source'],className='tiny')]
