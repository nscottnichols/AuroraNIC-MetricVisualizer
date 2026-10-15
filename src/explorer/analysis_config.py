"""Optional capture-specific latency calibration and explicit node topology."""
import json
import math
from pathlib import Path


def load_analysis_config(path):
    if path is None:return {}
    path=Path(path)
    if path.stat().st_size>1024*1024:raise ValueError('Analysis configuration exceeds 1 MiB.')
    config=json.loads(path.read_text())
    if not isinstance(config,dict):raise ValueError('Analysis config must be a JSON object.')
    if set(config)-{'latency_histograms','topology'}:raise ValueError('Unknown analysis configuration key.')
    if not isinstance(config.get('latency_histograms',{}),dict):raise ValueError('latency_histograms must map counter-family names to calibration objects.')
    for name,cal in config.get('latency_histograms',{}).items():
        count={'pct_req_rsp_latency':32,'pct_host_access_latency':16}.get(name)
        if not count or not isinstance(cal,dict) or not isinstance(cal.get('source'),str) or not cal['source'].strip():
            raise ValueError('Calibration needs a supported histogram name and a nonempty source.')
        if not isinstance(cal.get('run_id'),str) or not cal['run_id'].strip():raise ValueError('Each latency calibration must name its run_id; calibration is never silently reused on another trace.')
        edges=cal.get('edges_ns',[])
        if not isinstance(edges,list) or len(edges)!=count+1:raise ValueError(f'{name} requires {count+1} edges_ns, optionally null for the final open upper edge.')
        numeric=edges[:-1] if edges[-1] is None else edges
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in numeric):raise ValueError('Histogram edges must be finite nonnegative nanoseconds.')
        if any(b<=a for a,b in zip(numeric,numeric[1:])):raise ValueError('Histogram edges must be strictly increasing.')
    topo=config.get('topology',{})
    if not isinstance(topo,dict):raise ValueError('Topology must map node labels to explicit groups.')
    for node,groups in topo.items():
        if not isinstance(node,str) or not isinstance(groups,dict) or any(k not in {'cabinet','rack','switch','region'} or not isinstance(v,str) for k,v in groups.items()):raise ValueError('Topology fields are cabinet, rack, switch, and region, using string labels.')
    return config
