"""Plot builders. Plots share one physical population and canonical time selection."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from .model import MAX_HEATMAP_ROWS, human

TX_COLOR, RX_COLOR = "#007f88", "#6559d9"
CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["lasso2d", "autoScale2d"]}


def base(*, height=275):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=height,
        font={"family":"Arial, sans-serif", "size":12,"color":"#536178"},
        paper_bgcolor="white", plot_bgcolor="white", margin={"l":60,"r":25,"t":18,"b":46},
        hoverlabel={"bgcolor":"#182439","font_color":"white"},
        legend={"orientation":"h","y":1.16,"x":0,"font_size":12},
        hovermode="x unified")
    fig.update_xaxes(gridcolor="#edf1f5", zeroline=False)
    fig.update_yaxes(gridcolor="#edf1f5", zeroline=False, rangemode="tozero")
    return fig


def scale(values, base_unit="B/s"):
    a = np.asarray(values, dtype=float)
    good = a[np.isfinite(a)]
    maximum = np.max(np.abs(good)) if len(good) else 0
    for factor, prefix in ((1e12,"T"),(1e9,"G"),(1e6,"M"),(1e3,"k")):
        if maximum >= factor:
            return factor, prefix+base_unit
    return 1., base_unit


def empty(fig, message):
    fig.add_annotation(text=message,x=.5,y=.5,xref="paper",yref="paper",showarrow=False,
                       font={"size":13,"color":"#65738a"})
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig


def timeline(view):
    fig = base(height=278)
    q = view["query"]
    if q.metric != "traffic":
        ready = [r for r in view["records"] if r["raw_x"]][:12]
        for r in ready:
            fig.add_trace(go.Scatter(x=r["raw_x"],y=r["raw_y"],mode="lines+markers",
                name=f"{r['node']} / {r['nic']}",connectgaps=False,
                marker_size=3,line_width=1.8,
                hovertemplate="%{x:.4f} s<br>%{y:,.5g}<extra>%{fullData.name}</extra>"))
        unit = ready[0]["unit"] if ready else "source units"
        fig.update_yaxes(title_text=f"Observed value ({unit})")
        if not ready:
            empty(fig,"No observations in this selection. Check the counter, NIC, or collector.")
    else:
        key = "rates" if q.mode == "rate" else "cumulative"
        values = [a[key] for a in view["aggregate"].values()]
        factor, unit = scale(np.concatenate(values) if values else [], "B/s" if q.mode == "rate" else "B")
        any_data = False
        for direction,color in (("TX",TX_COLOR),("RX",RX_COLOR)):
            a=view["aggregate"][direction]
            y=np.asarray(a[key],dtype=float)/factor
            # Bin means are drawn as steps, not fictional smooth instantaneous peaks.
            x=view["x_s"]
            y=np.r_[y,y[-1]] if len(y) else y
            any_data |= bool(np.any(np.isfinite(y)))
            fig.add_trace(go.Scatter(x=x.tolist(),y=[float(v) if np.isfinite(v) else None for v in y],
                mode="lines",name=f"{direction} · {a['usable']}/{a['expected']} NICs",connectgaps=False,
                line={"color":color,"width":2.3,"shape":"hv"},
                hovertemplate="%{x:.4f} s<br>%{y:,.3f} "+unit+"<extra>"+direction+"</extra>"))
        fig.update_yaxes(title_text=unit)
        if not any_data:
            empty(fig,"No complete traffic intervals. Select a NIC to resolve its collector, or widen the time range.")
    fig.update_layout(dragmode="select",selectdirection="h",
        uirevision=f"{q.metric}/{q.start_ns}/{q.end_ns}/{q.node_ids}/{q.nic_ids}/{q.collectors}")
    fig.update_xaxes(title_text="Time from trace start (s)",range=[(q.start_ns-view['origin_ns'])/1e9,(q.end_ns-view['origin_ns'])/1e9])
    return fig


def heatmap(view):
    rows=view["ranked"][:MAX_HEATMAP_ROWS]
    if view["query"].metric != "traffic":
        return empty(base(height=305),"Raw counters are shown as observations above; no rate is inferred.")
    factor, unit=scale([r["rates"] for r in rows])
    z=[[float(v/factor) if np.isfinite(v) else None for v in r["rates"]] for r in rows]
    fig=base(height=max(294,min(700,92+len(rows)*21)))
    centers=(view["x_s"][:-1]+view["x_s"][1:])/2
    labels=[f"{r['node']} / {r['nic']}" for r in rows]
    fig.add_trace(go.Heatmap(x=centers.tolist(),y=labels,z=z,
        customdata=[[r["nic_id"]]*len(centers) for r in rows],
        colorscale=[[0,"#edf6f6"],[.2,"#c1e5e4"],[.5,"#57b6b5"],[1,"#00707a"]],
        zmin=0,colorbar={"title":{"text":unit},"thickness":10,"len":.88},
        hoverongaps=False,
        hovertemplate="%{y}<br>%{x:.4f} s · %{z:,.3f} "+unit+"<extra></extra>"))
    fig.update_layout(margin={"l":170,"r":16,"t":10,"b":44},hovermode="closest")
    fig.update_yaxes(autorange="reversed",rangemode="normal",tickfont_size=10,gridcolor="white")
    fig.update_xaxes(title_text="Time (s)",showgrid=False)
    if not rows:
        return empty(fig,"No NICs match this scope.")
    return fig


def ranking(view):
    rows=[r for r in view["ranked"] if r["total"] is not None][:10]
    fig=base(height=315)
    if not rows:
        return empty(fig,"No comparable values. Select a NIC to inspect availability.")
    is_traffic=view["query"].metric=="traffic"
    values=[float(r["total"]) for r in rows]
    factor,unit=scale(values,"B") if is_traffic else (1.,rows[0]["unit"])
    labels=[f"{r['node']} / {r['nic']}" for r in rows]
    fig.add_trace(go.Bar(x=[v/factor for v in values],y=labels,orientation="h",
        marker_color=TX_COLOR if view["query"].direction=="TX" else RX_COLOR,
        customdata=[r["nic_id"] for r in rows],
        hovertemplate="%{y}<br>%{x:,.3f} "+unit+"<extra></extra>",width=.58))
    fig.update_layout(margin={"l":155,"r":18,"t":8,"b":48},hovermode="closest",bargap=.32)
    fig.update_yaxes(autorange="reversed",tickfont_size=10,showgrid=False,rangemode="normal")
    fig.update_xaxes(title_text=("Observed volume (" if is_traffic else "Last observed value (")+unit+")")
    return fig


def reliability(view):
    rows=[]
    for record in view["records"]:
        for counter in record["counter_results"]:
            if counter["total"] is not None:
                rows.append({
                    "label":f"{record['node']} / {record['nic']} / {counter['counter']}",
                    "value":float(counter["total"]),
                    "nic_id":record["nic_id"],
                })
    fig=base(height=315)
    if not rows:
        return empty(fig,"No compatible reliability totals. Select a NIC to resolve its collector.")
    rows=rows[:24]
    labels=[r["label"] for r in rows]
    values=[r["value"] for r in rows]
    factor,unit=scale(values,"events")
    fig.add_trace(go.Bar(x=[value/factor for value in values],y=labels,orientation="h",
        marker_color="#b5563f",width=.58,
        customdata=[r["nic_id"] for r in rows],
        hovertemplate="%{y}<br>%{x:,.3f} "+unit+"<extra></extra>"))
    fig.update_layout(margin={"l":155,"r":18,"t":8,"b":48},hovermode="closest",bargap=.32)
    fig.update_yaxes(autorange="reversed",tickfont_size=10,showgrid=False,rangemode="normal")
    fig.update_xaxes(title_text=f"Observed errors ({unit})")
    return fig




def table_rows(view):
    rows=[]
    for row in view["ranked"]:
        rows.append({"id":str(row["nic_id"]),"node":row["node"],"nic":row["nic"],
            "total":float(row["total"]) if row["total"] is not None else None,
            "mean":row["mean"],"p99":row["p99"],"cv":row['cv'],
            "share":row.get('population_share'),"active":row.get('active_fraction'),
            "coverage":row["coverage"] if view["query"].metric=="traffic" else None,
            "state":row["state"],"samples":row["sample_count"],"collector":str(row["collector_id"]) if row["collector_id"] is not None else "—"})
    return rows
