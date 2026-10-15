"""Browser journeys for the new production-default explorer.

Default transport is local HTTP. Set NICVIZ_BROWSER_TRANSPORT=flask-route only in
an environment that prohibits Chromium localhost navigation. That mode serves the
same Flask responses to an isolated .invalid origin; it does not mock callbacks.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import threading
from urllib.parse import urlsplit

import pytest
pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright, expect
from explorer.app import create_explorer_app
from explorer.demo import make_demo_store


@pytest.fixture(scope='module')
def application():
    return create_explorer_app(make_demo_store())


@pytest.fixture(scope='module')
def server_url(application):
    from werkzeug.serving import make_server
    server=make_server('127.0.0.1',0,application.server,threaded=True)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    yield f'http://127.0.0.1:{server.server_port}/'
    server.shutdown(); thread.join(timeout=5)


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        executable=os.environ.get('NICVIZ_CHROMIUM') or shutil.which('chromium')
        b=p.chromium.launch(headless=True,**({'executable_path':executable} if executable else {}))
        yield b
        b.close()


def local_response_bootstrap(page,application):
    base='https://nicviz.invalid'
    def serve(route):
        request=route.request; split=urlsplit(request.url)
        path=split.path+('?' + split.query if split.query else '')
        if request.method=='OPTIONS':
            route.fulfill(status=204,headers={'Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'*','Access-Control-Allow-Methods':'GET,POST,OPTIONS'})
            return
        headers={k:v for k,v in request.headers.items() if k.lower() in ('content-type','accept')}
        with application.server.test_client() as client:
            result=client.open(path,method=request.method,data=request.post_data_buffer,headers=headers)
        route.fulfill(status=result.status_code,body=result.data,headers={'Content-Type':result.content_type,'Access-Control-Allow-Origin':'*'})
    page.route(base+'/**',serve)
    with application.server.test_client() as client:
        doc=client.get('/').data.decode()
    doc=doc.replace('<head>','<head><base href="'+base+'/">')
    doc=re.sub(r'"requests_pathname_prefix"\s*:\s*"/"','"requests_pathname_prefix":"'+base+'/"',doc)
    doc=re.sub(r'"url_base_pathname"\s*:\s*"/"','"url_base_pathname":"'+base+'/"',doc)
    page.set_content(doc,wait_until='networkidle',timeout=30000)


@pytest.fixture
def page(browser,application,server_url):
    p=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
    errors=[]; p.on('pageerror',lambda error: errors.append(str(error)))
    if os.environ.get('NICVIZ_BROWSER_TRANSPORT')=='flask-route':
        local_response_bootstrap(p,application)
    else:
        p.goto(server_url,wait_until='networkidle',timeout=30000)
    p.wait_for_selector('#ex-heatmap .main-svg',timeout=20000)
    p.locator('#ex-view').get_by_text('Trends',exact=True).click()
    p.locator('#ex-population > summary').click()
    p.wait_for_selector('#ex-timeline .main-svg',timeout=20000)
    yield p
    assert not errors,errors
    p.close()


def choose(page,selector,label):
    page.locator(selector).click()
    option=page.get_by_role('option',name=label,exact=True)
    if option.count() == 0:
        (page.locator('input[type=search]:visible').first if page.locator('input[type=search]:visible').count() else page.locator(selector).locator('input').first).fill(label)
    page.get_by_role('option',name=label,exact=True).click()
    # Multi-select remains open; escape closes it without altering the value.
    page.keyboard.press('Escape')


def export_doc(page,tmp_path):
    with page.expect_download(timeout=15000) as pending:
        page.locator('#ex-export').click()
    dest=tmp_path/'selection.json'; pending.value.save_as(dest)
    return json.loads(dest.read_text())


def graph(page):
    return page.evaluate("()=>document.querySelector('#ex-timeline .js-plotly-plot').data.map(t=>({name:t.name,x:Array.from(t.x),y:Array.from(t.y)}))")


def wait_idle(page):
    page.wait_for_timeout(150)
    page.wait_for_function("()=>!document.querySelector('[data-dash-is-loading=\"true\"]')",timeout=15000)


def test_overview_is_real_plotted_population_not_question_index(page):
    expect(page.locator('h1')).to_have_text('Traffic explorer')
    body=page.locator('body').inner_text()
    assert 'Question index' not in body and 'NIC-01' not in body
    curves=graph(page)
    assert [v['name'] for v in curves]==['TX · 15/16 NICs','RX · 15/16 NICs']
    assert all(len(v['x'])==241 for v in curves)
    assert max(v for v in curves[0]['y'] if v is not None)>40  # GB/s, not first NIC alone


def test_rank_click_opens_correct_physical_nic_and_export(page,tmp_path):
    bar=page.locator('#ex-ranking .barlayer .point').first
    bar.scroll_into_view_if_needed()
    box=bar.bounding_box()
    assert box
    # Plotly deliberately receives clicks on its transparent drag layer.
    page.mouse.click(box['x']+box['width']*.6,box['y']+box['height']*.5)
    expect(page.locator('#ex-scope')).to_have_text('node-c / cxi1',timeout=15000)
    expect(page.locator('#ex-nic-detail')).to_be_visible()
    expect(page.locator('#ex-comparison-grid')).not_to_be_visible()
    doc=export_doc(page,tmp_path)
    assert doc['selection']['nic_ids']==[9]
    assert {r['nic_id'] for r in doc['records']}=={9}
    assert {r['node_id'] for r in doc['records']}=={2}
    assert {r['direction'] for r in doc['records']}=={'TX','RX'}


def test_real_mouse_brush_updates_window_and_export(page,tmp_path):
    box=page.locator('#ex-timeline .nsewdrag').bounding_box()
    assert box
    page.mouse.move(box['x']+box['width']*.2,box['y']+box['height']*.35)
    page.mouse.down()
    page.mouse.move(box['x']+box['width']*.55,box['y']+box['height']*.6,steps=12)
    page.mouse.up()
    page.wait_for_function("()=>Number(document.querySelector('#ex-from').value)>1",timeout=15000)
    doc=export_doc(page,tmp_path)
    origin=1_700_000_000_123_456_789
    a=(int(doc['selection']['start_ns'])-origin)/1e9
    b=(int(doc['selection']['end_ns'])-origin)/1e9
    assert a==pytest.approx(2.4,abs=.08)
    assert b==pytest.approx(6.6,abs=.08)
    assert int(doc['selection']['start_ns'])!=round(float(doc['selection']['start_ns']))  # ns are not a JS float
    assert float(page.locator('#ex-from').input_value())==pytest.approx(a,abs=1e-9)


def test_missing_and_duplicate_collectors_are_actionable(page,tmp_path):
    choose(page,'#ex-nodes','node-d')
    choose(page,'#ex-nics','node-d / cxi3')
    expect(page.locator('#ex-scope')).to_have_text('node-d / cxi3',timeout=15000)
    expect(page.locator('#ex-notice')).to_contain_text('Multiple collectors')
    page.locator('.collector-details > summary').click()
    choose(page,'#ex-collector','Collector 1 · rank 1')
    page.wait_for_function("()=>document.querySelector('#ex-timeline .js-plotly-plot').data[0].name==='TX · 1/1 NICs'",timeout=15000)
    doc=export_doc(page,tmp_path)
    tx=next(r for r in doc['records'] if r['direction']=='TX')
    rx=next(r for r in doc['records'] if r['direction']=='RX')
    assert tx['collector_id']==1 and int(tx['total'])>0
    assert rx['total'] is None and rx['state']=='No samples'
    expect(page.locator('#ex-nic-detail')).to_contain_text('No samples')


def test_multiselect_and_separate_scope_time_resets(page,tmp_path):
    choose(page,'#ex-nodes','node-a'); choose(page,'#ex-nodes','node-b')
    expect(page.locator('#ex-scope')).to_have_text('2 selected nodes / All NICs')
    page.locator('#ex-from').fill('2'); page.locator('#ex-from').press('Tab')
    page.wait_for_function("()=>document.querySelector('#ex-window-label').textContent.startsWith('2.0000')")
    page.locator('#ex-reset-scope').click()
    expect(page.locator('#ex-scope')).to_have_text('All nodes / All NICs')
    assert page.locator('#ex-from').input_value()=='2'
    page.locator('#ex-reset-time').click()
    expect(page.locator('#ex-from')).to_have_value('0')


def test_counter_mode_is_raw_and_has_no_fake_rate_heatmap(page,tmp_path):
    choose(page,'#ex-metric','Raw · TX paused cycles · 0')
    expect(page.locator('h1')).to_have_text('Counter explorer',timeout=15000)
    expect(page.locator('#ex-table-wrap')).to_be_visible()
    expect(page.locator('#ex-map-wrap')).not_to_be_visible()
    expect(page.locator('#ex-notice')).to_contain_text('No bandwidth, event rate, latency')
    assert len(graph(page))==12
    doc=export_doc(page,tmp_path)
    assert doc['traffic_totals']=={}
    assert all(r['mean'] is None for r in doc['records'])


def test_octet_error_metric_is_a_compact_specialist_view(page):
    choose(page,'#ex-metric','Reliability · Bad octets')
    expect(page.locator('h1')).to_have_text('Error explorer',timeout=15000)
    expect(page.locator('#ex-timeline-title')).to_have_text('Bad octets by device')
    expect(page.locator('#ex-notice')).to_contain_text('Only sourced TX/RX bad-octet counters are included')
    curves=graph(page)
    assert len(curves)==1
    assert len(curves[0]['x'])>0
    assert page.locator('#ex-table th[data-dash-column="zero_counter_count"]').count()==1
    expect(page.locator('#ex-display')).not_to_be_visible()


def test_saved_intervals_compare_actual_mean_rates(page,tmp_path):
    page.locator('#ex-to').fill('4'); page.locator('#ex-to').press('Tab')
    page.wait_for_function("()=>document.querySelector('#ex-window-label').textContent.includes('4.0000')")
    page.locator('#ex-save-a').click()
    page.locator('#ex-reset-time').click()
    expect(page.locator('#ex-to')).to_have_value('12')
    page.locator('#ex-from').fill('8'); page.locator('#ex-from').press('Tab')
    page.wait_for_function("()=>document.querySelector('#ex-window-label').textContent.startsWith('8.0000')")
    page.locator('#ex-save-b').click()
    page.locator('#ex-interval-comparison > summary').click()
    expect(page.locator('#ex-compare-content')).to_contain_text('A: 0.0000–4.0000 s')
    expect(page.locator('#ex-compare-content')).to_contain_text('B: 8.0000–12.0000 s')
    expect(page.locator('#ex-compare-content tbody tr')).to_have_count(16)
    doc=export_doc(page,tmp_path)
    assert [item['label'] for item in doc['saved_intervals']]==['A','B']
    a,b=doc['saved_intervals']
    assert int(a['selection']['end_ns'])-int(a['selection']['start_ns'])==4_000_000_000
    assert int(b['selection']['start_ns'])-int(a['selection']['start_ns'])==8_000_000_000
    assert all(v['unit']=='B/s' for item in (a,b) for v in item['values'])
    assert any(v['value'] is not None for v in a['values'])


@pytest.mark.parametrize('width,height',[(1440,1000),(1024,768)])
def test_viewports_no_horizontal_overflow(page,width,height):
    page.set_viewport_size({'width':width,'height':height})
    page.wait_for_timeout(500)
    assert page.evaluate('document.documentElement.scrollWidth')<=width
    expect(page.locator('#ex-timeline')).to_be_visible()
    assert page.locator('.time-toolbar').bounding_box()['height']<90


def test_sorted_table_navigation_keeps_physical_identity(page,tmp_path):
    page.locator('#ex-display').get_by_text('Table',exact=True).click()
    expect(page.locator('#ex-table-wrap')).to_be_visible()
    # Numeric sorting followed by cell selection with stable row IDs.
    page.locator('#ex-table th[data-dash-column="total"]').click()
    expect(page.locator('#ex-table td[data-dash-column="node"]').first).to_have_text('node-c')
    expect(page.locator('#ex-table td[data-dash-column="nic"]').first).to_have_text('cxi1')
    cell=page.locator('#ex-table td[data-dash-column="nic"]').first
    page.wait_for_timeout(250)
    cell.click()
    expect(page.locator('#ex-nic-detail')).to_be_visible(timeout=15000)
    doc=export_doc(page,tmp_path)
    assert len(doc['selection']['nic_ids'])==1
    assert len({r['nic_id'] for r in doc['records']})==1
