"""Optional real-data acceptance: NICVIZ_REAL_TRACE_ROOT points at the supplied run.

This never creates an allocation or contacts Sunspot. It opens existing sources.
"""
from pathlib import Path
import json
import re
import os
import pytest
from playwright.sync_api import expect
from test_explorer_browser import browser, local_response_bootstrap, choose, export_doc
from dataset_lifecycle import DatasetLifecycleManager
from explorer.trace_browser import TracePathControls
from explorer.app import create_explorer_app

ROOT=os.environ.get('NICVIZ_REAL_TRACE_ROOT')
pytestmark=pytest.mark.skipif(not ROOT,reason='Set NICVIZ_REAL_TRACE_ROOT to an existing CTF/pftrace run for real-data acceptance.')

@pytest.fixture(scope='module')
def real_app():
    with DatasetLifecycleManager(Path(ROOT),[]) as manager:
        picker=TracePathControls(manager);request=picker.open_path(str(Path(ROOT)/'ctf'))
        assert manager.wait(request,60).state.value=='ready'
        app=create_explorer_app(None,dataset_manager=manager,trace_browser=True)
        yield app

@pytest.fixture
def real_page(browser,real_app):
    page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
    errors=[];failed=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.on('response',lambda response:failed.append((response.status,response.url)) if response.status>=500 else None)
    local_response_bootstrap(page,real_app)
    page.wait_for_selector('#ex-heatmap .main-svg',timeout=20000)
    page.locator('#ex-view').get_by_text('Trends',exact=True).click()
    page.wait_for_selector('#ex-timeline .main-svg',timeout=20000)
    yield page
    assert not errors,errors
    assert not failed,failed
    page.close()


def screenshot(page,name):
    root=os.environ.get('NICVIZ_SCREENSHOTS')
    if root:
        Path(root).mkdir(parents=True,exist_ok=True)
        page.screenshot(path=str(Path(root)/('legacyflow-'+name+'.png')),full_page=False)


def test_real_ctf_host_pressure_and_histogram_are_working_views(real_page,tmp_path):
    page=real_page
    expect(page.locator('#ex-cards')).to_contain_text('16 NICs')
    assert page.evaluate("()=>document.querySelector('#ex-timeline .js-plotly-plot').data.some(t=>Array.from(t.y).some(y=>y>0))")
    screenshot(page,'ctf-overview-1440')
    choose(page,'#ex-metric','Host / PCIe · Posted pushback')
    expect(page.locator('#ex-timeline-title')).to_have_text('Posted pushback')
    expect(page.locator('#ex-cards')).to_contain_text('cycles/packet')
    choose(page,'#ex-nics','x1921c6s3b0n0 / cxi0')
    expect(page.locator('#ex-nic-detail')).to_be_visible()
    expect(page.locator('#ex-cards')).to_contain_text('7.44')
    doc=export_doc(page,tmp_path)
    assert doc['records'][0]['node']=='x1921c6s3b0n0'
    assert doc['records'][0]['nic']=='cxi0'
    assert doc['records'][0]['total']==pytest.approx(7.439865503,rel=1e-6)
    assert doc['diagnostic']['unit']=='cycles/packet'
    screenshot(page,'ctf-posted-pushback-1440')
    choose(page,'#ex-metric','Latency · Request / response latency distribution')
    expect(page.locator('#ex-nic-detail')).to_contain_text('P99 bucket: 10')
    expect(page.locator('#ex-nic-detail')).to_contain_text('not nanoseconds')
    screenshot(page,'ctf-latency-1440')
    page.set_viewport_size({'width':1024,'height':900})
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    screenshot(page,'ctf-latency-1024')


def test_file_picker_switches_formats_and_rejects_bad_path_without_losing_data(real_page,real_app,tmp_path):
    page=real_page
    page.locator('#ex-open').click()
    expect(page.locator('#ex-file-path')).to_be_visible()
    page.locator('#ex-file-path').fill(str(Path(ROOT)/'timeline.pftrace'))
    page.locator('#ex-file-load').click()
    expect(page.locator('#ex-file-status')).to_contain_text('Trace ready',timeout=30000)
    expect(page.locator('#ex-source-format')).to_contain_text('pftrace',timeout=10000)
    page.locator('#ex-close-trace').click()
    choose(page,'#ex-metric','Host / PCIe · Posted pushback')
    expect(page.locator('#ex-cards')).to_contain_text('cycles/packet')
    doc=export_doc(page,tmp_path)
    assert 'pftrace' in doc['run']['source_format']
    assert sum(r['total'] is not None for r in doc['records'])==2
    assert real_app.explorer_engine().store.row_count==3871
    page.locator('#ex-open').click()
    page.locator('#ex-file-path').fill('/etc/passwd')
    page.locator('#ex-file-load').click()
    expect(page.locator('#ex-file-action-error')).to_contain_text('outside')
    assert real_app.explorer_engine().store.row_count==3871
    page.locator('#ex-file-path').fill(str(Path(ROOT)/'ctf'))
    page.locator('#ex-file-load').click()
    expect(page.locator('#ex-source-format')).to_contain_text(re.compile('ctf',re.I),timeout=30000)
    assert real_app.explorer_engine().store.row_count==93720
    page.locator('#ex-close-trace').click()
    choose(page,'#ex-metric','Reliability · Bad octets')
    expect(page.locator('#ex-cards')).to_contain_text('32 / 32')
    screenshot(page,'ctf-bad-octets-1440')


def test_empty_landing_presents_server_file_browser(browser):
    with DatasetLifecycleManager(Path(ROOT),[]) as manager:
        app=create_explorer_app(None,dataset_manager=manager,trace_browser=True)
        page=browser.new_page(viewport={'width':1440,'height':900})
        local_response_bootstrap(page,app)
        expect(page.get_by_role('heading',name='Start with your trace')).to_be_visible()
        expect(page.locator('#ex-file-path')).to_be_visible()
        page.locator('#ex-file-entry').click()
        expect(page.get_by_role('option',name='Directory · ctf',exact=True)).to_be_visible()
        page.keyboard.press('Escape')
        choose(page,'#ex-file-entry','Directory · ctf')
        expect(page.locator('#ex-file-path')).to_have_value(str(Path(ROOT)/'ctf'))
        page.locator('#ex-file-up').click()
        expect(page.locator('#ex-file-path')).to_have_value(str(Path(ROOT)))
        screenshot(page,'open-trace-1440')
        page.close()
