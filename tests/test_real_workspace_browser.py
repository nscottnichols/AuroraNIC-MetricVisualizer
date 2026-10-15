"""Optional recorded CTF/pftrace UI acceptance, without contacting Sunspot."""
from pathlib import Path
import os
import json
import re
import pytest
from playwright.sync_api import expect
from test_explorer_browser import browser,local_response_bootstrap,choose,export_doc,wait_idle
from dataset_lifecycle import DatasetLifecycleManager
from explorer.trace_browser import TracePathControls
from explorer.app import create_explorer_app

ROOT=os.environ.get('NICVIZ_REAL_TRACE_ROOT')
pytestmark=pytest.mark.skipif(not ROOT,reason='Set NICVIZ_REAL_TRACE_ROOT to the existing allreduce capture directory.')

@pytest.fixture(scope='module')
def real_app():
    with DatasetLifecycleManager(Path(ROOT),[]) as manager:
        picker=TracePathControls(manager);job=picker.open_path(str(Path(ROOT)/'ctf'))
        assert manager.wait(job,60).state.value=='ready'
        yield create_explorer_app(None,dataset_manager=manager,trace_browser=True)

@pytest.fixture
def real_page(browser,real_app):
    page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
    errors=[];failed=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('response',lambda r:failed.append((r.status,r.url)) if r.status>=500 else None)
    local_response_bootstrap(page,real_app)
    page.wait_for_selector('#ex-heatmap .main-svg',timeout=20000);wait_idle(page)
    yield page
    assert not errors,errors
    assert not failed,failed
    page.close()


def capture(page,name):
    if os.environ.get('NICVIZ_SCREENSHOTS'):
        p=Path(os.environ['NICVIZ_SCREENSHOTS']);p.mkdir(parents=True,exist_ok=True)
        page.screenshot(path=str(p/f'{name}.png'),full_page=False)


def test_real_ctf_metric_definitions_heatmap_and_correct_physical_nic(real_page,tmp_path):
    page=real_page
    expect(page.locator('#ex-cards')).to_contain_text('16 NICs')
    assert page.evaluate("()=>document.querySelector('#ex-heatmap .js-plotly-plot').data[1].z.some(row=>row.some(v=>v>0))")
    capture(page,'ctf-overview-1440')
    choose(page,'#ex-metric','Host / PCIe · Posted pushback')
    expect(page.locator('#ex-cards')).to_contain_text('cycles/packet')
    assert page.evaluate("()=>document.querySelector('#ex-heatmap .js-plotly-plot').data[1].z.some(row=>row.some(v=>v>0))")
    capture(page,'ctf-pushback-heatmap-1440')
    page.locator('#ex-definition-link').click()
    expect(page.locator('#ex-definition-content')).to_contain_text('parbs_tarb_pi_posted_blocked_cnt')
    expect(page.locator('#ex-definition-content')).to_contain_text('parbs_tarb_pi_posted_pkts')
    capture(page,'ctf-pushback-definition-1440')
    choose(page,'#ex-nics','x1921c6s3b0n0 / cxi0')
    expect(page.locator('#ex-nic-detail')).to_be_visible()
    doc=export_doc(page,tmp_path)
    assert len(doc['records'])==1
    assert doc['records'][0]['total']==pytest.approx(7.439865503,rel=1e-8)
    assert doc['records'][0]['node']=='x1921c6s3b0n0'
    assert doc['metric_definition']['inputs']==['parbs_tarb_pi_posted_blocked_cnt','parbs_tarb_pi_posted_pkts']
    page.locator('#ex-main').focus()
    capture(page,'ctf-nic-detail-1440')


def test_modal_focus_escape_failed_open_and_switch_to_pftrace(real_page,real_app,tmp_path):
    page=real_page
    page.locator('#ex-open').click()
    expect(page.locator('#ex-file-path')).to_be_focused()
    capture(page,'open-trace-modal')
    page.keyboard.press('Escape')
    expect(page.locator('#ex-open-trace')).not_to_be_visible()
    expect(page.locator('#ex-open')).to_be_focused()
    page.locator('#ex-open').click()
    page.locator('#ex-file-path').fill('/etc/passwd');page.locator('#ex-file-load').click()
    expect(page.locator('#ex-file-action-error')).to_contain_text('outside')
    assert real_app.explorer_engine().store.row_count==93720
    page.locator('#ex-file-path').fill(str(Path(ROOT)/'timeline.pftrace'));page.locator('#ex-file-load').click()
    expect(page.locator('#ex-file-status')).to_contain_text('Trace ready',timeout=30000)
    expect(page.locator('#ex-source-format')).to_contain_text('pftrace',timeout=15000)
    page.locator('#ex-close-trace').click()
    choose(page,'#ex-metric','Host / PCIe · Posted pushback')
    expect(page.locator('#ex-cards')).to_contain_text('cycles/packet')
    doc=export_doc(page,tmp_path)
    assert sum(r['total'] is not None for r in doc['records'])==2
    assert real_app.explorer_engine().store.row_count==3871
    capture(page,'pftrace-pushback-1440')
    page.locator('#ex-open').click();page.locator('#ex-file-path').fill(str(Path(ROOT)/'ctf'));page.locator('#ex-file-load').click()
    expect(page.locator('#ex-source-format')).to_contain_text(re.compile('ctf',re.I),timeout=30000)
    page.locator('#ex-close-trace').click()


def test_raw_observation_heatmap_uses_catalog_and_real_samples(real_page,real_app,tmp_path):
    page=real_page
    choose(page,'#ex-metric','Raw · Transmitted OK frame octets')
    expect(page.locator('h1')).to_have_text('Counter explorer')
    expect(page.locator('#ex-timeline-title')).to_have_text('Transmitted OK frame octets')
    expect(page.locator('#ex-page-note')).to_contain_text('Raw values are observations within bins')
    doc=export_doc(page,tmp_path)
    assert doc['selection']['metric']=='hni_sts_tx_ok_octets'
    assert all(r['mean'] is None for r in doc['records'])
    capture(page,'ctf-raw-observations-1440')
    page.set_viewport_size({'width':1024,'height':900});page.wait_for_timeout(300)
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    page.wait_for_function("()=>document.querySelector('#ex-heatmap').getBoundingClientRect().height>=400")
    assert page.evaluate("()=>document.querySelector('.map-footer').getBoundingClientRect().top>=document.querySelector('#ex-heatmap .svg-container').getBoundingClientRect().bottom-1")
    capture(page,'ctf-raw-observations-1024')
