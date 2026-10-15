"""Current default UX journeys using real browser JS and genuine Flask callbacks."""
import base64
import json
import os
from pathlib import Path
from dataclasses import replace
import pytest
from playwright.sync_api import expect
from test_explorer_browser import browser,local_response_bootstrap,choose,export_doc,wait_idle
from explorer.app import create_explorer_app
from explorer.demo import make_demo_store
from explorer.model import Query

@pytest.fixture(scope='module')
def ux_app():return create_explorer_app(make_demo_store())

@pytest.fixture
def ux_page(browser,ux_app):
    page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
    errors=[];failed=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('response',lambda r:failed.append((r.status,r.url)) if r.status>=500 else None)
    local_response_bootstrap(page,ux_app)
    page.wait_for_selector('#ex-heatmap .main-svg',timeout=20000)
    wait_idle(page)
    yield page
    assert not errors,errors
    assert not failed,failed
    page.close()


def test_default_map_click_and_back_restore_exact_population(ux_page,tmp_path):
    page=ux_page
    expect(page.locator('#ex-primary-map')).to_be_visible()
    expect(page.locator('#ex-trend-wrap')).not_to_be_visible()
    # Real mouse click at the first row, middle of its observed range.
    box=page.locator('#ex-heatmap .nsewdrag').bounding_box()
    page.mouse.click(box['x']+box['width']*.4,box['y']+box['height']/32)
    expect(page.locator('#ex-scope')).to_have_text('node-a / cxi0',timeout=15000)
    expect(page.locator('#ex-trend-wrap')).to_be_visible()
    doc=export_doc(page,tmp_path)
    assert doc['selection']['nic_ids']==[0]
    assert all(r['nic_id']==0 for r in doc['records'])
    page.locator('#ex-back').click()
    expect(page.locator('#ex-scope')).to_have_text('All nodes / All NICs')
    expect(page.locator('#ex-primary-map')).to_be_visible()
    expect(page.locator('#ex-back')).to_be_disabled()


def test_real_heatmap_brush_and_undo(ux_page,tmp_path):
    page=ux_page;box=page.locator('#ex-heatmap .nsewdrag').bounding_box()
    page.mouse.move(box['x']+box['width']*.2,box['y']+30);page.mouse.down()
    page.mouse.move(box['x']+box['width']*.6,box['y']+65,steps=12);page.mouse.up()
    page.wait_for_function("()=>Number(document.querySelector('#ex-from').value)>1",timeout=15000)
    doc=export_doc(page,tmp_path)
    delta=(int(doc['selection']['end_ns'])-int(doc['selection']['start_ns']))/1e9
    assert delta==pytest.approx(4.8,abs=.1)
    page.locator('#ex-back').click()
    expect(page.locator('#ex-from')).to_have_value('0')
    expect(page.locator('#ex-to')).to_have_value('12')


def test_grouped_node_drilldown_preserves_identity(ux_page):
    page=ux_page;choose(page,'#ex-row-group','Nodes')
    expect(page.locator('#ex-page-note')).to_contain_text('sum of complete NIC rates')
    box=page.locator('#ex-heatmap .nsewdrag').bounding_box()
    page.mouse.click(box['x']+box['width']*.5,box['y']+box['height']*.125)
    expect(page.locator('#ex-scope')).to_have_text('node-a / All NICs')
    expect(page.locator('#ex-cards')).to_contain_text('4 NICs')


def test_search_favorites_definition_and_csv(ux_page,tmp_path):
    page=ux_page
    page.locator('#ex-row-search').fill('node-c');page.locator('#ex-row-search').press('Tab')
    expect(page.locator('#ex-page-note')).to_contain_text('of 4 matching / 16 in scope')
    expect(page.locator('#ex-cards')).to_contain_text('16 NICs')
    page.locator('#ex-favorite').click();expect(page.locator('#ex-favorite')).to_contain_text('Favorited')
    choose(page,'#ex-family','Favorites')
    page.locator('#ex-definition-link').click()
    expect(page.locator('#ex-definition-content')).to_contain_text('hni_sts_tx_ok_octets')
    choose(page,'#ex-export-format','CSV summaries')
    with page.expect_download() as pending:page.locator('#ex-export').click()
    path=tmp_path/'out.csv';pending.value.save_as(path)
    import csv
    rows=list(csv.DictReader(path.open()))
    assert len(rows)==32  # Search is a display filter, never a changed analysis scope.
    assert '1700000000123456789' in rows[0]['start_ns']


def test_save_restore_and_import_view(ux_page,tmp_path):
    page=ux_page
    choose(page,'#ex-nodes','node-b')
    choose(page,'#ex-color','Logarithmic')
    page.locator('#ex-from').fill('3');page.locator('#ex-from').press('Tab')
    expect(page.locator('#ex-window-label')).to_contain_text('3.0000')
    page.locator('#ex-workspace > summary').click()
    page.locator('#ex-view-name').fill('Interesting burst')
    page.locator('#ex-save-view').click()
    expect(page.locator('#ex-workspace-status')).to_contain_text('Saved Interesting burst')
    with page.expect_download() as pending:page.locator('#ex-download-view').click()
    saved=tmp_path/'view.json';pending.value.save_as(saved)
    doc=json.loads(saved.read_text());assert doc['query']['node_ids']==[1]
    assert doc['settings']['color']=='log'
    page.locator('#ex-reset-scope').click()
    expect(page.locator('#ex-scope')).to_have_text('All nodes / All NICs')
    page.locator('#ex-reset-time').click()
    expect(page.locator('#ex-from')).to_have_value('0')
    expect(page.locator('#ex-scope')).to_have_text('All nodes / All NICs')
    choose(page,'#ex-bookmark','Interesting burst');page.locator('#ex-restore-view').click()
    expect(page.locator('#ex-scope')).to_have_text('node-b / All NICs')
    expect(page.locator('#ex-from')).to_have_value('3')
    page.locator('#ex-reset-scope').click();expect(page.locator('#ex-scope')).to_have_text('All nodes / All NICs')
    page.locator('#ex-upload-view input').set_input_files(saved)
    expect(page.locator('#ex-workspace-status')).to_contain_text('Restored imported view')
    expect(page.locator('#ex-scope')).to_have_text('node-b / All NICs')
    bad={**doc,'dataset':'wrong-trace'};saved.write_text(json.dumps(bad))
    page.locator('#ex-upload-view input').set_input_files(saved)
    expect(page.locator('#ex-workspace-status')).to_contain_text('different trace fingerprint')
    expect(page.locator('#ex-scope')).to_have_text('node-b / All NICs')


def test_keyboard_shortcuts_do_not_intercept_typing_and_help(ux_page):
    page=ux_page
    page.locator('#ex-main').focus();page.keyboard.press('f')
    expect(page.locator('#ex-row-search')).to_be_focused()
    page.keyboard.type('borf')
    expect(page.locator('#ex-row-search')).to_have_value('borf')
    expect(page.locator('#ex-open-trace')).not_to_be_visible()
    page.locator('#ex-main').focus();page.keyboard.press('?')
    expect(page.locator('#ex-help')).to_have_attribute('open','')


@pytest.mark.parametrize('width,height',[(1440,1000),(1024,900),(760,1000)])
def test_default_map_viewports_are_not_horizontally_clipped(ux_page,width,height):
    page=ux_page;page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(400)
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    expect(page.locator('#ex-primary-map')).to_be_visible()
    page.wait_for_function("()=>document.querySelector('#ex-heatmap').getBoundingClientRect().height>=300")
    assert page.evaluate("()=>document.querySelector('.map-footer').getBoundingClientRect().top>=document.querySelector('#ex-heatmap .svg-container').getBoundingClientRect().bottom-1")
