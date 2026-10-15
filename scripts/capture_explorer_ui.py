#!/usr/bin/env python3
"""Capture the explorer through an already-established local tunnel.

This is a browser smoke test, not independent validation of trace numerics.
It does not start/stop a server, capture a trace, or contact Sunspot directly.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import shutil
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright, expect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8050/')
    parser.add_argument('--expected-run-id',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--reject-synthetic',action='store_true')
    args=parser.parse_args()
    parsed=urlsplit(args.url)
    if parsed.scheme not in {'http','https'} or parsed.hostname not in {'localhost','127.0.0.1','::1'}:
        parser.error('--url must point to a local loopback server or tunnel')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={'url':args.url,'expected_run_id':args.expected_run_id,'viewports':[],'limits':'Browser smoke only; source-level numeric reference checks remain necessary.'}
    with sync_playwright() as p:
        executable=os.environ.get('NICVIZ_CHROMIUM') or shutil.which('chromium')
        browser=p.chromium.launch(headless=True,**({'executable_path':executable} if executable else {}))
        try:
            for width,height in [(1440,1000),(1024,768)]:
                page=browser.new_page(viewport={'width':width,'height':height},accept_downloads=True)
                errors=[]; page.on('pageerror',lambda e:errors.append(str(e)))
                navigation_started=time.perf_counter()
                page.goto(args.url,wait_until='networkidle',timeout=60000)
                navigation_seconds=time.perf_counter()-navigation_started
                plot_started=time.perf_counter()
                expect(page.locator('h1')).to_have_text('Traffic explorer')
                expect(page.locator('#ex-run-name')).to_have_text(args.expected_run_id)
                if args.reject_synthetic:
                    assert 'synthetic' not in page.locator('#ex-source-format').inner_text().lower()
                assert 'Question index' not in page.locator('body').inner_text()
                page.wait_for_selector('#ex-timeline .main-svg')
                first_plot_seconds=time.perf_counter()-plot_started
                assert page.evaluate('document.documentElement.scrollWidth')<=width
                points=page.evaluate("()=>document.querySelector('#ex-timeline .js-plotly-plot').data.map(t=>({name:t.name,count:t.x.length,finite:t.y.filter(v=>v!==null&&Number.isFinite(v)).length}))")
                assert all(r['count']<=241 for r in points)
                screenshot=args.output_dir/f'explorer-{width}x{height}.png'
                page.screenshot(path=str(screenshot),full_page=True)
                interaction_started=time.perf_counter()
                with page.expect_download() as result:
                    page.locator('#ex-export').click()
                export_path=args.output_dir/f'explorer-{width}x{height}.json'; result.value.save_as(str(export_path))
                export_seconds=time.perf_counter()-interaction_started
                exported=json.loads(export_path.read_text())
                assert exported['schema']=='aurora-nic-explorer/1'
                assert exported['run']['run_id']==args.expected_run_id
                assert isinstance(exported['selection']['start_ns'],str)
                assert not errors,errors
                report['viewports'].append({'width':width,'height':height,'plot_series':points,'screenshot':screenshot.name,'export':export_path.name,'javascript_errors':errors,
                                            'timings':{'navigation_seconds':navigation_seconds,'first_plot_seconds':first_plot_seconds,'export_seconds':export_seconds}})
                page.close()
        finally:
            browser.close()
    (args.output_dir/'browser-smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
