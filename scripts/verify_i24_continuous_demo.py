"""Optional Chromium verification of saved checkpoint arrays and UI controls.

Run with uv run --with playwright python scripts/verify_i24_continuous_demo.py.
No trajectories or model forecasts are regenerated.
"""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from playwright.async_api import async_playwright

from orchestra_wm.i24.data import sha


async def verify(root):
    root=Path(root);demo=root/'demo';payload=json.loads((demo/'payload.json').read_text())
    manifest=json.loads((root/'evaluation_manifest.json').read_text())
    for entry in payload['entries']:
        actual=next(e for e in manifest['entries'] if e['variant']=='full' and e['scene_number']==0 and e['regime']==entry['regime'])
        assert sha(root/actual['path'])==entry['rollout_sha256']
        assert actual['checkpoint_sha256']==entry['checkpoint_sha256']
        with np.load(root/actual['path']) as arrays:
            assert np.array_equal(np.asarray(entry['mean_agents']),arrays['agents'].mean(0)[:,:,:2])
            assert np.array_equal(np.asarray(entry['mean_fields']),arrays['fields'].mean(0))
            assert np.array_equal(np.asarray(entry['truth']),arrays['truth'][:,:,:2])
            assert np.array_equal(np.asarray(entry['predicted_valid']),arrays['predicted_valid'].mean(0)>=.5)
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
                                        args=['--no-sandbox','--disable-dev-shm-usage'],timeout=30000)
        page=await browser.new_page(viewport={'width':1280,'height':1100});errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        await page.set_content((demo/'index.html').read_text(),wait_until='load',timeout=30000)
        assert await page.locator('#regime option').count()==5
        for regime in range(5):
            await page.locator('#regime').select_option(str(regime))
            for frame in (0,24,49,99):
                await page.locator('#time').fill(str(frame));await page.locator('#time').dispatch_event('input')
                assert f'+{(frame+1)*.2:.1f} s' in await page.locator('#label').inner_text()
        await page.locator('#truth').uncheck();await page.locator('#truth').check()
        await page.locator('#time').fill('0');await page.locator('#time').dispatch_event('input')
        await page.locator('#play').click();await page.wait_for_timeout(500);await page.locator('#play').click()
        assert int(await page.locator('#time').input_value())>0
        await page.screenshot(path=str(demo/'browser_preview.png'),full_page=True)
        assert not errors,errors
        result=dict(pass_=True,browser='Chromium headless',javascript_errors=errors,
                    controls_exercised=['regime','time','truth','play'],regimes=5,
                    exact_saved_array_entries=5,html_sha256=sha(demo/'index.html'),payload_sha256=sha(demo/'payload.json'))
        result['pass']=result.pop('pass_')
        (demo/'browser_validation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
        await browser.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/i24_continuous/pilot_0700_wb')
    args=parser.parse_args();asyncio.run(verify(args.output))
