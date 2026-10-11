"""Exact saved-array comparison and browser control exercise for the private viewer."""
import argparse,asyncio,base64,gzip,json,re
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright
from orchestra_wm.i24.data import sha

async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/i24_phase4c/gallery')
    parser.add_argument('--evaluation-output',default='outputs/i24_phase4c/matched_macro_seed101_u300/development_test');args=parser.parse_args()
    root=Path(args.output);html=(root/'private/demo.html').read_text()
    encoded=re.search("atob\\('([^']+)'\\)",html).group(1)
    payload=json.loads(gzip.decompress(base64.b64decode(encoded)))
    evaluation=Path(args.evaluation_output)
    for e in payload['entries']:
        path=evaluation/e['source_path'];assert sha(path)==e['rollout_sha256']
        with np.load(path) as a:
            for key,original in [('agents','agents'),('valid','predicted_valid'),('fields','fields'),('truth','truth'),('truth_valid','valid'),('target_fields','target_fields'),('support','field_support'),('history','history'),('history_valid','history_valid')]:
                target=a[original][...,:2] if key in ('agents','truth','history') else a[original]
                assert np.array_equal(np.asarray(e[key]),target),(e['scene'],e['variant'],e['regime'],key)
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=await browser.new_page(viewport={'width':1300,'height':1350});errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        await page.set_content(html,wait_until='load',timeout=60000)
        await page.wait_for_function('window.ORCHESTRA_READY===true',timeout=30000)
        for scene in range(3):
            await page.locator('#scenario').select_option(str(scene))
            for variant in sorted(set(e['variant'] for e in payload['entries'])):
                await page.locator('#variant').select_option(variant)
                for regime in ('dense','outage'):
                    await page.locator('#regime').select_option(regime)
                    for frame in (-25,-1,0,24,49,99):
                        await page.locator('#time').fill(str(frame));await page.locator('#time').dispatch_event('input')
                        rendered=await page.evaluate('window.ORCHESTRA_RENDERED')
                        assert rendered==dict(scene=scene,variant=variant,regime=regime,frame=frame,samples=8)
        await page.locator('#scenario').select_option('0');await page.locator('#variant').select_option('full');await page.locator('#regime').select_option('dense')
        await page.locator('#time').fill('49');await page.locator('#time').dispatch_event('input')
        for sample in ('0','7','all'):await page.locator('#sample').select_option(sample)
        await page.locator('#truth').uncheck();await page.locator('#truth').check()
        await page.locator('#play').click();await page.wait_for_timeout(350);await page.locator('#play').click()
        assert int(await page.locator('#time').input_value())>49
        await page.screenshot(path=str(root/'browser_preview.png'),full_page=True)
        assert not errors,errors
        result=dict(passed=True,exact_array_entries=len(payload['entries']),stochastic_samples_each=8,scenarios=3,regimes=2,variants=len(set(e['variant'] for e in payload['entries'])),
                    javascript_errors=errors,private_html_sha256=sha(root/'private/demo.html'),controls=['scenario','variant','regime','sample','truth','time','play'])
        (root/'browser_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));await browser.close()

if __name__=='__main__':asyncio.run(main())
