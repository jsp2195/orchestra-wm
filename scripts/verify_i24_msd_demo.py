"""Optional browser proof and GIF from the saved, evaluated real-demo payload.
Run: uv run --with playwright python scripts/verify_i24_msd_demo.py
Requires a local Chromium executable. Does not train, download data or use live truth.
"""
import asyncio
import io
import json
from pathlib import Path
from PIL import Image
from playwright.async_api import async_playwright
from orchestra_wm.i24_msd.acquisition import hashes


async def main():
    out=Path('outputs/i24_msd/demo')
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=await browser.new_page(viewport={'width':1560,'height':1180});errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        await page.set_content((out/'index.html').read_text(),wait_until='load')
        await page.wait_for_function("document.getElementById('scene').options.length===3")
        await page.locator('#sample').select_option('3')
        await page.locator('#timeline').fill('19');await page.locator('#timeline').dispatch_event('input')
        await page.screenshot(path=str(out/'browser_preview.png'),full_page=True)
        frames=[]
        for step in range(30):
            await page.locator('#timeline').fill(str(step));await page.locator('#timeline').dispatch_event('input')
            data=await page.screenshot(clip={'x':0,'y':0,'width':1560,'height':970})
            frames.append(Image.open(io.BytesIO(data)).convert('RGB').resize((1170,728)))
        frames[0].save(out/'demo.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
        ids=await page.locator('#scene option').evaluate_all('(options)=>options.map(o=>o.value)')
        for scene in ids:
            await page.locator('#scene').select_option(scene)
            for regime in ['sparse50','outage','graph_off','reset_memory','dense']:
                await page.locator('#regime').select_option(regime)
        await page.locator('#truth').uncheck();await page.locator('#truth').check()
        await page.locator('#play').click();await page.wait_for_timeout(500);await page.locator('#play').click()
        assert not errors,errors
        result={'pass':True,'browser':'Chromium headless','html_sha256':hashes(out/'index.html')['SHA-256'],
                'payload_sha256':hashes(out/'data.json.gz')['SHA-256'],'scenario_options':len(ids),
                'sample_options':await page.locator('#sample option').count(),'javascript_errors':errors,
                'controls_exercised':['scene','regime','sample','timeline','truth','play'],
                'gif':'demo.gif','gif_frames':30,'gif_source':'saved checkpoint arrays rendered by the same verified HTML; no regenerated trajectories'}
        (out/'browser_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
        await browser.close()


if __name__=='__main__':asyncio.run(main())
