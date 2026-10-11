"""Verify every viewer array against preserved evaluated NPZs, then exercise UI."""
import asyncio,base64,gzip,json,re
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright
from orchestra_wm.i24.data import sha

async def main():
 root=Path('outputs/i24_phase4b/gallery');html=(root/'index.html').read_text();encoded=re.search("atob\\('([^']+)'\\)",html).group(1)
 payload=json.loads(gzip.decompress(base64.b64decode(encoded)));pilot=Path('outputs/i24_continuous/pilot_0700_wb')
 for e in payload['entries']:
  p=pilot/e['source_path'];assert sha(p)==e['rollout_sha256']
  with np.load(p) as a:
   for key,original in [('agents','agents'),('valid','predicted_valid'),('fields','fields'),('truth','truth'),('truth_valid','valid'),('target_fields','target_fields'),('support','field_support'),('history','history'),('history_valid','history_valid')]:
    target=a[original][...,:2] if key in ('agents','truth','history') else a[original]
    assert np.array_equal(np.asarray(e[key]),target),(e['scene'],e['regime'],key)
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
  page=await browser.new_page(viewport={'width':1300,'height':1300});errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
  await page.set_content(html,wait_until='load',timeout=30000);await page.wait_for_function('window.ORCHESTRA_READY===true',timeout=30000)
  for scene in range(3):
   await page.locator('#scenario').select_option(str(scene))
   for regime in ('dense','outage'):
    await page.locator('#regime').select_option(regime)
    for frame in (-25,-1,0,24,49,99):
     await page.locator('#time').fill(str(frame));await page.locator('#time').dispatch_event('input')
     rendered=await page.evaluate('window.ORCHESTRA_RENDERED');assert rendered['frame']==frame and rendered['scene']==scene and rendered['regime']==regime
  await page.locator('#scenario').select_option('0');await page.locator('#regime').select_option('dense');await page.locator('#time').fill('49');await page.locator('#time').dispatch_event('input')
  for sample in ('0','7','all'):await page.locator('#sample').select_option(sample)
  await page.locator('#truth').uncheck();await page.locator('#truth').check()
  await page.locator('#play').click();await page.wait_for_timeout(300);await page.locator('#play').click();assert int(await page.locator('#time').input_value())>49
  await page.screenshot(path=str(root/'browser_preview.png'),full_page=True);assert not errors,errors
  result=dict(pass_=True,exact_array_entries=len(payload['entries']),stochastic_samples_each=8,scenarios=3,regimes=2,javascript_errors=errors,html_sha256=sha(root/'index.html'),controls=['scenario','regime','sample','truth','time','play'])
  (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));await browser.close()

if __name__=='__main__':asyncio.run(main())
