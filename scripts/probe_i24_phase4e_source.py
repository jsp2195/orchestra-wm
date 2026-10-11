"""One credential check after a confirmed secure update; never log credentials."""
import argparse,datetime,json,os,re,urllib.error,urllib.request
from pathlib import Path
from orchestra_wm.i24_phase4e.ranges import NoRedirect
from orchestra_wm.i24_phase4e.auth import access_token
from orchestra_wm.i24_continuous.source import write_immutable
p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);a=p.parse_args()
if Path(a.output).exists():raise SystemExit('Existing access receipt preserved; select a new output')
known=json.loads(Path(a.metadata).read_text());token=access_token()
result=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),secret_variable_present=bool(token),metadata_executed=False,range_executed=False)
opener=urllib.request.build_opener(NoRedirect());base='https://www.googleapis.com/drive/v3/files/'+known['id']
try:
    if token:
        result['metadata_executed']=True
        req=urllib.request.Request(base+'?fields=id,name,size,mimeType,md5Checksum,version,capabilities(canDownload)',headers={'Authorization':'Bearer '+token})
        with opener.open(req,timeout=30) as response:meta=json.loads(response.read(65537));result['metadata_http_status']=response.status
        if any(meta[k]!=known[k] for k in ['id','name','size','md5Checksum','version']):raise ValueError('Known source identity/version changed')
        result.update(name=meta['name'],size=meta['size'],canDownload=meta.get('capabilities',{}).get('canDownload'))
        result['range_executed']=True
        req=urllib.request.Request(base+'?alt=media',headers={'Authorization':'Bearer '+token,'Range':'bytes=0-1023','Accept-Encoding':'identity'})
        with opener.open(req,timeout=30) as response:
            data=response.read(1025);result.update(range_http_status=response.status,content_range=response.headers.get('Content-Range'),read_bytes=len(data),zip_header=data[:4]==b'PK\x03\x04')
        result['range_verified']=result['range_http_status']==206 and result['content_range']==f"bytes 0-1023/{meta['size']}" and len(data)==1024 and result['zip_header']
except urllib.error.HTTPError as error:
    result['error_http_status']=error.code
    try:
        payload=json.loads(error.read(65536))
        result['reasons']=[x['reason'] for x in payload.get('error',{}).get('errors',[]) if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,64}',str(x.get('reason','')))]
    except (ValueError,KeyError):result['reason_unavailable']=True
    error.close()
except Exception as error:result['error_type']=type(error).__name__
write_immutable(a.output,result);print(json.dumps(result,indent=2))
raise SystemExit(0 if result.get('range_verified') else 2)
