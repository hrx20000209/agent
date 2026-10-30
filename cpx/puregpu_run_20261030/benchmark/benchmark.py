"""Identical three-image MAI requests; no emulator actions are executed."""
from pathlib import Path
import sys,json,time,concurrent.futures,statistics,hashlib,urllib.request,datetime
from PIL import Image
import requests
ROOT=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_single_rerun_20261030');S=ROOT/'experiments/trace_reachability_validation/scripts';sys.path.insert(0,str(S));import mai_protocol as M
SERVICE=Path('/home/pxchen/workspace/services/mai_ui_8b_gpu5_puregpu_20261030');RUN=ROOT/'experiments/trace_reachability_validation/raw/mai_official/mai8b_pair_single_c30x20_hostwatch_20261030'
out=SERVICE/'benchmark';out.mkdir(exist_ok=True)
ep=RUN/'BASE/BrowserDraw/p0';images=sorted((ep/'img').glob('*.jpg'))[-3:];assert len(images)==3
history=M.NativeHistory(3)
for p in images[:2]:history.append({'action':'wait'},'Wait for the page to settle.',Image.open(p).convert('RGB'))
prompt=(ROOT/'experiments/trace_reachability_validation/configs/mai_ui_official_system.txt').read_text().rstrip('\n');goal=json.load(open(ep/'meta.json'))['goal'];messages=history.messages(prompt,goal,Image.open(images[-1]).convert('RGB'))
payload=dict(model='MAI-UI-8B',messages=messages,max_tokens=2048,temperature=0,top_p=1,top_k=-1,frequency_penalty=0,presence_penalty=0,repetition_penalty=1,seed=42,stream=False)
b=json.dumps(payload,separators=(',',':')).encode();(out/'request.json').write_bytes(b)
metadata=dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),images=[str(p) for p in images],request_sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),note='Identical payload on two4090D services. PureGPU memory utilization0.97, CPUoffload0; old service0.9/offload2. Synthetic WAIT history built from saved screenshots; latency benchmark, not task success test. Warmups excluded from reported speedup.')
endpoints={'offload2':'http://127.0.0.1:8097','puregpu':'http://127.0.0.1:18197'}
for name,url in endpoints.items():
 for _ in range(180):
  try:
   op=urllib.request.build_opener(urllib.request.ProxyHandler({}));d=json.load(op.open(url+'/v1/models',timeout=3));assert any(m['id']=='MAI-UI-8B' for m in d['data']);break
  except Exception:time.sleep(2)
 else:raise RuntimeError('Server unavailable: '+name)
def call(name,stage,i):
 start=time.perf_counter();session=requests.Session();session.trust_env=False
 try:
  r=session.post(endpoints[name]+'/v1/chat/completions',data=b,headers={'Content-Type':'application/json'},timeout=180);r.raise_for_status();d=r.json();txt=d['choices'][0]['message']['content'];parsed=None;error=None
  try:parsed=M.parse_response(txt,(1080,2400))['native_arguments']
  except Exception as e:error=str(e)
  return dict(service=name,stage=stage,trial=i,latency_s=time.perf_counter()-start,usage=d.get('usage'),finish_reason=d['choices'][0].get('finish_reason'),parsed=parsed,parse_error=error,response=txt)
 except Exception as e:return dict(service=name,stage=stage,trial=i,latency_s=time.perf_counter()-start,error=repr(e))
rows=[]
with concurrent.futures.ThreadPoolExecutor(4) as pool:
 rows.extend(pool.map(lambda n:call(n,'warmup',0),endpoints))
 print('WARMUP',json.dumps([{k:r.get(k) for k in ['service','latency_s','error','parse_error']} for r in rows]),flush=True)
 for round_ in range(2):
  fs=[pool.submit(call,n,'measured',round_*2+i) for n in endpoints for i in range(2)]
  batch=[f.result() for f in fs];rows.extend(batch);(out/'results.json').write_text(json.dumps(dict(metadata=metadata,rows=rows),indent=2));print('ROUND',round_,json.dumps([{k:r.get(k) for k in ['service','latency_s','error','parse_error','usage']} for r in batch]),flush=True)
summary={}
for n in endpoints:
 rs=[r for r in rows if r['service']==n and r['stage']=='measured' and not r.get('error')];summary[n]=dict(requests=len(rs),mean_latency_s=statistics.mean(r['latency_s'] for r in rs) if rs else None,mean_completion_tokens=statistics.mean(r['usage']['completion_tokens'] for r in rs) if rs else None,all_actions_valid=all(not r.get('parse_error') and r.get('finish_reason')!='length' for r in rs))
if all(summary[n]['mean_latency_s'] for n in endpoints):summary['speedup']=summary['offload2']['mean_latency_s']/summary['puregpu']['mean_latency_s']
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
