from pathlib import Path
import sys,json,time,concurrent.futures,statistics,hashlib,copy,uuid
from PIL import Image
import requests
ROOT=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_single_rerun_20261030');sys.path.insert(0,str(ROOT/'experiments/trace_reachability_validation/scripts'));import mai_protocol as M
out=Path('/home/pxchen/workspace/services/mai_ui_8b_gpu5_puregpu_20261030/benchmark');ep=ROOT/'experiments/trace_reachability_validation/raw/mai_official/mai8b_pair_single_c30x20_hostwatch_20261030/BASE/BrowserDraw/p0';files=sorted((ep/'img').glob('*.jpg'))[-3:];prompt=(ROOT/'experiments/trace_reachability_validation/configs/mai_ui_official_system.txt').read_text().rstrip('\n');goal=json.load(open(ep/'meta.json'))['goal'];base=json.load(open(out/'request.json'));endpoints={'offload2':'http://127.0.0.1:8097','puregpu':'http://127.0.0.1:18197'}
payloads=[]
for i in range(4):
 ims=[]
 for j,p in enumerate(files):
  im=Image.open(p).convert('RGB');im.putpixel((0,0),(i+3,j+7,13));ims.append(im)
 h=M.NativeHistory(3)
 for im in ims[:2]:h.append({'action':'wait'},'Wait for the page to settle.',im)
 messages=h.messages(prompt,'Benchmark sample '+str(i)+' '+str(uuid.uuid4())+'\n'+goal,ims[-1]);payload=copy.deepcopy(base);payload['messages']=messages;b=json.dumps(payload,separators=(',',':')).encode();payloads.append(b);(out/f'fresh_request_{i}.json').write_bytes(b)
def call(name,i):
 session=requests.Session();session.trust_env=False;start=time.perf_counter()
 try:
  r=session.post(endpoints[name]+'/v1/chat/completions',data=payloads[i],headers={'Content-Type':'application/json'},timeout=180);r.raise_for_status();d=r.json();text=d['choices'][0]['message']['content'];error=None;parsed=None
  try:parsed=M.parse_response(text,(1080,2400))['native_arguments']
  except Exception as e:error=str(e)
  return dict(service=name,trial=i,request_sha256=hashlib.sha256(payloads[i]).hexdigest(),latency_s=time.perf_counter()-start,usage=d.get('usage'),finish_reason=d['choices'][0].get('finish_reason'),parse_error=error,parsed=parsed,response=text)
 except Exception as e:return dict(service=name,trial=i,latency_s=time.perf_counter()-start,error=repr(e))
rows=[]
with concurrent.futures.ThreadPoolExecutor(4) as pool:
 for round_ in range(2):
  fs=[pool.submit(call,name,round_*2+i) for name in endpoints for i in range(2)];batch=[f.result() for f in fs];rows+=batch;(out/'fresh_results.json').write_text(json.dumps(rows,indent=2));print('ROUND',round_,json.dumps([{k:r.get(k) for k in ['service','trial','latency_s','usage','error','parse_error']} for r in batch]),flush=True)
summary=dict(note='Four paired fresh-three-image requests. Request-specific goal prefix prevents full KV prefix reuse after goal; one corner pixel change per image invalidates multimodal image cache. Identical bytes per pair; two concurrent requests per server. No Android actions executed.',services={})
for n in endpoints:
 rs=[r for r in rows if r['service']==n and not r.get('error')];summary['services'][n]=dict(requests=len(rs),mean_latency_s=statistics.mean(r['latency_s'] for r in rs),median_latency_s=statistics.median(r['latency_s'] for r in rs),mean_completion_tokens=statistics.mean(r['usage']['completion_tokens'] for r in rs),all_actions_valid=all(not r.get('parse_error') and r['finish_reason']!='length' for r in rs))
summary['speedup']=summary['services']['offload2']['mean_latency_s']/summary['services']['puregpu']['mean_latency_s'];(out/'fresh_summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
