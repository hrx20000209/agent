from pathlib import Path
import json,base64,urllib.request,time,re,concurrent.futures,datetime
root=Path('/home/pxchen/workspace/services/gui_owl_15_8b_gpu45_20261031')
imgroot=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_puregpu_20261030/experiments/trace_reachability_validation/raw/mai_official/mai8b_pair_single_c30x20_puregpu_20261030/BASE/AudioRecorderRecordAudioWithFileName/p0/img')
images=sorted(imgroot.glob('*.jpg'))[:5];assert len(images)==5
prompt=(root/'official_mobile_system.txt').read_text().rstrip('\n')
urls=['data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode() for p in images]
def payload(n):
 messages=[dict(role='system',content=[dict(type='text',text=prompt)])]
 for i in range(n):
  content=[dict(type='image_url',image_url=dict(url=urls[i]))]
  if i==0:content.insert(0,dict(type='text',text='Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: Look at the current screen and wait one second.\n\nPrevious actions:\nNo previous action.'))
  messages.append(dict(role='user',content=content))
  if i<n-1:messages.append(dict(role='assistant',content=[dict(type='text',text='Action: Wait one second.\n<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait","time":1}}\n</tool_call>')]))
 return dict(model='GUI-Owl-1.5-8B-Instruct',messages=messages,temperature=0.0,top_p=1.0,top_k=-1,seed=42,max_tokens=256)
def call(label,n):
 p=payload(n);(root/(label+'.request.json')).write_text(json.dumps(p));start=time.monotonic();op=urllib.request.build_opener(urllib.request.ProxyHandler({}));req=urllib.request.Request('http://127.0.0.1:18197/v1/chat/completions',data=json.dumps(p).encode(),headers={'Content-Type':'application/json'})
 d=json.load(op.open(req,timeout=180));s=d['choices'][0]['message']['content'];m=re.search(r'<tool_call>\s*(.*?)\s*</tool_call>',s,re.S);action=json.loads(m.group(1)) if m else None
 row=dict(label=label,images=n,latency_s=time.monotonic()-start,usage=d.get('usage'),response=s,tool_call=action,valid=bool(action and action.get('name')=='mobile_use' and action.get('arguments',{}).get('action') in ['wait','terminate']),finish_reason=d['choices'][0].get('finish_reason'));(root/(label+'.response.json')).write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True);assert row['valid'],row
 return row
rows=[call('smoke_single_image',1)]
with concurrent.futures.ThreadPoolExecutor(2) as pool:rows+=list(pool.map(lambda i:call('smoke_concurrent5_'+str(i),5),range(2)))
summary=dict(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),note='Recorded screenshots and synthetic wait history, official upstream mobile prompt; service inference only, no Android actions executed.',results=rows)
(root/'smoke_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('SMOKE_PASSED',flush=True)
