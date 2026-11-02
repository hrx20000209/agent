"""Verify every archived nonimage log, actual source, record copy and key screenshot."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
count=0
request_counts={}
request_total=0
for archive,index in [('formal_logs.tar.gz','formal_files.json'),('actual_source.tar.gz','source_files.json'),('startup_history_logs.tar.gz','startup_history_files.json')]:
 manifest=json.loads((ROOT/index).read_text());rows=manifest['files'] if isinstance(manifest,dict) else manifest;expected={x['path']:x for x in rows if x['included']};seen=set()
 with tarfile.open(ROOT/archive) as tar:
  for member in tar:
   assert member.isfile(),member.name;assert member.name in expected and member.name not in seen,member.name
   row=expected[member.name];data=tar.extractfile(member).read();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],member.name;seen.add(member.name);count+=1
   if archive=='formal_logs.tar.gz' and member.name.endswith('/a11y_requests.jsonl'):
    requests=[json.loads(l) for l in data.decode().splitlines() if l.strip()]
    for x in requests:
     q=x['request'];assert x['images']==0 and all(isinstance(m['content'],str) for m in q['messages'])
     assert q['temperature']==0 and q['top_p']==1 and q['top_k']==-1 and q['seed']==42 and q['max_tokens']==2048 and q['chat_template_kwargs']['enable_thinking']==False
    parts=member.name.split('/');request_counts[(parts[1],parts[2])]=len(requests);request_total+=len(requests)

 assert seen==set(expected),(archive,set(expected)-seen)
formal=json.loads((ROOT/'formal_files.json').read_text());tag=formal['tag'];lookup={x['path'][len(tag)+1:]:x for x in formal['files']};source={x['path']:x for x in json.loads((ROOT/'source_files.json').read_text())}
for p in (ROOT/'records').rglob('*'):
 if not p.is_file():continue
 rel=str(p.relative_to(ROOT/'records'))
 if rel.startswith('episodes/'):
  _,arm,task,name=rel.split('/');rel=f'{arm}/{task}/p0/{name}'
 row=lookup[rel];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
for p in (ROOT/'key_screenshots').rglob('*.jpg'):
 row=lookup[str(p.relative_to(ROOT/'key_screenshots'))];assert row['key_screenshot'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
for p in (ROOT/'code').rglob('*'):
 if p.is_file():assert hashlib.sha256(p.read_bytes()).hexdigest()==source[str(p.relative_to(ROOT/'code'))]['sha256'],p
config=json.loads((ROOT/'records/run_config.json').read_text())
for name,h in config['source_sha256'].items():assert source[name]['sha256']==h,name
for arm in ['BASE','FULL45']:
 for line in (ROOT/'records'/(arm+'.runs.jsonl')).read_text().splitlines():
  row=json.loads(line);assert request_counts.get((arm,row['task']),0)==row['model_calls'],(arm,row['task'])
print('PURE_TEXT_REQUESTS_VERIFIED',request_total)
print('OK',count,'archive entries; record/source/screenshot copies and run source hashes match')
