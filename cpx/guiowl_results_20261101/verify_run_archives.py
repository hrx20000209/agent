"""Verify all archived raw records, actual launch sources and key screenshots."""
from pathlib import Path
import json,hashlib,tarfile
ROOT=Path(__file__).resolve().parent
for name in ['original','base7','full45_fixed']:
 manifest=json.loads((ROOT/(name+'_files.json')).read_text());files={x['path']:x for x in manifest['files']};expected={p for p,x in files.items() if x['in_archive']};seen=set();launch={}
 with tarfile.open(ROOT/(name+'_logs.tar.gz')) as tar:
  for m in tar:
   if not m.isfile():continue
   data=tar.extractfile(m).read()
   if m.name.startswith('actual_launch/'):
    launch[m.name.removeprefix('actual_launch/')]=hashlib.sha256(data).hexdigest();continue
   assert m.name.startswith(manifest['tag']+'/'),m.name
   rel=m.name.removeprefix(manifest['tag']+'/');assert rel in expected and rel not in seen,rel
   e=files[rel];assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],rel
   seen.add(rel)
 assert seen==expected,(name,expected-seen)
 config=json.loads((ROOT/'records'/name/'run_config.json').read_text())
 for rel,h in launch.items():
  source='experiments/trace_reachability_validation/scripts/'+rel.removeprefix('adapter/') if rel.startswith('adapter/') else 'experiments/trace_reachability_validation/'+rel if rel.startswith('configs/') else rel
  assert config['source_sha256'][source]==h,(name,source)
 for p in (ROOT/'records'/name).iterdir():
  if p.is_file():
   e=files[p.name];assert p.stat().st_size==e['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'],p
 for rel,e in files.items():
  if e.get('key_screenshot'):
   p=ROOT/'key_screenshots'/name/rel;assert p.stat().st_size==e['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'],p
 print(name,len(seen),'raw files and',len(launch),'actual source files verified; all screenshot hashes indexed')
