"""Archive GUI-Owl logs, UI trees, launch code and graph; index every screenshot."""
from pathlib import Path
import hashlib,json,tarfile
BASE=Path('/home/pxchen/workspace/android_world_mai8b_20261024')
OUT=Path('/home/pxchen/workspace/cpx_guiowl_results_20261101');OUT.mkdir(exist_ok=True)
RUNS=[('original','cpx_pair_guiowl_20261031','guiowl8b_pair_single_c10_official_20261031'),('base7','cpx_guiowl_base7_20261101','guiowl8b_base7_openapp_fix_20261101'),('full45_fixed','cpx_guiowl_full45_fix_20261101','guiowl8b_full45_openapp_fix_20261101')]
for name,folder,tag in RUNS:
 root=BASE/folder;run=root/'experiments/trace_reachability_validation/raw/mai_official'/tag
 errors=[]
 for arm in ['BASE','FULL45']:
  p=run/(arm+'.runs.jsonl')
  if p.exists():
   errors.extend((arm,r['task']) for r in map(json.loads,p.read_text().splitlines()) if r.get('error') or r.get('rc'))
 # Add last UI screenshots for every original/fixed execution error.
 selected=set()
 for arm,task in errors:
  for p in sorted((run/arm/task/'p0/img').glob('*.jpg'))[-2:]:selected.add(p)
 files=[]
 with tarfile.open(OUT/(name+'_logs.tar.gz'),'w:gz') as tar:
  for p in sorted(run.rglob('*')):
   if not p.is_file():continue
   rel=str(p.relative_to(run));included=p.suffix not in ['.jpg','.png']
   files.append(dict(path=rel,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),in_archive=included,key_screenshot=p in selected))
   if included:tar.add(p,arcname=tag+'/'+rel)
  for rel in ['host_supervisor.py','host_supervisor_test.py','readiness.py','episode_entry.py','owl_protocol_test.py','replay_openapp.py','upstream_sha256.json','upstream_modifications.json']:
   if (root/rel).exists():tar.add(root/rel,arcname='actual_launch/'+rel)
  for p in sorted((root/'experiments/trace_reachability_validation/scripts').glob('owl*.py')):tar.add(p,arcname='actual_launch/adapter/'+p.name)
  for p in [root/'experiments/trace_reachability_validation/configs/guiowl_8b_official.json',root/'experiments/trace_reachability_validation/configs/guiowl_official_mobile_system.txt']:
   tar.add(p,arcname='actual_launch/configs/'+p.name)
 (OUT/(name+'_files.json')).write_text(json.dumps(dict(tag=tag,server_root=str(run),files=files),indent=2)+'\n')
 keys=OUT/'key_screenshots'/name
 for p in selected:
  dst=keys/p.relative_to(run);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
 # Accessible top-level records for review and statistics without archive extraction.
 d=OUT/'records'/name;d.mkdir(parents=True,exist_ok=True)
 for p in run.iterdir():
  if p.is_file() and p.suffix in ['.json','.jsonl']:(d/p.name).write_bytes(p.read_bytes())
 print(name,(OUT/(name+'_logs.tar.gz')).stat().st_size,'files',len(files),'key_images',len(selected),flush=True)
