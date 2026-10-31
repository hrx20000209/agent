from pathlib import Path
import json,hashlib,tarfile
base=Path('/home/pxchen/workspace/android_world_mai8b_20261024');out=Path('/home/pxchen/workspace/cpx_results_pair_20261031');out.mkdir(exist_ok=True)
for model,sub,tag in [('8B','cpx_pair_puregpu_20261030','mai8b_pair_single_c30x20_puregpu_20261030'),('2B','cpx_pair_mai2b_puregpu_20261030','mai2b_pair_single_c20x10_puregpu_20261030')]:
 root=base/sub;run=root/'experiments/trace_reachability_validation/raw/mai_official'/tag;rows=[];name=out/(model+'_logs.tar.gz')
 with tarfile.open(name,'w:gz') as t:
  for p in sorted(run.rglob('*')):
   if not p.is_file():continue
   rel=str(p.relative_to(run));include=p.suffix not in ['.jpg','.png'];rows.append(dict(path=rel,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),in_archive=include))
   if include:t.add(p,arcname=tag+'/'+rel)
  for rel in ['host_supervisor.py','readiness.py','episode_entry.py','upstream_sha256.json','upstream_modifications.json']:
   t.add(root/rel,arcname='actual_launch/'+rel)
 (out/(model+'_files.json')).write_text(json.dumps(dict(server_root=str(run),files=rows),indent=2)+'\n')
 print(model,name.stat().st_size,flush=True)
