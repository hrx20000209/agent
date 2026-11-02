"""Snapshot formal logs/source/startup provenance without changing any experiment."""
from pathlib import Path
import json,hashlib,tarfile,io,shutil,datetime
ROOT=Path('/home/pxchen/workspace/qwen3_a11y_20261101');OUT=Path('/home/pxchen/workspace/qwen3_a11y_publish_20261102');OUT.mkdir(exist_ok=True)
RAW=ROOT/'experiments/trace_reachability_validation/raw/a11y_text';TAG='qwen3_8b_a11y_pair_20261101_full_v2';RUN=RAW/TAG
manifest=json.load(open(RUN/'run_config.json'))
for p,h in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
ledger=[]
def add(tar,p,name,include=True):
 data=p.read_bytes();row=dict(path=name,server_path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),included=include)
 if include:
  ti=tarfile.TarInfo(name);ti.size=len(data);ti.mtime=int(p.stat().st_mtime);ti.mode=0o644;tar.addfile(ti,io.BytesIO(data))
 return row
errors=[]
for a in ['BASE','FULL45']:
 rows=[json.loads(l) for l in (RUN/(a+'.runs.jsonl')).read_text().splitlines()]
 errors.extend((a,x['task']) for x in rows if x.get('error') or x.get('rc') or x.get('teardown_error'))
keys=set()
for a,task in errors:
 keys.update(sorted((RUN/a/task/'p0/img').glob('*.jpg'))[-2:])
files=[]
with tarfile.open(OUT/'formal_logs.tar.gz','w:gz') as tar:
 for p in sorted(RUN.rglob('*')):
  if not p.is_file():continue
  inc=p.suffix.lower() not in ['.jpg','.png','.jpeg','.webp'];row=add(tar,p,TAG+'/'+str(p.relative_to(RUN)),inc);row['key_screenshot']=p in keys;files.append(row)
  if p in keys:
   dst=OUT/'key_screenshots'/p.relative_to(RUN);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
(OUT/'formal_files.json').write_text(json.dumps(dict(server_root=str(RUN),tag=TAG,files=files),indent=2))
source=[];dirs={'android_world','apps','assets','docker_setup','docs','scripts','server','experiments'};rootnames={'Dockerfile','LICENSE','requirements.txt','pyproject.toml','setup.py','README.md','run_ma35.sh','run_guiowl15.sh'}
with tarfile.open(OUT/'actual_source.tar.gz','w:gz') as tar:
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(ROOT)
  if '__pycache__' in rel.parts or 'raw' in rel.parts or '.git' in rel.parts:continue
  if not (rel.parts[0] in dirs or (len(rel.parts)==1 and (p.suffix=='.py' or p.name in rootnames))):continue
  source.append(add(tar,p,str(rel)))
 for name,p in [('prepare_run.py',Path('/tmp/prepare_run.py')),('launch_stage.py',Path('/tmp/launch_stage.py')),('start_server.py',Path('/tmp/qwen_a11y_start_server.py')),('prepare_containers.py',Path('/tmp/qwen_prepare_containers.py'))]:
  if p.exists():source.append(add(tar,p,'actual_host_helpers/'+name))
(OUT/'source_files.json').write_text(json.dumps(source,indent=2))
history=[]
with tarfile.open(OUT/'startup_history_logs.tar.gz','w:gz') as tar:
 for phase in sorted(RAW.iterdir()):
  if not phase.is_dir() or phase.name==TAG:continue
  for p in sorted(phase.rglob('*')):
   if p.is_file():history.append(add(tar,p,'prior_trials/'+phase.name+'/'+str(p.relative_to(phase)),p.suffix.lower() not in ['.jpg','.jpeg','.png','.webp']))
 for folder in ['initialization','initialization_attempt1']:
  d=ROOT/folder
  for p in sorted(d.rglob('*')):
   if p.is_file():history.append(add(tar,p,folder+'/'+str(p.relative_to(d)),p.suffix.lower() not in ['.jpg','.jpeg','.png','.webp']))
 for p in sorted(ROOT.iterdir()):
  if p.is_file() and p.suffix in ['.json','.jsonl','.log']:history.append(add(tar,p,'root_state/'+p.name))
 for folder in ['qwen3_8b_a11y_20261101','qwen3_8b_download_20261101']:
  d=Path('/home/pxchen/workspace/services')/folder
  for p in sorted(d.iterdir()):
   if p.is_file():history.append(add(tar,p,'service/'+folder+'/'+p.name))
(OUT/'startup_history_files.json').write_text(json.dumps(history,indent=2))
records=OUT/'records';records.mkdir(exist_ok=True)
for p in RUN.iterdir():
 if p.is_file() and p.suffix in ['.json','.jsonl']:(records/p.name).write_bytes(p.read_bytes())
for a in ['BASE','FULL45']:
 for task in manifest['tasks']:
  for name in ['meta.json','result.json']:
   p=RUN/a/task/'p0'/name
   if p.exists():
    dst=records/'episodes'/a/task/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
readable=['episode_entry.py','host_supervisor.py','readiness.py','a11y_transport.py','initialize_a11y_device.py','finalize_a11y_device.py','test_a11y_protocol.py','test_a11y_runtime.py','host_supervisor_test.py','experiments/trace_reachability_validation/scripts/a11y_protocol.py','experiments/trace_reachability_validation/scripts/a11y_upstream_probe.py','experiments/trace_reachability_validation/scripts/full_run.py','experiments/trace_reachability_validation/scripts/full9.py','experiments/trace_reachability_validation/configs/qwen3_a11y.json','android_world/agents/t3a.py','android_world/env/actuation.py']
for rel in readable:
 p=ROOT/rel;dst=OUT/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
(OUT/'archive_snapshot.json').write_text(json.dumps(dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),formal_tag=TAG,formal_file_count=len(files),formal_nonimage_files=sum(x['included'] for x in files),formal_image_bytes=sum(x['bytes'] for x in files if not x['included']),key_screenshots=len(keys),source_file_count=len(source),source_hashes_verified=len(manifest['source_sha256']),earlier_trials_not_merged=True),indent=2))
for p in OUT.glob('*.tar.gz'):print(p.name,p.stat().st_size,flush=True)
# Bundle only the publish snapshot, not the live experiment tree.
with tarfile.open('/tmp/qwen3_publish_transfer.tgz','w:gz') as tar:
 for p in sorted(OUT.rglob('*')):
  if p.is_file():tar.add(p,arcname=str(p.relative_to(OUT)))
print('TRANSFER_READY',Path('/tmp/qwen3_publish_transfer.tgz').stat().st_size,flush=True)
