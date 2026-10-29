from pathlib import Path
import tarfile,json,hashlib,subprocess,datetime
root=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_single_20261029');assert not root.exists(),root
root.mkdir()
with tarfile.open('/tmp/cpx_pair_package_20261029.tar.gz') as t:t.extractall(root,filter='data')
exp=root/'experiments/trace_reachability_validation';run=exp/'raw/mai_official/mai8b_pair_single_c30x20_20261029';run.mkdir(parents=True)
tasks=(exp/'reports/e2e116n_task_order.txt').read_text().strip().split(',');assert len(tasks)==len(set(tasks))==116
up=json.loads((root/'upstream_sha256.json').read_text());mods=json.loads((root/'upstream_modifications.json').read_text())
for n,h in up.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==mods.get(n,{}).get('deployed',h),n
assert len(mods)==1 and 'experiments/trace_reachability_validation/scripts/full_run.py' in mods
source={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.txt'] and 'raw' not in p.parts}
manifest=dict(tag=run.name,tasks=tasks,task_seed=30,arms=['BASE','FULL45'],workers_per_arm=1,containers={'BASE':'pxchen-mai8b-base-w9-20261026','FULL45':'pxchen-mai8b-base-w10-20261026'},budget='min(30,int(20*task.complexity))',explicit_max_steps=0,upstream_commit='81a1531758da249c2acb74baa761d2da65e01b1c',upstream_patches=['Prevent recovery requery when model-call budget is already exhausted'],source_sha256=source,mai_config=json.loads((exp/'configs/mai_ui_8b_pair_c30x20.json').read_text()),graph_initialization='empty once, FULL45 sequential online accumulation over116 tasks',historical_observations=0,historical_statistics=True,baseline_graph=False,task_order='senior e2e116n_task_order.txt',task_order_sha256=hashlib.sha256((exp/'reports/e2e116n_task_order.txt').read_bytes()).hexdigest(),model='MAI-UI-8B',background_load=False,transport='official MAI nonstreaming',custom_recovery_guard=False,seenr=False,selection='all116 newly run, no preserved successes or retries',environment_note='Two existing idle pxchen emulators; normal benchmark task initialize/reset/tear_down; not recreated from pristine images',concurrency_note='One emulator per arm; both share same vLLM concurrently; inference load can change exploration window',prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(run/'run_config.json').write_text(json.dumps(manifest,indent=2))
for arm,c in manifest['containers'].items():
 result=subprocess.run(['docker','exec',c,'python','-m','py_compile','/androidworld/cpx_pair_single_20261029/run_pair_worker.py','/androidworld/cpx_pair_single_20261029/watch_pair.py','/androidworld/cpx_pair_single_20261029/experiments/trace_reachability_validation/scripts/mai_upstream_probe.py'],capture_output=True,text=True)
 assert result.returncode==0,result.stderr
print('DEPLOYED',root,'tasks116','upstreamfiles',len(up),flush=True)
