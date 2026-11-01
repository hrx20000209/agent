"""Rerun the complete ordered FULL45 suite with only the open_app parser fix."""
from pathlib import Path
import subprocess,json,tarfile,hashlib,datetime,urllib.request

ROOT=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_guiowl_full45_fix_20261101')
OLD=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_guiowl_20261031')
OLDTAG='guiowl8b_pair_single_c10_official_20261031'
TAG='guiowl8b_full45_openapp_fix_20261101'
C='pxchen-guiowl8b-full45-fix-20261101'
assert not ROOT.exists(),ROOT
assert subprocess.run(['docker','inspect',C],capture_output=True).returncode!=0,C
oldrun=OLD/'experiments/trace_reachability_validation/raw/mai_official'/OLDTAG
oldmanifest=json.loads((oldrun/'run_config.json').read_text())
tasks=oldmanifest['tasks'];assert len(tasks)==len(set(tasks))==116
source=json.loads(subprocess.check_output(['docker','inspect','pxchen-guiowl8b-full45-20261031'],text=True))[0]
assert source['Image']==oldmanifest['image_id']
assert source['Config']['Cmd']==oldmanifest['emulator_command']
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
models=json.load(op.open('http://127.0.0.1:18197/v1/models',timeout=10))
assert any(x['id']=='GUI-Owl-1.5-8B-Instruct' and x['max_model_len']==32768 for x in models['data'])
service=Path('/home/pxchen/workspace/services/gui_owl_15_8b_gpu45_20261031')
pid=int((service/'server.pid').read_text());vargs=Path(f'/proc/{pid}/cmdline').read_text().replace('\0',' ')
for flag in ['--port 18197','--cpu-offload-gb 0','--tensor-parallel-size 2','--max-num-seqs 2']:assert flag in vargs,flag
ROOT.mkdir()
with tarfile.open('/tmp/cpx_guiowl_full45_fix_20261101.tar.gz') as t:t.extractall(ROOT,filter='data')
exp=ROOT/'experiments/trace_reachability_validation';run=exp/'raw/mai_official'/TAG;run.mkdir(parents=True)
config=json.loads((exp/'configs/guiowl_8b_official.json').read_text());assert config==oldmanifest['mai_config']
order=exp/'reports/e2e116n_task_order.txt'
assert order.read_text().strip().split(',')==tasks
assert hashlib.sha256(order.read_bytes()).hexdigest()==oldmanifest['task_order_sha256']
assert (ROOT/'experiments/runtime_graph_exploration/graph/observations.jsonl').stat().st_size==0
assert not (run/'graph').exists()
proto='experiments/trace_reachability_validation/scripts/owl_protocol.py'
expected=(OLD/proto).read_text().replace(" if a not in {'key','click','long_press','swipe','type','system_button','open','wait','answer','terminate'}", " # Official mobile_agent_utils_new maps both spellings to OPEN_APP.\n if a not in {'key','click','long_press','swipe','type','system_button','open','open_app','wait','answer','terminate'}").replace(" if a=='open':", " if a in {'open','open_app'}:")
assert (ROOT/proto).read_text()==expected
allowed={proto,'host_supervisor.py','owl_protocol_test.py'}
for name,h in oldmanifest['source_sha256'].items():
 if name not in allowed:assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
expected_host=(OLD/'host_supervisor.py').read_text().replace("TAG='guiowl8b_pair_single_c10_official_20261031'",f'TAG={TAG!r}').replace("runners=[Runner(a.root,arm,manifest) for arm in ['BASE','FULL45']]", "runners=[Runner(a.root,arm,manifest) for arm in manifest.get('arms',['BASE','FULL45'])]").replace('with concurrent.futures.ThreadPoolExecutor(2) as pool:', 'with concurrent.futures.ThreadPoolExecutor(len(runners)) as pool:')
assert (ROOT/'host_supervisor.py').read_text()==expected_host
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.txt'] and 'raw' not in p.parts and '__pycache__' not in p.parts}
manifest=dict(oldmanifest)
manifest.update(tag=TAG,arms=['FULL45'],containers={'FULL45':C},source_sha256=hashes,prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_run=str(oldrun),selection='Fresh full116 FULL45 run in original order; no original results or final graph reused',adapter_fix='Accept official open_app alias; all other model/algorithm settings unchanged',graph_initialization='empty once; sequential online accumulation across all116; same historical statistics',arm_options={'FULL45':oldmanifest['arm_options']['FULL45']},actual_vllm_pid=pid,actual_vllm_args=vargs,source_difference_allowlist=sorted(allowed),supervisor_difference='new tag and FULL45-only arm selection; no episode policy changes',old_adapter_sha256=hashlib.sha256((OLD/proto).read_bytes()).hexdigest(),new_adapter_sha256=hashlib.sha256((ROOT/proto).read_bytes()).hexdigest())
(run/'run_config.json').write_text(json.dumps(manifest,indent=2))
cmd=['docker','run','-d','--name',C,'--label','mai-ui-experiment='+TAG,'--device','/dev/kvm','--memory','8g','--cpus','4','--shm-size','2g','--network','name=bridge,driver-opt=com.docker.network.endpoint.sysctls=net.ipv6.conf.IFNAME.disable_ipv6=0','--mount',f'type=bind,src={ROOT},dst=/androidworld','--env','NO_PROXY=*','--env','no_proxy=*','--entrypoint','/usr/local/bin/python',source['Image'],*source['Config']['Cmd']]
print('CREATED',subprocess.check_output(cmd,text=True).strip(),flush=True)
cmd=['docker','exec','-e','FULL_ARM=FULL45','-e','MAI_CONFIG=/androidworld/experiments/trace_reachability_validation/configs/guiowl_8b_official.json','-e','PYTHONPATH=/androidworld:/androidworld/experiments/trace_reachability_validation/scripts',C,'python','-c',"import owl_upstream_probe as O; from types import SimpleNamespace as N; assert O.F.ARM=='FULL45'; assert O.F.FullEpisode.max_steps(N(task=N(complexity=4.8)))==48; assert O.CONFIG['temperature']==0 and O.CONFIG['history_n']==5; import unittest,owl_protocol_test; r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(owl_protocol_test)); assert r.wasSuccessful(); print('RUNTIME_FULL45_IMPORT_PROTOCOL_BUDGET_OK')"]
check=subprocess.run(cmd,capture_output=True,text=True,timeout=120);(run/'runtime_import_check.log').write_text(check.stdout+check.stderr)
assert check.returncode==0,(check.stdout+check.stderr)[-3000:]
print(check.stdout,flush=True)
with (run/'host_supervisor.nohup.log').open('ab') as log:
 p=subprocess.Popen(['nohup','python3','-u',str(ROOT/'host_supervisor.py'),'--root',str(ROOT)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/'host_supervisor.pid').write_text(str(p.pid)+'\n')
print('STARTED',p.pid,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
