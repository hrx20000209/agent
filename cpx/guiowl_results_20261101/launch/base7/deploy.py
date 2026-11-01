"""Create an isolated BASE-only retry of the seven recorded interface errors."""
from pathlib import Path
import subprocess,json,tarfile,hashlib,datetime,urllib.request

ROOT=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_guiowl_base7_20261101')
OLD=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_guiowl_20261031')
OLDTAG='guiowl8b_pair_single_c10_official_20261031'
TAG='guiowl8b_base7_openapp_fix_20261101'
TASKS=['MarkorCreateNote','MarkorChangeNoteContent','MarkorMergeNotes','NotesMeetingAttendeeCount','SimpleCalendarDeleteEventsOnRelativeDay','SportsTrackerLongestDistanceActivity','TasksDueNextWeek']
C='pxchen-guiowl8b-base7-20261101'
assert not ROOT.exists(),ROOT
assert subprocess.run(['docker','inspect',C],capture_output=True).returncode!=0,C
oldrun=OLD/'experiments/trace_reachability_validation/raw/mai_official'/OLDTAG
rows=[json.loads(l) for l in (oldrun/'BASE.runs.jsonl').read_text().splitlines()]
assert len(rows)==len({r['task'] for r in rows})==116
assert [r['task'] for r in rows if r.get('error')]==TASKS
for r in rows:
 if r['task'] in TASKS:assert 'Unsupported GUI-Owl action: open_app' in r['error']
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
models=json.load(op.open('http://127.0.0.1:18197/v1/models',timeout=10))
assert any(x['id']=='GUI-Owl-1.5-8B-Instruct' and x['max_model_len']==32768 for x in models['data'])
source=json.loads(subprocess.check_output(['docker','inspect','pxchen-guiowl8b-base-20261031'],text=True))[0]
ROOT.mkdir()
with tarfile.open('/tmp/cpx_guiowl_base7_20261101.tar.gz') as t:t.extractall(ROOT,filter='data')
exp=ROOT/'experiments/trace_reachability_validation'
run=exp/'raw/mai_official'/TAG;run.mkdir(parents=True)
oldmanifest=json.loads((oldrun/'run_config.json').read_text())
config=json.loads((exp/'configs/guiowl_8b_official.json').read_text())
assert config==oldmanifest['mai_config'],'Model settings changed'
oldproto=OLD/'experiments/trace_reachability_validation/scripts/owl_protocol.py'
newproto=ROOT/'experiments/trace_reachability_validation/scripts/owl_protocol.py'
expected=oldproto.read_text().replace(" if a not in {'key','click','long_press','swipe','type','system_button','open','wait','answer','terminate'}", " # Official mobile_agent_utils_new maps both spellings to OPEN_APP.\n if a not in {'key','click','long_press','swipe','type','system_button','open','open_app','wait','answer','terminate'}").replace(" if a=='open':", " if a in {'open','open_app'}:")
assert newproto.read_text()==expected,'Unexpected protocol changes'
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.txt'] and 'raw' not in p.parts and '__pycache__' not in p.parts}
manifest=dict(oldmanifest)
manifest.update(tag=TAG,tasks=TASKS,arms=['BASE'],containers={'BASE':C},source_sha256=hashes,prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='Only seven BASE tasks with recorded open_app adapter exceptions; keep other109 original results',original_run=str(oldrun),original_base_runs_sha256=hashlib.sha256((oldrun/'BASE.runs.jsonl').read_bytes()).hexdigest(),adapter_fix='Accept official open_app alias; validate app text; preserve original response and native action history',graph_initialization='BASE graph/exploration disabled',arm_options={'BASE':{'graph_execution':False,'exploration':False}},old_adapter_sha256=hashlib.sha256(oldproto.read_bytes()).hexdigest(),new_adapter_sha256=hashlib.sha256(newproto.read_bytes()).hexdigest(),image_id=source['Image'],emulator_command=source['Config']['Cmd'])
(run/'run_config.json').write_text(json.dumps(manifest,indent=2))
(run/'original_BASE.runs.jsonl').write_bytes((oldrun/'BASE.runs.jsonl').read_bytes())
fixtures=[]
for task in TASKS:
 p=oldrun/'BASE'/task/'p0/mai_raw_responses.jsonl'
 for line in p.read_text().splitlines():fixtures.append({'task':task,**json.loads(line)})
(run/'original_openapp_responses.json').write_text(json.dumps(fixtures,ensure_ascii=False,indent=2))
cmd=['docker','run','-d','--name',C,'--label','mai-ui-experiment='+TAG,'--device','/dev/kvm','--memory','8g','--cpus','4','--shm-size','2g','--network','name=bridge,driver-opt=com.docker.network.endpoint.sysctls=net.ipv6.conf.IFNAME.disable_ipv6=0','--mount',f'type=bind,src={ROOT},dst=/androidworld','--env','NO_PROXY=*','--env','no_proxy=*','--entrypoint','/usr/local/bin/python',source['Image'],*source['Config']['Cmd']]
print('CREATED',subprocess.check_output(cmd,text=True).strip(),flush=True)
env=['docker','exec','-e','FULL_ARM=BASE','-e','MAI_CONFIG=/androidworld/experiments/trace_reachability_validation/configs/guiowl_8b_official.json','-e','PYTHONPATH=/androidworld:/androidworld/experiments/trace_reachability_validation/scripts',C]
check=subprocess.run(env+['python','/androidworld/replay_openapp.py','/androidworld/'+str(run.relative_to(ROOT))],capture_output=True,text=True,timeout=120)
(run/'adapter_replay.log').write_text(check.stdout+check.stderr)
assert check.returncode==0,(check.stdout+check.stderr)[-3000:]
print(check.stdout,flush=True)
with (run/'host_supervisor.nohup.log').open('ab') as log:
 p=subprocess.Popen(['nohup','python3','-u',str(ROOT/'host_supervisor.py'),'--root',str(ROOT)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/'host_supervisor.pid').write_text(str(p.pid)+'\n')
print('STARTED',p.pid,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
