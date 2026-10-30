from pathlib import Path
import subprocess,json,tarfile,hashlib,datetime,urllib.request,os
ROOT=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_pair_single_rerun_20261030');TAG='mai8b_pair_single_c30x20_hostwatch_20261030';assert not ROOT.exists(),ROOT;ROOT.mkdir()
with tarfile.open('/tmp/cpx_rerun_single_20261030.tar.gz') as t:t.extractall(ROOT,filter='data')
exp=ROOT/'experiments/trace_reachability_validation';run=exp/'raw/mai_official'/TAG;run.mkdir(parents=True)
tasks=(exp/'reports/e2e116n_task_order.txt').read_text().strip().split(',');assert len(tasks)==len(set(tasks))==116
up=json.load(open(ROOT/'upstream_sha256.json'));mods=json.load(open(ROOT/'upstream_modifications.json'))
for name,h in up.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==mods.get(name,{}).get('deployed',h),name
source=json.loads(subprocess.check_output(['docker','inspect','pxchen-mai8b-base-w9-20261026'],text=True))[0]
containers={'BASE':'pxchen-mai8b-base-single-20261030','FULL45':'pxchen-mai8b-full45-single-20261030'}
for c in containers.values():assert subprocess.run(['docker','inspect',c],capture_output=True).returncode!=0,c
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));models=json.load(opener.open('http://127.0.0.1:8097/v1/models',timeout=10));assert any(m['id']=='MAI-UI-8B' and m['max_model_len']==24576 for m in models['data'])
ps=subprocess.check_output(['ps','-eo','pid,lstart,args'],text=True);service=[l for l in ps.splitlines() if 'vllm.entrypoints.openai.api_server' in l and '--served-model-name MAI-UI-8B' in l]
configpath=exp/'configs/mai_ui_8b_pair_c30x20.json';config=json.load(open(configpath));config['evaluation_note']='20261030 fresh116 rerun per arm, two new containers; host-side supervision; no preserved old successes; min(30,int(20*complexity))';config['server']['gpu_memory_utilization']=0.9;config['server']['cpu_offload_gb']=2;configpath.write_text(json.dumps(config,indent=2))
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.txt'] and 'raw' not in p.parts and '__pycache__' not in p.parts}
manifest=dict(tag=TAG,prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks=tasks,task_seed=30,arms=['BASE','FULL45'],workers_per_arm=1,containers=containers,model='MAI-UI-8B',api_url='http://172.17.0.1:8097/v1/chat/completions',budget='min(30,int(20*complexity))',task_order_source='senior e2e116n_task_order.txt',task_order_sha256=hashlib.sha256((exp/'reports/e2e116n_task_order.txt').read_bytes()).hexdigest(),upstream_commit='81a1531758da249c2acb74baa761d2da65e01b1c',algorithm_patch='only recovery-requery budget guard',custom_recovery_guard=False,seenr=False,graph_initialization='empty once; only sequential FULL45 tasks update its graph',historical_statistics=True,historical_observations=0,source_sha256=hashes,mai_config=config,actual_vllm_process=service,model_service_restarted_by_this_run=False,image_id=source['Image'],emulator_command=source['Config']['Cmd'],host_supervision=dict(poll_s=5,heartbeat_s=15,no_progress_s=600,episode_hard_timeout_s=2700,environment_recovery_attempts=3,task_retry=False,consecutive_infra_failure_stop=3),environment='two fresh containers from identical preconfigured image; benchmark task initialization',selection='all116 newly run per arm; infrastructure errors retained; no best-attempt selection')
(run/'run_config.json').write_text(json.dumps(manifest,indent=2))
for arm,c in containers.items():
 args=['docker','run','-d','--name',c,'--label','mai-ui-experiment='+TAG,'--device','/dev/kvm','--memory','8g','--cpus','4','--shm-size','2g','--network','name=bridge,driver-opt=com.docker.network.endpoint.sysctls=net.ipv6.conf.IFNAME.disable_ipv6=0','--mount',f'type=bind,src={ROOT},dst=/androidworld','--env','NO_PROXY=*','--env','no_proxy=*','--entrypoint','/usr/local/bin/python',source['Image'],*source['Config']['Cmd']]
 print('CREATED',arm,subprocess.check_output(args,text=True).strip(),flush=True)
print('ROOT',ROOT,flush=True)
# Host supervision survives container exit and SSH logout.
with (run/'host_supervisor.nohup.log').open('ab') as log:
 p=subprocess.Popen(['nohup','python3','-u',str(ROOT/'host_supervisor.py'),'--root',str(ROOT)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/'host_supervisor.pid').write_text(str(p.pid)+'\n');print('HOST_SUPERVISOR',p.pid,'STARTED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
