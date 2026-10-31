from pathlib import Path
import json,subprocess,os,socket,datetime,shutil
root=Path('/home/pxchen/workspace/services/gui_owl_15_8b_gpu45_20261031');root.mkdir(exist_ok=True)
assert not (root/'server.pid').exists(),'Inspect existing service before launching another'
uuids=['GPU-d152c6a7-1dcf-0ff2-1802-80647045e446','GPU-4630f136-1e70-b954-b995-1f460998c448']
for uuid in uuids:
 m=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip();assert int(m)<200,(uuid,m)
s=socket.socket();s.bind(('127.0.0.1',18197));s.close()
model=Path('/home/pxchen/workspace/models/GUI-Owl-1.5-8B-Instruct');proof=json.loads((model/'DOWNLOAD_PROVENANCE.json').read_text());assert proof['revision']=='06d5faecff74840bab2be2425e9c42667a5d04fc'
python='/home/pxchen/workspace/services/minicpmo45_20260929/venv312/bin/python'
sampling=dict(temperature=0.0,top_p=1.0,top_k=-1,max_new_tokens=2048,repetition_penalty=1.0)
args=[python,'-m','vllm.entrypoints.openai.api_server','--model',str(model),'--served-model-name','GUI-Owl-1.5-8B-Instruct','--host','0.0.0.0','--port','18197','--tensor-parallel-size','2','--dtype','bfloat16','--max-model-len','32768','--gpu-memory-utilization','0.90','--max-num-seqs','2','--max-num-batched-tokens','4096','--limit-mm-per-prompt','{"image":5,"video":0}','--mm-processor-kwargs','{"size":{"longest_edge":3072000,"shortest_edge":65536}}','--seed','42','--enforce-eager','--attention-backend','TRITON_ATTN','--cpu-offload-gb','0','--generation-config','vllm','--override-generation-config',json.dumps(sampling)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids),CUDA_HOME='/usr/local/cuda-12.6',TRITON_PTXAS_PATH='/usr/local/cuda-12.6/bin/ptxas',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',VLLM_CACHE_ROOT=str(root/'vllm-cache'),NO_PROXY='*',no_proxy='*');env['PATH']='/usr/local/cuda-12.6/bin:'+str(Path(python).parent)+':'+env['PATH']
for k in ['http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']:env.pop(k,None)
with (root/'server.log').open('ab') as log:p=subprocess.Popen(['nohup',*args],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'server.pid').write_text(str(p.pid)+'\n');(root/'launch.json').write_text(json.dumps(dict(args=args,gpu_uuids=uuids,pid=p.pid,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_revision=proof['revision'],purpose='User requested replacement of three pxchen MAI services; pure GPU GUI-Owl',sampling=sampling,request_seed=42),indent=2)+'\n');print('STARTED',p.pid,root,flush=True)
