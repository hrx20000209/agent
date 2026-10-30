from pathlib import Path
import json,subprocess,os,socket,datetime
root=Path('/home/pxchen/workspace/services/mai_ui_8b_gpu5_puregpu_20261030');assert not root.exists(),root
uuid='GPU-4630f136-1e70-b954-b995-1f460998c448'
m=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip();assert int(m)<200,'GPU5 is no longer idle: '+m
s=socket.socket();s.bind(('127.0.0.1',18197));s.close();root.mkdir()
python='/home/pxchen/workspace/services/minicpmo45_20260929/venv312/bin/python'
args=[python,'-m','vllm.entrypoints.openai.api_server','--model','/home/pxchen/workspace/models/MAI-UI-8B','--served-model-name','MAI-UI-8B','--host','0.0.0.0','--port','18197','--tensor-parallel-size','1','--dtype','bfloat16','--max-model-len','24576','--gpu-memory-utilization','0.97','--max-num-seqs','8','--max-num-batched-tokens','1024','--limit-mm-per-prompt','{"image":3,"video":0}','--seed','0','--enforce-eager','--attention-backend','TRITON_ATTN','--cpu-offload-gb','0']
env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_HOME='/usr/local/cuda-12.6',TRITON_PTXAS_PATH='/usr/local/cuda-12.6/bin/ptxas',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',VLLM_CACHE_ROOT=str(root/'vllm-cache'),NO_PROXY='*',no_proxy='*');env['PATH']='/usr/local/cuda-12.6/bin:'+str(Path(python).parent)+':'+env['PATH']
for key in ['http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']:env.pop(key,None)
with (root/'server.log').open('ab') as log:p=subprocess.Popen(['nohup',*args],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'server.pid').write_text(str(p.pid)+'\n');(root/'launch.json').write_text(json.dumps(dict(args=args,gpu_uuid=uuid,pid=p.pid,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='Pure GPU benchmark; original service8097 preserved'),indent=2));print('STARTED',p.pid,root,flush=True)
