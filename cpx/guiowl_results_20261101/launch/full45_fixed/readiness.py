import os,sys,subprocess,time,json
from pathlib import Path
root=Path('/androidworld');sys.path.insert(0,str(root));sys.path.insert(0,str(root/'experiments/trace_reachability_validation/scripts'))
from mai_resume_20261025 import recover
from mai_warm_worker import sms_probe
c=json.load(open(root/'experiments/trace_reachability_validation/configs/guiowl_8b_official.json'))
for _ in range(60):
 try:
  p=subprocess.run(['adb','shell','getprop','sys.boot_completed'],capture_output=True,text=True,timeout=10)
  if p.stdout.strip()=='1':break
 except Exception:pass
 time.sleep(3)
else:raise RuntimeError('Boot not ready')
if not Path('/tmp/cpx_hostwatch_ready').exists():
 subprocess.run(['adb','root'],capture_output=True,timeout=20);subprocess.run(['adb','wait-for-device'],check=True,timeout=30)
for attempt in range(4):
 try:
  recover('http://172.17.0.1:18197/v1/chat/completions','GUI-Owl-1.5-8B-Instruct',c);break
 except Exception:
  if attempt==3:raise
  time.sleep(3)
if not Path('/tmp/cpx_hostwatch_ready').exists():sms_probe();Path('/tmp/cpx_hostwatch_ready').write_text('ready')
print('READY')
