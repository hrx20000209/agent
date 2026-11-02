import subprocess,time,json,urllib.request,socket
from pathlib import Path
for _ in range(60):
 try:
  p=subprocess.run(['adb','shell','getprop','sys.boot_completed'],capture_output=True,text=True,timeout=10)
  if p.stdout.strip()=='1':break
 except Exception:pass
 time.sleep(3)
else:raise RuntimeError('Boot not ready')
subprocess.run(['adb','root'],capture_output=True,timeout=20)
subprocess.run(['adb','wait-for-device'],check=True,timeout=30)
assert subprocess.check_output(['adb','shell','id','-u'],text=True,timeout=15).strip()=='0','adbd must be root after every boot'
assert subprocess.run(['adb','shell','test','-d','/data/data/android_world/snapshots/net.gsantner.markor'],timeout=15).returncode==0,'initialization snapshot missing'
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with op.open('http://172.17.0.1:18198/v1/models',timeout=10) as r:
 assert any(m['id']=='Qwen3-8B-A11Y' for m in json.load(r)['data'])
with socket.create_connection(('127.0.0.1',8554),timeout=10):pass
print('READY')
