import subprocess,json,pathlib,datetime
c='pxchen-mai8b-base-w11-20261026'
root=pathlib.Path('/home/pxchen/workspace/android_world_mai8b_20261024/experiments/trace_reachability_validation/raw/mai_official/mai8b_base_full116_calls50_12w_20261026')
assert not (root/'workers/BASE-11.status.json').exists()
d=root/'startup_diagnostics';d.mkdir(exist_ok=True)
for name,args in [('worker11_docker.log',['docker','logs','--tail','200',c]),('worker11_accessibility.txt',['docker','exec',c,'adb','shell','dumpsys','accessibility'])]:
 r=subprocess.run(args,capture_output=True,text=True,timeout=20);(d/name).write_text(r.stdout+r.stderr)
subprocess.run(['docker','stop','-t','10',c],check=True)
subprocess.run(['docker','rename',c,c+'-unready'],check=True)
source=json.loads(subprocess.check_output(['docker','inspect','pxchen-mai8b-base-w2-20261025'],text=True))[0]
args=['docker','run','-d','--name',c,'--label','mai-ui-experiment=base116-12w-20261026','--device','/dev/kvm','--memory','8g','--cpus','4','--shm-size','2g','--network','name=bridge,driver-opt=com.docker.network.endpoint.sysctls=net.ipv6.conf.IFNAME.disable_ipv6=0','--mount','type=bind,src=/home/pxchen/workspace/android_world_mai8b_20261024,dst=/androidworld','--entrypoint','/usr/local/bin/python','pxchen-mai8b-shared-ready:20261024',*source['Config']['Cmd']]
subprocess.run(args,check=True)
print('Replaced idle startup-only worker11; old stopped container retained for diagnosis',flush=True)
