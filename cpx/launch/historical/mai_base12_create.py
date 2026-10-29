import subprocess,json
source=json.loads(subprocess.check_output(['docker','inspect','pxchen-mai8b-base-w2-20261025'],text=True))[0]
for i in range(8,12):
 name=f'pxchen-mai8b-base-w{i}-20261026'
 exists=subprocess.run(['docker','inspect',name],capture_output=True).returncode==0
 if exists:raise RuntimeError('Container name already exists: '+name)
 args=['docker','run','-d','--name',name,'--label','mai-ui-experiment=base116-12w-20261026','--device','/dev/kvm','--memory','8g','--cpus','4','--shm-size','2g','--network','name=bridge,driver-opt=com.docker.network.endpoint.sysctls=net.ipv6.conf.IFNAME.disable_ipv6=0','--mount','type=bind,src=/home/pxchen/workspace/android_world_mai8b_20261024,dst=/androidworld','--entrypoint','/usr/local/bin/python','pxchen-mai8b-shared-ready:20261024',*source['Config']['Cmd']]
 r=subprocess.run(args,text=True,capture_output=True,check=True)
 print('CREATED',name,r.stdout.strip(),flush=True)
