import subprocess,json,pathlib,tarfile,hashlib
root=pathlib.Path('/home/pxchen/workspace/android_world_upstream81a1531_boundary_20261029')
assert not root.exists(),root
root.mkdir()
with tarfile.open('/tmp/cpx_latest_boundary_20261029.tar.gz') as t:t.extractall(root,filter='data')
m=json.loads((root/'upstream_sha256.json').read_text())
for name,sha in m.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
assert (root/'experiments/runtime_graph_exploration/graph/observations.jsonl').stat().st_size==0
name='pxchen-mai8b-upstream-boundary-20261029'
assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
source=json.loads(subprocess.check_output(['docker','inspect','pxchen-mai8b-base-w8-20261026'],text=True))[0]
args=['docker','run','-d','--name',name,'--label','mai-ui-experiment=upstream81a1531-boundary-20261029','--device','/dev/kvm','--memory','8g','--cpus','4','--shm-size','2g','--network','name=bridge,driver-opt=com.docker.network.endpoint.sysctls=net.ipv6.conf.IFNAME.disable_ipv6=0','--mount',f'type=bind,src={root},dst=/androidworld','--entrypoint','/usr/local/bin/python','pxchen-mai8b-shared-ready:20261024',*source['Config']['Cmd']]
print(subprocess.check_output(args,text=True),flush=True)
print('Verified upstream files',len(m),flush=True)
