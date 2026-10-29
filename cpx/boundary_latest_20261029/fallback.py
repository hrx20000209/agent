from pathlib import Path
import shutil,json,subprocess,datetime
src=Path('/home/pxchen/workspace/android_world_upstream81a1531_boundary_20261029');dst=Path('/home/pxchen/workspace/android_world_mai8b_20261024/cpx_upstream81a1531_boundary_20261029');assert not dst.exists()
# Preserve the failed fresh-emulator attempt separately; no episode was started.
status=src/'experiments/trace_reachability_validation/raw/mai_official/mai8b_upstream81a1531_boundary_complexity_20261029/status.json';s=json.loads(status.read_text());assert not s['completed'];s.update(status='FAILED_ENV_BOOT',error='Fresh emulator container exited139, QEMU CPU3 thread hang; no tasks attempted');(src/'fresh_attempt_failure.json').write_text(json.dumps(s,indent=2))
(src/'fresh_emulator_inspect.json').write_text(subprocess.check_output(['docker','inspect','pxchen-mai8b-upstream-boundary-20261029'],text=True))
with (src/'fresh_emulator_boot.log').open('w') as f:subprocess.run(['docker','logs','pxchen-mai8b-upstream-boundary-20261029'],stdout=f,stderr=subprocess.STDOUT)
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('mai_official','__pycache__','boundary_driver.log'))
shutil.copy2('/tmp/cpx_boundary_driver_updated.py',dst/'run_boundary.py')
meta=dict(container='pxchen-mai8b-base-w8-20261026',reason='Fresh emulator139 before any tasks; use idle existing emulator with benchmark task initialization; not a pristine-image formal comparison',fresh_attempt=str(src),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(dst/'environment_provenance.json').write_text(json.dumps(meta,indent=2))
cmd='cd /androidworld/cpx_upstream81a1531_boundary_20261029 && CPX_PROBE_ROOT=/androidworld/cpx_upstream81a1531_boundary_20261029 exec nohup python -u run_boundary.py > boundary_driver.log 2>&1 < /dev/null'
subprocess.run(['docker','exec','-d','pxchen-mai8b-base-w8-20261026','sh','-c',cmd],check=True)
print('STARTED',dst)
