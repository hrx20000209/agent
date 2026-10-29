"""Evaluate BASE versus FULL45, each episode starting from a fixed snapshot."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import mai_androidworld_driver as D
from mai_resume_20261025 import recover


def now():
  return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_snapshot(snapshot):
  m = json.loads((snapshot / 'manifest.json').read_text())
  for name, expected in m['files_sha256'].items():
    p = snapshot / name
    if not p.resolve().is_relative_to(snapshot.resolve()):
      raise RuntimeError('Snapshot path escaped root')
    if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
      raise RuntimeError('Snapshot hash mismatch: ' + name)
  return hashlib.sha256((snapshot / 'manifest.json').read_bytes()).hexdigest()


def make_episode(run, arm, task, snapshot):
  bundle = run / 'episodes' / arm / task
  if bundle.exists():
    raise RuntimeError('Partial episode retained; inspect before retrying: ' + str(bundle))
  out = bundle / 'raw' / task / 'p0'
  if arm == 'FULL45':
    shutil.copytree(snapshot / 'graph/FULL45', bundle / 'graph/FULL45')
    for p in (bundle / 'graph').rglob('*'):
      p.chmod(0o755 if p.is_dir() else 0o644)
  out.parent.mkdir(parents=True, exist_ok=True)
  canonical = run / arm / task / 'p0'
  canonical.parent.mkdir(parents=True, exist_ok=True)
  canonical.symlink_to(out, target_is_directory=True)
  return bundle, out


def sms_probe():
  # Must run before the first episode, never while the model is acting.
  socket.getaddrinfo('::1', 1234, 0, socket.SOCK_STREAM, 0, socket.AI_ADDRCONFIG)
  def adb(*args):
    return subprocess.run(['adb', *args], check=True, capture_output=True,
                          text=True, timeout=20).stdout
  adb('emu', 'sms', 'send', '15551234567', 'MAI_WARM_READINESS_PROBE')
  for _ in range(12):
    time.sleep(1)
    if 'MAI_WARM_READINESS_PROBE' in adb('shell', 'content', 'query', '--uri', 'content://sms/inbox'):
      break
  else:
    raise RuntimeError('Actual incoming SMS delivery did not pass')
  adb('shell', 'content delete --uri content://sms --where "body=\'MAI_WARM_READINESS_PROBE\'"')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--tag', required=True)
  p.add_argument('--arm', choices=['BASE', 'FULL45'], required=True)
  p.add_argument('--worker-id', type=int, required=True)
  p.add_argument('--workers', type=int, default=4)
  p.add_argument('--snapshot', type=Path, required=True)
  p.add_argument('--tasks')
  p.add_argument('--evaluation-arms', default='BASE,FULL45')
  p.add_argument('--repair-reconciliation', action='store_true')
  a = p.parse_args()
  if not a.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.tag):
    p.error('Invalid tag')
  if not 0 <= a.worker_id < a.workers:
    p.error('Invalid worker ID')
  arms = a.evaluation_arms.split(',')
  if len(set(arms)) != len(arms) or any(x not in ['BASE', 'FULL45'] for x in arms) or a.arm not in arms:
    p.error('Invalid evaluation arms')
  snapshot = a.snapshot.resolve()
  snap_sha = verify_snapshot(snapshot)
  source = Path(json.loads((snapshot / 'manifest.json').read_text())['source_run'])
  original = json.loads((source / 'run_config.json').read_text())
  tasks = a.tasks.split(',') if a.tasks else original['tasks']
  assert len(set(tasks)) == len(tasks) and set(tasks) <= set(original['tasks'])
  run = D.ROOT / 'raw/mai_official' / a.tag
  run.mkdir(parents=True, exist_ok=True)
  settings = dict(arms=arms, tasks=tasks, workers_per_arm=a.workers,
      task_seed=original['task_seed'], api_url=original['api_url'], model=original['model'],
      mai_config=original['mai_config'], snapshot=str(snapshot), snapshot_sha256=snap_sha,
      graph_initialization='BASE no graph; FULL45 identical snapshot per episode with private within-episode writes',
      cross_evaluation_episode_learning=False,
      cohort_note='Same-task warm graph evaluation; BASE no graph; FULL45 final full232 snapshot per episode. Not a cold-start or held-out-task result.',
      source_sha256={str(f.relative_to(D.REPO)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in [Path(__file__), D.REPO / 'android_world/shared_state.py',
                               D.REPO / 'android_world/env/interface.py',
                               *[D.ROOT / 'scripts' / n for n in original['source_sha256']]]})
  if a.repair_reconciliation:
    assert arms == ['FULL45'] and len(tasks) == 7
    settings['repair_reconciliation'] = dict(original_tag=source.name,
        baseline_sms_tag='mai8b_sms_repair14_20261025', repair_tasks=tasks)
  def establish(old):
    if old is not None and old != settings:
      raise RuntimeError('Existing warm run has a different manifest')
    return settings
  D.SS.update_json(run / 'run_config.json', establish, None)
  name = f'{a.arm}-{a.worker_id}'
  status_path = run / 'workers' / (name + '.status.json')
  state = dict(status='RUNNING', started_at=now(), pid=os.getpid(), arm=a.arm,
               worker_id=a.worker_id, completed=[], snapshot_sha256=snap_sha)
  def save():
    D.SS.atomic_text(status_path, json.dumps(state, indent=2))
  with D.SS.lock(Path('/tmp/mai_device_experiment.claim'), blocking=False), \
       D.SS.lock(run / 'workers' / (name + '.claim'), blocking=False):
    save()
    try:
      recover(original['api_url'], original['model'], original['mai_config'])
      sms_probe()
      for _, task in D.worker_tasks(tasks, a.workers, a.worker_id):
        canonical = run / a.arm / task / 'p0'
        if canonical.exists() and (canonical / '.shared_complete').exists():
          continue
        verify_snapshot(snapshot)
        recover(original['api_url'], original['model'], original['mai_config'])
        bundle, out = make_episode(run, a.arm, task, snapshot)
        env = dict(os.environ, FULL_ARM=a.arm, SEQ_APPDIR=str(bundle / 'graph'),
            MAI_CONFIG=str(D.ROOT / 'configs/mai_ui_8b_full_20261025.json'),
            ONDEVICE_LOAD='0', MAI_SMOKE_EMPTY_STATS='0',
            MAI_SHARED_STATE='1' if a.arm == 'FULL45' else '0',
            MAI_WORKER_ID=str(a.worker_id),
            MAI_EPISODE_ROOTS_JSON=str(snapshot / 'episode_roots.json'))
        cmd = [sys.executable, str(D.ROOT / 'scripts/mai_run.py'), '--task', task,
            '--seed', str(original['task_seed']), '--variant', f'{a.tag}_{a.arm}',
            '--mode', 'agent', '--out', str(out), '--api_url', original['api_url'],
            '--model', original['model']]
        t = time.time()
        with (run / f'{a.arm}_{task}.log').open('w') as log:
          rc = subprocess.run(cmd, cwd=D.REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=2400).returncode
        result = json.loads((out / 'result.json').read_text()) if (out / 'result.json').exists() else {'error': 'no result'}
        attempts = out / 'mai_attempts.jsonl'
        row = dict(task=task, arm=a.arm, worker_id=a.worker_id, rc=rc,
            e2e_s=round(time.time() - t, 2), success=result.get('is_successful'),
            model_calls=result.get('model_calls'), error=result.get('error'),
            http_attempts=len(attempts.read_text().splitlines()) if attempts.exists() else 0,
            snapshot_sha256=snap_sha, episode_output=str(out))
        D.SS.append_json(run / 'runs.jsonl', row)
        print(json.dumps(row), flush=True)
        if rc != 0 or row['error']:
          raise RuntimeError('Episode error retained: ' + task)
        D.SS.atomic_text(out / '.shared_complete', json.dumps(row))
        state['completed'].append(task)
        save()
      verify_snapshot(snapshot)
      state['status'] = 'FINISHED'
    except BaseException as exc:
      state.update(status='FAILED', error=repr(exc))
      raise
    finally:
      state['ended_at'] = now()
      save()


if __name__ == '__main__':
  main()
