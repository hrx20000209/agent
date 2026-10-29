"""Continue only unattempted episodes; retain original driver and run manifest."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import mai_androidworld_driver as D


def now():
  return datetime.datetime.now(datetime.timezone.utc).isoformat()


def recover(api, model, config):
  """Recover observation transport only between episodes, never during one."""
  try:
    D.preflight(api, model, config)
    return
  except Exception as exc:
    print('PREFLIGHT_RECOVERY', repr(exc), flush=True)
  def adb(*args):
    return subprocess.run(['adb', '-s', 'emulator-5554', *args], check=True,
                          capture_output=True, text=True, timeout=20).stdout.strip()
  services = adb('shell', 'settings', 'get', 'secure', 'enabled_accessibility_services')
  if 'com.androidworld.fasta11y/' not in services:
    raise RuntimeError('Expected installed accessibility service missing')
  adb('shell', 'settings', 'put', 'secure', 'enabled_accessibility_services',
      ':'.join(s for s in services.split(':') if not s.startswith('com.androidworld.fasta11y/')))
  time.sleep(1)
  adb('shell', 'settings', 'put', 'secure', 'enabled_accessibility_services', services)
  adb('shell', 'settings', 'put', 'secure', 'accessibility_enabled', '1')
  adb('shell', 'input', 'keyevent', 'KEYCODE_HOME')
  for i in range(5):
    time.sleep(2)
    try:
      D.preflight(api, model, config)
      return
    except Exception:
      if i == 4:
        raise


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--tag', required=True)
  p.add_argument('--arm', choices=['BASE', 'FULL45'], required=True)
  p.add_argument('--worker-id', type=int, required=True)
  p.add_argument('--retry-root', type=Path)
  p.add_argument('--retry-task')
  a = p.parse_args()
  run = D.ROOT / 'raw/mai_official' / a.tag
  config = json.loads((run / 'run_config.json').read_text())
  # Refuse to continue if any original algorithm/client/statistics source changed.
  for name, expected in config['source_sha256'].items():
    assert hashlib.sha256((D.ROOT / 'scripts' / name).read_bytes()).hexdigest() == expected, name
  for name, expected in config['shared_source_sha256'].items():
    assert hashlib.sha256((D.REPO / name).read_bytes()).hexdigest() == expected, name
  args = SimpleNamespace(tag=a.tag, workers=config['workers_per_arm'], worker_id=a.worker_id,
      seed=config['task_seed'], config=D.ROOT / 'configs/mai_ui_8b_full_20261025.json', smoke=False)
  assert json.loads(args.config.read_text()) == config['mai_config']
  status_path = run / 'workers' / f'{a.arm}-{a.worker_id}.continuation.status.json'
  state = dict(status='RUNNING', started_at=now(), pid=os.getpid(), arm=a.arm,
               worker_id=a.worker_id, completed=[], retained_errors=[])
  def save():
    D.SS.atomic_text(status_path, json.dumps(state, indent=2))
  with D.SS.lock(run / 'workers' / f'{a.arm}-{a.worker_id}.claim', blocking=False):
    save()
    try:
      targets = D.worker_tasks(config['tasks'], args.workers, a.worker_id)
      target_run = run
      if a.retry_task:
        assert a.retry_root and a.retry_task in config['tasks']
        target_run = a.retry_root
        target_run.mkdir(parents=True, exist_ok=True)
        if not (target_run / 'graph').exists():
          (target_run / 'graph').symlink_to(run / 'graph', target_is_directory=True)
        targets = [(config['tasks'].index(a.retry_task), a.retry_task)]
      for _, task in targets:
        result = target_run / a.arm / task / 'p0/result.json'
        if result.exists():
          if json.loads(result.read_text()).get('error'):
            state['retained_errors'].append(task)
          continue
        recover(config['api_url'], config['model'], config['mai_config'])
        # Keep the original worker ID and shared FULL45 behavior. The original
        # driver's modulo partition selects exactly one entry in this list.
        try:
          D.run_tasks(args, [a.arm], [task] * (a.worker_id + 1), target_run,
                      config['api_url'], config['model'], config['mai_config'])
          state['completed'].append(task)
        except Exception as exc:
          if not result.exists():
            raise
          state['retained_errors'].append(task)
          print('EPISODE_ERROR_RETAINED', task, repr(exc), flush=True)
        save()
      state['status'] = 'FINISHED_WITH_ERRORS' if state['retained_errors'] else 'FINISHED'
    except BaseException as exc:
      state.update(status='FAILED', error=repr(exc))
      raise
    finally:
      state['ended_at'] = now()
      save()


if __name__ == '__main__':
  main()
