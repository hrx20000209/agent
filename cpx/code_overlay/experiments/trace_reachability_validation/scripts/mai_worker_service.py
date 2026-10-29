"""Detached worker wrapper with persistent logs and exit status."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from android_world import shared_state as SS


def main():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--tag', required=True)
  parser.add_argument('--arms', choices=['BASE', 'FULL45'], required=True)
  parser.add_argument('--worker-id', type=int, required=True)
  args, _ = parser.parse_known_args()
  if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.tag):
    parser.error('Invalid tag')
  root = Path(__file__).resolve().parents[1] / 'raw/mai_official' / args.tag
  workers = root / 'workers'
  workers.mkdir(parents=True, exist_ok=True)
  name = f'{args.arms}-{args.worker_id}'
  status_path = workers / (name + '.status.json')
  command = [sys.executable, '-u', str(Path(__file__).with_name('mai_androidworld_driver.py')), *sys.argv[1:]]
  def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
  def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Worker signal {signum}')
  signal.signal(signal.SIGTERM, interrupted)
  signal.signal(signal.SIGINT, interrupted)
  with SS.lock(workers / (name + '.service'), blocking=False):
    row = dict(arm=args.arms, worker_id=args.worker_id, command=command,
               started_at=now(), wrapper_pid=os.getpid(), container=os.environ.get('HOSTNAME'), status='RUNNING')
    proc = None
    try:
      with (workers / (name + '.log')).open('a', buffering=1) as logfile:
        proc = subprocess.Popen(command, cwd=REPO, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        row['driver_pid'] = proc.pid
        SS.atomic_text(status_path, json.dumps(row, indent=2))
        code = proc.wait()
      row.update(returncode=code, status='FINISHED' if code == 0 else 'FAILED')
    except BaseException as exc:
      row.update(status='INTERRUPTED', error=repr(exc))
    finally:
      if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
          proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
          os.killpg(proc.pid, signal.SIGKILL)
          proc.wait()
      row['ended_at'] = now()
      SS.atomic_text(status_path, json.dumps(row, indent=2))
    if row['status'] != 'FINISHED':
      raise SystemExit(1)


if __name__ == '__main__':
  main()
