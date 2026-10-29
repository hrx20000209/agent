"""FULL45 full-suite 50-call evaluation with one initially empty shared graph."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import mai_androidworld_driver as D
from mai_resume_20261025 import recover
from mai_warm_worker import sms_probe


def now():
  return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--tag', required=True)
  p.add_argument('--worker-id', type=int, required=True)
  p.add_argument('--workers', type=int, default=8)
  p.add_argument('--max-calls', type=int, default=50)
  p.add_argument('--config', type=Path, required=True)
  p.add_argument('--source-run', default='mai8b_full232_shared_20261025')
  a = p.parse_args()
  for tag in [a.tag, a.source_run]:
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in tag):
      p.error('Invalid tag')
  if not 0 <= a.worker_id < a.workers or a.max_calls != 50:
    p.error('Require valid worker ID and explicit 50-call diagnostic')
  parent = D.ROOT / 'raw/mai_official'
  original = json.loads((parent / a.source_run / 'run_config.json').read_text())
  tasks = original['tasks']
  assert len(tasks) == len(set(tasks)) == 116
  config = json.loads(a.config.read_text())
  api, model = original['api_url'], original['model']
  run = parent / a.tag
  run.mkdir(parents=True, exist_ok=True)
  settings = dict(arms=['FULL45'], workers_per_arm=a.workers, tasks=tasks,
      task_seed=original['task_seed'], api_url=api, model=model, mai_config=config,
      max_model_calls=a.max_calls, task_complexity_budget=False,
      graph_initialization='empty_once_on_new_tag_shared_across_all_workers_and_tasks',
      shared_full45_graph=True, cross_evaluation_episode_learning=True,
      selection='all 116 tasks, same order and seed as original full suite',
      task_order_source=a.source_run,
      cohort_note='Fresh FULL45 full-suite evaluation: 116 tasks, 50 model calls per task, eight workers sharing one initially empty graph. No reused task outcomes or historical graph.',
      watch_timeout_s=8 * 3600, full_suite_summary=True,
      source_sha256={str(f.relative_to(D.REPO)): hashlib.sha256(f.read_bytes()).hexdigest()
          for f in [Path(__file__), D.REPO / 'android_world/shared_state.py',
              D.REPO / 'android_world/env/interface.py',
              D.REPO / 'android_world/agents/gelab_agent.py',
              D.ROOT / 'scripts/recovery_guard.py',
              D.REPO / 'experiments/runtime_graph_study/scripts/rgs_run.py',
              *[D.ROOT / 'scripts' / n for n in original['source_sha256']]]})
  def establish(old):
    if old is not None and old != settings:
      raise RuntimeError('Existing tag uses different settings')
    return settings
  D.SS.update_json(run / 'run_config.json', establish, None)
  with D.SS.lock(run / 'graph_initialization.claim'):
    marker = run / 'graph_initialization.json'
    graph = run / 'graph' / 'FULL45'
    if not marker.exists():
      if graph.exists() and any(graph.rglob('*')):
        raise RuntimeError('New cold-start run already contains graph data')
      graph.mkdir(parents=True, exist_ok=True)
      D.SS.atomic_text(marker, json.dumps(dict(initialized_at=now(), nodes=0,
          observations=0, functions=0, historical_graph_loaded=False)))
  name = f'FULL45-{a.worker_id}'
  status_path = run / 'workers' / (name + '.status.json')
  state = dict(status='RUNNING', started_at=now(), pid=os.getpid(), worker_id=a.worker_id,
               max_model_calls=a.max_calls, completed=[], errors=[])
  def save():
    D.SS.atomic_text(status_path, json.dumps(state, indent=2))
  with D.SS.lock(Path('/tmp/mai_device_experiment.claim'), blocking=False), \
       D.SS.lock(run / 'workers' / (name + '.claim'), blocking=False):
    save()
    try:
      recover(api, model, config)
      sms_probe()
      for _, task in D.worker_tasks(tasks, a.workers, a.worker_id):
        out = run / 'FULL45' / task / 'p0'
        result_path = out / 'result.json'
        if out.exists():
          if (out / '.shared_complete').exists():
            continue
          raise RuntimeError('Prior incomplete/error episode retained: ' + str(out))
        recover(api, model, config)
        env = dict(os.environ, FULL_ARM='FULL45', SEQ_APPDIR=str(run / 'graph'),
            MAI_CONFIG=str(a.config.resolve()), ONDEVICE_LOAD='0',
            MAI_SMOKE_EMPTY_STATS='0', MAI_SHARED_STATE='1', MAI_WORKER_ID=str(a.worker_id))
        env.pop('MAI_EPISODE_ROOTS_JSON', None)
        cmd = [sys.executable, str(D.ROOT / 'scripts/mai_run.py'), '--task', task,
            '--seed', str(original['task_seed']), '--variant', a.tag + '_FULL45',
            '--mode', 'agent', '--out', str(out), '--api_url', api, '--model', model,
            '--max_steps', str(a.max_calls)]
        t = time.time()
        timeout_error = None
        with (run / f'FULL45_{task}.log').open('w') as log:
          try:
            rc = subprocess.run(cmd, cwd=D.REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=5400).returncode
          except subprocess.TimeoutExpired:
            rc, timeout_error = 124, 'episode exceeded 5400 seconds'
        result = json.loads(result_path.read_text()) if result_path.exists() else {'error': 'no result'}
        attempts = out / 'mai_attempts.jsonl'
        error = timeout_error or result.get('error') or result.get('eval_error')
        calls = result.get('model_calls')
        if calls is not None and calls > a.max_calls:
          error = 'Model-call budget exceeded'
        row = dict(task=task, arm='FULL45', worker_id=a.worker_id, rc=rc,
            e2e_s=round(time.time() - t, 2), success=result.get('is_successful'),
            model_calls=calls, max_model_calls=a.max_calls,
            http_attempts=len(attempts.read_text().splitlines()) if attempts.exists() else 0,
            error=error, teardown_error=result.get('teardown_error'),
            runtime_stop_reason=result.get('runtime_stop_reason'),
            termination='execution_error' if error or rc else
              'budget' if calls is not None and calls >= a.max_calls else
              'recovery_guard' if result.get('runtime_stop_reason') else 'model_done')
        D.SS.append_json(run / 'runs.jsonl', row)
        print(json.dumps(row), flush=True)
        if error or rc:
          state['errors'].append(task)
        else:
          D.SS.atomic_text(out / '.shared_complete', json.dumps(row))
        state['completed'].append(task)
        save()
      state['status'] = 'FINISHED_WITH_ERRORS' if state['errors'] else 'FINISHED'
    except BaseException as exc:
      state.update(status='FAILED', error=repr(exc))
      raise
    finally:
      state['ended_at'] = now()
      save()


if __name__ == '__main__':
  main()
