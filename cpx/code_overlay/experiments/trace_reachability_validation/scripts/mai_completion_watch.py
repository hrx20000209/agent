"""Persist completion audits for detached workers, without changing episodes."""
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import time

import mai_androidworld_driver as D


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--tag', required=True)
  a = p.parse_args()
  if not a.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.tag):
    p.error('Invalid tag')
  root = D.ROOT / 'raw/mai_official' / a.tag
  cfg = json.loads((root / 'run_config.json').read_text())
  expected = {(arm, task) for arm in cfg['arms'] for task in cfg['tasks']}
  deadline = time.monotonic() + cfg.get('watch_timeout_s', 3 * 3600)
  with D.SS.lock(root / 'nohup_supervisor.claim', blocking=False):
    D.SS.atomic_text(root / 'nohup_supervisor.pid', str(os.getpid()))
    while True:
      rows = list(D.SS.rows(root / 'runs.jsonl'))
      statuses = []
      for arm in cfg['arms']:
        for w in range(cfg['workers_per_arm']):
          path = root / 'workers' / f'{arm}-{w}.status.json'
          statuses.append(json.loads(path.read_text()) if path.exists() else {'status': 'PENDING'})
      terminal = all(s['status'] in {'FINISHED', 'FINISHED_WITH_ERRORS', 'FAILED', 'INTERRUPTED'} for s in statuses)
      keys = [(r['arm'], r['task']) for r in rows]
      problems = [r for r in rows if r.get('error') or r.get('rc') != 0]
      missing = sorted(expected - set(keys))
      valid = not missing and not problems and len(keys) == len(set(keys)) == len(expected)
      for row in rows:
        if not (root / row['arm'] / row['task'] / 'p0/.shared_complete').exists():
          valid = False
      success = collections.Counter(r['arm'] for r in rows if r.get('success') == 1)
      audit = dict(tag=a.tag, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   status='RUNNING', expected=len(expected), completed=len(keys), missing=missing,
                   errors=problems, worker_statuses=[s['status'] for s in statuses],
                   full_success_counts=dict(success),
                   cohort_note=cfg.get('cohort_note', 'Independent SMS repair supplement with its own initially empty graph; keep separate from original full232.'))
      if terminal:
        audit['status'] = 'FINISHED' if valid and all(s['status'] == 'FINISHED' for s in statuses) else 'FAILED'
        if audit['status'] == 'FINISHED' and cfg.get('full_suite_summary'):
          total_calls = sum(r.get('model_calls') or 0 for r in rows)
          summary = dict(tasks=len(rows), complete_success=sum(success.values()),
              full_success_rate=sum(success.values()) / len(rows),
              score_sum=sum(r['success'] for r in rows),
              total_model_calls=total_calls, mean_model_calls=total_calls / len(rows),
              budget_exhausted=sum(r.get('termination') == 'budget' for r in rows),
              cohort_note=cfg['cohort_note'])
          D.SS.atomic_text(root / 'summary.json', json.dumps(summary, indent=2))
          audit['summary'] = summary
        if audit['status'] == 'FINISHED' and cfg.get('repair_reconciliation'):
          try:
            from mai_reconcile_repair import reconcile
            audit['reconciled_summary'] = reconcile(root, cfg)
          except Exception as exc:
            audit.update(status='FAILED_RECONCILIATION', reconciliation_error=repr(exc))
        if audit['status'] == 'FINISHED' and cfg.get('budget_reconciliation'):
          try:
            from mai_reconcile_repair import reconcile_budget
            audit['budget_comparison'] = reconcile_budget(root, cfg)
          except Exception as exc:
            audit.update(status='FAILED_RECONCILIATION', reconciliation_error=repr(exc))
        if audit['status'] == 'FINISHED' and cfg.get('full45_budget_reconciliation'):
          try:
            from mai_reconcile_repair import reconcile_full45_budget
            audit['budget_comparison'] = reconcile_full45_budget(root, cfg)
          except Exception as exc:
            audit.update(status='FAILED_RECONCILIATION', reconciliation_error=repr(exc))
      elif time.monotonic() >= deadline:
        audit['status'] = 'WATCH_TIMEOUT'
      D.SS.atomic_text(root / 'completion_audit.json', json.dumps(audit, indent=2))
      print(json.dumps(audit), flush=True)
      if audit['status'] != 'RUNNING':
        return
      time.sleep(30)


if __name__ == '__main__':
  main()
