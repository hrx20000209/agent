"""Reconcile explicit environment repairs, retaining each row's source cohort."""
import json
from pathlib import Path

import mai_androidworld_driver as D


def reconcile(run, config):
  run = Path(run)
  recipe = config['repair_reconciliation']
  base = run.parent
  original = base / recipe['original_tag']
  baseline = base / recipe['baseline_sms_tag']
  original_config = json.loads((original / 'run_config.json').read_text())
  tasks = original_config['tasks']
  repair = set(recipe['repair_tasks'])
  assert len(repair) == 7 and repair <= set(tasks)
  source_rows = {}
  for root in [original, baseline, run]:
    rows = list(D.SS.rows(root / 'runs.jsonl'))
    keyed = {(r['arm'], r['task']): r for r in rows}
    if len(keyed) != len(rows):
      raise RuntimeError('Duplicate task results in ' + str(root))
    source_rows[root.name] = keyed
  selected = []
  summary = {}
  for arm in ['BASE', 'FULL45']:
    for task in tasks:
      tag = (baseline.name if arm == 'BASE' else run.name) if task in repair else original.name
      row = dict(source_rows[tag][arm, task])
      if row.get('error') or row.get('rc') != 0 or row.get('success') is None:
        raise RuntimeError('Invalid selected result: ' + repr((tag, arm, task)))
      row['source_run'] = tag
      row['environment_repair'] = task in repair
      selected.append(row)
    rows = [r for r in selected if r['arm'] == arm]
    summary[arm] = dict(tasks=len(rows), complete_success=sum(r['success'] == 1 for r in rows),
        score_sum=sum(r['success'] for r in rows),
        full_success_rate=sum(r['success'] == 1 for r in rows) / len(rows),
        replacement_tasks=sorted(repair))
  result = dict(arms=summary, total=232,
      interpretation='Environment repair reconciliation authorized by user: retain BASE and ordinary task outcomes; replace only seven SMS tasks. FULL45 replacement starts from frozen final full232 graph. Not an uninterrupted cold-start run.',
      source_recipe=recipe, snapshot_sha256=config['snapshot_sha256'])
  D.SS.atomic_text(run / 'reconciled_runs.jsonl', ''.join(json.dumps(r) + '\n' for r in selected))
  D.SS.atomic_text(run / 'reconciled_summary.json', json.dumps(result, indent=2))
  return result


def reconcile_budget(run, config):
  """Replace exactly the previous zero-score BASE rows, preserving other rows."""
  run = Path(run)
  prior = run.parent / config['budget_reconciliation']['prior_run']
  old = [r for r in D.SS.rows(prior / 'reconciled_runs.jsonl') if r['arm'] == 'BASE']
  assert len(old) == len({r['task'] for r in old}) == 116
  expected = {r['task'] for r in old if r['success'] == 0}
  new = list(D.SS.rows(run / 'runs.jsonl'))
  replacement = {r['task']: r for r in new}
  assert len(new) == len(replacement) == len(expected) == 67
  assert set(replacement) == expected == set(config['tasks'])
  assert all(r['arm'] == 'BASE' and r['rc'] == 0 and not r.get('error') and r.get('success') is not None for r in new)
  chosen = []
  for original in old:
    row = dict(replacement[original['task']] if original['task'] in expected else original)
    row['budget_rerun'] = original['task'] in expected
    if row['budget_rerun']:
      row['source_run'] = run.name
    chosen.append(row)
  successes = sum(r['success'] == 1 for r in chosen)
  result = dict(tasks=116, previous_full_success=sum(r['success'] == 1 for r in old),
      rerun_tasks=67, rescued_full_success=sum(r['success'] == 1 for r in new),
      full_success=successes, full_success_rate=successes / 116,
      score_sum=sum(r['success'] for r in chosen),
      budget_exhausted=sum(r['termination'] == 'budget' for r in new),
      total_rerun_model_calls=sum(r.get('model_calls') or 0 for r in new),
      interpretation='Targeted BASE budget diagnostic: rerun only prior zero-score tasks with up to 50 calls. Retain 48 prior successes and one partial. FULL45 unchanged. Not a fresh uniform-budget full-suite run.')
  D.SS.atomic_text(run / 'reconciled_base_runs.jsonl', ''.join(json.dumps(r) + '\n' for r in chosen))
  D.SS.atomic_text(run / 'budget_comparison.json', json.dumps(result, indent=2))
  return result
