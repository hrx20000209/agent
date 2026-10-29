"""Snapshot isolation and historical capture routing checks (no model needed)."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mai_warm_worker as W


class WarmSnapshotTest(unittest.TestCase):
  def test_episode_changes_do_not_change_source_or_next_episode(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      snapshot = root / 'snapshot'
      g = snapshot / 'graph/FULL45'
      g.mkdir(parents=True)
      (g / 'graph.jsonl').write_text('old\n')
      first, out = W.make_episode(root / 'run', 'FULL45', 'TaskA', snapshot)
      out.mkdir()
      (out / 'result.json').write_text('{}')
      (first / 'graph/FULL45/graph.jsonl').write_text('old\nnew\n')
      second, out2 = W.make_episode(root / 'run', 'FULL45', 'TaskB', snapshot)
      self.assertEqual((g / 'graph.jsonl').read_text(), 'old\n')
      self.assertEqual((second / 'graph/FULL45/graph.jsonl').read_text(), 'old\n')
      self.assertEqual(list(out2.parents[1].glob('*/p0')), [])
      self.assertEqual((root / 'run/FULL45/TaskA/p0/result.json').read_text(), '{}')
      with self.assertRaises(RuntimeError):
        W.make_episode(root / 'run', 'FULL45', 'TaskA', snapshot)

  def test_historical_and_current_same_task_use_different_captures(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      mapping = root / 'sources.json'
      mapping.write_text(json.dumps({'old_FULL45': {'raw': str(root / 'history'), 'episode': 'p0'}}))
      with patch.dict(os.environ, MAI_EPISODE_ROOTS_JSON=str(mapping), MAI_SHARED_STATE='1'):
        self.assertEqual(W.D.SS.episode_path(root / 'current', 'SameTask', 'old_FULL45'), root / 'history/SameTask/p0')
        self.assertEqual(W.D.SS.episode_path(root / 'current', 'SameTask', 'new_FULL45'), root / 'current/SameTask/p0')

  def test_baseline_does_not_receive_snapshot_graph(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      bundle, _ = W.make_episode(root / 'run', 'BASE', 'Task', root / 'unused')
      self.assertFalse((bundle / 'graph').exists())

  def test_snapshot_hash_mismatch_stops_evaluation(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      (root / 'graph.jsonl').write_text('changed')
      (root / 'manifest.json').write_text(json.dumps({'files_sha256': {'graph.jsonl': 'wrong'}}))
      with self.assertRaises(RuntimeError):
        W.verify_snapshot(root)


if __name__ == '__main__':
  unittest.main()
