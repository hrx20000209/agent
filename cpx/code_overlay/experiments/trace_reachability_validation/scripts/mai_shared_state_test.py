"""Multiprocess checks of production graph indexes and worker partitioning."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import mai_run as M
from android_world import shared_state as SS
from mai_androidworld_driver import worker_tasks


def observation(worker, i):
  page = {'struct_sig': f'page-{worker}-{i}', 'layout_sig': f'layout-{worker}-{i}',
          'activity': 'TestActivity', 'package': 'test.app'}
  return {'src': page, 'dst': page, 'slot_id': 'tap', 'instance': f'Task{worker}/shared_FULL45',
          'origin': 'agent', 'selector': {}}


def writer(root, worker, count, barrier):
  root = Path(root)
  graph = M.F.R.Graph(root / 'graph.jsonl', root / 'unrestorable.jsonl')
  store = M.F.S.Store(root / 'store.json')
  barrier.wait(timeout=30)
  for i in range(count):
    graph.add(observation(worker, i))
    store.append([], f'{worker}-{i}')
    SS.append_wait(root / 'edge_wait.json', f'worker-{worker}', i)
    SS.append_wait(root / 'edge_wait.json', 'common', i)


def function_builder(root, barrier):
  root = Path(root)
  with mock.patch.dict(os.environ, MAI_SHARED_STATE='1'), \
       mock.patch.object(M.F.full9.FG, 'self_reported_done', return_value=True), \
       mock.patch.object(M.F.full9.FG, 'extract', return_value=[{'entry_act': 'Entry', 'fields': [], 'steps': [{'op': 'nav', 'fam': 'next'}]}]):
    barrier.wait(timeout=30)
    M.F.full9.FG.build_online(root / 'raw', root / 'functions.jsonl')


class SharedStateTest(unittest.TestCase):
  def test_four_writers_no_lost_edges_trajectories_or_waits(self):
    with tempfile.TemporaryDirectory() as root:
      context = mp.get_context('fork')
      barrier = context.Barrier(4)
      processes = [context.Process(target=writer, args=(root, i, 25, barrier)) for i in range(4)]
      for process in processes:
        process.start()
      for process in processes:
        process.join(45)
        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)
      root = Path(root)
      graph = M.F.R.Graph(root / 'graph.jsonl')
      self.assertEqual(len(graph.nodes), 100)
      self.assertEqual(sum(map(len, graph.slots.values())), 100)
      data = SS.read_json(root / 'store.json', [])
      self.assertEqual(len(data), 100)
      self.assertEqual(len({x['task'] for x in data}), 100)
      waits = SS.read_json(root / 'edge_wait.json', {})
      self.assertEqual(set(waits), {'common'} | {f'worker-{i}' for i in range(4)})
      self.assertTrue(all(len(v) == 9 for v in waits.values()))

  def test_existing_reader_observes_peer_once_and_shares_misuse(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      a, b = [M.F.R.Graph(root / 'graph.jsonl') for _ in range(2)]
      a.add(observation(0, 0))
      self.assertEqual(len(b.nodes), 0)
      b.refresh()
      b.refresh()
      self.assertEqual(len(b.nodes), 1)
      self.assertEqual(sum(map(len, b.slots.values())), 1)
      ga, gb = [M.F.GX.GraphReuse(root) for _ in range(2)]
      ga.record('ACTION', ('page', 'tap'), 'Task0/v', 0, True)
      gb.refresh()
      self.assertEqual(gb.evidence('ACTION', ('page', 'tap')), 1)
      ga.check_undo('Task0/v', 1, {'action_type': 'navigate_back'})
      gb.refresh()
      self.assertTrue(gb.braked('ACTION', ('page', 'tap')))
      gb.refresh()
      self.assertEqual(gb.misuse[('ACTION', ('page', 'tap'))], 1)

  def test_intent_refresh_keeps_leave_one_task_out(self):
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / 'intent.jsonl'
      intent = M.F.IG.EdgeIntent(path, 'Own')
      for task in ['Own', 'Peer']:
        SS.append_json(path, {'key': ['app', 'act', 'tap'], 'task': task, 'g': {'add': 1}})
      intent.refresh()
      self.assertEqual(intent.n_valid[('app', 'act', 'tap')], 1)
      self.assertEqual(intent.use[('app', 'act', 'tap')]['add'], 1)

  def test_function_extraction_is_once_and_skips_inflight_episode(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      for task in ['Finished', 'Inflight']:
        ep = root / 'raw' / task / 'p0'
        ep.mkdir(parents=True)
        (ep / 'result.json').write_text('{}')
        (ep / 'steps.jsonl').write_text('{}\n')
        (ep / 'meta.json').write_text(json.dumps({'goal': 'Add a recipe'}))
      (root / 'raw/Finished/p0/.shared_complete').touch()
      context = mp.get_context('fork')
      barrier = context.Barrier(4)
      processes = [context.Process(target=function_builder, args=(folder, barrier)) for _ in range(4)]
      for process in processes:
        process.start()
      for process in processes:
        process.join(30)
        self.assertEqual(process.exitcode, 0)
      rows = SS.rows(root / 'functions.jsonl')
      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]['src'], 'online/Finished')
      self.assertEqual((root / 'functions.done').read_text(), 'Finished')

  def test_refresh_functions_preserves_episode_local_learning(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      ep = root / 'Task/p0'
      ep.mkdir(parents=True)
      cache = root / 'functions.jsonl'
      entry = {'entry_act': 'Entry', 'fields': [], 'steps': [{'op': 'nav', 'fam': 'next'}],
               'template': 'Peer', 'src': 'online/Peer', 'verb': 'add'}
      SS.append_json(cache, entry)
      fx = M.F.full9.FuncExec.__new__(M.F.full9.FuncExec)
      fx.ep = types.SimpleNamespace(out=ep, args=types.SimpleNamespace(task='Task'))
      fx.online_cache, fx.recs = cache, {}
      fx.lib = types.SimpleNamespace(funcs=[dict(entry, src='self')])
      with mock.patch.dict(os.environ, MAI_SHARED_STATE='1'):
        fx.refresh_shared()
      self.assertEqual({f['src'] for f in fx.lib.funcs}, {'self', 'online/Peer'})

  def test_partial_tail_not_consumed(self):
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / 'events.jsonl'
      path.write_bytes(b'{"first":1}\n{"second":')
      rows, offset = SS.read_delta(path)
      self.assertEqual(rows, [{'first': 1}])
      SS.append_text(path, '2}\n')
      rows, offset = SS.read_delta(path, offset)
      self.assertEqual(rows, [{'second': 2}])

  def test_partition_is_exact_and_paired_between_arms(self):
    tasks = [f'Task{i}' for i in range(116)]
    shards = [worker_tasks(tasks, 4, i) for i in range(4)]
    self.assertEqual([len(s) for s in shards], [29] * 4)
    self.assertEqual(sorted(i for s in shards for i, task in s), list(range(116)))

  def test_claim_excludes_duplicate_worker_and_releases(self):
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / 'worker.claim'
      with SS.lock(path):
        with self.assertRaises(BlockingIOError):
          with SS.lock(path, blocking=False):
            pass
      with SS.lock(path, blocking=False):
        pass


if __name__ == '__main__':
  unittest.main()
