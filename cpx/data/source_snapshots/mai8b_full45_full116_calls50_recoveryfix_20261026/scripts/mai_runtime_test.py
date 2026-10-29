"""Execution-boundary regressions; mocks hardware and model service calls."""
import json
import threading
import types
import unittest
import tempfile
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
import mai_run as M


def episode():
  return types.SimpleNamespace(_mai_active=True, _mai_exploring=False, _mai_in_execute=False,
      _mai_history=M.M.NativeHistory(3), _mai_cap={"pixels": np.zeros((8, 8, 3), dtype=np.uint8), "cid": "test"},
      env=types.SimpleNamespace(logical_screen_size=(1080, 2400), controller=object()),
      args=types.SimpleNamespace(api_url="http://test", model="MAI-UI-8B"), log=lambda *args: None)


class RuntimeTest(unittest.TestCase):
  def test_empty_statistics_uses_neutral_weights_without_offline_reads(self):
    with mock.patch.object(M.F.EV, "w_app"), mock.patch.object(M.F.EV, "w_ui"), mock.patch.object(M.F.full9, "goal_idf"), mock.patch.object(M.F.full9, "rev_prior"):
      M.configure_empty_statistics()
      self.assertEqual(M.F.EV.w_app("test.package")("save"), 1.0)
      self.assertEqual(M.F.EV.w_ui("save"), 1.0)
      self.assertEqual(M.F.full9.goal_idf()("save"), 1.0)
      self.assertEqual(M.F.full9.rev_prior(), {})

  def test_online_pages_start_empty_and_only_reuse_current_run(self):
    fx = M.F.full7
    with tempfile.TemporaryDirectory() as root, mock.patch.object(fx, "_known_from_history", side_effect=AssertionError("offline read")):
      first_run = Path(root) / "first"
      first_run.mkdir()
      ep = episode()
      first = fx.StuckExplorer(ep, "task", first_run, hard=True, online_only=True)
      self.assertEqual(first.known, set())
      self.assertEqual(first.mem, {})
      first.visited.add("runtime-page")
      first.finish()
      next_task = fx.StuckExplorer(ep, "task", first_run, hard=True, online_only=True)
      self.assertEqual(next_task.known, {"runtime-page"})
      other_run = Path(root) / "other"
      other_run.mkdir()
      fresh = fx.StuckExplorer(ep, "task", other_run, hard=True, online_only=True)
      self.assertEqual(fresh.known, set())

  def test_legacy_pages_still_use_historical_source(self):
    ep = episode()
    ep.args.task = "legacy-task"
    with tempfile.TemporaryDirectory() as root, mock.patch.object(M.F.full7, "_known_from_history", return_value={"historical-page"}) as history:
      old = M.F.full7.StuckExplorer(ep, "task", root)
      self.assertEqual(old.known, {"historical-page"})
      history.assert_called_once_with("legacy-task")

  def test_failure_signals_decision_ready(self):
    ep = episode()
    ep._early = {"ev": threading.Event()}
    ep._mai_history = M.M.NativeHistory(3)
    agent = types.SimpleNamespace(vllm=types.SimpleNamespace(predict_mm=mock.Mock(side_effect=RuntimeError("test outage"))))
    with self.assertRaises(RuntimeError):
      M.decide(ep, agent, "task", ep._mai_cap["pixels"])
    self.assertTrue(ep._early["ev"].is_set())
    self.assertEqual(agent.vllm.predict_mm.call_count, 3)
    self.assertEqual(ep._mai_history.steps, [])

  def test_menu_and_recovery_replay_use_key_82(self):
    ep = episode()
    plain = types.SimpleNamespace(_execute_action=mock.Mock(), _summaries=[])
    response = types.SimpleNamespace(status=M.F.adb_utils.adb_pb2.AdbResponse.Status.OK)
    with mock.patch.object(M, "_make_agent", return_value=plain), mock.patch.object(M.F.adb_utils, "press_keyboard_generic", return_value=response) as press:
      agent = M.make_agent(ep)
      action = M.F.json_action.JSONAction(action_type="unknown", keycode="KEYCODE_MENU")
      agent._execute_action(action, {"menu_keycode": 82})
      agent._execute_action(action, {})
      self.assertEqual(press.call_count, 2)
      press.assert_called_with("82", ep.env.controller)
    self.assertEqual(len(ep._mai_history.steps), 2)
    self.assertIn('"button":"menu"', ep._mai_history.steps[0]["response"])

  def test_failed_execution_is_not_recorded(self):
    ep = episode()
    plain = types.SimpleNamespace(_execute_action=mock.Mock(side_effect=RuntimeError("ADB failed")), _summaries=[])
    with mock.patch.object(M, "_make_agent", return_value=plain):
      agent = M.make_agent(ep)
      with self.assertRaises(RuntimeError):
        agent._execute_action(M.F.json_action.JSONAction(action_type="wait"), {})
    self.assertEqual(ep._mai_history.steps, [])
    self.assertFalse(ep._mai_in_execute)

  def test_probe_actions_do_not_enter_main_history(self):
    ep = episode()
    with M.exploration(ep):
      M.record(ep, M.F.json_action.JSONAction(action_type="click", x=10, y=20), {})
    self.assertEqual(ep._mai_history.steps, [])
    self.assertFalse(ep._mai_exploring)

  def test_model_history_uses_native_not_pixel_coordinates(self):
    ep = episode()
    action = M.F.json_action.JSONAction(action_type="click", x=540, y=1201)
    ep._mai_model_decision = {"mai_initial_action": dict(action.__dict__),
        "parsed": {"native_arguments": {"action": "click", "coordinate": [500, 500]}, "explain": "Click."},
        "mai_image": Image.new("RGB", (8, 8))}
    M.record(ep, action, {})
    self.assertIn('"coordinate":[500,500]', ep._mai_history.steps[-1]["response"])
    action.x = 800  # FULL45 may relocate the grounded target after restoration.
    M.record(ep, action, {})
    self.assertNotIn('"coordinate":[500,500]', ep._mai_history.steps[-1]["response"])


class RecoveryRuntimeTest(unittest.TestCase):
  def test_full45_probe_filter_rejects_anonymous_toolbar_button(self):
    ep=episode()
    ui=types.SimpleNamespace(nodes=[types.SimpleNamespace(cls="ImageButton", checkable=False)])
    slot=types.SimpleNamespace(node_idx=0, sel=dict(cls="ImageButton",rid="",text="",label="",desc="",ancestry=["ViewGroup|event_toolbar"]))
    self.assertTrue(M.F._risky_slot(ep,ui,slot))

  def test_failed_recovery_is_reported_with_current_image(self):
    ep = episode()
    M.recovery_feedback(ep, "Restore failed; device is on Desktop.")
    response = '<thinking>Reopen the app.</thinking><tool_call>{"name":"mobile_use","arguments":{"action":"open","text":"Calendar"}}</tool_call>'
    agent = types.SimpleNamespace(vllm=types.SimpleNamespace(predict_mm=mock.Mock(return_value=(response, True, {}))))
    with mock.patch.object(M.F, "ARM", "FULL45"):
      M.decide_native(ep, agent, "Create event", ep._mai_cap["pixels"])
    messages = agent.vllm.predict_mm.call_args.kwargs["messages"]
    self.assertIn("Restore failed", messages[-1]["content"][0]["text"])
    self.assertEqual(messages[-1]["content"][1]["type"], "image_url")
    self.assertEqual(ep._mai_history.steps, [])

  def test_baseline_does_not_receive_exploration_feedback(self):
    ep = episode()
    M.recovery_feedback(ep, "Restore failed")
    response = '<tool_call>{"name":"mobile_use","arguments":{"action":"wait"}}</tool_call>'
    agent = types.SimpleNamespace(vllm=types.SimpleNamespace(predict_mm=mock.Mock(return_value=(response, True, {}))))
    with mock.patch.object(M.F, "ARM", "BASE"):
      M.decide_native(ep, agent, "task", ep._mai_cap["pixels"])
    content=agent.vllm.predict_mm.call_args.kwargs["messages"][-1]["content"]
    self.assertEqual([x["type"] for x in content], ["image_url"])

  def test_actual_relaunch_rejects_content_drift_and_accepts_exact_restore(self):
    target=types.SimpleNamespace(activity="Editor", layout_sig="layout", struct_sig="tree", content_sig="original", package="test.app")
    restored=types.SimpleNamespace(**vars(target))
    ep=episode()
    ep._exec_log=[{"action":{"action_type":"open_app", "app_name":"Calendar"},"commit":False}]
    ep._agent=types.SimpleNamespace(_execute_action=mock.Mock())
    ep.dev=types.SimpleNamespace(capture=lambda *a,**k:{"ui":restored})
    with mock.patch.dict("os.environ", {"FULL_ARM":"FULL45"}), mock.patch("subprocess.run"), mock.patch("time.sleep"):
      restored.content_sig="modified"
      self.assertFalse(M.F.full9.relaunch_restore(ep,target,4))
      self.assertFalse(ep._relaunch_content_same)
      restored.content_sig="original"
      self.assertTrue(M.F.full9.relaunch_restore(ep,target,4))

  def test_relaunch_does_not_repeat_commit(self):
    ep=episode();ep._exec_log=[{"action":{"action_type":"open_app"},"commit":False},{"action":{"action_type":"click"},"commit":True}]
    with mock.patch("subprocess.run") as adb:
      self.assertFalse(M.F.full9.relaunch_restore(ep,types.SimpleNamespace(),0))
    adb.assert_not_called()


if __name__ == "__main__":
  unittest.main()
