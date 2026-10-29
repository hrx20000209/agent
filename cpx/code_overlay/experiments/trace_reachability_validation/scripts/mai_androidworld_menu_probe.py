"""Exercise native MAI menu parsing -> actual ADB key 82 on a dedicated emulator.

No model request or benchmark task initialization; does execute menu then Back.
"""
import argparse
import json
from pathlib import Path
import types
import mai_run as M

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", required=True)
a = p.parse_args()
args = types.SimpleNamespace(task="RecipeAddSingleRecipe", seed=30, variant="menu_probe", mode="agent",
    out=a.out, ref=None, graph=None, explore=False, budget=6.0, max_probes=6,
    no_revalidate=False, recover=False, no_graph_restore=False, app_memory=None,
    max_steps=0, api_url=M.CONFIG["api_url"], model=M.CONFIG["model"])
ep = M.F.R.Episode(args)
try:
  ep.setup()
  ep.env.reset(go_home=True)
  ep.env.hide_automation_ui()
  cap = ep.dev.capture("menu_before", full_res=True)
  agent = ep.make_agent()
  response = '<thinking>Open menu.</thinking><tool_call>{"name":"mobile_use","arguments":{"action":"system_button","button":"menu"}}</tool_call>'
  agent.vllm = types.SimpleNamespace(predict_mm=lambda *args, **kwargs: (response, True, {"usage": {}}))
  decision = M.decide(ep, agent, "Open menu.", cap["pixels"])
  assert decision["extras"] == {"menu_keycode": 82}
  M.commit(ep, agent, decision, "Open menu.")
  assert '"button":"menu"' in ep._mai_history.steps[-1]["response"]
  agent._execute_action(M.F.json_action.JSONAction(action_type="navigate_back"), {})
  result = {"success": True, "keycode": 82, "history_actions": len(ep._mai_history.steps),
            "action": decision["action"].__dict__}
  (Path(a.out) / "menu_probe.json").write_text(json.dumps(result, indent=2))
  print("PASS: native menu -> ADB 82 -> official structured history", json.dumps(result))
finally:
  if getattr(ep, "env", None):
    ep.env.close()
