"""Regression: FULL45 waiter is released even when decision generation fails."""
import sys,threading
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent/'experiments/trace_reachability_validation/scripts'))
import a11y_upstream_probe as A
obj=object.__new__(A.F.FullEpisode);obj._early={'ev':threading.Event()};obj.dev=SimpleNamespace(by_pixels={})
try:obj.decide(None,'goal',object())
except RuntimeError as e:assert 'Observation/tree correspondence' in str(e)
assert obj._early['ev'].wait(.1),'FULL45 completion waiter was never released'
print('FULL45_DECISION_COMPLETION_SIGNAL_OK')

from unittest.mock import patch
class Ep:
 env=None
 _a11y_decision={}
 _a11y_active=False
 def log(self,*args):pass
ep=Ep();agent=A.Agent(ep)
with patch.object(A,'_agent_execute',side_effect=RuntimeError('unknown application File Manager')):
 agent._execute_action(A.F.json_action.JSONAction(action_type='open_app',app_name='File Manager'),{})
assert 'File Manager' in ep._a11y_decision['execution_error']
assert 'Inspect the current tree' in ep._a11y_decision['execution_feedback']
ep._a11y_decision=None
with patch.object(A,'_agent_execute',side_effect=RuntimeError('graph failure')):
 try:agent._execute_action(A.F.json_action.JSONAction(action_type='navigate_back'),{})
 except RuntimeError:pass
 else:raise AssertionError('Graph executor error must keep its original recovery path')
print('MODEL_ACTION_ERROR_FEEDBACK_AND_GRAPH_ERROR_PATH_OK')
