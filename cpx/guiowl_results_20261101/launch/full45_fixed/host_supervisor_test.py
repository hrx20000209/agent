import tempfile,json,unittest,types,signal
from pathlib import Path
from unittest.mock import patch,Mock
import host_supervisor as H
class Tests(unittest.TestCase):
 def test_malformed_result_is_recorded_without_crashing(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'result.json').write_text('{');x=H.harvest(p,'T','BASE',0,None,1);self.assertIsNone(x['success']);self.assertIsNone(x['model_calls']);self.assertIn('invalid/missing',x['error'])
 def test_budget_breach_is_an_error(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'result.json').write_text(json.dumps({'model_calls':31,'is_successful':1}));(p/'meta.json').write_text(json.dumps({'complexity':1.6}));self.assertEqual(H.harvest(p,'T','FULL45',0,None,1)['error'],'budget exceeded')
 def make_runner(self,d):return H.Runner(Path(d),'BASE',{'containers':{'BASE':'owned-test'},'tasks':['T']})
 def test_recorded_tasks_are_not_retried(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make_runner(d);r.commit({'task':'T','success':0,'error':None,'rc':0});r.ready=Mock();r.execute=Mock();r.run_all();r.ready.assert_not_called();r.execute.assert_not_called();self.assertEqual(len(r.rows),1)
 def test_container_exit_is_detected_and_recorded(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make_runner(d);r.inspect=Mock(return_value={'Running':False,'Status':'exited','ExitCode':139});r.preserve_logs=Mock();r.command=Mock();proc=Mock(pid=123);proc.poll.return_value=None;proc.wait.return_value=137
   with patch.object(H.subprocess,'Popen',return_value=proc),patch.object(H.os,'killpg') as kill:x=r.execute('T',r.run/'BASE/T/p0')
   self.assertEqual(x['termination'],'infrastructure_error');self.assertIn('139',x['error']);kill.assert_called_once_with(123,signal.SIGTERM);r.preserve_logs.assert_called_once()
 def test_failed_environment_recovery_is_bounded(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.make_runner(d);r.inspect=Mock(return_value={'Running':False});r.command=Mock(return_value=types.SimpleNamespace(returncode=1,stdout='',stderr='boot failed'))
   with patch.object(H.time,'sleep'):self.assertFalse(r.ready())
   self.assertEqual(r.state['status'],'BLOCKED_ENV');self.assertEqual(r.state['recovery_attempt'],3)
 def test_explore_flag_only_for_full45(self):
  for arm in ['BASE','FULL45']:
   with tempfile.TemporaryDirectory() as d:
    r=H.Runner(Path(d),arm,{'containers':{arm:'owned-test'},'tasks':['T']});r.inspect=Mock(return_value={'Running':False,'Status':'exited','ExitCode':139});r.preserve_logs=Mock();r.command=Mock();proc=Mock(pid=123);proc.poll.return_value=None;proc.wait.return_value=137
    with patch.object(H.subprocess,'Popen',return_value=proc) as popen,patch.object(H.os,'killpg'):
     r.execute('T',r.run/arm/'T/p0')
    cmd=popen.call_args.args[0];self.assertEqual('--explore' in cmd,arm=='FULL45')
    if arm=='FULL45':self.assertEqual(cmd[cmd.index('--budget')+1],'30')
if __name__=='__main__':unittest.main()
