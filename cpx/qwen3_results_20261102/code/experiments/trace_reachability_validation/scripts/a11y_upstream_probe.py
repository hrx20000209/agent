"""Shared pure-a11y Qwen client; official Android core plus existing FULL45 method."""
import sys,os,json,time,hashlib,shutil,gzip,contextlib,threading
from pathlib import Path
from types import SimpleNamespace
os.environ['STREAM_ACT']='0';os.environ['STREAM_IO']='0'
import full_run as F
import a11y_protocol as P
import a11y_transport as T
from android_world.agents import t3a
from android_world.env import actuation,android_world_controller
import numpy as np
from PIL import Image
import urllib.request

CONFIG_PATH=Path(os.environ.get('A11Y_CONFIG',str(F.HERE.parent/'configs/qwen3_a11y.json')))
CONFIG=json.loads(CONFIG_PATH.read_text())
F.STREAM=F.ARM=='FULL45';F.STREAM_IO=False
F.ADB=F.R.ADB=shutil.which('adb') or '/opt/android/platform-tools/adb'
F.R.subprocess_forward=lambda:None
# Pure text does not need the visual-model stale screenshot wait.
F.FullEpisode._pixels_caught_up=lambda self,prev,pre,k,m:pre
# Indexed tree actions require the same observed layout/bounds after exploration.
F.R.same_state=lambda a,b:a.exact_sig==b.exact_sig and a.content_sig==b.content_sig

class NativeDevice:
 def __init__(self,env,out):
  self.env=env;self.ctrl=env.controller;self.out=out;self.screen=tuple(env.logical_screen_size);self.n=0;self.by_pixels={};self.latest=None;self._last_payload=None
  (out/'img').mkdir(parents=True,exist_ok=True);(out/'a11y').mkdir(exist_ok=True)
 def activity(self):
  try:return F.adb_utils.get_current_activity(self.ctrl)[0] or ''
  except Exception:return ''
 def quick_ui(self):
  T.ensure(self.env)
  forest=android_world_controller.get_a11y_tree(self.ctrl.env)
  return F.U.parse_flat(P.flatten_forest(forest),activity='',screen=self.screen)
 def settle(self,timeout=3.0,interval=.25):
  start=time.time();prev=None
  while time.time()-start<timeout:
   ui=self.quick_ui();sig=(ui.exact_sig,ui.content_sig)
   if sig==prev:break
   prev=sig;time.sleep(interval)
  return time.time()-start
 def capture(self,tag,settle=True,pause=None,full_res=False):
  start=time.time()
  if pause:time.sleep(pause)
  t_settle=self.settle() if settle else 0
  T.ensure(self.env)
  tick=time.time();state=self.env.get_state(wait_to_stabilize=False);payload=P.flatten_forest(state.forest);act=self.activity();ui=F.U.parse_flat(payload,activity=act,screen=self.screen)
  pixels=np.array(state.pixels,copy=True);self.n+=1;cid=f'{self.n:04d}_{tag}'
  img=Image.fromarray(pixels)
  if not full_res:img=img.resize((img.width//2,img.height//2))
  img.convert('RGB').save(self.out/'img'/f'{cid}.jpg',quality=82)
  with gzip.open(self.out/'a11y'/f'{cid}.json.gz','wt',encoding='utf-8') as f:json.dump({'activity':act,'payload':payload},f,ensure_ascii=False)
  self._last_payload=json.dumps(payload);cap=dict(cid=cid,t=start,settle_s=round(t_settle,3),capture_s=round(time.time()-tick,3),pixels=pixels,ui=ui,summary=ui.summary())
  self.latest=cap;self.by_pixels[id(pixels)]=cap
  while len(self.by_pixels)>24:self.by_pixels.pop(next(iter(self.by_pixels)))
  return cap
 def tap(self,x,y):
  result=F.adb_utils.tap_screen(int(x),int(y),self.ctrl);record(self.ep,F.json_action.JSONAction(action_type='click',x=int(x),y=int(y)));return result
 def back(self):
  result=F.adb_utils.press_back_button(self.ctrl);record(self.ep,F.json_action.JSONAction(action_type='navigate_back'));return result
F.R.Device=NativeDevice
_load_env=F.R.env_launcher.load_and_setup_env
def load_env(*args,**kwargs):
 env=_load_env(*args,**kwargs)
 # An attached running emulator may have relaunch_count=0 and retain an old gRPC port.
 from a11y_transport import connect
 connect(env)
 return env
F.R.env_launcher.load_and_setup_env=load_env

def record(ep,action,reason='Executed action'):
 if not getattr(ep,'_a11y_active',False) or getattr(ep,'_a11y_exploring',False):return
 decision=getattr(ep,'_a11y_decision',None)
 obj=decision['parsed']['native'] if decision and not decision.get('parse_error') else {k:v for k,v in action.__dict__.items() if v is not None}
 note=(decision['parsed']['reason'] if decision and not decision.get('parse_error') else reason)
 ep._a11y_history.append(note+' Action: '+json.dumps(obj,ensure_ascii=False))
 ep.log('a11y_executed',dict(origin='model' if decision else 'graph',action=action.__dict__,summary=ep._a11y_history[-1]))

class Agent:
 def __init__(self,ep):self.ep=ep;self.env=ep.env;self._actions=[];self._summaries=[]
 def _history_text(self):return '\n'.join(self._summaries)
 def _execute_action(self,action,extras):
  if extras.get('no_action'):
   self.ep._a11y_history.append('No action executed: '+extras['parse_error']);return
  if action.action_type=='answer':self.env.interaction_cache=action.text
  elif action.action_type=='status':pass
  else:
   native=action
   if action.action_type=='swipe':
    if extras.get('start_coordinate') and extras.get('end_coordinate'):points=extras['start_coordinate']+extras['end_coordinate']
    else:
     w,h=self.env.logical_screen_size;sx,sy=w//2,h//2;ex,ey={'up':(sx,0),'down':(sx,h),'left':(0,sy),'right':(w,sy)}[action.direction];points=[sx,sy,ex,ey]
    native=SimpleNamespace(action_type='swipe',direction=points)
   elements=extras.get('ui_elements',[])
   actuation.execute_adb_action(native,elements,self.env.logical_screen_size,self.env.controller)
  record(self.ep,action)

_agent_execute=Agent._execute_action
def execute_with_feedback(self,action,extras):
 try:return _agent_execute(self,action,extras)
 except Exception as exc:
  d=getattr(self.ep,'_a11y_decision',None)
  if d is None:raise  # Preserve graph executor's existing recovery behavior.
  d['execution_error']=repr(exc)
  d['execution_feedback']='Some error happened executing the action '+action.action_type+': '+str(exc)[:400]+'. Inspect the current tree before the next action.'
  self.ep.log('a11y_execution_errors',dict(action=action.__dict__,error=repr(exc),feedback=d['execution_feedback']))
Agent._execute_action=execute_with_feedback

def post(path,payload):
 req=urllib.request.Request(CONFIG['base_url']+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(req,timeout=180) as r:return json.load(r)

def decide(self,agent,goal,pixels,early=None):
 if early is None:early=getattr(self,'_early',None)
 try:
  cap=self.dev.by_pixels.get(id(pixels))
  if cap is None:raise RuntimeError('Observation/tree correspondence lost')
  ui=cap['ui'];tree=P.tree_text(ui);history=list(agent._summaries[-24:]);dropped=max(0,len(agent._summaries)-24)
  guidelines=['Use the provided accessibility indexes for click, long_press and input_text. Do not infer screen coordinates.','The answer action submits your answer and ends the episode.','Return exactly one Reason and one Action JSON.']
  hint=F.full7.EXTRA.get('text','') if F.ARM=='FULL45' else ''
  if hint:guidelines.append('Runtime graph feedback: '+hint)
  while True:
   prompt=t3a._action_selection_prompt(goal,history,tree,guidelines).replace('screenshot','accessibility tree')
   messages=[{'role':'system','content':'You control Android using only an accessibility tree and text history. No screenshot is available. Follow the indexed-action JSON protocol.'},{'role':'user','content':prompt}];P.assert_text_messages(messages)
   tok=post('/tokenize',dict(model=CONFIG['model'],messages=messages,add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}))['count']
   if tok+CONFIG['max_tokens']+128<=CONFIG['max_model_len']:break
   if not history:raise RuntimeError('Full accessibility tree and goal exceed context; no tree nodes were silently removed')
   history.pop(0);dropped+=1
  with self._a11y_lock:
   if self._a11y_calls>=self.max_steps():raise RuntimeError('LLM call budget exhausted before request')
   self._a11y_calls+=1;call=self._a11y_calls
  request=dict(model=CONFIG['model'],messages=messages,temperature=0,top_p=1,top_k=-1,seed=42,max_tokens=CONFIG['max_tokens'],chat_template_kwargs={'enable_thinking':False})
  self.log('a11y_requests',dict(call=call,observation=cap['cid'],images=0,tree_nodes=len(ui.nodes),prompt_tokens=tok,history_dropped=dropped,request=request))
  start=time.time();raw=post('/v1/chat/completions',request);response=raw['choices'][0]['message']['content'];latency=time.time()-start
  self.log('a11y_raw_responses',dict(call=call,response=response,usage=raw.get('usage'),latency_s=latency))
  try:
   parsed=P.parse_output(response,ui);parsed['summary']=parsed['reason']+' Action: '+json.dumps(parsed['native'],ensure_ascii=False);action=F.json_action.JSONAction(**parsed['action']);error=None
   elements=[SimpleNamespace(bbox_pixels=SimpleNamespace(x_min=n.bounds[0],y_min=n.bounds[1],x_max=n.bounds[2],y_max=n.bounds[3])) for n in ui.nodes]
   extras={'ui_elements':elements}
   if action.action_type=='answer':extras['return_text']=action.text
  except Exception as exc:
   error=str(exc);parsed={'native':{},'reason':'','summary':'No action executed: '+error};action=F.json_action.JSONAction(action_type=F.json_action.UNKNOWN);extras={'no_action':True,'parse_error':error}
  d=dict(response=response,parsed=parsed,action=action,tool_call={'name':'a11y_action','arguments':parsed['native']},extras=extras,model_latency_s=round(latency,3),parse_error=error,observed_cid=cap['cid'])
  self.log('a11y_decisions',dict(call=call,observation=cap['cid'],action=action.__dict__,parsed=parsed,parse_error=error,latency_s=latency));F._T_ACTION.append(latency)
  return d
 finally:
  if early is not None:early['ev'].set()

_setup=F.FullEpisode.setup;_commit=F.FullEpisode.commit;_run=F.FullEpisode.run;_run_agent=F.FullEpisode.run_agent
def setup(self):
 self.args.seed=P.instance_seed(self.args.task,30)
 meta=_setup(self);self.dev.ep=self;self._a11y_history=[];self._a11y_active=False;self._a11y_exploring=False;self._a11y_decision=None;self._a11y_calls=0;self._a11y_lock=threading.Lock()
 meta.update(decision_backend='qwen3-pure-a11y',images_to_model=0,thinking=False,temperature=0,task_seed_protocol='official hashed task seed from30',allocated_model_calls=int(10*self.task.complexity),graph_execution_enabled=F.ARM=='FULL45',exploration_enabled=F.ARM=='FULL45',a11y_config=CONFIG,observation_backend='official native accessibility forest; no FastA11y',state_equality='exact a11y signature including bounds',visual_stale_wait=False)
 (self.out/'meta.json').write_text(F.R.jdump(meta));return meta
def make_agent(self):self._a11y_active=True;return Agent(self)
def commit(self,agent,d,goal):
 self._a11y_decision=d
 try:
  summary,elapsed=_commit(self,agent,d,goal)
  if d.get('execution_error'):
   summary=d['execution_feedback'];agent._summaries[-1]=summary;agent._actions[-1]['summary']=summary
  return summary,elapsed
 finally:self._a11y_decision=None
def run_agent(self,*args,**kwargs):
 try:return _run_agent(self,*args,**kwargs)
 finally:self._a11y_active=False
def run(self):
 result=_run(self);result['model_calls_internal']=result.get('model_calls');result['model_calls']=getattr(self,'_a11y_calls',0)
 (self.out/'result.json').write_text(F.R.jdump(result));return result
@contextlib.contextmanager
def exploration(ep):
 old=ep._a11y_exploring;ep._a11y_exploring=True
 try:yield
 finally:ep._a11y_exploring=old
_window=F.FullEpisode.window;_stuck=F.full7.StuckExplorer.window
def window(self,*a,**kw):
 with exploration(self):return _window(self,*a,**kw)
def stuck_window(self,*a,**kw):
 with exploration(self.ep):return _stuck(self,*a,**kw)
F.FullEpisode.setup=setup;F.FullEpisode.make_agent=make_agent;F.FullEpisode.decide=decide;F.FullEpisode.commit=commit;F.FullEpisode.run=run;F.FullEpisode.run_agent=run_agent;F.FullEpisode.max_steps=lambda self:int(10*self.task.complexity)
F.FullEpisode.window=window;F.full7.StuckExplorer.window=stuck_window
if __name__=='__main__':F.R.main()
