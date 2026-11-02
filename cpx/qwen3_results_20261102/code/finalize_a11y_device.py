"""Official24-app initialization with audited onboarding fixes; never used for scoring."""
import sys,os,json,time,datetime,subprocess,re,logging
from pathlib import Path
sys.path.insert(0,'/androidworld')
from android_world.env import env_launcher,adb_utils
from android_world.env.setup_device import setup
from android_world.utils import app_snapshot,file_utils
from absl import logging as L
L.set_verbosity(L.INFO)
ARM=os.environ['FULL_ARM'];out=Path('/androidworld/initialization')/ARM;out.mkdir(parents=True,exist_ok=True)
report=json.loads((out/'report.json').read_text());report['apps']=[];report['verified']=False;report['manual_actions']=report.get('manual_actions',[])
class Handler(logging.Handler):
 def emit(self,r):
  if r.levelno>=logging.WARNING:report['warnings'].append(r.getMessage())
logging.getLogger().addHandler(Handler());logging.getLogger().addHandler(logging.StreamHandler())
def save(): (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
def adb(*args):return subprocess.check_output(['adb',*args],text=True,timeout=40)
def observe():
 s=env.get_state(wait_to_stabilize=True)
 with (out/'onboarding_states.jsonl').open('a') as f:f.write(json.dumps([dict(index=i,text=n.text,description=n.content_description,resource=n.resource_name,bounds=repr(n.bbox_pixels)) for i,n in enumerate(s.ui_elements)],ensure_ascii=False)+'\n')
 return s.ui_elements
def tap_labels(labels,ids=()):
 for n in observe():
  if (n.text or '').replace('’',"'") in labels or any((n.resource_name or '').endswith(x) for x in ids):
   if n.bbox_pixels is None:continue
   x,y=n.bbox_pixels.center;adb_utils.tap_screen(int(x),int(y),env.controller);report['manual_actions'].append(dict(text=n.text,resource=n.resource_name,x=int(x),y=int(y)));save();time.sleep(3);return True
 return False
try:
 for _ in range(90):
  try:
   if adb('shell','getprop','sys.boot_completed').strip()=='1':break
  except Exception:pass
  time.sleep(3)
 else:raise RuntimeError('boot timeout')
 adb('root');adb('wait-for-device');assert adb('shell','id','-u').strip()=='0'
 env=env_launcher.load_and_setup_env(adb_path='/opt/android/platform-tools/adb',emulator_setup=False)
 from a11y_transport import connect
 connect(env)
 # Official24-app setup already completed; resume only audited onboarding and verification.
 unknown=[x for x in report['warnings'] if 'Failed to automatically setup app' in x and not any(a in x.lower() for a in ['chrome','contacts','markor','open tracks','simple sms','vlc','simple gallery'])]
 if unknown:raise RuntimeError('Unresolved setup warnings: '+repr(unknown))
 for name in ['chrome','contacts','markor','open tracks sports tracker','simple sms messenger','simple gallery pro','vlc']:
  pkg=adb_utils.extract_package_name(adb_utils.get_adb_activity(name))
  if name in ['markor','vlc','simple gallery pro']:
   adb('shell','cmd','appops','set',pkg,'MANAGE_EXTERNAL_STORAGE','allow');report['manual_actions'].append(dict(app=name,config='MANAGE_EXTERNAL_STORAGE allow (official intended onboarding permission)'))
  if name=='simple sms messenger':
   adb('shell','cmd','role','add-role-holder','android.app.role.SMS',pkg,'0');assert re.search(r'name=android\.app\.role\.SMS\s+holders='+re.escape(pkg),adb('shell','dumpsys','role'))
  adb_utils.launch_app(name,env.controller);time.sleep(3)
  if name=='chrome':
   for _ in range(4):
    if not tap_labels(['Use without an account','No thanks']):break
   assert not any((n.text or '')=='Welcome to Chrome' for n in observe())
  elif name=='contacts':
   for _ in range(3):
    if not tap_labels(['Skip',"Don't allow"],['/permission_deny_button']):break
   assert 'android.permission.POST_NOTIFICATIONS: granted=false' in adb('shell','dumpsys','package',pkg)
  elif name=='simple gallery pro':
   # Resolve the official all-files-access onboarding intent; permission is already granted above.
   tap_labels(['All files'])
   adb_utils.press_back_button(env.controller);time.sleep(2)
   adb_utils.launch_app(name,env.controller);time.sleep(2)
   assert 'allow' in adb('shell','cmd','appops','get',pkg,'MANAGE_EXTERNAL_STORAGE')
  elif name=='vlc':
   for _ in range(4):
    if not tap_labels(['Skip','GRANT PERMISSION','OK']):break
   assert not any((n.text or '') in ['GRANT PERMISSION','Permission not granted'] for n in observe())
  elif name=='open tracks sports tracker':
   text=adb('shell','dumpsys','package',pkg)
   for p in ['ACCESS_COARSE_LOCATION','ACCESS_FINE_LOCATION','POST_NOTIFICATIONS','BLUETOOTH_CONNECT','BLUETOOTH_SCAN']:assert 'android.permission.'+p+': granted=true' in text,p
  adb_utils.close_app(name,env.controller);app_snapshot.save_snapshot(name,env.controller)
 for app in setup._APPS:
  path=app_snapshot._snapshot_path(app.app_name);assert file_utils.check_directory_exists(path,env.controller),path
  app_snapshot.restore_snapshot(app.app_name,env.controller);report['apps'].append(dict(app=app.app_name,package=app.package_name(),snapshot=path,restore_verified=True));save()
 adb_utils.press_home_button(env.controller)
 report.pop('error',None);report['verified']=True;report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();env.close();print('ALL24_INITIALIZATION_VERIFIED',flush=True)
except BaseException as e:
 report['error']=repr(e);save();raise
