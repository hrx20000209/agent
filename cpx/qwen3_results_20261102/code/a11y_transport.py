"""Connect the official native accessibility forwarder over adb reverse."""
import subprocess,time

def connect(env):
 wrapper=env.controller.env;port=wrapper._port
 def adb(*args):
  p=subprocess.run(['adb',*args],capture_output=True,text=True,timeout=25)
  if p.returncode:raise RuntimeError(p.stdout+p.stderr)
  return p.stdout
 adb('reverse',f'tcp:{port}',f'tcp:{port}')
 component='com.google.androidenv.accessibilityforwarder/com.google.androidenv.accessibilityforwarder.FlagsBroadcastReceiver'
 for action,extras in [('SET_GRPC',['--es','host','127.0.0.1','--ei','port',str(port)]),('ENABLE_GRPC',[]),('ENABLE_ACCESSIBILITY_TREE_LOGS',[])]:
  adb('shell','am','broadcast','--receiver-foreground','-a','accessibility_forwarder.intent.action.'+action,'-n',component,*extras)
 time.sleep(1)
 return port

def ensure(env):
 """ADB restart clears reverse mappings; restore transport before reading a tree."""
 port=env.controller.env._port
 p=subprocess.run(['adb','reverse','--list'],capture_output=True,text=True,timeout=10)
 if p.returncode or not any(line.split()[-2:]==[f'tcp:{port}',f'tcp:{port}'] for line in p.stdout.splitlines()):
  p=subprocess.run(['adb','reverse',f'tcp:{port}',f'tcp:{port}'],capture_output=True,text=True,timeout=15)
  if p.returncode:raise RuntimeError('a11y reverse recovery failed: '+p.stderr)
