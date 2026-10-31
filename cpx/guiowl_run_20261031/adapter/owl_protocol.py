"""GUI-Owl official five-image history and 0..999 action/device bridge."""
import ast,base64,copy,io,json,math,re
from pathlib import Path
from owl_coordinate_resize import update_image_size_
import mai_protocol as legacy

# Extract unchanged official methods without importing its Android dependencies.
source=Path(__file__).with_name('owl_official_agent_source.py')
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GUIOwl')
methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
ns={'copy':copy}
for name in ['cut_current_messages','convert_format']:
 exec(compile(ast.fix_missing_locations(ast.Module(body=[methods[name]],type_ignores=[])),str(source),'exec'),ns)

def image_content(image):
 size=update_image_size_({'width':image.width,'height':image.height})
 im=image.convert('RGB').resize((size['resized_width'],size['resized_height']))
 b=io.BytesIO();im.save(b,format='PNG')
 return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()}}

def action_line(response):
 m=re.search(r'Action:\s*(.+)',response)
 return m.group(1).strip() if m else response.split('<tool_call>')[0].strip()

class NativeHistory:
 cut_current_messages=ns['cut_current_messages'];convert_format=ns['convert_format']
 def __init__(self,history_n=5):
  if history_n!=5:raise ValueError('This evaluation pins official five-image history')
  self.history_n=history_n;self.steps=[]
 def append(self,args,thinking,image,origin='model',response=None):
  if response is None:
   label='Execute '+json.dumps(args,ensure_ascii=False)
   response='Action: '+label+'\n<tool_call>\n'+json.dumps({'name':'mobile_use','arguments':args})+'\n</tool_call>'
  self.steps.append(dict(response=response,image=image_content(image),origin=origin,results=[]))
  for s in self.steps[:max(0,len(self.steps)-4)]:s['image']=None
 def messages(self,prompt,goal,image):
  initial='Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: '+goal+'\n\nPrevious actions:\nNo previous action.'
  messages=[{'role':'system','content':[{'type':'text','text':prompt}]}]
  for i,s in enumerate(self.steps+[dict(image=image_content(image))]):
   content=[{'type':'text','text':initial}] if i==0 else []
   if s['image'] is not None:content.append(s['image'])
   messages.append({'role':'user','content':content})
   if i<len(self.steps):messages.append({'role':'assistant','content':[{'type':'text','text':s['response']}]})
  return self.convert_format(goal,self.cut_current_messages(messages,self.history_n))

def keycode(text):
 if isinstance(text,int):return str(text)
 if not isinstance(text,str) or not re.fullmatch(r'[A-Za-z0-9_]+',text):raise ValueError('Invalid key event')
 return text if text.isdigit() or text.startswith('KEYCODE_') else 'KEYCODE_'+text.upper()

def parse_response(response,screen_size):
 calls=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',response,re.S)
 if len(calls)!=1:raise ValueError('Expected exactly one complete GUI-Owl tool_call')
 call=json.loads(calls[0]);args=call.get('arguments');assert call.get('name')=='mobile_use' and isinstance(args,dict),'Expected mobile_use arguments'
 args=copy.deepcopy(args);a=args.get('action');a='click' if a=='tap' else a;args['action']=a
 if a not in {'key','click','long_press','swipe','type','system_button','open','wait','answer','terminate'}:raise ValueError('Unsupported GUI-Owl action: '+str(a))
 native=copy.deepcopy(args);w,h=screen_size
 def pt(value):
  if not isinstance(value,list) or len(value)!=2 or any(type(v) not in [int,float] or not math.isfinite(v) or v<0 or v>1000 for v in value):raise ValueError('Expected finite normalized coordinate pair')
  return [max(0,min(w-1,int(value[0]/999*w))),max(0,min(h-1,int(value[1]/999*h)))]
 if a in {'click','long_press'}:args['coordinate']=pt(args.get('coordinate'))
 if a=='swipe':
  s,e=pt(args.get('coordinate')),pt(args.get('coordinate2'));dx,dy=e[0]-s[0],e[1]-s[1]
  if dx==dy==0:raise ValueError('Swipe endpoints identical')
  args={'action':'swipe','start_coordinate':s,'end_coordinate':e,'direction':('right' if dx>0 else 'left') if abs(dx)>abs(dy) else ('down' if dy>0 else 'up')}
 if a=='open':
  if not isinstance(args.get('text'),str) or not args['text'].strip():raise ValueError('Open missing app')
  args['action']='open_app'
 if a in {'type','answer'} and not isinstance(args.get('text'),str):raise ValueError('Missing action text')
 if a=='system_button':
  if args.get('button') not in ['Back','Home','Menu','Enter']:raise ValueError('Unknown system button')
  args['button']=args['button'].lower()
 if a=='terminate':
  if args.get('status') not in ['success','failure','fail']:raise ValueError('Invalid termination status')
  if args['status']=='failure':args['status']='fail'
 if a=='key':keycode(args.get('text'))
 if 'time' in args:
  if type(args['time']) not in [int,float] or not math.isfinite(args['time']) or not 0<args['time']<=180:raise ValueError('Invalid action time')
 return dict(action='__TOOL_CALL__',tool_call={'name':'mobile_use','arguments':args},summary=json.dumps(args,ensure_ascii=False),explain=action_line(response),native_arguments=native)

def native_from_action(action,extras,screen_size):
 if action.action_type=='unknown' and action.keycode and action.keycode!='KEYCODE_MENU':return {'action':'key','text':str(action.keycode)}
 a=legacy.native_from_action(action,extras,screen_size)
 if a['action']=='system_button':a['button']=a['button'].capitalize()
 elif a['action']=='terminate' and a['status']=='fail':a['status']='failure'
 elif a['action']=='drag':a={'action':'swipe','coordinate':a['start_coordinate'],'coordinate2':a['end_coordinate']}
 elif a['action']=='swipe':
  x,y=500,500;dx,dy={'up':(0,-400),'down':(0,400),'left':(-400,0),'right':(400,0)}[a['direction']];a={'action':'swipe','coordinate':[x,y],'coordinate2':[x+dx,y+dy]}
 return a
