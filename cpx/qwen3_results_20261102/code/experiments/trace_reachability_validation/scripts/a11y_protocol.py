"""Pure text accessibility observation and indexed Android action protocol."""
import json,re,math,hashlib,copy

ALLOWED={'click','long_press','input_text','keyboard_enter','navigate_home','navigate_back','scroll','open_app','wait','answer','status'}

def instance_seed(task,seed=30):return int(hashlib.sha256(f'{seed}_{task}_0'.encode()).hexdigest(),16)%(2**32)

def flatten_forest(forest):
 out=[]
 for wi,window in enumerate(forest.windows):
  source=list(window.tree.nodes);by_id={n.unique_id:n for n in source};children={c for n in source for c in n.child_ids};seen=set()
  def visit(n,depth):
   if n.unique_id in seen:return
   seen.add(n.unique_id);b=n.bounds_in_screen
   out.append(dict(windowIndex=wi,depth=depth,**{'class':n.class_name},package=n.package_name,resourceId=n.view_id_resource_name,text=n.text,contentDescription=n.content_description,hint=n.hint_text,bounds=[b.left,b.top,b.right,b.bottom],clickable=n.is_clickable,longClickable=n.is_long_clickable,scrollable=n.is_scrollable,editable=n.is_editable,enabled=n.is_enabled,visible=n.is_visible_to_user,checkable=n.is_checkable,checked=n.is_checked,selected=n.is_selected,focused=n.is_focused,childCount=len(n.child_ids)))
   for cid in n.child_ids:
    if cid in by_id:visit(by_id[cid],depth+1)
  for n in source:
   if n.unique_id not in children:visit(n,0)
  for n in source:
   if n.unique_id not in seen:visit(n,int(n.depth))
 return {'ok':True,'nodes':out}

def tree_text(ui):
 rows=[]
 for n in ui.nodes:
  if not n.visible:continue
  flags=[k for k in ['clickable','long_clickable','scrollable','editable','enabled','checkable','checked','selected','focused'] if getattr(n,k)]
  row=dict(index=n.idx,parent=n.parent,depth=n.depth,class_name=n.short_cls,package=n.pkg,resource_id=n.rid,text=n.text,description=n.desc,hint=n.hint,bounds=list(n.bounds),states=flags)
  rows.append(json.dumps(row,ensure_ascii=False,separators=(',',':')))
 return '\n'.join(rows)

def parse_output(response,ui):
 # Empty think wrappers are permitted; actual hidden reasoning is not enabled.
 s=response.strip();s=re.sub(r'^<think>\s*</think>\s*','',s)
 marker=re.search(r'(?<!\w)Action:\s*',s)
 reason=s[:marker.start()].strip() if marker else ''
 body=s[marker.end():].strip() if marker else s
 if body.startswith('```'):body=re.sub(r'^```(?:json)?\s*','',body);body=re.sub(r'\s*```$','',body)
 obj,end=json.JSONDecoder().raw_decode(body)
 if body[end:].strip():raise ValueError('Trailing content after action JSON')
 if not isinstance(obj,dict):raise ValueError('Action must be an object')
 native=copy.deepcopy(obj);a=obj.get('action_type')
 if a not in ALLOWED:raise ValueError('Unsupported action_type '+str(a))
 fields={'action_type','index','text','direction','app_name','goal_status'}
 if set(obj)-fields:raise ValueError('Unsupported action parameters '+repr(set(obj)-fields))
 grounded=None
 if 'index' in obj:
  idx=obj['index']
  if type(idx) is not int or not 0<=idx<len(ui.nodes):raise ValueError('Invalid accessibility index')
  n=ui.nodes[idx];l,t,r,b=n.bounds;w,h=ui.screen;l,t,r,b=max(0,l),max(0,t),min(w,r),min(h,b)
  if not n.visible or not n.enabled or l>=r or t>=b:raise ValueError('Target element is not visible/enabled')
  grounded={'index':idx,'resource_id':n.rid,'text':n.text,'description':n.desc,'bounds':list(n.bounds)}
  if a in ['click','long_press','input_text']:
   obj.pop('index');obj.update(x=(l+r)//2,y=(t+b)//2)
  elif a!='scroll':raise ValueError('This action cannot take an index')
 elif a in ['click','long_press']:raise ValueError('Pointer action requires accessibility index')
 if a in ['input_text','answer'] and not isinstance(obj.get('text'),str):raise ValueError('Action requires text')
 if a=='input_text' and grounded is None and not any(n.visible and n.editable and n.focused for n in ui.nodes):raise ValueError('input_text without index requires focused editable node')
 if a=='open_app' and (not isinstance(obj.get('app_name'),str) or not obj['app_name'].strip()):raise ValueError('Missing app_name')
 if a=='scroll' and obj.get('direction') not in ['up','down','left','right']:raise ValueError('Invalid scroll direction')
 if a=='status' and obj.get('goal_status') not in ['complete','infeasible']:raise ValueError('Invalid goal_status')
 return dict(action=obj,native=native,reason=reason,grounded=grounded)

def assert_text_messages(messages):
 assert all(isinstance(m.get('content'),str) and m.get('role') in ['system','user','assistant'] for m in messages),'Only text messages are permitted'
 assert not any('data:image/' in m['content'] for m in messages),'Image payload forbidden'
