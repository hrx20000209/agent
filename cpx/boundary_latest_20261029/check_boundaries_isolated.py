"""Execute extracted upstream boundary functions with mocked device I/O."""
import ast,re,json,types,sys
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'upstream';p=root/'experiments/trace_reachability_validation/scripts/full_run.py';tree=ast.parse(p.read_text());nodes=[]
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name in {'_risky_probe','_risky_slot'}:nodes.append(n)
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_RISKY_WORDS' for t in n.targets):nodes.append(n)
ns={'C':types.SimpleNamespace(family_of_sel=lambda sel:'anonymous-toolbar')};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
ui=types.SimpleNamespace(nodes=[types.SimpleNamespace(cls='ImageButton',checkable=False)])
sl=types.SimpleNamespace(node_idx=0,sel=dict(cls='ImageButton',rid='',text='',desc='',label='',ancestry=['ViewGroup|event_toolbar']))
risky=ns['_risky_slot'](types.SimpleNamespace(),ui,sl)
p=root/'experiments/trace_reachability_validation/scripts/full9.py';tree=ast.parse(p.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='relaunch_restore');ns={'R':types.SimpleNamespace(ADB='/mock/adb'),'json_action':types.SimpleNamespace(JSONAction=lambda **kw:kw)};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
log=[];actual=types.SimpleNamespace(activity='Notes',layout_sig='layout',content_sig='changed');target=types.SimpleNamespace(activity='Notes',layout_sig='layout',content_sig='original',package='app')
ep=types.SimpleNamespace(_exec_log=[{'action':{'action_type':'open_app','app_name':'notes'},'commit':False}],_agent=types.SimpleNamespace(_execute_action=lambda *a:None),dev=types.SimpleNamespace(capture=lambda *a,**kw:{'ui':actual}),log=lambda name,row:log.append(row))
with patch('subprocess.run'),patch('time.sleep'):
 ok=ns['relaunch_restore'](ep,target,0)
result=dict(anonymous_toolbar_rejected=risky,content_changed_relaunch_returns_ok=ok,content_same_flag=ep._relaunch_content_same,source_commit='81a1531758da249c2acb74baa761d2da65e01b1c')
assert risky is False and ok is True and ep._relaunch_content_same is False
out=Path(__file__).parent/'isolated_results.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
