import ast,re,json,types,importlib.util
from pathlib import Path
r=Path('/androidworld');p=r/'experiments/runtime_graph_exploration/scripts/fgraph.py';tree=ast.parse(p.read_text());names={'norm','_key','parse_goal'};nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];verbs=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='VERBS' for y in x.targets));ns=dict(re=re,STOP=set(),_ARM_N=45);exec(compile(ast.Module(body=[verbs,*nodes],type_ignores=[]),str(p),'exec'),ns)
goal='Reply to +18490934066 with message: The library book is due back on the 15th. in Simple SMS Messenger';parsed=[x for x in ns['parse_goal'](goal)['slots'] if x['field']=='message']
source=(r/'experiments/trace_reachability_validation/scripts/full_run.py').read_text()
result=dict(upstream_commit='81a1531758da249c2acb74baa761d2da65e01b1c',sms_message=parsed,has_repetition_guard='RepetitionGuard' in source,has_strict_content_recovery='exact_page_restored' in source,has_recovery_unresolved='recovery_unresolved' in source)
print(json.dumps(result,indent=2))
