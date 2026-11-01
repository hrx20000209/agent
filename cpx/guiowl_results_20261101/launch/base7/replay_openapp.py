"""Check all original failing responses through the real runtime action bridge."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import owl_upstream_probe as O
run=Path(sys.argv[1])
rows=json.loads((run/'original_openapp_responses.json').read_text())
assert len({r['task'] for r in rows})==7
for r in rows:
 p=O.M.parse_response(r['response'],(1080,2400))
 a,call,extras=O.F.R.gelab_agent.gelab_action_to_json_action(p,(1080,2400))
 assert a.action_type=='open_app' and a.app_name==p['native_arguments']['text'],(r['task'],a)
 assert p['native_arguments']['action']=='open_app'
assert O.F.FullEpisode.max_steps(SimpleNamespace(task=SimpleNamespace(complexity=4.8)))==48
assert O.CONFIG['temperature']==0 and O.CONFIG['history_n']==5
print('PASS:',len(rows),'original responses from seven tasks parsed and translated; budget/config unchanged')
