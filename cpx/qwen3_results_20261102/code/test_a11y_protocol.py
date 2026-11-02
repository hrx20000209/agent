import sys,unittest,json
from pathlib import Path
from types import SimpleNamespace as N
ROOT=Path(__file__).parent
sys.path[:0]=[str(ROOT/'experiments/trace_reachability_validation/scripts'),str(ROOT/'experiments/runtime_graph_study/scripts')]
import a11y_protocol as P
import rgs_ui as U
class ProtocolTest(unittest.TestCase):
 def setUp(self):
  def node(id,children,bounds,**kw):
   d=dict(unique_id=id,child_ids=children,depth=0,bounds_in_screen=N(left=bounds[0],top=bounds[1],right=bounds[2],bottom=bounds[3]),class_name='android.widget.TextView',package_name='test.app',view_id_resource_name='',text='',content_description='',hint_text='',is_clickable=False,is_long_clickable=False,is_scrollable=False,is_editable=False,is_enabled=True,is_visible_to_user=True,is_checkable=False,is_checked=False,is_selected=False,is_focused=False);d.update(kw);return N(**d)
  self.forest=N(windows=[N(tree=N(nodes=[node(2,[],[10,20,80,70],text='Save',is_clickable=True),node(1,[2,3],[0,0,100,200]),node(3,[],[0,220,100,240],text='Hidden',is_visible_to_user=False)]))])
  self.ui=U.parse_flat(P.flatten_forest(self.forest),activity='A',screen=(100,200))
 def test_hierarchy_and_visible_grounding(self):
  rows=[json.loads(l) for l in P.tree_text(self.ui).splitlines()]
  self.assertEqual([r['index'] for r in rows],[0,1]);self.assertEqual(rows[1]['parent'],0)
  d=P.parse_output('Reason: tap save\nAction: {"action_type":"click","index":1}',self.ui)
  self.assertEqual((d['action']['x'],d['action']['y']),(45,45));self.assertEqual(d['native']['index'],1)
 def test_invalid_actions_never_become_guessed_gestures(self):
  for x in ['{"action_type":"click","index":2}','{"action_type":"click","index":true}','{"action_type":"click","x":10,"y":20}','{"action_type":"input_text","text":"foo"}','{"action_type":"wait"} {"action_type":"wait"}']:
   with self.assertRaises(ValueError):P.parse_output(x,self.ui)
 def test_inline_action_from_real_model_response(self):
  d=P.parse_output('Reason: Input the filename. Action: {"action_type":"input_text","text":"wise_yacht.md","index":1}',self.ui)
  self.assertEqual(d['action']['text'],'wise_yacht.md')
  d=P.parse_output('Reason: Save a note. Action: {"action_type":"input_text","text":"Action: do it","index":1}',self.ui)
  self.assertEqual(d['action']['text'],'Action: do it')
 def test_multiple_actions_are_rejected(self):
  with self.assertRaises(ValueError):P.parse_output('Reason: two actions. Action: {"action_type":"wait"} Action: {"action_type":"wait"}',self.ui)
 def test_text_only_guard(self):
  P.assert_text_messages([dict(role='user',content='tree')])
  with self.assertRaises(AssertionError):P.assert_text_messages([dict(role='user',content=[dict(type='image_url')])])
 def test_seed_matches_official(self):
  import hashlib
  self.assertEqual(P.instance_seed('AudioRecorderRecordAudio'),int(hashlib.sha256(b'30_AudioRecorderRecordAudio_0').hexdigest(),16)%(2**32))
if __name__=='__main__':unittest.main()
