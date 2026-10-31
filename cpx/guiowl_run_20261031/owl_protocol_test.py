import unittest,json,sys,copy,types
from pathlib import Path
from PIL import Image
S=Path(__file__).parent/'experiments/trace_reachability_validation/scripts';sys.path.insert(0,str(S))
import owl_protocol as O
class Tests(unittest.TestCase):
 def response(self,args):return 'Action: Test action.\n<tool_call>\n'+json.dumps({'name':'mobile_use','arguments':args})+'\n</tool_call>'
 def test_swipe_endpoints_preserved(self):
  p=O.parse_response(self.response({'action':'swipe','coordinate':[500,800],'coordinate2':[500,200]}),(1080,2400));a=p['tool_call']['arguments'];self.assertEqual(a['direction'],'up');self.assertEqual(a['start_coordinate'],[540,1921]);self.assertEqual(a['end_coordinate'],[540,480])
 def test_official_button_and_failure(self):
  self.assertEqual(O.parse_response(self.response({'action':'system_button','button':'Back'}),(100,200))['tool_call']['arguments']['button'],'back')
  self.assertEqual(O.parse_response(self.response({'action':'terminate','status':'failure'}),(100,200))['tool_call']['arguments']['status'],'fail')
 def test_reject_unknown_or_ambiguous_action(self):
  for text in [self.response({'action':'count'}),self.response({'action':'click','coordinate':[float('nan'),1]}),self.response({'action':'wait'})*2]:
   with self.assertRaises((ValueError,AssertionError)):O.parse_response(text,(100,200))
 def test_key_event_is_not_shell_code(self):
  self.assertEqual(O.keycode('volume_up'),'KEYCODE_VOLUME_UP')
  with self.assertRaises(ValueError):O.keycode('1;echo x')
 def test_history_matches_official_methods(self):
  im=Image.new('RGB',(96,192));h=O.NativeHistory();raw=[{'role':'system','content':[{'type':'text','text':'SYS'}]}]
  stub=types.SimpleNamespace()
  for i in range(12):
   content=[{'type':'text','text':'Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: GOAL\n\nPrevious actions:\nNo previous action.'}] if i==0 else []
   content.append(O.image_content(im));raw.append({'role':'user','content':content})
   expected=O.ns['convert_format'](stub,'GOAL',O.ns['cut_current_messages'](stub,copy.deepcopy(raw),5))
   actual=h.messages('SYS','GOAL',im);self.assertEqual(actual,expected)
   self.assertEqual(sum(x.get('type')=='image_url' for m in actual for x in m['content']),min(i+1,5))
   response=self.response({'action':'wait'});raw.append({'role':'assistant','content':[{'type':'text','text':response}]});h.append({'action':'wait'},'Test action.',im,response=response)
 def test_graph_swipe_history_uses_owl_endpoints(self):
  a=types.SimpleNamespace(action_type='swipe',direction='up');n=O.native_from_action(a,{'start_coordinate':[500,1800],'end_coordinate':[500,600]},(1080,2400));self.assertEqual(n['action'],'swipe');self.assertIn('coordinate2',n);self.assertNotIn('direction',n)
if __name__=='__main__':unittest.main()
