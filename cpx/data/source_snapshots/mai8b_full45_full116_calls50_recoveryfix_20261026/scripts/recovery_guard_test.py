import types
import unittest
from recovery_guard import RepetitionGuard, exact_page_restored, unsafe_navigation_probe


def ui(content='original'):
  return types.SimpleNamespace(activity='Editor', layout_sig='layout', struct_sig='tree', content_sig=content)


class RecoveryGuardTest(unittest.TestCase):
  def test_calendar_anonymous_toolbar_navigation_is_not_explored(self):
    self.assertTrue(unsafe_navigation_probe(dict(cls='ImageButton', rid='', label='', desc='', text='', ancestry=['ViewGroup|event_toolbar'])))
    self.assertTrue(unsafe_navigation_probe(dict(desc='Navigate up')))
    self.assertFalse(unsafe_navigation_probe(dict(label='Event color', rid='event_color_holder', ancestry=['ScrollView|event_holder'])))

  def test_same_page_with_changed_data_is_not_restored(self):
    self.assertFalse(exact_page_restored(ui('modified'), ui()))
    self.assertTrue(exact_page_restored(ui(), ui()))
    different=ui();different.activity='Desktop'
    self.assertFalse(exact_page_restored(different, ui()))

  def test_ineffective_repeat_is_detected_but_real_progress_resets(self):
    guard=RepetitionGuard();action=dict(action_type='click', x=993, y=1256)
    self.assertEqual([guard.observe(ui(),action) for _ in range(6)], [1,2,3,4,5,6])
    # Repeated delete clicks with a decreasing list are useful, not a loop.
    self.assertEqual(guard.observe(ui('shorter list'),action),1)
    self.assertEqual(guard.observe(ui('even shorter list'),action),1)
    # Identical rows may disappear without changing visible text; do not stop commits.
    self.assertEqual(guard.observe(ui(),action,effectful=True),0)
    self.assertEqual(guard.observe(ui(),dict(action_type='wait')),0)
    self.assertEqual(guard.observe(ui(),action),1)


if __name__=='__main__':
  unittest.main()
