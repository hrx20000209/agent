"""Conservative FULL45 exploration and repeated-action checks."""
import json
import re


def unsafe_navigation_probe(sel):
  text = ' '.join(str(sel.get(k) or '') for k in ('label', 'desc', 'rid', 'text')).lower()
  words = set(re.findall(r'[a-z]+', text))
  if words & {'back', 'navigate', 'navigation', 'up', 'close', 'cancel', 'dismiss'}:
    return True
  ancestry = ' '.join(sel.get('ancestry') or []).lower()
  anonymous = not any(str(sel.get(k) or '').strip() for k in ('label', 'desc', 'rid', 'text'))
  return anonymous and any(w in ancestry for w in ('toolbar', 'action_bar', 'appbar'))


def exact_page_restored(actual, target):
  return (actual.activity == target.activity and actual.layout_sig == target.layout_sig
          and actual.struct_sig == target.struct_sig and actual.content_sig == target.content_sig)


class RepetitionGuard:
  """Count identical pointer actions only while observed task state is unchanged."""
  def __init__(self):
    self.last = None
    self.count = 0

  def observe(self, ui, action, effectful=False):
    if effectful or action.get('action_type') not in {'click', 'long_press', 'swipe'}:
      self.last, self.count = None, 0
      return 0
    key = (ui.activity, ui.struct_sig, ui.content_sig,
           json.dumps(action, sort_keys=True))
    self.count = self.count + 1 if key == self.last else 1
    self.last = key
    return self.count
