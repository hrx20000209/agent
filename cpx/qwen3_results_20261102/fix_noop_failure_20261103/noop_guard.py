"""Stop an episode when the model repeats the same action with no effect.

The FULL45 controller already brakes *navigation* controls that are tapped
three times without a screen change (``full7.StuckExplorer.needs_back``), but
hard mode deliberately ignores every other control. A text model can then keep
tapping the same non-navigation element (e.g. a ``TextView``) until the call
budget is exhausted -- observed on Qwen3-8B / RecipeDeleteMultipleRecipesWith-
Constraint, where steps 9-40 were 32 identical clicks at (563, 1033).

This guard is arm-agnostic (it does not depend on the graph or on hard mode)
and only fires on *consecutive identical* actions that leave the screen
unchanged, so legitimate repetition (typing, incrementing a counter, scrolling,
where the content changes) keeps its count at zero.
"""


class NoopGuard:
  """Counts consecutive identical actions that did not change the screen."""

  def __init__(self, limit=5):
    if int(limit) < 2:
      raise ValueError("limit must be >= 2")
    self.limit = int(limit)
    self.sig = None
    self.n = 0
    self.fired = False

  def observe(self, sig, unchanged):
    """Record one executed action; True exactly once when the limit is reached.

    Args:
      sig: hashable action signature (type + target); identical means the same
        action on the same element.
      unchanged: whether the screen after the action equals the screen before.
    """
    if not unchanged:
      self.sig, self.n, self.fired = None, 0, False
    elif sig == self.sig:
      self.n += 1
    else:
      self.sig, self.n, self.fired = sig, 1, False
    if self.n >= self.limit and not self.fired:
      self.fired = True
      return True
    return False
