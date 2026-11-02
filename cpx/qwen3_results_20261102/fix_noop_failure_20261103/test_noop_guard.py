"""Unit tests for the arm-agnostic no-op stuck guard."""
import unittest

from noop_guard import NoopGuard


class NoopGuardTest(unittest.TestCase):

  def test_identical_no_effect_actions_stop_at_limit(self):
    guard = NoopGuard(5)
    hits = [i for i in range(40)
            if guard.observe(("click", 563, 1033, None, None), True)]
    self.assertEqual(hits, [4])          # 5th observation ends the episode

  def test_screen_change_resets(self):
    guard = NoopGuard(3)
    self.assertFalse(guard.observe(("click", 1, 2), True))
    self.assertFalse(guard.observe(("click", 1, 2), True))
    self.assertFalse(guard.observe(("click", 1, 2), False))   # the page moved
    self.assertFalse(guard.observe(("click", 1, 2), True))
    self.assertFalse(guard.observe(("click", 1, 2), True))
    self.assertTrue(guard.observe(("click", 1, 2), True))

  def test_alternating_actions_never_fire(self):
    guard = NoopGuard(3)
    for _ in range(20):
      self.assertFalse(guard.observe(("click", 1, 2), True))
      self.assertFalse(guard.observe(("click", 9, 9), True))

  def test_limit_must_be_sane(self):
    with self.assertRaises(ValueError):
      NoopGuard(1)

  def test_replay_of_recorded_loop(self):
    # RecipeDeleteMultipleRecipesWithConstraint / FULL45: steps 9-40 are one
    # unchanged click; the guard must cut the episode at the 5th repetition.
    guard = NoopGuard(5)
    fired = []
    for step in range(9, 41):
      if guard.observe(("click", 563, 1033, None, None), True):
        fired.append(step)
    self.assertEqual(fired, [13])


if __name__ == "__main__":
  unittest.main()
