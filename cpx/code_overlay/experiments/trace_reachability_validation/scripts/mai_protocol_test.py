"""Protocol regressions at the MAI normalized / Android pixel boundary."""
import json
import unittest
import mai_protocol as M
from mai_client import OfficialClient


class ImageStub:
  def convert(self, mode):
    return self
  def save(self, buf, format):
    buf.write(b"test-image")


class ProtocolTest(unittest.TestCase):
  def parse(self, args):
    response = '<thinking>next</thinking><tool_call>' + json.dumps(
        {"name": "mobile_use", "arguments": args}) + '</tool_call>'
    return M.parse_response(response, (1080, 2400))["tool_call"]["arguments"]

  def test_pointer_grounding(self):
    self.assertEqual(self.parse({"action": "click", "coordinate": [500, 500]})["coordinate"], [540, 1201])
    self.assertEqual(self.parse({"action": "long_press", "coordinate": [999, 999]})["coordinate"], [1079, 2399])

  def test_text_and_completion(self):
    self.assertEqual(self.parse({"action": "type", "text": 'a\n"b"'})["text"], 'a\n"b"')
    for status in ("success", "fail"):
      self.assertEqual(self.parse({"action": "terminate", "status": status})["status"], status)
    self.assertEqual(self.parse({"action": "open", "text": "Settings"})["action"], "open_app")
    self.assertEqual(self.parse({"action": "system_button", "button": "menu"})["button"], "menu")

  def test_bbox_and_native_coordinate_history(self):
    args = {"action": "click", "coordinate": [100, 200, 900, 800]}
    response = '<tool_call>' + json.dumps({"name": "mobile_use", "arguments": args}) + '</tool_call>'
    parsed = M.parse_response(response, (1080, 2400))
    self.assertEqual(parsed["native_arguments"], args)
    self.assertEqual(parsed["tool_call"]["arguments"]["coordinate"], [540, 1201])

  def test_official_history_retains_text_and_caps_images(self):
    h = M.NativeHistory(3)
    for i in range(6):
      h.append({"action": "click", "coordinate": [i, i]}, f"step {i}", ImageStub())
      messages = h.messages("prompt", "task", ImageStub())
      self.assertEqual(sum(x["role"] == "assistant" for x in messages), i + 1)
      self.assertEqual(sum(c["type"] == "image_url" for x in messages for c in x["content"]), min(i + 2, 3))
    self.assertIn('"coordinate":[0,0]', messages[2]["content"][0]["text"])
    self.assertIsNone(h.steps[0]["image"])

  def test_official_sampling_parameters(self):
    from pathlib import Path
    config = json.loads((Path(__file__).parents[1] / "configs/mai_ui_8b_official.json").read_text())
    payload = OfficialClient("http://test", "MAI-UI-8B", config).payload([])
    self.assertEqual({k: payload[k] for k in ("temperature", "top_k", "top_p", "max_tokens", "seed")},
                     {"temperature": 0.0, "top_k": -1, "top_p": 1.0, "max_tokens": 2048, "seed": 42})
    self.assertNotIn("cache_salt", payload)

  def test_targeted_swipe_and_drag(self):
    args = self.parse({"action": "swipe", "direction": "up", "coordinate": [500, 500]})
    self.assertEqual(args["start_coordinate"], [540, 1201])
    self.assertEqual(args["end_coordinate"], [540, 241])
    args = self.parse({"action": "drag", "start_coordinate": [0, 0], "end_coordinate": [999, 999]})
    self.assertEqual(args["action"], "swipe")
    self.assertEqual(args["end_coordinate"], [1079, 2399])

  def test_reject_ambiguous_and_incomplete(self):
    call = '<tool_call>{"name":"mobile_use","arguments":{"action":"wait"}}</tool_call>'
    for response in (call + call, call[:-4], '{}'):
      with self.subTest(response=response), self.assertRaises(ValueError):
        M.parse_response(response, (1080, 2400))

  def test_reject_unsafe_values(self):
    cases = [{"action": "click"}, {"action": "click", "coordinate": [1000, 1]},
             {"action": "click", "coordinate": [True, 1]},
             {"action": "click", "coordinate": [float("nan"), 1]},
             {"action": "terminate"}, {"action": "system_button", "button": "volume_up"},
             {"action": "type"}, {"action": "swipe", "direction": "diagonal"}]
    for args in cases:
      with self.subTest(args=args), self.assertRaises(ValueError):
        self.parse(args)


if __name__ == "__main__":
  unittest.main()
