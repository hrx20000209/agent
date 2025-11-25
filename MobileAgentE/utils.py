import json
import re


def parse_bounds(bounds):
    """
    支持 "[x1,y1][x2,y2]" 或 "x1,y1,x2,y2" 或其它常见格式。
    始终返回 [x1, y1, x2, y2] (int)
    """
    if bounds is None:
        return None

    # 提取所有数字（包括负数）
    nums = list(map(int, re.findall(r"-?\d+", str(bounds))))

    if len(nums) == 4:
        return nums
    else:
        print(f"[WARN] Unexpected bounds format: {bounds}")
        return None


def safe_json_parse(response: str):
    """
    Parse LLM output into normalized structure:
    {
      "action": {
         "name": str,
         "arguments": dict
      }
    }
    Handles cases where action is a natural language string.
    """
    if not response:
        return None

    # --------------------------------------------
    # 1) Extract JSON object
    # --------------------------------------------
    match = re.search(r'\{.*\}', response, re.S)
    if not match:
        print("⚠️ No JSON object found.")
        return None

    candidate = match.group(0).strip()

    # Clean markdown wrappers
    candidate = re.sub(r'```[a-zA-Z]*', '', candidate)
    candidate = candidate.strip('` \n')

    # normalize single quotes
    candidate = candidate.replace("'", '"')
    candidate = re.sub(r'\n+', ' ', candidate)

    # --------------------------------------------
    # 2) Try load JSON
    # --------------------------------------------
    try:
        parsed = json.loads(candidate)
    except Exception as e:
        print("⚠️ JSON load failed:", e)
        return None

    # --------------------------------------------
    # 3) Normalize action
    # --------------------------------------------
    action = parsed.get("action")

    # Case A: action is string (your case)
    if isinstance(action, str):
        return {
            "action": {
                "name": action.strip(),
                "arguments": {}
            }
        }

    # Case B: action is dict (old format)
    if isinstance(action, dict):
        name = action.get("name", "")
        args = action.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        return {
            "action": {
                "name": name,
                "arguments": args
            }
        }

    # Case C: unexpected format
    return {
        "action": {
            "name": str(action),
            "arguments": {}
        }
    }
