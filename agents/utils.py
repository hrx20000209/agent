from typing import Dict, Any, Optional, Tuple
import time

from MobileAgentE.controller import (
    tap, swipe, type as adb_type, back, home, switch_app, enter, launch_app
)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _norm_to_pixel(x_norm: float, y_norm: float, width: int, height: int) -> Tuple[int, int]:
    x_norm = _clamp01(x_norm)
    y_norm = _clamp01(y_norm)
    x = int(x_norm * width)
    y = int(y_norm * height)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def _coord_to_pixel(
    x_raw: float,
    y_raw: float,
    width: int,
    height: int,
    coord_space: str = "auto",
) -> Tuple[int, int]:
    """
    Accept one of:
    - normalized coords in [0,1]
    - normalized coords in [0,1000] (common in UI agent prompts)
    - absolute pixel coords
    """
    x = float(x_raw)
    y = float(y_raw)
    mode = str(coord_space or "auto").strip().lower()

    if mode in {"norm1", "normalized", "0_1"}:
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return _norm_to_pixel(x, y, width, height)
    elif mode in {"norm1000", "1000", "0_1000"}:
        if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
            return _norm_to_pixel(x / 1000.0, y / 1000.0, width, height)
    elif mode in {"pixel", "px"}:
        px = int(round(x))
        py = int(round(y))
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        return px, py
    else:
        # auto: keep old behavior to avoid breaking existing pixel outputs.
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return _norm_to_pixel(x, y, width, height)

    # absolute pixel format
    px = int(round(x))
    py = int(round(y))
    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))
    return px, py


def _apply_coord_scale(px: int, py: int, width: int, height: int, coord_scale: float) -> Tuple[int, int]:
    if coord_scale is None or abs(float(coord_scale) - 1.0) < 1e-6:
        return px, py
    s = float(coord_scale)
    x = int(round(float(px) * s))
    y = int(round(float(py) * s))
    max_w = max(1, int(round(float(width) * s)))
    max_h = max(1, int(round(float(height) * s)))
    x = max(0, min(max_w - 1, x))
    y = max(0, min(max_h - 1, y))
    return x, y


def extract_action_inputs(action_obj: dict):
    # 1. 优先标准字段（如果解析器填了）
    inputs = action_obj.get("action_inputs")
    if inputs:
        return inputs

    # 2. 有些框架直接放在顶层 arguments
    inputs = action_obj.get("arguments")
    if inputs:
        return inputs

    # 3. 你的 MobileAgentE / tool_call 常见结构：raw.arguments
    raw = action_obj.get("raw") or {}
    inputs = raw.get("arguments")
    if inputs:
        return inputs

    # 4. 最后兜底
    return {}


def execute_action(
    action_obj: Dict[str, Any],
    width, height,
    adb,
    coord_scale: float = 1.0,
) -> Optional[str]:
    """
    Execute canonical action schema (standalone function).

    Args:
        action_obj: {
            "action_type": "...",
            "action_inputs": {...}
        }
        width: width of the screenshot
        height: height of the screenshot
        adb: adb controller / path (same as previous self.adb)

    Returns:
        Optional[str]: executed action string (for logging / replay)
    """

    if not action_obj:
        print("⚠️ No valid action.")
        return None

    action_type = action_obj.get("action_type", "").lower()
    args = extract_action_inputs(action_obj)
    coord_space = "1000"

    print(f"[EXEC] {action_type} → {args} (coord_space={coord_space})")

    # ==============================
    # 🟢 TERMINATE
    # ==============================
    if action_type in ["finished", "done", "terminate"]:
        return "finished"

    # ==============================
    # 🟡 WAIT
    # ==============================
    if action_type == "wait":
        sec = float(args.get("seconds", 1))
        time.sleep(sec)
        return f"wait({sec})"

    # ==============================
    # 🔵 SYSTEM BUTTONS
    # ==============================
    if action_type == "press_back":
        back(adb)
        return "press_back"

    if action_type == "press_home":
        home(adb)
        return "press_home"

    if action_type == "press_enter":
        enter(adb)
        return "press_enter"

    # ==============================
    # 🟣 TYPE
    # ==============================
    if action_type == "type":
        text = args.get("content", "")
        adb_type(adb, text)
        return f"type({text})"

    # ==============================
    # 🔴 CLICK
    # ==============================
    if action_type == "click":
        coord = args.get("coordinate")
        if coord is None:
            print("⚠️ click missing coordinate")
            return None

        x_raw, y_raw = coord
        x, y = _coord_to_pixel(x_raw, y_raw, width, height, coord_space=coord_space)
        x, y = _apply_coord_scale(x, y, width, height, coord_scale)
        tap(adb, x, y)
        return f"click({x},{y})"

    # ==============================
    # 🟠 LONG PRESS
    # ==============================
    if action_type == "long_press":
        coord = args.get("coordinate")
        if coord is None:
            print("⚠️ long_press missing coordinate")
            return None

        x_raw, y_raw = coord
        x, y = _coord_to_pixel(x_raw, y_raw, width, height, coord_space=coord_space)
        x, y = _apply_coord_scale(x, y, width, height, coord_scale)
        duration_ms = int(args.get("duration_ms", 600))
        swipe(adb, x, y, x, y)  # press via swipe hold
        return f"long_press({x},{y},{duration_ms}ms)"

    # ==============================
    # 🟡 SWIPE / DRAG
    # ==============================
    if action_type in ["swipe", "drag"]:
        s = args.get("start_coordinate") or args.get("coordinate")
        e = args.get("end_coordinate")

        if not s:
            print("⚠️ swipe missing start coordinate")
            return None

        if not e:
            # directional fallback
            direction = args.get("direction", "down")
            sx_raw, sy_raw = s
            sx, sy = _coord_to_pixel(sx_raw, sy_raw, width, height, coord_space=coord_space)
            sx, sy = _apply_coord_scale(sx, sy, width, height, coord_scale)

            if direction == "down":
                step = int(round(400 * float(coord_scale)))
                ex, ey = sx, min(max(1, int(round(height * float(coord_scale)))) - 1, sy + step)
            elif direction == "up":
                step = int(round(400 * float(coord_scale)))
                ex, ey = sx, max(0, sy - step)
            elif direction == "left":
                step = int(round(400 * float(coord_scale)))
                ex, ey = max(0, sx - step), sy
            else:  # right
                step = int(round(400 * float(coord_scale)))
                ex, ey = min(max(1, int(round(width * float(coord_scale)))) - 1, sx + step), sy
        else:
            sx_raw, sy_raw = s
            ex_raw, ey_raw = e
            sx, sy = _coord_to_pixel(sx_raw, sy_raw, width, height, coord_space=coord_space)
            ex, ey = _coord_to_pixel(ex_raw, ey_raw, width, height, coord_space=coord_space)
            sx, sy = _apply_coord_scale(sx, sy, width, height, coord_scale)
            ex, ey = _apply_coord_scale(ex, ey, width, height, coord_scale)

        swipe(adb, sx, sy, ex, ey)
        return f"swipe({sx},{sy}→{ex},{ey})"

    # ==============================
    # 🟢 OPEN APP
    # ==============================
    if action_type == "open_app":
        nested = args.get("arguments") if isinstance(args, dict) else None
        if isinstance(nested, dict):
            app_name = (
                nested.get("app_name")
                or nested.get("value")
                or nested.get("text")
                or nested.get("content")
            )
        else:
            app_name = None

        if not app_name:
            if "app_name" in args:
                app_name = args.get("app_name")
            elif "value" in args:
                app_name = args.get("value")
            elif "text" in args:
                app_name = args.get("text")
            else:
                app_name = args.get("content")

        app = launch_app(adb, app_name)
        return f"open_app({app_name})"

    print(f"⚠️ Unsupported action_type={action_type}")
    return None
