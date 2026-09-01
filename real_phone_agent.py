"""Standalone MobileExplorer runner for a physical Android phone.

This entry point deliberately does not import AndroidWorld.  The phone is
observed and controlled only through ADB, while the repository's deterministic
belief graph, candidate matrix, distiller, and gate remain reusable.

Example::

    python real_phone_agent.py \
      --task "Delete the recipes named Lentil Soup and Garlic Butter Shrimp from Broccoli" \
      --serial ABC123 \
      --api_url http://127.0.0.1:8100/v1/chat/completions \
      --model GELAB-ZERO-4B \
      --exploration on --graph on --skip off --out_dir runs/broccoli
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from Explorer.graph_distiller import (
    GraphDistiller,
    GraphDistillerConfig,
    GraphMode,
    GraphReasoningGate,
)
from Explorer.progressive_belief_graph import (
    GenerationGuardedGraph,
    GraphSnapshot,
    ProgressiveBeliefGraph,
    UIStateDescriptor,
    describe_ui_state,
)
from Explorer.state_graph_information import (
    InformationNeed,
    MatrixAblations,
    PredictiveElementScorer,
    StateGraphInformationMatrix,
    element_identity,
    parse_reasoning_prior,
)
from Explorer.utils import node_to_text
from MobileAgentE.tree import Node, parse_a11y_tree
from agents.mai_ui_agent import MAIOneStepAgent


try:
    import psutil
except Exception:  # pragma: no cover - optional on a minimal host
    psutil = None


DEFAULT_ADB_TIMEOUT = 12.0
RECOVERY_HASH_THRESHOLD = 8


def now_ms() -> float:
    return time.time() * 1000.0


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


class AdbError(RuntimeError):
    pass


class AdbClient:
    """Small argument-vector-only ADB wrapper.

    Keeping commands as argv lists avoids shell quoting bugs for real device
    identifiers and text input.  Every device operation in this module goes
    through this class.
    """

    def __init__(self, serial: str, adb_path: str = "adb", timeout: float = DEFAULT_ADB_TIMEOUT, retries: int = 1, allow_emulator: bool = False):
        if not serial:
            raise ValueError("--serial is required for real-phone execution")
        self.serial = str(serial)
        if not allow_emulator and (self.serial.startswith("emulator-") or self.serial.startswith("emulator")):
            raise ValueError("emulator serial rejected; use a physical phone or pass --allow_emulator for smoke tests")
        self.adb_path = str(adb_path or "adb")
        self.timeout = float(timeout)
        self.retries = max(1, int(retries))
        self.prefix = [self.adb_path, "-s", self.serial]
        self._reported_slow_a11y = False

    def run(
        self,
        *args: str,
        timeout: Optional[float] = None,
        check: bool = True,
        binary: bool = False,
    ) -> subprocess.CompletedProcess:
        command = self.prefix + [str(arg) for arg in args]
        last: Optional[subprocess.CompletedProcess] = None
        for attempt in range(self.retries):
            try:
                last = subprocess.run(
                    command,
                    capture_output=True,
                    text=not binary,
                    timeout=float(timeout or self.timeout),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or (b"" if binary else "")
                stderr = exc.stderr or (b"" if binary else f"timeout after {timeout or self.timeout}s")
                last = subprocess.CompletedProcess(command, 124, stdout, stderr)
            if last.returncode == 0:
                return last
            if attempt + 1 < self.retries:
                time.sleep(0.25)
        assert last is not None
        if check:
            stdout = last.stdout.decode(errors="replace") if isinstance(last.stdout, bytes) else str(last.stdout or "")
            stderr = last.stderr.decode(errors="replace") if isinstance(last.stderr, bytes) else str(last.stderr or "")
            raise AdbError(f"ADB failed ({last.returncode}): {' '.join(command)}\n{stdout}\n{stderr}")
        return last

    def assert_ready(self) -> None:
        result = self.run("get-state", timeout=8)
        state = (result.stdout or "").strip()
        if state != "device":
            raise AdbError(f"ADB target is not ready: {state!r}")

    def screenshot(self, path: Path) -> None:
        result = self.run("exec-out", "screencap", "-p", timeout=self.timeout, binary=True)
        data = result.stdout
        if not isinstance(data, bytes) or not data:
            raise AdbError("ADB returned an empty screenshot")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        with Image.open(io.BytesIO(data)) as image:
            image.verify()

    def dump_ui(self, path: Path) -> None:
        if not self._reported_slow_a11y:
            print("[A11y] transport=uiautomator_dump_fallback; fast socket service is not installed/configured", flush=True)
            self._reported_slow_a11y = True
        remote = f"/sdcard/mobile_explorer_{os.getpid()}_{int(time.time() * 1000)}.xml"
        result = self.run("shell", "uiautomator", "dump", "--compressed", remote, timeout=self.timeout, check=False)
        if result.returncode != 0:
            result = self.run("shell", "uiautomator", "dump", remote, timeout=self.timeout, check=False)
        if result.returncode != 0:
            raise AdbError("uiautomator dump failed")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.run("pull", remote, str(path), timeout=self.timeout)
        finally:
            self.run("shell", "rm", "-f", remote, timeout=4, check=False)
        if not path.exists() or path.stat().st_size == 0:
            raise AdbError("ADB returned an empty accessibility tree")

    def current_package(self) -> str:
        for command in (("shell", "dumpsys", "activity", "activities"), ("shell", "dumpsys", "window", "windows")):
            result = self.run(*command, timeout=8, check=False)
            text = result.stdout or ""
            patterns = (
                r"mResumedActivity:.*?\s([A-Za-z0-9._]+)/(?:[A-Za-z0-9_.$]+)",
                r"mFocusedApp:.*?\s([A-Za-z0-9._]+)/(?:[A-Za-z0-9_.$]+)",
                r"topResumedActivity=ActivityRecord\{[^ ]+\s([A-Za-z0-9._]+)/",
            )
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1)
        return ""

    def screen_size(self) -> Tuple[int, int]:
        result = self.run("shell", "wm", "size", timeout=5, check=False)
        match = re.search(r"(?:Physical|Override) size:\s*(\d+)x(\d+)", result.stdout or "")
        if match:
            return int(match.group(1)), int(match.group(2))
        return 1080, 2400

    def packages(self) -> List[str]:
        result = self.run("shell", "pm", "list", "packages", "-3", timeout=10)
        return sorted({line.split(":", 1)[-1].strip() for line in (result.stdout or "").splitlines() if line.strip()})

    def package_label(self, package: str) -> str:
        result = self.run("shell", "dumpsys", "package", package, timeout=10, check=False)
        text = result.stdout or ""
        for pattern in (r"application-label[^=]*=([^\r\n]+)", r"nonLocalizedLabel=([^,\s}]+)", r"label=([^,\s}]+)"):
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(1).strip().strip('"')
        return ""

    def launch_package(self, package: str) -> None:
        if not package:
            raise AdbError("refusing to launch an empty package")
        result = self.run(
            "shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1",
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            raise AdbError(f"could not launch package {package}")

    def tap(self, x: int, y: int) -> None:
        self.run("shell", "input", "tap", str(int(x)), str(int(y)))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 450) -> None:
        self.run("shell", "input", "swipe", str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(duration_ms)))

    def back(self) -> None:
        self.run("shell", "input", "keyevent", "KEYCODE_BACK", timeout=5)

    def home(self) -> None:
        self.run("shell", "input", "keyevent", "KEYCODE_HOME", timeout=5)

    def enter(self) -> None:
        self.run("shell", "input", "keyevent", "KEYCODE_ENTER", timeout=5)

    def input_text(self, value: str) -> None:
        # Android's input tool uses %s for spaces.  The argv call keeps shell
        # metacharacters from being interpreted by the host shell.
        text = str(value).replace("%", "%25").replace(" ", "%s").replace("\n", "%s")
        result = self.run("shell", "input", "text", text, timeout=15, check=False)
        if result.returncode != 0:
            for char in str(value):
                if char == "\n":
                    self.enter()
                else:
                    self.run("shell", "input", "text", char, timeout=5, check=False)

    def device_memory(self, package: str = "") -> Dict[str, Any]:
        payload: Dict[str, Any] = {"package": package or None}
        if package:
            result = self.run("shell", "dumpsys", "meminfo", package, timeout=15, check=False)
            payload["app_meminfo"] = (result.stdout or "")[-12000:]
        result = self.run("shell", "cat", "/proc/meminfo", timeout=8, check=False)
        payload["proc_meminfo"] = (result.stdout or "")[-12000:]
        return payload


KNOWN_APP_PACKAGES = {
    "broccoli": "com.flauschcode.broccoli",
    "broccoli app": "com.flauschcode.broccoli",
    "chrome": "com.android.chrome",
    "google chrome": "com.android.chrome",
    "settings": "com.android.settings",
    "clock": "com.google.android.deskclock",
    "contacts": "com.google.android.contacts",
    "dialer": "com.google.android.dialer",
    "phone": "com.google.android.dialer",
    "messages": "com.google.android.apps.messaging",
    "gmail": "com.google.android.gm",
    "maps": "com.google.android.apps.maps",
    "photos": "com.google.android.apps.photos",
    "youtube": "com.google.android.youtube",
    "spotify": "com.spotify.music",
    "tasks": "org.tasks",
    "markor": "net.gsantner.markor",
    "vlc": "org.videolan.vlc",
    "joplin": "net.cozic.joplin",
    "audio recorder": "com.dimowner.audiorecorder",
    "pro expense": "com.arduia.expense",
    "osmand": "net.osmand",
}


class AppResolver:
    def __init__(self, device: AdbClient):
        self.device = device
        self.installed = set(device.packages())
        self.labels: Dict[str, str] = {}

    def resolve(self, name: str) -> Optional[str]:
        raw = str(name or "").strip()
        if not raw:
            return None
        if raw in self.installed:
            return raw
        low = raw.lower()
        for alias, package in sorted(KNOWN_APP_PACKAGES.items(), key=lambda item: len(item[0]), reverse=True):
            if alias in low and package in self.installed:
                return package
        compact = re.sub(r"[^a-z0-9]", "", low)
        for package in self.installed:
            leaf = package.rsplit(".", 1)[-1].lower()
            if compact and (compact in re.sub(r"[^a-z0-9]", "", leaf) or re.sub(r"[^a-z0-9]", "", leaf) in compact):
                return package
        # Package names are not a reliable user-facing identifier.  Resolve
        # labels from the live device as a final fallback for arbitrary apps.
        for package in sorted(self.installed):
            label = self.labels.get(package)
            if label is None:
                label = self.device.package_label(package)
                self.labels[package] = label
            if label and (low == label.lower() or label.lower() in low or low in label.lower()):
                return package
        return None

    def target_from_task(self, task: str, explicit: str = "") -> Optional[str]:
        if explicit:
            package = self.resolve(explicit)
            if package:
                return package
            if explicit in self.installed:
                return explicit
        package_match = re.search(r"\b([a-z][a-z0-9_]+(?:\.[a-z0-9_]+){1,})\b", task.lower())
        if package_match and package_match.group(1) in self.installed:
            return package_match.group(1)
        return self.resolve(task)


def _iter_nodes(root: Optional[Node]) -> Iterable[Node]:
    if root is None:
        return
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(getattr(node, "children", None) or []))


def _strict_tree_hash(root: Optional[Node]) -> str:
    records: List[str] = []
    for node in _iter_nodes(root):
        records.append("|".join([
            str(getattr(node, "package", "") or ""),
            str(getattr(node, "class_name", "") or ""),
            str(getattr(node, "resource_id", "") or ""),
            str(getattr(node, "text", "") or ""),
            str(getattr(node, "content_desc", "") or ""),
            str(getattr(node, "bounds", "") or ""),
            str(bool(getattr(node, "clickable", False))),
            str(bool(getattr(node, "scrollable", False))),
        ]))
    return hashlib.sha256("\n".join(records).encode("utf-8", errors="ignore")).hexdigest()[:24]


def _parse_bounds_silent(value: Any) -> Optional[Tuple[int, int, int, int]]:
    numbers = re.findall(r"-?\d+", str(value or ""))
    if len(numbers) != 4:
        return None
    return tuple(int(number) for number in numbers)  # type: ignore[return-value]


def _image_hash(path: Path) -> int:
    with Image.open(path).convert("L") as image:
        image = image.resize((16, 16), Image.Resampling.BILINEAR)
        pixels = list(image.getdata())
    mean = sum(pixels) / max(1, len(pixels))
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)
    return value


def _hash_distance(left: int, right: int) -> int:
    return (int(left) ^ int(right)).bit_count()


@dataclass
class UICapture:
    screenshot_path: str
    xml_path: str
    width: int
    height: int
    root: Node
    descriptor: UIStateDescriptor
    strict_hash: str
    visual_hash: int
    package: str

    def wire(self) -> Dict[str, Any]:
        return {
            "screenshot_path": self.screenshot_path,
            "width": self.width,
            "height": self.height,
            "descriptor": asdict(self.descriptor),
            "strict_hash": self.strict_hash,
            "visual_hash": self.visual_hash,
            "package": self.package,
        }


def _capture_ui(device: AdbClient, directory: Path, stem: str) -> UICapture:
    directory.mkdir(parents=True, exist_ok=True)
    screenshot = directory / f"{stem}.png"
    xml = directory / f"{stem}.xml"
    device.screenshot(screenshot)
    device.dump_ui(xml)
    try:
        root = parse_a11y_tree(str(xml))
    except Exception:
        root = Node(class_name="LEAF_ROOT", uid="ROOT", children=[])
    candidates = list(_iter_nodes(root))
    descriptor = describe_ui_state(root, len(candidates))
    package = device.current_package() or descriptor.package
    if package and package != descriptor.package:
        descriptor = replace(descriptor, package=package)
    with Image.open(screenshot) as image:
        width, height = image.size
    return UICapture(
        screenshot_path=str(screenshot),
        xml_path=str(xml),
        width=int(width),
        height=int(height),
        root=root,
        descriptor=descriptor,
        strict_hash=_strict_tree_hash(root),
        visual_hash=_image_hash(screenshot),
        package=package,
    )


def _same_capture(current: UICapture, baseline: Dict[str, Any]) -> bool:
    expected = baseline.get("descriptor") or {}
    if str(current.descriptor.signature) != str(expected.get("signature", "")):
        return False
    baseline_package = str(baseline.get("package") or "")
    if baseline_package and current.package and current.package != baseline_package:
        return False
    return _hash_distance(current.visual_hash, int(baseline.get("visual_hash", 0))) <= RECOVERY_HASH_THRESHOLD


def _descriptor_from_wire(raw: Dict[str, Any]) -> UIStateDescriptor:
    data = dict(raw or {})
    data["labels"] = tuple(data.get("labels") or ())
    return UIStateDescriptor(**data)


def _coord_pair(action: Dict[str, Any], width: int, height: int, start: bool = True) -> Optional[Tuple[int, int]]:
    inputs = action.get("action_inputs") or {}
    raw = inputs.get("start_coordinate" if start else "end_coordinate")
    if raw is None:
        raw = inputs.get("coordinate")
    if raw is None and start:
        raw = inputs.get("start_box")
    if not isinstance(raw, (list, tuple)):
        return None
    if len(raw) == 4:
        raw = ((float(raw[0]) + float(raw[2])) / 2.0, (float(raw[1]) + float(raw[3])) / 2.0)
    if len(raw) != 2:
        return None
    x, y = float(raw[0]), float(raw[1])
    mode = str(action.get("coord_space") or inputs.get("coord_space") or "norm1000").lower()
    if mode in {"norm1", "normalized", "0_1"}:
        x, y = x * width, y * height
    elif mode in {"norm1000", "1000", "0_1000", "auto"} and 0 <= x <= 1000 and 0 <= y <= 1000:
        x, y = x / 1000.0 * width, y / 1000.0 * height
    return max(0, min(width - 1, int(round(x)))), max(0, min(height - 1, int(round(y))))


def _direction_delta(direction: str, width: int, height: int) -> Tuple[int, int]:
    distance = max(180, int(min(width, height) * 0.32))
    direction = str(direction or "down").lower()
    return {
        "up": (0, -distance),
        "down": (0, distance),
        "left": (-distance, 0),
        "right": (distance, 0),
    }.get(direction, (0, distance))


def _execute_device_action(device: AdbClient, action: Dict[str, Any], width: int, height: int, resolver: AppResolver) -> str:
    action_type = str(action.get("action_type") or "wait").lower()
    inputs = action.get("action_inputs") or {}
    if action_type in {"finished", "done", "terminate", "status", "infeasible"}:
        return action_type
    if action_type == "wait":
        time.sleep(max(0.0, min(30.0, float(inputs.get("seconds", 1.0)))))
        return "wait"
    if action_type in {"press_back", "navigate_back", "back"}:
        device.back()
        return "navigate_back"
    if action_type in {"press_home", "navigate_home", "home"}:
        device.home()
        return "navigate_home"
    if action_type in {"press_enter", "enter"}:
        device.enter()
        return "enter"
    if action_type in {"type", "input_text"}:
        device.input_text(str(inputs.get("content", inputs.get("text", ""))))
        return "input_text"
    if action_type in {"click", "tap", "long_press"}:
        point = _coord_pair(action, width, height)
        if point is None:
            raise ValueError("click-like action has no coordinate")
        if action_type == "long_press":
            duration = int(inputs.get("duration_ms", 700))
            device.swipe(point[0], point[1], point[0], point[1], duration)
        else:
            device.tap(*point)
        return action_type
    if action_type in {"scroll", "swipe", "drag"}:
        start = _coord_pair(action, width, height, start=True) or (width // 2, height // 2)
        end = _coord_pair(action, width, height, start=False)
        if end is None:
            dx, dy = _direction_delta(str(inputs.get("direction", "down")), width, height)
            end = (max(0, min(width - 1, start[0] + dx)), max(0, min(height - 1, start[1] + dy)))
        device.swipe(start[0], start[1], end[0], end[1])
        return action_type
    if action_type in {"open_app", "open"}:
        name = inputs.get("app_name") or inputs.get("content") or inputs.get("text") or inputs.get("value")
        package = resolver.resolve(str(name or ""))
        if not package:
            raise ValueError(f"cannot resolve installed app: {name!r}")
        device.launch_package(package)
        return f"open_app:{package}"
    raise ValueError(f"unsupported action type: {action_type}")


def _parse_model_action(text: str, coord_space: str) -> Dict[str, Any]:
    raw_text = str(text or "")
    function_match = re.search(r"\b(click|tap|long_press|input_text|type|scroll|navigate_back|navigate_home|open_app|wait)\s*\((.*?)\)", raw_text, re.I | re.S)
    if function_match:
        name = function_match.group(1).lower()
        args_text = function_match.group(2).strip()
        if name in {"click", "tap", "long_press"}:
            coordinates = re.findall(r"-?\d+(?:\.\d+)?", args_text)
            if len(coordinates) >= 2:
                return {"action_type": "long_press" if name == "long_press" else "click", "action_inputs": {"coordinate": [float(coordinates[0]), float(coordinates[1])]}, "coord_space": coord_space, "raw": {"function_call": raw_text}}
        if name in {"input_text", "type", "open_app"}:
            value_match = re.search(r"(?:content|text|app_name|value)\s*=\s*(['\"])(.*?)\1", args_text, re.S)
            value = value_match.group(2) if value_match else args_text.strip().strip("'\"")
            return {"action_type": "type" if name in {"input_text", "type"} else "open_app", "action_inputs": {"content": value}, "coord_space": coord_space, "raw": {"function_call": raw_text}}
        if name == "scroll":
            direction_match = re.search(r"(?:direction\s*=\s*)?['\"]?(up|down|left|right)['\"]?", args_text, re.I)
            return {"action_type": "scroll", "action_inputs": {"direction": (direction_match.group(1).lower() if direction_match else "down")}, "coord_space": coord_space, "raw": {"function_call": raw_text}}
        if name in {"navigate_back", "navigate_home"}:
            return {"action_type": "press_back" if name.endswith("back") else "press_home", "action_inputs": {}, "coord_space": coord_space, "raw": {"function_call": raw_text}}
        if name == "wait":
            seconds = re.search(r"-?\d+(?:\.\d+)?", args_text)
            return {"action_type": "wait", "action_inputs": {"seconds": float(seconds.group(0)) if seconds else 1.0}, "coord_space": coord_space, "raw": {"function_call": raw_text}}
    parser = MAIOneStepAgent("adb", coord_space=coord_space)
    action = parser.parse_action(raw_text)
    if not isinstance(action, dict):
        return {"action_type": "wait", "action_inputs": {"seconds": 1}, "coord_space": coord_space, "raw": {"parse_error": True}}
    action = copy.deepcopy(action)
    action.setdefault("coord_space", coord_space)
    action_type = str(action.get("action_type") or "wait").lower()
    inputs = action.setdefault("action_inputs", {}) or {}
    action["action_inputs"] = inputs
    status = str(inputs.get("status", "")).lower()
    if action_type in {"terminate", "done", "exit", "stop", "answer"} and status in {"infeasible", "fail", "failed"}:
        action["action_type"] = "infeasible"
    elif action_type in {"terminate", "done", "exit", "stop", "answer"}:
        action["action_type"] = "finished"
    elif action_type == "status" and status in {"complete", "success", "done"}:
        action["action_type"] = "finished"
    elif action_type == "status" and status in {"infeasible", "fail", "failed"}:
        action["action_type"] = "infeasible"
    return action


def _control_at(root: Node, action: Dict[str, Any], width: int, height: int) -> Tuple[Optional[Node], Optional[str], Optional[Tuple[int, int, int, int]]]:
    point = _coord_pair(action, width, height)
    if point is None:
        return None, None, None
    x, y = point
    matches: List[Tuple[int, Node, Tuple[int, int, int, int]]] = []
    for node in _iter_nodes(root):
        bounds = _parse_bounds_silent(getattr(node, "bounds", ""))
        if not bounds:
            continue
        x1, y1, x2, y2 = bounds
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            matches.append((area, node, tuple(bounds)))
    if not matches:
        return None, None, None
    _, node, bounds = min(matches, key=lambda item: item[0])
    return node, _control_key(node, width, height), bounds


def _has_filled_input(root: Node) -> bool:
    for node in _iter_nodes(root):
        cls = str(getattr(node, "class_name", "") or "").lower()
        text = str(getattr(node, "text", "") or "").strip()
        if text and ("edittext" in cls or "textfield" in cls or bool(getattr(node, "focused", False))):
            return True
    return False


def _control_key(node: Node, width: int, height: int) -> str:
    """Stable control identity used by edges; it intentionally omits bounds."""
    resource_id = str(getattr(node, "resource_id", "") or "").strip().lower()
    text = re.sub(r"\s+", " ", str(getattr(node, "text", "") or "").strip().lower())
    description = re.sub(r"\s+", " ", str(getattr(node, "content_desc", "") or "").strip().lower())
    class_name = str(getattr(node, "class_name", "") or "").strip().lower()
    if resource_id or text or description:
        return "|".join((resource_id, text, description, class_name))
    bounds = _parse_bounds_silent(getattr(node, "bounds", ""))
    if bounds:
        center_x = (bounds[0] + bounds[2]) // 2
        center_y = (bounds[1] + bounds[3]) // 2
        return f"||{class_name}|{center_x // 96}:{center_y // 96}"
    return f"|||{class_name}"


def _safe_candidates(root: Node, step: int, node_id: Optional[str], node_ready: bool, allow_exploration: bool, log_path: Path) -> List[Dict[str, Any]]:
    """Hard safety filter; every removed candidate is auditable."""
    candidates: List[Dict[str, Any]] = []
    risky = re.compile(r"\b(delete|remove|save|submit|send|purchase|buy|uninstall|clear all|erase)\b|删除|保存|提交|发送|购买|卸载|清除", re.I)
    for index, node in enumerate(_iter_nodes(root)):
        merged = " ".join(str(getattr(node, key, "") or "") for key in ("text", "content_desc", "resource_id", "class_name"))
        bounds = _parse_bounds_silent(getattr(node, "bounds", ""))
        reason = None
        if not (getattr(node, "clickable", False) or getattr(node, "scrollable", False)):
            reason = "not_clickable_or_scrollable"
        elif not bounds:
            reason = "missing_bounds"
        elif getattr(node, "enabled", True) is False:
            reason = "disabled"
        elif not any(str(getattr(node, key, "") or "").strip() for key in ("text", "content_desc", "resource_id")):
            reason = "missing_control_identity"
        elif "systemui" in merged.lower() or "statusbar" in merged.lower():
            reason = "system_ui_noise"
        elif risky.search(merged):
            reason = "irreversible_or_action_button"
        elif bool(getattr(node, "checkable", False)) or any(token in merged.lower() for token in ("checkbox", "radiobutton", "switch", "toggle")):
            reason = "context_dependent_selection"
        if reason:
            append_jsonl(log_path, {"event": "filtered", "step": step, "index": index, "reason": reason, "text": merged[:240], "bounds": bounds})
            continue
        action_type = "scroll" if bool(getattr(node, "scrollable", False)) else "click"
        center = ((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2)
        candidates.append({
            "index": index,
            "action_type": action_type,
            "label": (getattr(node, "text", "") or getattr(node, "content_desc", "") or getattr(node, "resource_id", "") or "").strip()[:80],
            "text": str(getattr(node, "text", "") or ""),
            "content_desc": str(getattr(node, "content_desc", "") or ""),
            "class_name": str(getattr(node, "class_name", "") or ""),
            "role": str(getattr(node, "class_name", "") or "").rsplit(".", 1)[-1].lower(),
            "bounds": tuple(bounds),
            "pixel": center,
            "node": node,
        })
    if not allow_exploration or _has_filled_input(root):
        return []
    if step < 2:
        return []
    if not node_ready:
        candidates = [candidate for candidate in candidates if candidate["action_type"] == "scroll"]
    return candidates


def _candidate_action(candidate: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    x, y = candidate["pixel"]
    action_type = candidate["action_type"]
    if action_type == "scroll":
        return {
            "action_type": "scroll",
            "coord_space": "pixel",
            "action_inputs": {"coordinate": [x, y], "direction": "down", "label": candidate.get("label", "")},
            "control_key": candidate.get("control_key"),
        }
    return {
        "action_type": "click",
        "coord_space": "pixel",
        "action_inputs": {"coordinate": [x, y], "label": candidate.get("label", "")},
        "control_key": candidate.get("control_key"),
    }


def _explorer_recover(
    device: AdbClient,
    baseline: Dict[str, Any],
    work_dir: Path,
    resolver: AppResolver,
    max_back_steps: int,
    probe_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def check(label: str) -> Tuple[bool, Optional[UICapture]]:
        capture = _capture_ui(device, work_dir, f"recover_{label}_{int(time.time() * 1000)}")
        return _same_capture(capture, baseline), capture

    same, capture = check("noop")
    if same:
        return {"restored": True, "level": "NOOP", "package": capture.package if capture else "", "left_app": False}

    original_package = str(baseline.get("package") or "")
    # A one-step inverse is intentionally explicit.  For SCROLL it uses the
    # opposite gesture; for TAP_NAV it uses Back without assuming a fixed
    # activity stack.
    probe_action = probe_action or {}
    probe_inputs = probe_action.get("action_inputs") or {}
    if probe_action.get("action_type") == "scroll":
        width = int(baseline.get("width", 1080))
        height = int(baseline.get("height", 2400))
        point = _coord_pair(probe_action, width, height) or (width // 2, height // 2)
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}.get(str(probe_inputs.get("direction", "down")).lower(), "up")
        dx, dy = _direction_delta(opposite, width, height)
        device.swipe(point[0], point[1], max(0, min(width - 1, point[0] + dx)), max(0, min(height - 1, point[1] + dy)))
    else:
        device.back()
    same, capture = check("inverse")
    if same:
        return {"restored": True, "level": "INVERSE", "package": capture.package if capture else "", "left_app": False}

    for count in range(1, max(0, int(max_back_steps)) + 1):
        device.back()
        current_package = device.current_package()
        left_app = bool(original_package and current_package and current_package != original_package)
        if left_app and original_package:
            # Re-enter immediately after leaving the target app; do not keep
            # pressing Back from the launcher.
            try:
                device.launch_package(original_package)
            except Exception:
                pass
        same, capture = check(f"back_{count}")
        if same:
            return {"restored": True, "level": "BACK_N", "back_count": count, "package": capture.package if capture else current_package, "left_app": left_app}

    # Deeplinks are intentionally not attempted: many real apps do not export
    # their activities.  A package relaunch is a safe last resort only when a
    # concrete baseline package is known.
    if original_package:
        try:
            device.launch_package(original_package)
            same, capture = check("relaunch")
            if same:
                return {"restored": True, "level": "TRAJECTORY_REPLAY", "package": capture.package if capture else original_package, "left_app": False}
        except Exception as exc:
            return {"restored": False, "level": "FAILED", "reason": str(exc), "left_app": True}
    return {"restored": False, "level": "FAILED", "reason": "baseline_not_restored", "left_app": bool(original_package and device.current_package() != original_package)}


def _explorer_worker(config: Dict[str, Any], stop_event: Any, result_queue: Any) -> None:
    """Process target.  It never mutates the parent's graph."""
    try:
        device = AdbClient(config["serial"], config.get("adb_path", "adb"), config.get("adb_timeout", DEFAULT_ADB_TIMEOUT), config.get("adb_retries", 1), config.get("allow_emulator", False))
        resolver = AppResolver(device)
        baseline = config["baseline"]
        worker_dir = Path(config["worker_dir"])
        results: List[Dict[str, Any]] = []
        for number, candidate in enumerate(config.get("candidates", []), start=1):
            if stop_event.is_set():
                break
            started = now_ms()
            action = dict(candidate["action"])
            before_package = device.current_package()
            try:
                if action["action_type"] == "click":
                    x, y = action["action_inputs"]["coordinate"]
                    device.tap(x, y)
                else:
                    x, y = action["action_inputs"]["coordinate"]
                    dx, dy = _direction_delta(action["action_inputs"].get("direction", "down"), config["width"], config["height"])
                    device.swipe(x, y, max(0, min(config["width"] - 1, x + dx)), max(0, min(config["height"] - 1, y + dy)))
                time.sleep(float(config.get("settle_sec", 0.25)))
                after = _capture_ui(device, worker_dir, f"probe_{number}_after")
                recovery = _explorer_recover(device, baseline, worker_dir, resolver, int(config.get("max_back_steps", 2)), action)
                results.append({
                    "probe_number": number,
                    "ok": True,
                    "action": action,
                    "candidate": {key: value for key, value in candidate.items() if key != "node"},
                    "destination": asdict(after.descriptor),
                    "destination_package": after.package,
                    "destination_labels": list(after.descriptor.labels[:12]),
                    "before_package": before_package,
                    "after_package": after.package,
                    "cross_package": bool(before_package and after.package and before_package != after.package),
                    "recovery": recovery,
                    "latency_ms": now_ms() - started,
                })
                if not recovery.get("restored"):
                    break
            except Exception as exc:
                recovery = _explorer_recover(device, baseline, worker_dir, resolver, int(config.get("max_back_steps", 2)), action)
                results.append({
                    "probe_number": number,
                    "ok": False,
                    "action": action,
                    "candidate": {key: value for key, value in candidate.items() if key != "node"},
                    "destination": None,
                    "recovery": recovery,
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": now_ms() - started,
                })
                break
        result_queue.put({"ok": True, "results": results, "stopped": bool(stop_event.is_set())})
    except Exception as exc:
        result_queue.put({"ok": False, "results": [], "error": f"{type(exc).__name__}: {exc}"})


class MemorySampler:
    def __init__(self, path: Path, graph_supplier, interval_sec: float = 0.2):
        self.path = path
        self.graph_supplier = graph_supplier
        self.interval_sec = float(interval_sec)
        self.child_pid: Optional[int] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if psutil is None:
            return
        self.thread = threading.Thread(target=self._run, name="memory-profile", daemon=True)
        self.thread.start()

    def set_child_pid(self, pid: Optional[int]) -> None:
        self.child_pid = int(pid) if pid else None

    @staticmethod
    def _process_memory(pid: Optional[int]) -> Dict[str, Any]:
        if psutil is None or not pid:
            return {"rss_bytes": None, "uss_bytes": None}
        try:
            process = psutil.Process(int(pid))
            full = process.memory_full_info()
            return {"rss_bytes": int(full.rss), "uss_bytes": int(getattr(full, "uss", 0) or 0)}
        except Exception:
            return {"rss_bytes": None, "uss_bytes": None}

    def _run(self) -> None:
        while not self.stop_event.is_set():
            graph = self.graph_supplier() or {}
            sample = {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "main": self._process_memory(os.getpid()),
                "explorer_pid": self.child_pid,
                "explorer": self._process_memory(self.child_pid),
                **graph,
            }
            append_jsonl(self.path, sample)
            self.stop_event.wait(self.interval_sec)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)


class RealPhoneAgent:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.out_dir)
        self.tmp_dir = self.run_dir / "tmp"
        self.screens_dir = self.run_dir / "screens"
        self.prompts_dir = self.run_dir / "prompts"
        self.run_events_path = self.run_dir / "run_events.jsonl"
        self.filtered_path = self.run_dir / "filtered_elements.jsonl"
        self.probe_trace_path = self.run_dir / "probe_trace.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = AdbClient(args.serial, args.adb_path, args.adb_timeout, args.adb_retries, args.allow_emulator)
        self.resolver = AppResolver(self.device)
        self.target_package = self.resolver.target_from_task(args.task, args.target_package)
        self.graph_enabled = args.graph == "on"
        graph_path = self.run_dir / "belief_graph.json"
        self.graph = ProgressiveBeliefGraph.load(str(graph_path)) if self.graph_enabled else ProgressiveBeliefGraph()
        self.guarded = GenerationGuardedGraph(self.graph)
        self.matrix = StateGraphInformationMatrix()
        self.scorer = PredictiveElementScorer()
        self.distiller = GraphDistiller(GraphDistillerConfig(args.max_graph_facts, args.max_graph_context_tokens))
        self.gate = GraphReasoningGate(self.distiller)
        self.model = MAIOneStepAgent("adb", coord_space=args.coord_space)
        self.last_model_output = ""
        self.history: List[str] = []
        self.recent_nodes: List[str] = []
        self.taken_edges: List[str] = []
        self.edge_execution_graph_sizes: Dict[str, List[int]] = {}
        self.recovery_ready_nodes: set[str] = set()
        self.exploration_disabled = False
        self.total_probes_started = 0
        self.skip_disabled = False
        self.consecutive_skips = 0
        self.skip_attempts = 0
        self.skip_successes = 0
        self.profiler = MemorySampler(self.run_dir / "memory_profile.jsonl", self._graph_metrics)

    def _graph_metrics(self) -> Dict[str, Any]:
        if not self.graph_enabled:
            return {"graph_enabled": False, "graph_nodes": 0, "graph_edges": 0, "graph_serialized_bytes": 0, "graph_python_bytes": 0}
        return {
            "graph_enabled": True,
            "graph_nodes": len(self.graph.nodes),
            "graph_edges": len(self.graph.edges),
            "graph_serialized_bytes": self.graph.approximate_serialized_bytes(),
            "graph_python_bytes": self.graph.approximate_python_bytes(),
        }

    def _event(self, event: str, step: int, **payload: Any) -> None:
        append_jsonl(self.run_events_path, {"event": event, "step": step, "timestamp": time.time(), **payload})

    def _save_graph(self) -> None:
        if self.graph_enabled:
            budget = int(max(0.0, float(self.args.graph_memory_budget_mb)) * 1024 * 1024)
            prune = self.graph.prune_to_budget(budget, self.recent_nodes[-4:]) if budget else {"pruned_edges": 0, "pruned_nodes": 0}
            self.graph.save(str(self.run_dir / "belief_graph.json"))
            return
        (self.run_dir / "belief_graph.json").write_text(json.dumps({"disabled": True}, indent=2), encoding="utf-8")

    def _save_first_screen(self, node_id: Optional[str], capture: UICapture, step: int) -> None:
        destination = self.screens_dir / (f"{node_id}.png" if node_id else f"step_{step:02d}.png")
        if not destination.exists():
            shutil.copyfile(capture.screenshot_path, destination)

    def _gate(self, step: int, source_snapshot_id: Optional[str], snapshot: GraphSnapshot, need: InformationNeed) -> Any:
        if not self.graph_enabled:
            from Explorer.graph_distiller import GraphGateDecision
            return GraphGateDecision(GraphMode.NORMAL_INFERENCE, reason="graph_off")
        mode = "distill_and_skip" if self.args.skip == "on" and not self.skip_disabled else "distill"
        decision = self.gate.decide(source_snapshot_id, snapshot, need, self.taken_edges, self.recent_nodes[-4:], mode, self.args.max_graph_context_tokens)
        if decision.mode == GraphMode.SKIP_INFERENCE and not self._skip_admission(source_snapshot_id, snapshot, decision.reusable_edge):
            decision = self.gate.decide(source_snapshot_id, snapshot, need, self.taken_edges, self.recent_nodes[-4:], "distill", self.args.max_graph_context_tokens)
            decision.reason = "skip_rejected_by_episode_admission"
        return decision

    def _skip_admission(self, node_id: Optional[str], snapshot: GraphSnapshot, edge: Any) -> bool:
        if edge is None or self.skip_disabled or self.consecutive_skips >= 3:
            return False
        node = snapshot.node(node_id)
        if node is None or node.visit_count < 2 or edge.execution_hit_count < 2:
            return False
        if edge.skip_attempt_count >= 2:
            return False
        sizes = self.edge_execution_graph_sizes.get(edge.edge_id, [])
        if len(sizes) < 2 or not any(later > earlier for earlier, later in zip(sizes, sizes[1:])):
            return False
        if edge.destination_node_id and edge.destination_node_id in set(self.recent_nodes[-4:]):
            return False
        if edge.rollback_success_rate is not None and edge.rollback_success_rate < 0.8:
            return False
        rate = self.skip_successes / self.skip_attempts if self.skip_attempts else 1.0
        return rate >= 0.5

    def _exploration_candidates(self, step: int, capture: UICapture, source_snapshot_id: Optional[str], snapshot: GraphSnapshot, need: InformationNeed) -> List[Dict[str, Any]]:
        if self.exploration_disabled or self.args.exploration != "on":
            return []
        node_ready = (not self.graph_enabled) or bool(source_snapshot_id and source_snapshot_id in self.recovery_ready_nodes)
        node = snapshot.node(source_snapshot_id)
        visit_count = node.visit_count if node else 0
        safe = _safe_candidates(capture.root, step, source_snapshot_id, node_ready, True, self.filtered_path)
        if not safe or step < 2 or _has_filled_input(capture.root):
            return []
        if self.graph_enabled and (not source_snapshot_id or visit_count < 2):
            return []
        for candidate in safe:
            candidate["matrix_key"] = element_identity(candidate["node"], capture.width, capture.height, "click")
            candidate["control_key"] = _control_key(candidate["node"], capture.width, capture.height)
        rows = self.matrix.build(
            capture.descriptor,
            source_snapshot_id,
            [candidate["node"] for candidate in safe],
            snapshot,
            need,
            None,
            self.recent_nodes[-4:],
            capture.width,
            capture.height,
            self.graph.blocked_recovery_contexts if self.graph_enabled else (),
            MatrixAblations(),
        )
        ranked = self.scorer.score(rows)
        ranked_by_identity = {row.element_identity: row for row in ranked}
        for candidate in safe:
            row = ranked_by_identity.get(candidate["matrix_key"])
            candidate["score"] = float(row.final_exploration_score if row else 0.0)
            candidate["row"] = row.to_log_dict() if row else {}
        safe.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        remaining_budget = max(0, 12 - self.total_probes_started)
        if remaining_budget <= 0:
            return []
        chosen = safe[: min(max(1, int(self.args.explore_max_steps)), remaining_budget)]
        # If all model-derived scores are zero, safe exploration still gets a
        # deterministic top candidate rather than silently becoming random.
        for candidate in chosen:
            candidate["action"] = _candidate_action(candidate, capture.width, capture.height)
            candidate["action"]["control_key"] = candidate["control_key"]
            self._event("candidate", step, score=candidate["score"], selected=True, candidate=candidate["row"])
        return chosen

    def _spawn_explorer(self, step: int, capture: UICapture, snapshot: GraphSnapshot, candidates: List[Dict[str, Any]], source_node_id: str) -> Tuple[Any, Any]:
        context = mp.get_context("spawn")
        stop_event = context.Event()
        result_queue = context.Queue()
        process_dir = self.tmp_dir / f"explorer_{step:02d}_{uuid_hex()}"
        wire_candidates = []
        for candidate in candidates:
            wire_candidates.append({key: value for key, value in candidate.items() if key not in {"node", "row"}})
        config = {
            "serial": self.args.serial,
            "adb_path": self.args.adb_path,
            "adb_timeout": self.args.adb_timeout,
            "adb_retries": self.args.adb_retries,
            "allow_emulator": self.args.allow_emulator,
            "baseline": capture.wire(),
            "worker_dir": str(process_dir),
            "candidates": wire_candidates,
            "width": capture.width,
            "height": capture.height,
            "max_back_steps": self.args.max_back_steps,
            "settle_sec": self.args.explore_settle_sec,
            "source_node_id": source_node_id,
            "snapshot_generation": snapshot.generation,
            # A serialized snapshot is intentionally handed to the child even
            # though selected candidates are already ranked.  It documents the
            # generation boundary and keeps future worker ranking extensions safe.
            "graph_snapshot": snapshot,
        }
        process = context.Process(target=_explorer_worker, args=(config, stop_event, result_queue), daemon=True)
        process.start()
        self.profiler.set_child_pid(process.pid)
        return process, (stop_event, result_queue, process_dir)

    def _finish_explorer(self, step: int, process: Any, controls: Any) -> Dict[str, Any]:
        stop_event, result_queue, _ = controls
        stop_event.set()
        process.join(timeout=max(1.0, float(self.args.explore_stop_wait_sec)))
        needs_repair = False
        if process.is_alive():
            # The worker checks the event between probes and recovers after each
            # probe.  A stuck ADB command is repaired by the parent below.
            process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            needs_repair = True
        self.profiler.set_child_pid(None)
        try:
            output = result_queue.get(timeout=1.0)
            output["needs_repair"] = needs_repair
            return output
        except queue.Empty:
            return {"ok": False, "results": [], "error": "explorer did not return", "needs_repair": needs_repair}

    def _ingest_explorer(self, step: int, source_live_id: Optional[str], source_snapshot_id: Optional[str], output: Dict[str, Any], source_capture: UICapture) -> bool:
        results = output.get("results") or []
        all_restored = bool(output.get("ok", False)) and not bool(output.get("needs_repair"))
        if not output.get("ok", False) or output.get("needs_repair"):
            self.exploration_disabled = True
        for result in results:
            recovery = result.get("recovery") or {}
            restored = bool(recovery.get("restored"))
            all_restored = all_restored and restored
            self.total_probes_started += 1
            if result.get("cross_package") or recovery.get("left_app"):
                self.exploration_disabled = True
            self._event("explore", step, source_node_id=source_snapshot_id, probe=result, restored=restored)
            append_jsonl(self.probe_trace_path, {"step": step, **result})
            if not self.graph_enabled or not source_live_id or not result.get("ok"):
                continue
            destination_wire = result.get("destination")
            destination_id = None
            if destination_wire:
                destination = _descriptor_from_wire(destination_wire)
                destination_id = self.graph.observe_state(destination)
            candidate = result.get("candidate") or {}
            action = copy.deepcopy(result.get("action") or {})
            action["control_key"] = candidate.get("control_key") or action.get("control_key")
            bounds = tuple(candidate.get("bounds") or ()) or None
            source_node = self.graph.nodes.get(source_live_id)
            labels = result.get("destination_labels") or []
            source_labels = set(source_node.discovered_labels if source_node else [])
            realized_ig = min(1.0, len(set(labels) - source_labels) / 6.0) if labels else 0.0
            edge_id = self.graph.record_probe(
                source_live_id,
                str(action.get("control_key") or candidate.get("control_key") or "unmapped"),
                str(action.get("action_type") or "click"),
                str(candidate.get("role") or "node"),
                "SCROLL" if action.get("action_type") == "scroll" else "TAP_NAV",
                source_capture.descriptor.coarse_context,
                parse_reasoning_prior(self.last_model_output, self.args.task).need_type,
                action,
                destination_id,
                labels,
                float(result.get("latency_ms", 0.0)) / 1000.0,
                realized_ig,
                bounds,
                0.0,
            )
            self.graph.record_rollback_result(edge_id, restored, deep_recovery=str(recovery.get("level")) not in {"NOOP", "INVERSE", ""})
            edge = self.graph.edges.get(edge_id)
            if edge and restored and str(recovery.get("level")) in {"NOOP", "INVERSE"}:
                self.recovery_ready_nodes.add(source_live_id)
            self._event("graph", step, operation="probe_ingest", edge_id=edge_id, nodes=len(self.graph.nodes), edges=len(self.graph.edges), serialized_bytes=self.graph.approximate_serialized_bytes())
        if not all_restored and results:
            self.exploration_disabled = True
        return all_restored

    def _repair_after_explorer(self, step: int, baseline: UICapture) -> bool:
        current_package = self.device.current_package()
        if self.target_package and current_package != self.target_package:
            try:
                self.device.launch_package(self.target_package)
            except Exception:
                pass
        repaired = False
        try:
            check = _capture_ui(self.device, self.tmp_dir, f"repair_{step}")
            repaired = _same_capture(check, baseline.wire())
        except Exception:
            repaired = False
        self._event("repair", step, package_before=current_package, target_package=self.target_package, repaired=repaired, method="relaunch" if current_package != self.target_package else "verify")
        if not repaired:
            self.exploration_disabled = True
        return repaired

    def _model_chat(self, capture: UICapture, graph_context: str) -> List[Dict[str, Any]]:
        system = (
            "You are a mobile GUI agent controlling a real Android phone. Output exactly one "
            "<tool_call> JSON object per turn. Coordinates are normalized to [0,1000]. "
            "Allowed actions: click, long_press, type, scroll, drag, open, system_button, wait, "
            "terminate, answer. Use terminate success only when the task is complete."
        )
        user = f"Task:\n{self.args.task}\n\nAction history:\n{chr(10).join(self.history[-12:]) or 'None'}"
        if graph_context:
            user += f"\n\n{graph_context}"
        user += "\n\nChoose the next single action."
        image_b64 = base64.b64encode(Path(capture.screenshot_path).read_bytes()).decode("ascii")
        return [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]},
        ]

    def _call_vlm(self, messages: List[Dict[str, Any]]) -> str:
        import urllib.error
        import urllib.request

        body = json.dumps({"model": self.args.model, "messages": messages, "max_tokens": self.args.max_tokens, "temperature": 0.0, "stream": False}).encode("utf-8")
        request = urllib.request.Request(self.args.api_url, data=body, method="POST", headers={"Content-Type": "application/json"})
        if self.args.api_key:
            request.add_header("Authorization", f"Bearer {self.args.api_key}")
        started = now_ms()
        try:
            with urllib.request.urlopen(request, timeout=self.args.api_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM HTTP {exc.code}: {details[:1000]}") from exc
        content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        self._last_inference_latency_ms = now_ms() - started
        return str(content)

    def _execute_and_record(self, step: int, before: UICapture, source_live_id: Optional[str], action: Dict[str, Any], resolver: AppResolver) -> Tuple[UICapture, Optional[str], bool, float]:
        started = now_ms()
        executed = _execute_device_action(self.device, action, before.width, before.height, resolver)
        time.sleep(float(self.args.post_action_wait_sec))
        after = _capture_ui(self.device, self.tmp_dir, f"step_{step:02d}_after")
        destination_id = None
        edge_id = None
        matched = False
        if self.graph_enabled:
            destination_id = self.graph.observe_state(after.descriptor)
            if source_live_id:
                edge_id = self.graph.find_edge_for_action(source_live_id, action, before.width, before.height)
                if edge_id is None and action.get("action_type") not in {"wait", "finished", "infeasible"}:
                    node, control_key, bounds = _control_at(before.root, action, before.width, before.height)
                    edge_action = copy.deepcopy(action)
                    if control_key:
                        edge_action["control_key"] = control_key
                    label = node_to_text(node) if node else ""
                    if label:
                        edge_action.setdefault("action_inputs", {})["label"] = label[:80]
                    edge_id = self.graph.add_speculative_transition(
                        source_live_id,
                        edge_action,
                        destination_id,
                        role=(str(getattr(node, "class_name", "") or "").rsplit(".", 1)[-1].lower() if node else "action"),
                        probe_type="execution",
                        coarse_context=before.descriptor.coarse_context,
                        information_need_type=parse_reasoning_prior(self.last_model_output, self.args.task).need_type,
                        discovered_labels=after.descriptor.labels,
                        bounds=bounds,
                    )
                if edge_id:
                    edge = self.graph.edges.get(edge_id)
                    expected = set(edge.observed_destinations if edge else [])
                    if edge and edge.destination_node_id:
                        expected.add(edge.destination_node_id)
                    matched = not expected or destination_id in expected
                    self.graph.record_execution_verification(edge_id, matched, destination_id if matched else None)
                    if action.get("action_type") not in {"wait", "finished", "infeasible"}:
                        self.edge_execution_graph_sizes.setdefault(edge_id, []).append(len(self.graph.nodes))
                    self._event("graph", step, operation="execution", edge_id=edge_id, matched=matched, nodes=len(self.graph.nodes), edges=len(self.graph.edges), serialized_bytes=self.graph.approximate_serialized_bytes())
        self._event("step", step, node_id=source_live_id, action=action, executed=executed, edge_id=edge_id, successor_node_id=destination_id, successor_matched=matched, execution_latency_ms=now_ms() - started)
        return after, edge_id, matched, now_ms() - started

    def run(self) -> int:
        self.device.assert_ready()
        if self.args.profile_memory == "on":
            self.profiler.start()
        else:
            (self.run_dir / "memory_profile.jsonl").write_text(
                json.dumps({"disabled": True, "reason": "--profile_memory off"}) + "\n",
                encoding="utf-8",
            )
        config = vars(self.args).copy()
        config["git_commit"] = _git_commit()
        config["target_package"] = self.target_package
        (self.run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        current = _capture_ui(self.device, self.tmp_dir, "initial")
        exit_code = 1
        try:
            for step in range(max(0, int(self.args.max_steps))):
                step_started = now_ms()
                before = _capture_ui(self.device, self.tmp_dir, f"step_{step:02d}_before")
                frozen = self.guarded.snapshot_for_step(step) if self.graph_enabled else GraphSnapshot(0, {}, {})
                source_snapshot_id = frozen.match_state(before.descriptor) if self.graph_enabled else None
                source_live_id = self.graph.observe_state(before.descriptor) if self.graph_enabled else None
                self._save_first_screen(source_live_id, before, step)
                if source_snapshot_id:
                    self.recent_nodes.append(source_snapshot_id)
                    self.recent_nodes = self.recent_nodes[-16:]

                # Fold explicit app startup into step zero.
                if step == 0 and self.target_package and before.package != self.target_package:
                    try:
                        if "launcher" in before.package.lower() or before.package != self.target_package:
                            _execute_device_action(self.device, {"action_type": "open_app", "action_inputs": {"content": self.target_package}}, before.width, before.height, self.resolver)
                            before = _capture_ui(self.device, self.tmp_dir, "bootstrap_after")
                            if self.graph_enabled:
                                source_live_id = self.graph.observe_state(before.descriptor)
                                # The frozen step snapshot still describes the
                                # pre-bootstrap screen; do not use its node id
                                # for the newly opened app.
                                source_snapshot_id = frozen.match_state(before.descriptor)
                            self._event("step", step, operation="bootstrap_open_app", target_package=self.target_package, folded=True)
                    except Exception as exc:
                        self._event("repair", step, operation="bootstrap_open_app", repaired=False, error=str(exc))

                need = parse_reasoning_prior(self.last_model_output, self.args.task)
                decision = self._gate(step, source_snapshot_id, frozen, need)
                context = decision.distillation.context if decision.distillation else ""
                node_snapshot = frozen.node(source_snapshot_id)
                node_stats = ProgressiveBeliefGraph._node_to_dict(node_snapshot) if node_snapshot else None
                self._event("gate", step, mode=decision.mode.value, reason=decision.reason, source_node_id=source_snapshot_id, snapshot_generation=frozen.generation, reusable_edge_id=getattr(decision.reusable_edge, "edge_id", None), node_stats=node_stats)
                self._event("graph_context", step, injected=bool(context), token_count=decision.distillation.estimated_graph_context_tokens if decision.distillation else 0, text=context, selected_fact_types=decision.distillation.selected_fact_types if decision.distillation else [], selected_edge_ids=decision.distillation.selected_edge_ids if decision.distillation else [])

                if decision.mode == GraphMode.SKIP_INFERENCE and decision.reusable_edge is not None:
                    action = copy.deepcopy(decision.reusable_edge.action)
                    after, edge_id, matched, execution_latency = self._execute_and_record(step, before, source_live_id, action, self.resolver)
                    self.skip_attempts += 1
                    self.consecutive_skips += 1
                    success = bool(matched)
                    self.skip_successes += int(success)
                    self.graph.record_skip_result(decision.reusable_edge.edge_id, success)
                    self._event("skip", step, edge_id=decision.reusable_edge.edge_id, hit=success, replay_number=decision.reusable_edge.skip_attempt_count, cumulative_hit_rate=self.skip_successes / max(1, self.skip_attempts))
                    if not success:
                        self.skip_disabled = True
                        self.consecutive_skips = 0
                    self.history.append(f"skip:{action.get('action_type')} {'hit' if success else 'miss'}")
                    current = after
                    self.guarded.commit_step(step)
                    self._save_graph()
                    continue

                self.consecutive_skips = 0
                candidates = self._exploration_candidates(step, before, source_snapshot_id, frozen, need)
                explorer = None
                explorer_controls = None
                if candidates:
                    explorer, explorer_controls = self._spawn_explorer(step, before, frozen, candidates, source_live_id)
                graph_context = context if self.graph_enabled and self.args.graph == "on" else ""
                messages = self._model_chat(before, graph_context)
                (self.prompts_dir / f"step{step:02d}.txt").write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
                inference_started = now_ms()
                try:
                    output = self._call_vlm(messages)
                    self.last_model_output = output
                    action = _parse_model_action(output, self.args.coord_space)
                except Exception as exc:
                    output = f"model_error: {type(exc).__name__}: {exc}"
                    self.last_model_output = output
                    action = {"action_type": "wait", "action_inputs": {"seconds": 1}, "coord_space": self.args.coord_space, "raw": {"error": str(exc)}}
                    self._event("repair", step, operation="model_error", error=str(exc), repaired=False)
                inference_latency = now_ms() - inference_started

                explorer_output = {"ok": True, "results": []}
                if explorer is not None:
                    explorer_output = self._finish_explorer(step, explorer, explorer_controls)
                    restored = self._ingest_explorer(step, source_live_id, source_snapshot_id, explorer_output, before)
                    if not restored or explorer_output.get("needs_repair"):
                        self._repair_after_explorer(step, before)
                if action.get("action_type") in {"finished", "infeasible", "status"}:
                    self._event("step", step, operation="terminal", action=action, model_output=output, inference_latency_ms=inference_latency, end_to_end_latency_ms=now_ms() - step_started)
                    exit_code = 0 if action.get("action_type") == "finished" else 2
                    self.guarded.commit_step(step)
                    self._save_graph()
                    break

                try:
                    after, edge_id, matched, execution_latency = self._execute_and_record(step, before, source_live_id, action, self.resolver)
                except Exception as exc:
                    after = _capture_ui(self.device, self.tmp_dir, f"step_{step:02d}_error")
                    self._event("repair", step, operation="action_execution", error=str(exc), repaired=False)
                    self.history.append(f"execution_error:{exc}")
                    current = after
                    self.guarded.commit_step(step)
                    self._save_graph()
                    continue
                self.history.append(f"{action.get('action_type')}:{(action.get('action_inputs') or {}).get('label', '')}")
                self._event("step", step, model_output=output, inference_latency_ms=inference_latency, exploration_results=len(explorer_output.get("results") or []), end_to_end_latency_ms=now_ms() - step_started)
                try:
                    append_jsonl(self.run_events_path, {"event": "device_memory", "step": step, **self.device.device_memory(self.target_package or before.package)})
                except Exception as exc:
                    self._event("device_memory", step, error=str(exc))
                current = after
                self.guarded.commit_step(step)
                self._save_graph()
            else:
                exit_code = 3
        finally:
            self.profiler.stop()
            self._save_graph()
        return exit_code


def uuid_hex() -> str:
    return hashlib.sha1(f"{os.getpid()}:{time.time_ns()}".encode()).hexdigest()[:10]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone MobileExplorer agent for a real Android phone")
    parser.add_argument("--task", required=True)
    parser.add_argument("--serial", required=True, help="ADB device serial")
    parser.add_argument("--api_url", required=True, help="OpenAI-compatible /v1/chat/completions endpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--exploration", choices=["on", "off"], default="on")
    parser.add_argument("--graph", choices=["on", "off"], default="on")
    parser.add_argument("--skip", choices=["on", "off"], default="off")
    parser.add_argument("--profile_memory", choices=["on", "off"], default="on")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--target_package", default="")
    parser.add_argument("--adb_path", default=os.environ.get("ADB", "adb"))
    parser.add_argument("--adb_timeout", type=float, default=DEFAULT_ADB_TIMEOUT)
    parser.add_argument("--adb_retries", type=int, default=1)
    parser.add_argument("--allow_emulator", action="store_true", help="Allow emulator serials for non-final smoke tests")
    parser.add_argument("--api_timeout", type=float, default=180.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--coord_space", choices=["auto", "pixel", "norm1", "norm1000"], default="norm1000")
    parser.add_argument("--graph_memory_budget_mb", type=float, default=0.0)
    parser.add_argument("--max_graph_facts", type=int, default=3)
    parser.add_argument("--max_graph_context_tokens", type=int, default=64)
    parser.add_argument("--explore_max_steps", type=int, default=6)
    parser.add_argument("--explore_max_depth", type=int, default=1)
    parser.add_argument("--max_back_steps", type=int, default=2)
    parser.add_argument("--explore_settle_sec", type=float, default=0.25)
    parser.add_argument("--explore_stop_wait_sec", type=float, default=20.0)
    parser.add_argument("--post_action_wait_sec", type=float, default=0.35)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.profile_memory == "off":
        # The sampler is intentionally disabled without changing run logic.
        args.profile_memory = "off"
    agent = RealPhoneAgent(args)
    if args.profile_memory == "off":
        agent.profiler = MemorySampler(agent.run_dir / "memory_profile.jsonl", agent._graph_metrics)
    return agent.run()


if __name__ == "__main__":
    raise SystemExit(main())
