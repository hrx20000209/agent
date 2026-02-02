import threading
import random
import time
import os

from MobileAgentE.controller import tap, swipe, back, home, get_a11y_tree, get_screenshot
from MobileAgentE.tree import parse_a11y_tree, print_tree
from MobileAgentE.utils import parse_bounds
from Explorer.utils import _mark_and_save_explore_click


class A11yTreeOnlineExplorer:
    """
    Minimal online exploration (thread):
    - repeatedly fetch accessibility tree XML
    - sample a random clickable node (with bounds)
    - interact with it (tap)
    """

    def __init__(self, adb_path: str, args, xml_path: str,
                 explore_vis_dir: str = "explore_results",
                 screenshot_path: str = "explore_screenshot.png"):
        self.adb = adb_path
        self.args = args
        self.xml_path = xml_path

        self._stop = threading.Event()
        self._thread = None

        self.explore_vis_dir = explore_vis_dir
        self.explore_screenshot_path = screenshot_path
        self._vis_step = 0


    def start(self, max_steps: int = 2, sleep_sec: float = 0.25):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker,
            args=(max_steps, sleep_sec),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _collect_clickable_nodes(self, root):
        """
        Return list of nodes that:
        - have bounds
        - are likely clickable
        Your Node schema may vary, so I write it robustly.
        """
        candidates = []

        def dfs(n):
            if n is None:
                return

            # --- robust attribute access ---
            bounds = getattr(n, "bounds", None)
            text = getattr(n, "text", None) or ""
            clickable = getattr(n, "clickable", None)
            enabled = getattr(n, "enabled", None)

            # bounds must exist
            if bounds and isinstance(bounds, str) and "[" in bounds and "]" in bounds:
                # heuristic clickable check
                # 1) node.clickable == True
                # 2) or contains some text (often indicates actionable)
                ok = False
                if clickable is True:
                    ok = True
                elif isinstance(text, str) and len(text.strip()) > 0:
                    ok = True

                # enabled check (optional)
                if enabled is False:
                    ok = False

                if ok:
                    candidates.append(n)

            # children traversal
            children = getattr(n, "children", None) or []
            for c in children:
                dfs(c)

        dfs(root)
        return candidates

    def _click_node_center(self, node):
        b = parse_bounds(node.bounds)
        if b is None:
            return False, None, None

        x1, y1, x2, y2 = b
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)

        tap(self.adb, x, y)
        return True, (x, y), b

    def _worker(self, max_steps: int, sleep_sec: float):
        for k in range(max_steps):
            if self._stop.is_set():
                return

            # try:
            # 1) pull latest a11y tree
            get_a11y_tree(self.args, self.xml_path)

            # 2) parse
            root = parse_a11y_tree(xml_path=self.xml_path)

            # 3) sample a node
            candidates = self._collect_clickable_nodes(root)
            if len(candidates) == 0:
                # fallback action to avoid freeze
                if random.random() < 0.5:
                    back(self.adb)
                else:
                    home(self.adb)
                time.sleep(sleep_sec)
                continue

            node = random.choice(candidates)

            ok, xy, b = self._click_node_center(node)

            if ok:
                self._vis_step += 1
                text = getattr(node, "text", "") or ""

                # ① 拉 screenshot（点击后截图更直观）
                get_screenshot(self.args, self.explore_screenshot_path)

                # ② 保存可视化结果
                out = _mark_and_save_explore_click(
                    screenshot_path=self.explore_screenshot_path,
                    save_dir=self.explore_vis_dir,
                    step_idx=self._vis_step,
                    xy=xy,
                    bounds=b,
                    text=text,
                )

                print(
                    f"[Explore] step={k + 1}/{max_steps}, click xy={xy}, bounds={node.bounds}, text={text} -> saved {out}"
                )
            else:
                print(f"[Explore] step={k + 1}/{max_steps}, failed bounds parse")


            # except Exception as e:
            #     print("[Explore] error:", e)

            # time.sleep(sleep_sec)
