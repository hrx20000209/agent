import threading
import random
import time
import os
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from MobileAgentE.controller import tap, swipe, back, home, get_a11y_tree, get_screenshot
from MobileAgentE.tree import parse_a11y_tree, print_tree
from MobileAgentE.utils import parse_bounds
from Explorer.utils import _mark_and_save_explore_click, extract_task_queries


class A11yTreeOnlineExplorer:
    """
    Online exploration (thread), goal-conditioned semantic matching version:

    - repeatedly fetch accessibility tree XML
    - collect clickable candidates
    - pick candidate with max semantic similarity to task_text
    - interact with it (tap)
    - record every step (console + jsonl log + in-memory history)
    """

    def __init__(
        self,
        adb_path: str,
        args,
        xml_path: str,
        task_text: str,
        explore_vis_dir: str = "explore_results",
        screenshot_path: str = "explore_screenshot.png",
        embed_model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ):
        self.adb = adb_path
        self.args = args
        self.xml_path = xml_path
        self.task_text = task_text

        self._stop = threading.Event()
        self._thread = None

        self.explore_vis_dir = explore_vis_dir
        self.explore_screenshot_path = screenshot_path
        self._vis_step = 0

        # --- logging ---
        self._history = []
        os.makedirs(self.explore_vis_dir, exist_ok=True)
        self.log_path = os.path.join(self.explore_vis_dir, "explore_log.jsonl")

        # --- embedding model (for semantic matching) ---
        self.embed_model = SentenceTransformer(embed_model_name)
        self._emb_cache = {}

        # precompute goal embedding
        self.task_queries = extract_task_queries(self.task_text)
        self.goal_emb = [self._embed_text(q) for q in self.task_queries if q.strip()]
        print(f"[Explorer:init] task_queries={self.task_queries}")

        # avoid repeating same click area
        self._clicked_bounds = set()

        print(f"[Explorer:init] task_text='{self.task_text}'")
        print(f"[Explorer:init] explore_vis_dir='{self.explore_vis_dir}'")
        print(f"[Explorer:init] log_path='{self.log_path}'")
        print(f"[Explorer:init] xml_path='{self.xml_path}'")

        # ---- latency profiling ----
        self.lat_total = []
        self.lat_tree = []
        self.lat_semantic = []
        self.lat_selection = []
        self.lat_action = []

    # -----------------------------
    # Thread control
    # -----------------------------
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

    # -----------------------------
    # Embedding helpers
    # -----------------------------
    def _embed_text(self, text: str):
        text = (text or "").strip()
        if len(text) == 0:
            return None
        if text in self._emb_cache:
            return self._emb_cache[text]

        emb = self.embed_model.encode([text])[0].astype(np.float32)
        self._emb_cache[text] = emb
        return emb

    def _cosine_sim(self, a, b):
        if a is None or b is None:
            return 0.0

        b = np.asarray(b, dtype=np.float32)

        # --- 情况 1：a 是单个向量 ---
        if isinstance(a, np.ndarray):
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)

        # --- 情况 2：a 是“多个 goal 向量”的 list ---
        best = 0.0
        for g in a:
            if g is None:
                continue
            g = np.asarray(g, dtype=np.float32)
            denom = np.linalg.norm(g) * np.linalg.norm(b)
            if denom == 0:
                continue
            s = float(np.dot(g, b) / denom)
            if s > best:
                best = s

        return best

    def _node_to_text(self, node):
        """
        Convert node into a string for semantic matching.

        NOTE:
        - icon-only nodes may have empty text but non-empty content-desc or resource-id
        """
        parts = []

        t = getattr(node, "text", "") or ""
        d = getattr(node, "content_desc", "") or ""
        rid = getattr(node, "resource_id", "") or ""
        cls = getattr(node, "class_name", "") or ""

        if t.strip():
            parts.append(t.strip())
        if d.strip():
            parts.append(d.strip())

        if rid.strip():
            parts.append(rid.strip().split("/")[-1])

        if cls.strip():
            parts.append(cls.strip().split(".")[-1])

        return " | ".join(parts).strip()

    # -----------------------------
    # Candidate collection
    # -----------------------------
    def _collect_clickable_nodes(self, root):
        """
        Return list of nodes that:
        - have bounds
        - are likely clickable (node.clickable == True) OR has text
        """
        candidates = []

        def dfs(n):
            if n is None:
                return

            bounds = getattr(n, "bounds", None)
            text = getattr(n, "text", None) or ""
            clickable = getattr(n, "clickable", None)
            enabled = getattr(n, "enabled", None)

            if bounds and isinstance(bounds, str) and "[" in bounds and "]" in bounds:
                ok = False
                if clickable is True:
                    ok = True
                elif isinstance(text, str) and len(text.strip()) > 0:
                    ok = True

                if enabled is False:
                    ok = False

                if ok:
                    candidates.append(n)

            for c in getattr(n, "children", None) or []:
                dfs(c)

        dfs(root)
        return candidates

    # -----------------------------
    # Selection: max semantic similarity
    # -----------------------------
    def _select_node_by_semantic(self, candidates):
        """
        Select candidate with maximum semantic similarity to task_text.
        Avoid repeated bounds.
        """
        best = None
        best_score = -1.0
        best_txt = ""

        for n in candidates:
            bounds = getattr(n, "bounds", None) or ""
            if bounds in self._clicked_bounds:
                continue

            cand_txt = self._node_to_text(n)
            if not cand_txt:
                continue

            cand_emb = self._embed_text(cand_txt)
            score = self._cosine_sim(self.goal_emb, cand_emb)

            # tiny heuristic bonus
            if getattr(n, "clickable", False):
                score += 0.05

            if score > best_score:
                best_score = score
                best = n
                best_txt = cand_txt

        return best, best_score, best_txt

    # -----------------------------
    # Action
    # -----------------------------
    def _click_node_center(self, node):
        b = parse_bounds(node.bounds)
        if b is None:
            return False, None, None

        x1, y1, x2, y2 = b
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)

        tap(self.adb, x, y)
        return True, (x, y), b

    # -----------------------------
    # Logging
    # -----------------------------
    def _log_step(self, record: dict):
        """
        record to:
        - in-memory history
        - jsonl log file
        """
        self._history.append(record)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print("[Explorer:log] failed to write jsonl:", e)

    # -----------------------------
    # Worker
    # -----------------------------
    def _worker(self, max_steps: int, sleep_sec: float):
        step_t0 = time.time()
        for k in range(max_steps):
            if self._stop.is_set():
                return

            step_t0 = time.time()

            # try:
            # 1) pull latest a11y tree
            t0 = time.time()
            get_a11y_tree(self.args, self.xml_path)
            root = parse_a11y_tree(xml_path=self.xml_path)
            candidates = self._collect_clickable_nodes(root)
            lat_tree = time.time() - t0

            if len(candidates) == 0:
                # fallback action to avoid freeze
                if random.random() < 0.5:
                    back(self.adb)
                    fallback = "back"
                else:
                    home(self.adb)
                    fallback = "home"

                rec = {
                    "step": k + 1,
                    "type": "fallback",
                    "fallback": fallback,
                    "n_candidates": 0,
                    "time_sec": round(time.time() - step_t0, 4),
                }
                self._log_step(rec)
                print(f"[Explore] step={k+1}/{max_steps} candidates=0 -> fallback={fallback}")
                time.sleep(sleep_sec)
                continue

            # 4) select semantic-best node
            t0 = time.time()
            node, sim, node_txt = self._select_node_by_semantic(candidates)
            lat_semantic = time.time() - t0
            lat_selection = 0.0

            if node is None:
                # all candidates repeated / empty
                if random.random() < 0.5:
                    back(self.adb)
                    fallback = "back"
                else:
                    home(self.adb)
                    fallback = "home"

                rec = {
                    "step": k + 1,
                    "type": "fallback",
                    "fallback": fallback,
                    "n_candidates": len(candidates),
                    "reason": "no_valid_semantic_candidate",
                    "time_sec": round(time.time() - step_t0, 4),
                }
                self._log_step(rec)
                print(
                    f"[Explore] step={k+1}/{max_steps} candidates={len(candidates)} "
                    f"-> fallback={fallback} (no semantic candidate)"
                )
                time.sleep(sleep_sec)
                continue

            t0 = time.time()
            ok, xy, b = self._click_node_center(node)
            lat_action = time.time() - t0

            if not ok:
                lat_total = time.time() - step_t0
                rec = {
                    "step": k + 1,
                    "type": "fail",
                    "reason": "bounds_parse_failed",
                    "node_bounds": getattr(node, "bounds", None),
                    "node_text": getattr(node, "text", None),
                    "node_desc": getattr(node, "content_desc", None),
                    "node_resource_id": getattr(node, "resource_id", None),
                    "node_class": getattr(node, "class_name", None),
                    "best_sim": float(sim),
                    "n_candidates": len(candidates),
                    "time_sec": round(time.time() - step_t0, 4),
                }
                self._log_step(rec)
                print(f"[Explore] step={k+1}/{max_steps} bounds parse failed: {getattr(node,'bounds',None)}")
                time.sleep(sleep_sec)
                continue

            # mark as clicked
            self._clicked_bounds.add(getattr(node, "bounds", ""))

            # 5) visualization
            self._vis_step += 1
            ui_text = getattr(node, "text", "") or ""
            ui_desc = getattr(node, "content_desc", "") or ""

            # click then screenshot
            get_screenshot(self.args, self.explore_screenshot_path)

            out = _mark_and_save_explore_click(
                screenshot_path=self.explore_screenshot_path,
                save_dir=self.explore_vis_dir,
                step_idx=self._vis_step,
                xy=xy,
                bounds=b,
                text=f"{ui_text} {ui_desc} | sim={sim:.3f}",
            )

            # record
            lat_total = time.time() - step_t0
            rec = {
                "step": k + 1,
                "type": "click",
                "best_sim": float(sim),
                "xy": list(xy),
                "bounds": getattr(node, "bounds", None),
                "node_text": ui_text,
                "node_desc": ui_desc,
                "node_resource_id": getattr(node, "resource_id", None),
                "node_class": getattr(node, "class_name", None),
                "node_match_text": node_txt,
                "n_candidates": len(candidates),
                "saved_vis": out,
                "time_sec": round(time.time() - step_t0, 4),
                "lat_total": round(lat_total, 4),
                "lat_tree": round(lat_tree, 4),
                "lat_semantic": round(lat_semantic, 4),
                "lat_selection": round(lat_selection, 6),
                "lat_action": round(lat_action, 4),
            }
            self._log_step(rec)
            self.lat_total.append(lat_total)
            self.lat_tree.append(lat_tree)
            self.lat_semantic.append(lat_semantic)
            self.lat_selection.append(lat_selection)
            self.lat_action.append(lat_action)

            # rich print
            print("=" * 90)
            print(f"[Explore] step={k + 1}/{max_steps}  ✅ semantic click")
            print(f"  task_text     : {self.task_text}")
            print(f"  best_sim      : {sim:.4f}")
            print(f"  candidates    : {len(candidates)}")
            print(f"  click_xy      : {xy}")
            print(f"  bounds        : {getattr(node, 'bounds', None)}")
            print(f"  node_text     : {ui_text}")
            print(f"  node_desc     : {ui_desc}")
            print(f"  node_class    : {getattr(node, 'class_name', None)}")
            print(f"  resource_id   : {getattr(node, 'resource_id', None)}")
            print(f"  match_text    : {node_txt}")
            print(f"  saved_vis     : {out}")
            print(f"  log_path      : {self.log_path}")
            print("=" * 90)

        if self.lat_total:
            print("\n" + "=" * 90)
            print("📊 Exploration latency summary")
            print(f"Avg TOTAL      : {sum(self.lat_total) / len(self.lat_total):.4f}s")
            print(f"Avg Tree parse : {sum(self.lat_tree) / len(self.lat_tree):.4f}s")
            print(f"Avg Semantic   : {sum(self.lat_semantic) / len(self.lat_semantic):.4f}s")
            print(f"Avg Selection  : {sum(self.lat_selection) / len(self.lat_selection):.6f}s")
            print(f"Avg Action     : {sum(self.lat_action) / len(self.lat_action):.4f}s")
            print("=" * 90 + "\n")
            # except Exception as e:
            #     rec = {
            #         "step": k + 1,
            #         "type": "error",
            #         "error": str(e),
            #         "time_sec": round(time.time() - step_t0, 4),
            #     }
            #     self._log_step(rec)
            #     print("[Explore] error:", e)

            # time.sleep(sleep_sec)

    def fast_rollback(self, sleep_sec=0.2):
        """
        Fast rollback: back N times, N = number of exploration clicks
        """

        if self._vis_step == 0:
            return

        print(f"⚡ Fast rollback: backing {self._vis_step} steps...")

        for _ in range(self._vis_step):
            back(self.adb)
            time.sleep(sleep_sec)

        print("⚡ Fast rollback done.")


    # -----------------------------
    # Compress exploration → prompt clues
    # -----------------------------
    def build_prompt_clues(self, max_items: int = 4):
        """
        Convert exploration history into short environment clues for VLM prompt.
        """

        if not self._history:
            return ""

        # 只取 click 记录
        clicks = [r for r in self._history if r.get("type") == "click"]

        # 过滤低相关度点击
        clicks = [r for r in clicks if r.get("best_sim", 0.0) > 0.35]

        if not clicks:
            return ""

        # 去重（text + desc + resource id）
        uniq = {}
        for r in clicks:
            key = (
                r.get("node_text", ""),
                r.get("node_desc", ""),
                r.get("node_resource_id", ""),
            )
            if key not in uniq or r["best_sim"] > uniq[key]["best_sim"]:
                uniq[key] = r

        # 按相似度排序
        items = sorted(uniq.values(), key=lambda x: x["best_sim"], reverse=True)

        role_map = {
            "settings": "likely related to configuration options",
            "search": "used to locate content or features",
            "profile": "user account and personal information",
            "menu": "contains additional app functions",
            "more": "contains additional app functions",
        }

        lines = []
        lines.append("[Environment Clues from Parallel Exploration]")
        lines.append("The system briefly explored the interface and found possible functional areas:")

        for r in items[:max_items]:
            name = r.get("node_text") or r.get("node_desc") or ""
            name = name.strip()
            if not name:
                continue

            low = name.lower()
            role = "potentially useful"
            for k in role_map:
                if k in low:
                    role = role_map[k]
                    break

            lines.append(f"• {name} — {role}")

        lines.append("Use these clues if they help with the task.\n")

        return "\n".join(lines)
