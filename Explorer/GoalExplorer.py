import threading
import time
import os
import shutil
import re

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from MobileAgentE.controller import get_a11y_tree, get_screenshot, back, home
from agents.utils import execute_action
from MobileAgentE.tree import parse_a11y_tree
from Explorer.utils import (
    ensure_dir,
    append_jsonl,
    embed_text,
    cosine_sim,
    node_to_text,
    collect_clickable_nodes,
    click_node_center,
    build_prompt_clues,
    print_latency_summary,
    extract_task_queries,
    mark_and_save_explore_click,
    check_same_image,
    phash,
)


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
        args=None,
        xml_path=None,
        explore_vis_dir="explore_results",
        embed_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        ui_lock=None,
        stop_event=None,
        rollback_done_event=None,
        width=None,
        height=None,
        explorer_mode="collect_demo",
        adb_path=None,
        task_text=None,
    ):
        if args is None:
            class _Args:
                pass

            args = _Args()
            setattr(args, "adb_path", adb_path or "adb")
            setattr(args, "task", task_text or "")
            setattr(args, "scale", 1.0)
            setattr(args, "on_device", False)

        if xml_path is None:
            raise ValueError("A11yTreeOnlineExplorer requires xml_path.")

        self.args = args
        self.adb = str(adb_path or getattr(args, "adb_path", "adb"))
        self.xml_path = xml_path
        self.task_text = str(task_text if task_text is not None else getattr(args, "task", ""))

        self.stop_event = stop_event or threading.Event()
        self.ui_lock = ui_lock or threading.Lock()
        self.rollback_done_event = rollback_done_event
        self.rollback_lock = threading.Lock()
        self.thread = None

        self.explore_vis_dir = explore_vis_dir
        self.explore_debug_vis_dir = os.path.join(self.explore_vis_dir, "debug")
        self.explore_raw_dir = os.path.join(self.explore_vis_dir, "raw")
        self.explore_screenshot_path = "screenshot/explore_screenshot.png"
        self.rollback_screenshot_path = "rollback/explore_screenshot.png"
        self.rollback_root_path = "rollback/root.png"
        self.leaf_before_screenshot_path = "rollback/leaf_before.png"
        self.leaf_after_screenshot_path = "rollback/leaf_after.png"
        self.vis_step = 0
        self.debug_step = 0
        self.cur_steps = 0
        self.height, self.width = height, width
        self.coord_scale = float(getattr(args, "scale", 1.0) or 1.0)
        self.no_effect_delta_threshold = 1.2
        self._explore_trigger_reason = "periodic_probe"
        self.runtime_history_tail = []
        self.explorer_mode = str(explorer_mode or "collect_demo").strip().lower()
        self.collection_mode = self.explorer_mode in {"collect_demo", "collection", "demo", "real_collect"}

        # --- logging ---
        self.history = []
        ensure_dir(self.explore_vis_dir)
        ensure_dir(self.explore_debug_vis_dir)
        ensure_dir(self.explore_raw_dir)
        self.log_path = os.path.join(self.explore_vis_dir, "explore_log.jsonl")

        # --- embedding model ---
        self.embed_model = SentenceTransformer(embed_model_name)
        self.emb_cache = {}

        # precompute goal embedding
        self.task_queries = extract_task_queries(self.task_text)
        self.goal_emb = []
        for q in self.task_queries:
            q = (q or "").strip()
            if q:
                self.goal_emb.append(embed_text(q, self.embed_model, self.emb_cache))

        print(f"[Explorer:init] task_queries={self.task_queries}")
        print(f"[Explorer:init] task_text='{self.task_text}'")
        print(f"[Explorer:init] explore_vis_dir='{self.explore_vis_dir}'")
        print(f"[Explorer:init] log_path='{self.log_path}'")
        print(f"[Explorer:init] xml_path='{self.xml_path}'")

        # avoid repeating same click area
        self.clicked_bounds = set()
        # Cross-iteration visit memory for explore/exploit balance.
        self.bound_visit_count = {}
        self.bound_effect_ema = {}
        self.bound_effect_count = {}
        self.bound_seen_count = {}
        self.bound_skip_count = {}
        self.recent_clicked_bounds = []
        self.recent_clicked_window = 12
        self._last_filter_stats = {}

        # ---- latency profiling ----
        self.total_latency = []
        self.adb_tree_latency = []
        self.tree_latency = []
        self.selection_latency = []
        self.action_latency = []
        self.screenshot_latency = []

        # Keep reasoning actions for rollback replay (home + replay).
        self.action_history = []
        # Runtime focus from previous reasoning/history/clues, refreshed every iteration.
        self.runtime_focus_queries = []
        self.runtime_goal_emb = []
        # Snapshot of action_history for current exploration worker.
        self.replay_action_history = []
        # Trunk actions for current branch, only for debug/inspection.
        self.branch_action_history = []
        # Branch candidates captured during the latest exploration worker.
        self.iteration_candidates = []
        # Last clue generation diagnostics for debugging k->k+1 matching.
        self.last_clue_debug = {}

    def _decay_visit_counts(self, decay=0.92):
        if not self.bound_visit_count:
            return
        new_count = {}
        for b, v in self.bound_visit_count.items():
            nv = float(v) * float(decay)
            if nv >= 0.15:
                new_count[b] = nv
        self.bound_visit_count = new_count
        active = set(new_count.keys())
        for key in list(self.bound_effect_ema.keys()):
            if key not in active:
                self.bound_effect_ema.pop(key, None)
                self.bound_effect_count.pop(key, None)
                self.bound_seen_count.pop(key, None)
                self.bound_skip_count.pop(key, None)


    # -----------------------------
    # Thread control
    # -----------------------------
    def _recent_no_effect_repeat(self, max_tail=4):
        cnt = 0
        for item in reversed(self.runtime_history_tail[-max_tail:]):
            txt = str(item).lower()
            if "no_effect" in txt or "unchanged" in txt:
                cnt += 1
            else:
                break
        return cnt

    def _infer_explore_trigger_reason(self):
        no_effect_repeat = self._recent_no_effect_repeat(max_tail=4)
        if no_effect_repeat >= 1:
            return "no_effect_repeat"
        merged = " ".join([str(x).lower() for x in self.runtime_history_tail[-3:]])
        if "fallback" in merged:
            return "fallback_recovery"
        if "parse" in merged or "json" in merged:
            return "parse_uncertain"
        return "periodic_probe"

    def _compute_explore_budget(self, max_steps, max_depth, leaf_width, trigger_reason=None):
        max_depth = max(2, int(max_depth))
        max_steps = max(1, int(max_steps))
        leaf_width = max(1, int(leaf_width))

        reason = str(trigger_reason or "periodic_probe").strip().lower()

        if self.collection_mode:
            # Collection/demo mode prioritizes coverage and trace diversity over tight budget compression.
            base_steps = max(4, max_steps)
            base_depth = max(2, min(max_depth, 3))
            base_leaf = max(2, leaf_width)
            if reason in {"no_effect_repeat", "repeat_loop", "page_loop", "fallback_recovery", "parse_uncertain"}:
                repeats = max(1, self._recent_no_effect_repeat(max_tail=6))
                boost = min(8, 2 * repeats)
                return min(max_steps + 8, base_steps + boost), base_depth, min(base_leaf + 1, 5), boost
            return base_steps, base_depth, base_leaf, 0

        compact_depth = 2
        compact_leaf = max(1, min(leaf_width, 2))
        compact_steps = max(2, min(max_steps, 3))

        if reason in {"periodic_probe", "bootstrap", "read_only_bootstrap"}:
            return compact_steps, compact_depth, compact_leaf, 0

        if reason in {"no_effect_repeat", "repeat_loop", "page_loop", "fallback_recovery", "parse_uncertain"}:
            repeats = max(1, self._recent_no_effect_repeat(max_tail=6))
            boost = min(8, 2 * repeats)
            stressed_steps = min(max_steps, max(4, compact_steps + boost))
            stressed_depth = 2
            stressed_leaf = max(compact_leaf, min(leaf_width, 3))
            return stressed_steps, stressed_depth, stressed_leaf, boost

        return compact_steps, compact_depth, compact_leaf, 0

    @staticmethod
    def _deadline_exceeded(deadline_ts):
        return deadline_ts is not None and time.time() >= float(deadline_ts)

    def start(
        self,
        max_steps,
        max_depth=2,
        leaf_width=3,
        max_branches=None,
        time_budget_sec=None,
        trigger_reason=None,
    ):
        self.stop_event.clear()
        reason = trigger_reason or self._infer_explore_trigger_reason()
        self._explore_trigger_reason = str(reason)
        self.thread = threading.Thread(
            target=self.worker,
            args=(max_steps, max_depth, leaf_width, max_branches, time_budget_sec, reason),
            daemon=True,
        )
        self.thread.start()

    @staticmethod
    def _tokenize(text):
        return [t for t in re.findall(r"[a-z0-9_\\-]{2,}", (text or "").lower())]

    @staticmethod
    def _normalize_text(text):
        return re.sub(r"[^a-z0-9_\\-]+", " ", (text or "").lower()).strip()

    @staticmethod
    def _node_bounds_key(node):
        return (getattr(node, "bounds", "") or "").strip()

    @staticmethod
    def _node_merged_text(node):
        text = (getattr(node, "text", "") or "").strip().lower()
        desc = (getattr(node, "content_desc", "") or "").strip().lower()
        rid = (getattr(node, "resource_id", "") or "").strip().lower()
        cls = (getattr(node, "class_name", "") or "").strip().lower()
        hint = (getattr(node, "hint", "") or "").strip().lower()
        return " ".join([x for x in [text, desc, rid, cls, hint] if x]).strip()

    @staticmethod
    def _query_keywords(goal_queries, runtime_queries, limit=24):
        merged = " ".join([*(goal_queries or []), *(runtime_queries or [])])
        out = []
        seen = set()
        for tok in A11yTreeOnlineExplorer._tokenize(merged):
            if len(tok) < 3 or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= max(1, int(limit)):
                break
        return out

    @staticmethod
    def _query_overlap_score(query_keywords, merged_text):
        if not query_keywords or not merged_text:
            return 0.0
        hit = 0
        for kw in query_keywords:
            if kw and kw in merged_text:
                hit += 1
        if hit <= 0:
            return 0.0
        denom = max(2.0, min(6.0, float(len(query_keywords))))
        return min(1.0, float(hit) / denom)

    @staticmethod
    def _is_submit_or_dismiss_node(node):
        merged = A11yTreeOnlineExplorer._node_merged_text(node)
        if not merged:
            return False
        submit_tokens = {
            "save", "ok", "okay", "confirm", "done", "apply", "finish",
            "submit", "allow", "accept", "continue", "close", "dismiss",
            "确定", "保存", "完成", "确认", "继续", "关闭",
        }
        return any(tok in merged for tok in submit_tokens)

    @staticmethod
    def _is_back_navigation_node(node):
        merged = A11yTreeOnlineExplorer._normalize_text(A11yTreeOnlineExplorer._node_merged_text(node))
        if any(phrase in merged for phrase in ("navigate up", "go back", " back ")):
            return True
        rid = (getattr(node, "resource_id", "") or "").lower()
        return any(marker in rid for marker in ("btn_back", "navigate_up", "up_button", "nav_back"))

    @staticmethod
    def _navigation_helpfulness(node):
        cls = (getattr(node, "class_name", "") or "").lower()
        merged = A11yTreeOnlineExplorer._node_merged_text(node)
        nav_tokens = {"menu", "settings", "search", "tab", "category", "folder", "more"}
        bonus = 0.0
        if any(tok in merged for tok in nav_tokens):
            bonus += 0.22
        if "button" in cls or "tab" in cls or "toolbar" in cls:
            bonus += 0.06
        if bool(getattr(node, "scrollable", False)):
            bonus += 0.06
        if bool(getattr(node, "clickable", False)):
            bonus += 0.04
        return min(0.34, float(bonus))

    def _task_wants_back_navigation(self, goal_queries, runtime_queries):
        merged_goal = " ".join(goal_queries or []).lower()
        merged_runtime = " ".join(runtime_queries or []).lower()
        merged = " ".join([merged_goal, merged_runtime]).strip()
        return any(tok in merged for tok in {"back", "previous", "return", "cancel", "undo", "close"})

    @staticmethod
    def _intent_flags(goal_queries, runtime_queries):
        merged_goal = " ".join(goal_queries or []).lower()
        merged_runtime = " ".join(runtime_queries or []).lower()
        merged = " ".join([merged_goal, merged_runtime]).strip()
        input_tokens = {"type", "input", "enter", "write", "search", "fill", "name", "title"}
        select_tokens = {"select", "choose", "pick", "option", "set", "toggle", "switch"}
        nav_tokens = {"open", "go", "navigate", "back", "menu", "settings", "tab"}
        return {
            "input": any(tok in merged for tok in input_tokens),
            "select": any(tok in merged for tok in select_tokens),
            "nav": any(tok in merged_goal for tok in nav_tokens),
        }

    @staticmethod
    def _is_system_ui_noise_node(node):
        rid = (getattr(node, "resource_id", "") or "").lower()
        cls = (getattr(node, "class_name", "") or "").lower()
        merged = A11yTreeOnlineExplorer._node_merged_text(node)
        if "com.android.systemui" in rid or "systemui" in merged:
            return True
        noise_tokens = {
            "wifi signal", "phone signal", "battery", "quick settings",
            "privacy chip", "notification shade",
        }
        return any(tok in merged for tok in noise_tokens) or "statusbar" in cls

    @staticmethod
    def _is_keyboard_key_node(node):
        text = (getattr(node, "text", "") or "").strip()
        rid = (getattr(node, "resource_id", "") or "").lower()
        cls = (getattr(node, "class_name", "") or "").lower()
        merged = f"{rid} {cls}".strip()
        if any(tok in merged for tok in ("keyboard", "ime", "key_pos")):
            return True
        return len(text) == 1 and re.fullmatch(r"[A-Za-z0-9]", text) is not None

    @staticmethod
    def _is_information_only_node(node):
        text = (getattr(node, "text", "") or "").lower().strip()
        merged = A11yTreeOnlineExplorer._node_merged_text(node)
        if text:
            allowed_chars = set("0123456789:./- \t")
            numeric_like = all(ch in allowed_chars for ch in text)
        else:
            numeric_like = False
        info_tokens = {"clock", "status", "date only", "time only", "notification"}
        return bool(numeric_like or any(tok in merged for tok in info_tokens))

    @staticmethod
    def _is_low_value_node(node):
        cls = (getattr(node, "class_name", "") or "").lower()
        merged = A11yTreeOnlineExplorer._node_merged_text(node)
        text = (getattr(node, "text", "") or "").strip()
        low_cls = any(tok in cls for tok in ("checkbox", "switch", "radiobutton", "toggle"))
        low_text = any(tok in merged for tok in ("status", "clock", "signal", "battery"))
        checkable = bool(getattr(node, "checkable", False))
        return bool(low_cls or checkable or (low_text and not text))

    def _is_meaningless_node(self, node, intent_flags=None):
        intent_flags = intent_flags or {"input": False, "select": False, "nav": False}
        cls = (getattr(node, "class_name", "") or "").lower()
        if self._is_keyboard_key_node(node):
            return not bool(intent_flags.get("input"))
        if self._is_system_ui_noise_node(node):
            return True
        if self._is_information_only_node(node):
            return True
        if "progressbar" in cls:
            return True
        if "seekbar" in cls and not bool(intent_flags.get("select")):
            return True
        return False

    def _collect_candidates(
        self,
        root,
        avoid_bounds=None,
        hard_avoid=False,
        intent_flags=None,
        query_keywords=None,
        allow_back_navigation=False,
    ):
        intent_flags = intent_flags or {"input": False, "select": False, "nav": False}
        query_keywords = list(query_keywords or [])
        avoid = set(avoid_bounds or set())
        candidates = collect_clickable_nodes(root)
        stats = {
            "total": len(candidates),
            "removed_risky": 0,
            "removed_meaningless": 0,
            "removed_intent_mismatch": 0,
            "removed_visited": 0,
            "hard_avoid_fallback": 0,
            "candidates": 0,
        }

        filtered = []
        for node in candidates:
            if self._is_risky_node(node):
                stats["removed_risky"] += 1
                continue
            if self._is_meaningless_node(node, intent_flags=intent_flags):
                stats["removed_meaningless"] += 1
                continue
            merged = self._node_merged_text(node)
            overlap = self._query_overlap_score(query_keywords, merged)
            nav_bonus = self._navigation_helpfulness(node)
            keep_back = bool(allow_back_navigation and self._is_back_navigation_node(node))
            keep_common = bool(self._is_submit_or_dismiss_node(node))
            keep = bool(
                not query_keywords
                or overlap >= 0.12
                or nav_bonus >= 0.22
                or keep_back
                or keep_common
            )
            if self.collection_mode:
                # Keep broader coverage in collection mode; avoid over-pruning by intent.
                keep = bool(
                    not query_keywords
                    or overlap >= 0.06
                    or nav_bonus >= 0.12
                    or keep_back
                    or keep_common
                )
            if not keep:
                stats["removed_intent_mismatch"] += 1
                continue
            filtered.append(node)

        if not filtered:
            # Fallback: keep all non-destructive clickable nodes to avoid dead exploration.
            filtered = [n for n in candidates if not self._is_risky_node(n)]

        if self.collection_mode and len(filtered) < 3:
            relaxed = []
            for node in candidates:
                if self._is_risky_node(node):
                    continue
                if self._is_system_ui_noise_node(node):
                    continue
                relaxed.append(node)
            if relaxed:
                filtered = relaxed

        if avoid:
            keep = []
            for node in filtered:
                b = self._node_bounds_key(node)
                if b in avoid:
                    stats["removed_visited"] += 1
                    self.bound_skip_count[b] = int(self.bound_skip_count.get(b, 0)) + 1
                    if hard_avoid:
                        continue
                keep.append(node)
            if keep:
                filtered = keep
            else:
                stats["hard_avoid_fallback"] = 1

        stats["candidates"] = len(filtered)
        return filtered, stats

    def _select_depth_candidate(self, picked, semantic_low=0.25, avoid_bounds=None, hard_avoid=False):
        if not picked:
            return None, 0
        avoid = set(avoid_bounds or set())
        skipped = 0
        pool = []
        for item in picked:
            node = item[0]
            bounds = self._node_bounds_key(node)
            if bounds in avoid and hard_avoid:
                skipped += 1
                self.bound_skip_count[bounds] = int(self.bound_skip_count.get(bounds, 0)) + 1
                continue
            pool.append(item)
        if not pool:
            pool = list(picked)

        semantic_pool = []
        for item in pool:
            score_detail = item[3] if len(item) >= 4 else {}
            if float((score_detail or {}).get("similarity", 0.0)) >= float(semantic_low):
                semantic_pool.append(item)
        if semantic_pool:
            pool = semantic_pool

        recent = set(self.recent_clicked_bounds[-int(self.recent_clicked_window):])
        best = None
        best_score = -1e9
        for item in pool:
            node, score, _, score_detail = item
            bounds = self._node_bounds_key(node)
            adjusted = float(score)
            if bounds in avoid and not hard_avoid:
                adjusted -= 0.12
            if bounds in recent:
                adjusted -= 0.18
            visits = float((score_detail or {}).get("visits", 0.0))
            if visits > 0.0:
                adjusted -= min(0.25, visits * 0.05)
            effect_ema = self.bound_effect_ema.get(bounds)
            if effect_ema is not None and float(effect_ema) <= float(self.no_effect_delta_threshold) * 1.5:
                adjusted -= 0.10
            if best is None or adjusted > best_score:
                best = item
                best_score = adjusted
        return best, skipped

    def _score_candidate(self, node, intent_flags=None, query_keywords=None):
        intent_flags = intent_flags or {"input": False, "select": False, "nav": False}
        query_keywords = list(query_keywords or [])
        bounds = self._node_bounds_key(node)
        cand_txt = node_to_text(node)
        has_semantic_text = bool(cand_txt and cand_txt.strip())
        if not has_semantic_text:
            rid = (getattr(node, "resource_id", "") or "").strip()
            cls = (getattr(node, "class_name", "") or "").strip()
            desc = (getattr(node, "content_desc", "") or "").strip()
            hint = (getattr(node, "hint", "") or "").strip()
            cand_txt = " | ".join([x for x in [desc, hint, rid, cls, bounds] if x]) or "[icon_only]"

        cand_emb = embed_text(cand_txt, self.embed_model, self.emb_cache)
        task_score = float(cosine_sim(self.goal_emb, cand_emb))
        reason_score = float(cosine_sim(self.runtime_goal_emb, cand_emb)) if self.runtime_goal_emb else 0.0
        query_overlap = float(self._query_overlap_score(query_keywords, self._node_merged_text(node)))
        similarity = float(max(task_score, reason_score, min(1.0, query_overlap * 0.92)))

        visits = float(self.bound_visit_count.get(bounds, 0.0))
        novelty_score = 1.0 / (1.0 + visits)
        total_visits = float(sum(float(v) for v in self.bound_visit_count.values()))
        ucb_bonus = np.sqrt(np.log(1.0 + total_visits + 1.0) / (1.0 + visits))

        in_curr_run_penalty = 0.22 if bounds in self.clicked_bounds else 0.0
        recent_penalty = 0.18 if bounds in self.recent_clicked_bounds[-self.recent_clicked_window:] else 0.0
        effect_ema = self.bound_effect_ema.get(bounds)
        low_effect_penalty = 0.0
        if effect_ema is not None and float(effect_ema) <= float(self.no_effect_delta_threshold) * 1.5:
            low_effect_penalty = 0.10
        repeat_penalty = min(0.35, float(visits) * 0.07)

        is_clickable = bool(getattr(node, "clickable", False))
        clickable_bonus = 0.06 if is_clickable else 0.0
        non_clickable_penalty = 0.28 if not is_clickable else 0.0
        icon_explore_bonus = 0.10 if (not has_semantic_text and visits < 1.0) else 0.0
        low_value_penalty = 0.18 if self._is_low_value_node(node) and similarity < 0.22 else 0.0
        overlap_bonus = min(0.18, query_overlap * 0.18)
        nav_bonus = 0.35 * float(self._navigation_helpfulness(node)) if bool(intent_flags.get("nav")) else 0.0
        common_bonus = 0.05 if self._is_submit_or_dismiss_node(node) else 0.0

        total_score = (
            0.42 * similarity
            + 0.18 * novelty_score
            + 0.14 * float(ucb_bonus)
            + 0.16 * task_score
            + 0.10 * reason_score
            + clickable_bonus
            + icon_explore_bonus
            + overlap_bonus
            + nav_bonus
            + common_bonus
            - non_clickable_penalty
            - in_curr_run_penalty
            - recent_penalty
            - repeat_penalty
            - low_effect_penalty
            - low_value_penalty
        )
        total_score = float(max(0.0, min(1.0, total_score)))
        return {
            "node": node,
            "node_txt": cand_txt,
            "score": total_score,
            "task_score": float(task_score),
            "reason_score": float(reason_score),
            "query_overlap": float(query_overlap),
            "similarity": float(similarity),
            "novelty_score": float(novelty_score),
            "ucb_bonus": float(ucb_bonus),
            "clickable_bonus": float(clickable_bonus),
            "non_clickable_penalty": float(non_clickable_penalty),
            "icon_explore_bonus": float(icon_explore_bonus),
            "in_curr_run_penalty": float(in_curr_run_penalty),
            "recent_penalty": float(recent_penalty),
            "repeat_penalty": float(repeat_penalty),
            "low_effect_penalty": float(low_effect_penalty),
            "low_value_penalty": float(low_value_penalty),
            "overlap_bonus": float(overlap_bonus),
            "nav_bonus": float(nav_bonus),
            "common_bonus": float(common_bonus),
            "visits": float(visits),
            "has_semantic_text": bool(has_semantic_text),
            "is_clickable": bool(is_clickable),
            "effect_ema": None if effect_ema is None else float(effect_ema),
        }

    def _compact_score_detail(self, score_detail):
        if not isinstance(score_detail, dict):
            return {}
        keep_keys = [
            "score",
            "task_score",
            "reason_score",
            "query_overlap",
            "similarity",
            "novelty_score",
            "ucb_bonus",
            "clickable_bonus",
            "non_clickable_penalty",
            "icon_explore_bonus",
            "in_curr_run_penalty",
            "recent_penalty",
            "repeat_penalty",
            "low_effect_penalty",
            "low_value_penalty",
            "overlap_bonus",
            "nav_bonus",
            "common_bonus",
            "visits",
            "has_semantic_text",
            "is_clickable",
            "effect_ema",
            "node_txt",
        ]
        out = {}
        for k in keep_keys:
            if k not in score_detail:
                continue
            v = score_detail.get(k)
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
        return out

    def _semantic_pick(self, root, avoid_bounds=None, hard_avoid=False, semantic_low=0.30):
        picked, n_candidates = self._pick_topk_nodes(
            root=root,
            k=5,
            avoid_bounds=avoid_bounds,
            hard_avoid=hard_avoid,
        )
        if not picked:
            return None, None, None, n_candidates, None
        selected, _ = self._select_depth_candidate(
            picked,
            semantic_low=semantic_low,
            avoid_bounds=avoid_bounds,
            hard_avoid=hard_avoid,
        )
        if selected is None:
            return None, None, None, n_candidates, None
        node, score, node_txt, score_detail = selected
        return node, float(score), node_txt, n_candidates, score_detail

    def _pick_topk_nodes(self, root, k, avoid_bounds=None, hard_avoid=False):
        goal_queries = list(self.task_queries or [])
        runtime_queries = list(self.runtime_focus_queries or [])
        intent_flags = self._intent_flags(goal_queries, runtime_queries)
        query_keywords = self._query_keywords(goal_queries, runtime_queries, limit=24)
        allow_back_navigation = self._task_wants_back_navigation(goal_queries, runtime_queries)

        candidates, filter_stats = self._collect_candidates(
            root=root,
            avoid_bounds=avoid_bounds,
            hard_avoid=hard_avoid,
            intent_flags=intent_flags,
            query_keywords=query_keywords,
            allow_back_navigation=allow_back_navigation,
        )
        self._last_filter_stats = dict(filter_stats or {})

        n_candidates = len(candidates)
        if n_candidates == 0:
            return [], 0

        scored = []
        seen_bounds = set()
        for n in candidates:
            bounds = self._node_bounds_key(n)
            if bounds in seen_bounds:
                continue
            seen_bounds.add(bounds)
            self.bound_seen_count[bounds] = int(self.bound_seen_count.get(bounds, 0)) + 1
            s = self._score_candidate(n, intent_flags=intent_flags, query_keywords=query_keywords)
            if s is None:
                continue
            scored.append(s)

        scored.sort(key=lambda x: x["score"], reverse=True)
        picked = []
        for s in scored[:max(1, int(k))]:
            picked.append((s["node"], float(s["score"]), s["node_txt"], s))
        return picked, n_candidates

    def _is_risky_node(self, node):
        merged = self._normalize_text(self._node_merged_text(node))
        if not merged:
            return False
        risky_patterns = [
            r"\bdelete\b",
            r"\bremove\b",
            r"\bclear all\b",
            r"\berase\b",
            r"\bwipe\b",
            r"\bdiscard\b",
            r"\buninstall\b",
            r"\bfactory reset\b",
        ]
        return any(re.search(pattern, merged) for pattern in risky_patterns)

    def set_runtime_focus(self, history_tail=None, clues_text=None):
        self.runtime_focus_queries = []
        self.runtime_goal_emb = []
        self.runtime_history_tail = list(history_tail or []) if isinstance(history_tail, list) else ([history_tail] if history_tail else [])

        sources = []
        if history_tail:
            if isinstance(history_tail, list):
                sources.extend([str(x) for x in history_tail if x is not None])
            else:
                sources.append(str(history_tail))
        if clues_text:
            sources.append(str(clues_text))

        for src in sources:
            for q in extract_task_queries(src):
                q = (q or "").strip()
                if not q:
                    continue
                if q.lower() in {x.lower() for x in self.runtime_focus_queries}:
                    continue
                self.runtime_focus_queries.append(q)
                self.runtime_goal_emb.append(embed_text(q, self.embed_model, self.emb_cache))

    def _record_click_action_norm(self, coordinate_pixel, action_list):
        if self.width is None or self.height is None:
            return

        x, y = coordinate_pixel
        x_norm = float(x) / float(self.width)
        y_norm = float(y) / float(self.height)
        action_list.append({
            "action_type": "click",
            "action_inputs": {"coordinate": [x_norm, y_norm]}
        })

    def _click_and_record(self, node, sim, node_txt, n_candidates, depth, branch_id, branch_actions, score_detail=None):
        step_t0 = time.time()
        before_path = os.path.join(
            self.explore_raw_dir,
            f"before_{self.vis_step + 1:03d}.png",
        )

        act_t0 = time.time()
        with self.ui_lock:
            get_screenshot(self.args, self.explore_screenshot_path)
            shutil.copyfile(self.explore_screenshot_path, before_path)
            coordinate, bounds = click_node_center(self.adb, node)
        self.action_latency.append(time.time() - act_t0)

        if not coordinate:
            self.log_step({
                "step": self.cur_steps + 1,
                "type": "fail",
                "reason": "click_failed",
                "node_bounds": getattr(node, "bounds", None),
                "node_text": getattr(node, "text", None),
                "node_desc": getattr(node, "content_desc", None),
                "best_sim": float(sim) if sim is not None else None,
                "n_candidates": n_candidates,
                "depth": depth,
                "branch": branch_id,
                "time_sec": round(time.time() - step_t0, 4),
            })
            return {"ok": False}

        self.cur_steps += 1
        node_bounds = self._node_bounds_key(node)
        self.clicked_bounds.add(node_bounds)
        self.bound_visit_count[node_bounds] = float(self.bound_visit_count.get(node_bounds, 0.0)) + 1.0
        self.recent_clicked_bounds.append(node_bounds)
        if len(self.recent_clicked_bounds) > 128:
            self.recent_clicked_bounds = self.recent_clicked_bounds[-128:]

        if branch_actions is not None:
            self._record_click_action_norm(coordinate, branch_actions)

        self.vis_step += 1
        ui_text = getattr(node, "text", "") or ""
        ui_desc = getattr(node, "content_desc", "") or ""

        # Give UI a short settle window before observing transition.
        time.sleep(0.22)
        shot_t0 = time.time()
        get_screenshot(self.args, self.explore_screenshot_path)
        self.screenshot_latency.append(time.time() - shot_t0)
        before_hash = None
        after_hash = None
        page_hash_diff = None
        try:
            before_hash = phash(before_path)
            after_hash = phash(self.explore_screenshot_path)
            page_hash_diff = int(before_hash - after_hash)
        except Exception:
            page_hash_diff = None

        if page_hash_diff is not None:
            prev_ema = self.bound_effect_ema.get(node_bounds)
            if prev_ema is None:
                new_ema = float(page_hash_diff)
            else:
                new_ema = 0.65 * float(prev_ema) + 0.35 * float(page_hash_diff)
            self.bound_effect_ema[node_bounds] = float(new_ema)
            self.bound_effect_count[node_bounds] = int(self.bound_effect_count.get(node_bounds, 0)) + 1

        raw_out = os.path.join(
            self.explore_raw_dir,
            f"raw_{self.vis_step + 1:03d}.png",
        )
        shutil.copyfile(self.explore_screenshot_path, raw_out)

        sim_value = float(sim) if sim is not None else 0.0
        semantic_rel = float((score_detail or {}).get("similarity", sim_value))
        low_value_hit = self._is_low_value_node(node)
        useful_by_change = bool(page_hash_diff is not None and page_hash_diff >= 6)
        useful_by_semantic = bool(semantic_rel >= 0.35)
        is_useful = bool(useful_by_change or useful_by_semantic)
        if low_value_hit and semantic_rel < 0.22 and not useful_by_change:
            is_useful = False

        out = mark_and_save_explore_click(
            screenshot_path=self.explore_screenshot_path,
            save_dir=self.explore_vis_dir,
            step_idx=self.vis_step,
            xy=coordinate,
            bounds=bounds,
            text=f"action=click depth={depth} branch={branch_id} | {ui_text} {ui_desc} | sim={sim_value:.3f}",
            extra_lines=self._build_action_history_lines() + [
                f"page_hash_diff={page_hash_diff}",
                f"is_useful={is_useful} by_change={useful_by_change} by_semantic={useful_by_semantic}",
                f"trigger={self._explore_trigger_reason}",
            ],
        )

        total_latency = time.time() - step_t0
        self.total_latency.append(total_latency)

        self.log_step({
            "step": self.cur_steps,
            "type": "click",
            "best_sim": round(sim_value, 3),
            "score_detail": self._compact_score_detail(score_detail),
            "coordinate": list(coordinate),
            "bounds": getattr(node, "bounds", None),
            "node_text": ui_text,
            "node_desc": ui_desc,
            "node_resource_id": getattr(node, "resource_id", None),
            "node_class": getattr(node, "class_name", None),
            "node_match_text": node_txt,
            "n_candidates": n_candidates,
            "saved_vis": out,
            "depth": depth,
            "branch": branch_id,
            "is_useful": bool(is_useful),
            "useful_by_change": bool(useful_by_change),
            "useful_by_semantic": bool(useful_by_semantic),
            "low_value_hit": bool(low_value_hit),
            "page_hash_diff": page_hash_diff,
            "effect_ema": self.bound_effect_ema.get(node_bounds),
            "time_sec": round(total_latency, 3),
            "raw_screenshot": raw_out,
        })
        return {
            "ok": True,
            "raw_screenshot": raw_out,
            "best_sim": round(sim_value, 3),
            "score_detail": self._compact_score_detail(score_detail),
            "coordinate": list(coordinate),
            "bounds": getattr(node, "bounds", None) or bounds,
            "node_text": ui_text,
            "node_desc": ui_desc,
            "node_resource_id": getattr(node, "resource_id", None),
            "node_class": getattr(node, "class_name", None),
            "node_match_text": node_txt,
            "depth": depth,
            "branch": branch_id,
            "is_useful": bool(is_useful),
            "useful_by_change": bool(useful_by_change),
            "useful_by_semantic": bool(useful_by_semantic),
            "low_value_hit": bool(low_value_hit),
            "page_hash_diff": page_hash_diff,
            "effect_ema": self.bound_effect_ema.get(node_bounds),
            "hash": after_hash,
        }

    def _build_action_history_lines(self, max_items=6, max_len=120):
        actions = self.replay_action_history or self.action_history or []
        if not actions:
            return ["action_history: (empty)"]

        tail = actions[-max_items:]
        start_idx = len(actions) - len(tail) + 1
        lines = ["action_history (latest):"]
        for i, act in enumerate(tail):
            idx = start_idx + i
            if isinstance(act, dict):
                action_type = (
                    act.get("action_type")
                    or (act.get("raw") or {}).get("name")
                    or "unknown"
                )
                inputs = (
                    act.get("action_inputs")
                    or act.get("arguments")
                    or ((act.get("raw") or {}).get("arguments"))
                    or {}
                )
                s = f"{idx}. {action_type} {inputs}"
            else:
                s = f"{idx}. {act}"
            lines.append(s[:max_len])
        return lines

    def _save_debug_frame(self, action_text, xy=None, bounds=None, extra_lines=None):
        self.debug_step += 1
        get_screenshot(self.args, self.explore_screenshot_path)
        lines = list(extra_lines or [])
        lines.extend(self._build_action_history_lines())
        return mark_and_save_explore_click(
            screenshot_path=self.explore_screenshot_path,
            save_dir=self.explore_debug_vis_dir,
            step_idx=self.debug_step,
            xy=xy,
            bounds=bounds,
            text=action_text,
            extra_lines=lines,
        )

    def _same_root_page(self, root_path, curr_path, phash_thr=18, mae_thr=14.0):
        # 1) Fast hash check (current behavior)
        try:
            if check_same_image(root_path, curr_path, threshold=phash_thr):
                return True, "phash"
        except Exception:
            pass

        # 2) Robust fallback for same-page dynamic content (timer/text animation)
        # Compare low-res grayscale MAE to focus on layout-level similarity.
        try:
            a = np.asarray(Image.open(root_path).convert("L").resize((18, 40)), dtype=np.float32)
            b = np.asarray(Image.open(curr_path).convert("L").resize((18, 40)), dtype=np.float32)
            mae = float(np.mean(np.abs(a - b)))
            if mae <= mae_thr:
                return True, f"mae:{mae:.2f}"
            return False, f"mae:{mae:.2f}"
        except Exception:
            return False, "mae:err"

    def run_exploration_policy(
        self,
        max_steps,
        max_depth,
        leaf_width,
        max_branches=None,
        deadline_ts=None,
        trigger_reason=None,
    ):
        """
        树干：每层只走 top1，直到 depth = max_depth-1
        叶子层：在父节点处取 top leaf_width，逐个点叶子；叶子之间用 back 回到父节点
        每个分支结束：调用 fast_rollback() 回根
        """
        max_depth = max(2, int(max_depth))
        if leaf_width <= 0:
            leaf_width = 1

        budget_reason = str(trigger_reason or self._explore_trigger_reason or "periodic_probe")
        max_steps, max_depth, leaf_width, budget_boost = self._compute_explore_budget(
            max_steps=max_steps,
            max_depth=max_depth,
            leaf_width=leaf_width,
            trigger_reason=budget_reason,
        )
        if max_branches is None:
            max_branches = max(1, max_steps // max_depth)
            if self.collection_mode:
                max_branches = max(max_branches, min(max_steps, 4))
        self.log_step(
            {
                "type": "budget",
                "reason": budget_reason,
                "max_steps": int(max_steps),
                "max_depth": int(max_depth),
                "leaf_width": int(leaf_width),
                "max_branches": int(max_branches),
                "boost": int(budget_boost),
                "time_budget_sec": None if deadline_ts is None else max(0.0, float(deadline_ts - time.time())),
            }
        )

        self.cur_steps = 0
        self.clicked_bounds = set()

        branches_done = 0
        self.iteration_candidates = []
        explored_root_bounds = set()

        while branches_done < max_branches and self.cur_steps < max_steps:
            if self.stop_event.is_set() or self._deadline_exceeded(deadline_ts):
                break

            branch_id = branches_done + 1
            branch_actions = []
            branch_candidate = {
                "branch_id": branch_id,
                "trunk": None,
                "leaf_observations": [],
            }
            branch_path_bounds = set()

            depth = 0
            trunk_ok = True

            while depth < max_depth - 1 and self.cur_steps < max_steps:
                if self.stop_event.is_set() or self._deadline_exceeded(deadline_ts):
                    trunk_ok = False
                    break

                t0 = time.time()
                get_a11y_tree(self.args, self.xml_path)

                t1 = time.time()
                self.adb_tree_latency.append(t1 - t0)

                root = parse_a11y_tree(xml_path=self.xml_path)

                t2 = time.time()
                self.tree_latency.append(t2 - t1)

                s0 = time.time()
                avoid_bounds = explored_root_bounds if depth == 0 else branch_path_bounds
                hard_avoid = bool(depth == 0)
                node, sim, node_txt, n_candidates, score_detail = self._semantic_pick(
                    root,
                    avoid_bounds=avoid_bounds,
                    hard_avoid=hard_avoid,
                    semantic_low=0.30 if depth == 0 else 0.35,
                )
                self.selection_latency.append(time.time() - s0)

                if node is None:
                    # Avoid side-effect navigation during exploration.
                    # Random fallback (back/home) can introduce unexpected page drift.
                    fb = "noop"
                    self.log_step({
                        "step": self.cur_steps + 1,
                        "type": "fallback",
                        "fallback": fb,
                        "n_candidates": n_candidates,
                        "reason": "no_valid_semantic_candidate",
                        "depth": depth + 1,
                        "branch": branch_id,
                    })
                    trunk_ok = False
                    break

                node_bounds = self._node_bounds_key(node)
                if node_bounds:
                    branch_path_bounds.add(node_bounds)
                    if depth == 0:
                        explored_root_bounds.add(node_bounds)

                click_ret = self._click_and_record(
                    node=node,
                    sim=float(sim) if sim is not None else 0.0,
                    node_txt=node_txt,
                    n_candidates=n_candidates,
                    depth=depth + 1,
                    branch_id=branch_id,
                    branch_actions=branch_actions,
                    score_detail=score_detail,
                )
                if not click_ret.get("ok", False):
                    trunk_ok = False
                    break
                if depth == 0:
                    branch_candidate["trunk"] = click_ret

                depth += 1

            self.branch_action_history = list(branch_actions)
            if not trunk_ok or self.stop_event.is_set() or self._deadline_exceeded(deadline_ts):
                rollback_depth = max(1, min(max_depth, max(1, depth)))
                self.fast_rollback(max_depth=rollback_depth, step=branch_id, enable_replay=True)
                trunk = branch_candidate.get("trunk") or {}
                if trunk and bool(trunk.get("is_useful")):
                    self.iteration_candidates.append(branch_candidate)
                branches_done += 1
                time.sleep(0.05)
                continue

            if self.cur_steps < max_steps and not self.stop_event.is_set() and not self._deadline_exceeded(deadline_ts):
                get_a11y_tree(self.args, self.xml_path)
                root = parse_a11y_tree(xml_path=self.xml_path)

                s0 = time.time()
                leaf_nodes, n_candidates = self._pick_topk_nodes(
                    root=root,
                    k=leaf_width,
                    avoid_bounds=branch_path_bounds,
                    hard_avoid=False,
                )
                self.selection_latency.append(time.time() - s0)

                for leaf_idx, (leaf_node, leaf_sim, leaf_txt, leaf_score_detail) in enumerate(leaf_nodes):
                    if (
                        self.cur_steps >= max_steps
                        or self.stop_event.is_set()
                        or self._deadline_exceeded(deadline_ts)
                    ):
                        break

                    # Snapshot before leaf click: used to decide whether we need a "between leafs" back.
                    get_screenshot(self.args, self.leaf_before_screenshot_path)
                    leaf_ret = self._click_and_record(
                        node=leaf_node,
                        sim=leaf_sim,
                        node_txt=leaf_txt,
                        n_candidates=n_candidates,
                        depth=max_depth,
                        branch_id=branch_id,
                        branch_actions=None,
                        score_detail=leaf_score_detail,
                    )
                    leaf_ok = leaf_ret.get("ok", False)
                    if leaf_ok and (self.collection_mode or bool(leaf_ret.get("is_useful"))):
                        branch_candidate["leaf_observations"].append(leaf_ret)

                    # Only use this for debug now. Do not navigate back between leafs.
                    # Between-leaf back can overshoot to upper-level pages in launcher/task transitions.
                    is_last_leaf = (leaf_idx == len(leaf_nodes) - 1)
                    page_changed = False
                    if leaf_ok:
                        time.sleep(0.25)
                        get_screenshot(self.args, self.leaf_after_screenshot_path)
                        page_changed = not check_same_image(
                            self.leaf_before_screenshot_path,
                            self.leaf_after_screenshot_path,
                            threshold=12,
                        )
                    self._save_debug_frame(
                        action_text=f"skip_back_between_leafs branch={branch_id}",
                        extra_lines=[
                            f"leaf_ok={leaf_ok}",
                            f"is_last_leaf={is_last_leaf}",
                            f"page_changed={page_changed}",
                        ],
                    )

            rollback_depth = max(1, min(max_depth, max(1, depth)))
            self.fast_rollback(max_depth=rollback_depth, step=branch_id, enable_replay=True)
            trunk = branch_candidate.get("trunk") or {}
            trunk_useful = bool(trunk.get("is_useful"))
            has_useful_leaf = bool(branch_candidate.get("leaf_observations"))
            if trunk and (self.collection_mode or trunk_useful or has_useful_leaf):
                self.iteration_candidates.append(branch_candidate)
            branches_done += 1
            time.sleep(0.05)

    def worker(
        self,
        max_steps,
        max_depth=2,
        leaf_width=3,
        max_branches=None,
        time_budget_sec=None,
        trigger_reason=None,
    ):
        start_time = time.time()
        self._decay_visit_counts(decay=0.92)
        deadline_ts = None
        if time_budget_sec is not None:
            try:
                t_budget = float(time_budget_sec)
                if t_budget > 0:
                    deadline_ts = start_time + t_budget
            except Exception:
                deadline_ts = None

        # Capture root screenshot at worker start to avoid stale file races.
        if os.path.exists(self.rollback_root_path):
            os.remove(self.rollback_root_path)
        with self.ui_lock:
            get_screenshot(self.args, self.rollback_root_path)
        self.replay_action_history = list(self.action_history)

        self.run_exploration_policy(
            max_steps=max_steps,
            max_depth=max_depth,
            leaf_width=leaf_width,
            max_branches=max_branches,
            deadline_ts=deadline_ts,
            trigger_reason=trigger_reason,
        )

        exploration_latency = time.time() - start_time
        print_latency_summary(
            total_latency=self.total_latency,
            get_tree_latency=self.adb_tree_latency,
            parse_tree_latency=self.tree_latency,
            selection_latency=self.selection_latency,
            action_latency=self.action_latency,
            screenshot_latency=self.screenshot_latency,
            exploration_latency=exploration_latency
        )

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()

    # -----------------------------
    # Logging
    # -----------------------------
    def log_step(self, record: dict):
        self.history.append(record)
        try:
            append_jsonl(self.log_path, record)
        except Exception as e:
            print("[Explorer:log] failed to write jsonl:", e)

    # -----------------------------
    # Rollback
    # -----------------------------
    def fast_rollback(self, max_depth=2, step=0, enable_replay=True):
        if self.cur_steps == 0:
            if self.rollback_done_event:
                self.rollback_done_event.set()
            return

        rollback_success = False

        with self.rollback_lock:
            with self.ui_lock:
                for i in range(max_depth):
                    start_time = time.time()

                    # 1. 先截图判断是否已经回到起始页
                    get_screenshot(self.args, self.rollback_screenshot_path)

                    file_path = f"explore_debug/step_{step}_explore_{self.cur_steps}_back_{i}.png"
                    shutil.copyfile(self.rollback_screenshot_path, file_path)

                    is_same, same_reason = self._same_root_page(
                        self.rollback_root_path,
                        self.rollback_screenshot_path,
                        phash_thr=18,
                        mae_thr=14.0,
                    )

                    end_time = time.time()
                    check_latency = end_time - start_time

                    if is_same:
                        self._save_debug_frame(
                            action_text=f"rollback_match_root step={step} i={i}",
                            extra_lines=[
                                f"cur_steps={self.cur_steps}",
                                f"img_same={is_same}",
                                f"same_reason={same_reason}",
                            ],
                        )
                        print("⚡ Fast rollback done (matched root page).")
                        rollback_success = True
                        break

                    print(f" Fast rollback step {i + 1} / {max_depth} in {check_latency * 1000:.3f} ms")

                    # 2. 执行 back
                    back(self.adb)
                    time.sleep(1.0)
                    self._save_debug_frame(
                        action_text=f"action=back (rollback) step={step} i={i}",
                        extra_lines=[f"cur_steps={self.cur_steps}"],
                    )

                # rollback失败时进行replay
                if not rollback_success and enable_replay:
                    print("⚠️ Fast rollback failed. Start replay previous exploration actions...")

                    home(self.adb)
                    time.sleep(0.8)
                    replay_actions = self.replay_action_history or self.action_history
                    for action in replay_actions:
                        execute_action(
                            action,
                            self.width,
                            self.height,
                            self.adb,
                            coord_scale=self.coord_scale,
                        )
                        act_type = (action or {}).get("action_type", "")
                        if act_type == "open_app":
                            time.sleep(1.2)
                        else:
                            time.sleep(0.35)

        if self.rollback_done_event:
            self.rollback_done_event.set()

    # -----------------------------
    # Clues for VLM prompt
    # -----------------------------
    def consume_iteration_candidates(self):
        return list(self.iteration_candidates or [])

    def _parse_bounds(self, bounds):
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            try:
                x1, y1, x2, y2 = [int(float(v)) for v in bounds]
                return x1, y1, x2, y2
            except Exception:
                return None
        if not isinstance(bounds, str):
            return None
        nums = re.findall(r"\d+", bounds)
        if len(nums) != 4:
            return None
        x1, y1, x2, y2 = map(int, nums)
        return x1, y1, x2, y2

    def _region_from_bounds(self, bounds):
        b = self._parse_bounds(bounds)
        if b:
            x1, y1, x2, y2 = b
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            return self._region_from_point(cx, cy)
        # No bounds: fallback to screen center instead of "unknown area".
        w = max(1, int(self.width or 1080))
        h = max(1, int(self.height or 2400))
        return self._region_from_point(w / 2.0, h / 2.0)

    def _region_from_point(self, cx, cy):
        w = max(1, int(self.width or 1080))
        h = max(1, int(self.height or 2400))
        horiz = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
        vert = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
        if vert == "middle" and horiz == "center":
            return "center"
        return f"{vert}-{horiz}"

    def _region_from_record(self, rec):
        if not isinstance(rec, dict):
            return self._region_from_bounds(None)
        bounds = rec.get("bounds")
        if bounds:
            return self._region_from_bounds(bounds)
        coord = rec.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            try:
                return self._region_from_point(float(coord[0]), float(coord[1]))
            except Exception:
                pass
        return self._region_from_bounds(None)

    def _extract_keywords(self, text, limit=6):
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\\-]{2,}", (text or "").lower())
        stop = {
            "the", "and", "for", "with", "from", "this", "that", "using", "click",
            "button", "open", "close", "page", "screen", "app", "action", "record",
            "audio", "clip", "task", "matched", "branch", "observed", "likely",
        }
        out = []
        seen = set()
        for t in tokens:
            if t in stop:
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= limit:
                break
        return out

    def _clean_clue_text(self, text):
        t = (text or "").strip()
        if not t:
            return ""
        t = re.sub(r"\s+", " ", t)
        # Drop command-like or low-value noisy lines.
        low = t.lower()
        noisy_patterns = [
            "am broadcast",
            "adb shell",
            "content://",
            "[ ]",
            "- [ ]",
            "yyyy-mm-dd",
            "date only",
            "time only",
        ]
        if any(p in low for p in noisy_patterns):
            return ""
        # Avoid ultra-long unstructured strings.
        if len(t) > 120:
            t = t[:120].rstrip() + "..."
        return t

    def _infer_ui_effect(self, text):
        low = (text or "").lower()
        mapping = [
            (["more options", "menu", "settings"], "open options/menu"),
            (["search"], "open search"),
            (["record", "resume", "stop", "finish"], "control recording state"),
            (["save", "ok", "confirm", "done"], "confirm/save action"),
            (["new", "create", "add", "folder"], "create new item"),
            (["back", "up", "navigate"], "navigate backward"),
        ]
        for keys, eff in mapping:
            if any(k in low for k in keys):
                return eff
        return "possible next action"

    def build_prompt_clues_from_candidates(self, candidates, current_screenshot_path, max_items=4, last_reasoning_action=None):
        self.last_clue_debug = {
            "status": "start",
            "n_candidates": len(candidates or []),
            "best_diff": None,
            "best_action_hit": None,
            "confidence": None,
            "n_leaves": 0,
            "n_selected": 0,
        }
        if not candidates:
            self.last_clue_debug["status"] = "empty_candidates"
            return ""

        cur_hash = phash(current_screenshot_path)
        last_action_text = (last_reasoning_action or "").lower().strip()
        ranked = []
        for cand in candidates:
            trunk = cand.get("trunk") or {}
            shot = trunk.get("raw_screenshot")
            if not shot or not os.path.exists(shot):
                continue
            try:
                diff = cur_hash - phash(shot)
            except Exception:
                continue
            trunk_text = " ".join(
                [
                    str(trunk.get("node_text") or ""),
                    str(trunk.get("node_desc") or ""),
                    str(trunk.get("node_match_text") or ""),
                ]
            ).lower()
            action_hit = 0
            if last_action_text and trunk_text:
                if any(k in trunk_text for k in self._extract_keywords(last_action_text, limit=5)):
                    action_hit = 1
            rank_score = float(diff) - (2.0 * action_hit)
            ranked.append((rank_score, diff, action_hit, cand))

        if not ranked:
            self.last_clue_debug["status"] = "no_ranked_candidates"
            return ""

        ranked.sort(key=lambda x: x[0])
        top_ranked = ranked[: max(1, min(3, len(ranked)))]
        _, best_diff, best_action_hit, best = top_ranked[0]
        self.last_clue_debug["best_diff"] = int(best_diff)
        self.last_clue_debug["best_action_hit"] = int(best_action_hit)
        # Multi-level guard for k+1 matching.
        if best_diff <= 12:
            match_conf = "high"
        elif best_diff <= 24:
            match_conf = "medium"
        elif best_diff <= 36:
            match_conf = "low"
        else:
            match_conf = "very_low"
        if match_conf == "very_low" and int(best_action_hit or 0) <= 0:
            self.last_clue_debug["status"] = "reject_by_diff"
            return ""

        self.last_clue_debug["confidence"] = match_conf
        trunk = best.get("trunk") or {}
        leaves = []
        branch_ids = []
        for _, _, _, cand in top_ranked:
            branch_id = cand.get("branch_id")
            if branch_id is not None:
                branch_ids.append(branch_id)
            branch_leaves = cand.get("leaf_observations") or []
            if branch_leaves:
                leaves.extend(branch_leaves)
            else:
                trunk_only = dict(cand.get("trunk") or {})
                if trunk_only:
                    trunk_only["_from_trunk_only"] = True
                    leaves.append(trunk_only)
        self.last_clue_debug["n_leaves"] = len(leaves)

        lines = [
            "[Environment Clues from Parallel Exploration]",
            f"[K+1 Match] branches={branch_ids or [best.get('branch_id')]}, diff={best_diff}, action_hit={best_action_hit}, confidence={match_conf}",
            "[K+1 Elements -> K+2 Hints]",
        ]

        ttxt = (trunk.get("node_text") or trunk.get("node_desc") or trunk.get("node_match_text") or "").strip()
        tpos = self._region_from_record(trunk)
        ttxt = self._clean_clue_text(ttxt)
        if ttxt:
            lines.append(f"• entry @ {tpos}: {ttxt}")

        # Filter leaf clues by task/reason relevance to reduce off-task app pollution.
        scored = []
        goal_vecs = self.goal_emb + self.runtime_goal_emb
        for leaf in leaves:
            txt = (
                leaf.get("node_text")
                or leaf.get("node_desc")
                or leaf.get("node_match_text")
                or ""
            ).strip()
            txt = self._clean_clue_text(txt)
            if not txt:
                continue
            txt_emb = embed_text(txt, self.embed_model, self.emb_cache)
            rel = float(cosine_sim(goal_vecs, txt_emb))
            pri = float((leaf.get("score_detail") or {}).get("score", 0.0))
            if pri <= 0.0:
                pri = float(leaf.get("best_sim") or 0.0)
            if rel < 0.10 and pri < 0.05:
                continue
            rank = rel * 0.75 + pri * 0.25
            scored.append((rank, rel, pri, leaf))

        # Degrade gracefully: if semantic filter rejects all leaves, still keep top observed leaves.
        if not scored:
            for leaf in leaves:
                txt = (
                    leaf.get("node_text")
                    or leaf.get("node_desc")
                    or leaf.get("node_match_text")
                    or ""
                ).strip()
                txt = self._clean_clue_text(txt)
                if not txt:
                    continue
                pri = float((leaf.get("score_detail") or {}).get("score", 0.0))
                if pri == 0.0:
                    pri = float(leaf.get("best_sim") or 0.0)
                if pri <= 0.0:
                    continue
                scored.append((pri, 0.0, pri, leaf))
        scored.sort(key=lambda x: x[0], reverse=True)
        added = 0
        seen = set()
        k2_texts = []
        for _, rel, pri, leaf in scored:
            txt = (
                leaf.get("node_text")
                or leaf.get("node_desc")
                or leaf.get("node_match_text")
                or ""
            ).strip()
            txt = self._clean_clue_text(txt)
            if not txt:
                continue
            if len(txt) > 140:
                txt = txt[:140].rstrip(" ,.;:") + "..."
            key = txt.lower()
            if key in seen:
                continue
            seen.add(key)
            pos = self._region_from_record(leaf)
            kws = self._extract_keywords(txt, limit=3)
            effect = self._infer_ui_effect(txt)
            if kws:
                lines.append(
                    f"• branch={leaf.get('branch')} {pos}: {txt} -> {effect}; "
                    f"k+2: {', '.join(kws)}; rel={rel:.3f}, score={pri:.3f}"
                )
            else:
                lines.append(
                    f"• branch={leaf.get('branch')} {pos}: {txt} -> {effect}; "
                    f"rel={rel:.3f}, score={pri:.3f}"
                )
            k2_texts.append(txt)
            added += 1
            if added >= max_items:
                break
        self.last_clue_debug["n_selected"] = int(added)
        self.last_clue_debug["status"] = "ok" if added > 0 else "no_selected_clues"

        if added == 0:
            lines.append("• no reliable K+1 element clues from matched branch.")
        else:
            k2_keywords = self._extract_keywords(" ".join(k2_texts), limit=6)
            if k2_keywords:
                lines.append(f"[K+2 Keywords] {', '.join(k2_keywords)}")
        lines.append("")
        return "\n".join(lines)

    def get_last_clue_debug_lines(self):
        d = self.last_clue_debug or {}
        return [
            f"[ClueDebug] status={d.get('status')}",
            f"[ClueDebug] n_candidates={d.get('n_candidates')} n_leaves={d.get('n_leaves')} selected={d.get('n_selected')}",
            f"[ClueDebug] best_diff={d.get('best_diff')} action_hit={d.get('best_action_hit')} confidence={d.get('confidence')}",
        ]

    def build_prompt_clues(self, max_items: int = 4):
        return build_prompt_clues(self.history, max_items=max_items)
