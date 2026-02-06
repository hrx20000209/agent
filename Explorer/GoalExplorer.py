import threading
import time
import os
import shutil

import numpy as np
from sentence_transformers import SentenceTransformer

from MobileAgentE.controller import get_a11y_tree, get_screenshot, back
from MobileAgentE.tree import parse_a11y_tree
from Explorer.utils import (
    ensure_dir,
    append_jsonl,
    embed_text,
    collect_clickable_nodes,
    select_node_by_semantic,
    click_node_center,
    fallback_action,
    build_prompt_clues,
    print_latency_summary,
    extract_task_queries,
    mark_and_save_explore_click,
    check_same_image
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
        adb_path: str,
        args,
        xml_path: str,
        task_text: str,
        explore_vis_dir: str = "explore_results",
        screenshot_path: str = "screenshot.png",
        embed_model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ):
        self.adb = adb_path
        self.args = args
        self.xml_path = xml_path
        self.task_text = task_text

        self.stop_event = threading.Event()
        self.thread = None

        self.explore_vis_dir = explore_vis_dir
        self.explore_screenshot_path = "screenshot/explore_screenshot.png"
        self.rollback_screenshot_path = "rollback/explore_screenshot.png"
        self.rollback_root_path = "rollback/root.png"
        self.vis_step = 0
        self.cur_steps = 0

        # --- logging ---
        self.history = []
        ensure_dir(self.explore_vis_dir)
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

        # ---- latency profiling ----
        self.total_latency = []
        self.adb_tree_latency = []
        self.tree_latency = []
        self.selection_latency = []
        self.action_latency = []
        self.screenshot_latency = []


    # -----------------------------
    # Thread control
    # -----------------------------
    def start(self, max_steps, max_depth=2, max_branches=None):
        """
        max_steps: 总点击预算（与你现在一致）
        max_depth: 每条分支走几步，2 就是“点两次就回滚”
        max_branches: 分支数上限，None 就由 max_steps/max_depth 推出来
        """
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self.worker,
            args=(max_steps, max_depth, max_branches),
            daemon=True,
        )
        self.thread.start()

    def _semantic_pick(self, root):
        candidates = collect_clickable_nodes(root)
        if len(candidates) == 0:
            return None, None, None, 0

        node, sim, node_txt = select_node_by_semantic(
            candidates=candidates,
            goal_emb_list=self.goal_emb,
            embed_model=self.embed_model,
            emb_cache=self.emb_cache,
            clicked_bounds=self.clicked_bounds,
        )
        return node, sim, node_txt, len(candidates)

    def worker(self, max_steps, max_depth=2, max_branches=None):
        """
        BFS-like exploration:
          - Explore many shallow branches: depth=2 (or configurable)
          - After reaching depth, rollback to start, continue next branch
        """
        start_time = time.time()
        shutil.copyfile(self.args.screenshot_path, self.rollback_root_path)
        self.cur_steps = 0
        self.clicked_bounds = set()  # 你也可以不重置，看你要不要跨轮复用

        # 计算分支数：每条分支最多 max_depth 次点击
        if max_depth <= 0:
            max_depth = 1
        if max_branches is None:
            max_branches = max(1, max_steps // max_depth)

        # 为了 “更 BFS”，我们先收集第一层的候选（只收集，不点击）
        # 但 UI 是动态的，所以做成“每个分支开始前再取一次 tree”更稳
        branches_done = 0

        while branches_done < max_branches and self.cur_steps < max_steps:
            if self.stop_event.is_set():
                break

            # ====== 每条分支从起点开始 ======
            branch_start_t = time.time()
            branch_path = []  # 用于记录这条分支走过的节点信息（也会进入 history 里）

            depth = 0
            while depth < max_depth and self.cur_steps < max_steps:
                if self.stop_event.is_set():
                    break

                step_t0 = time.time()

                # 1) tree
                get_a11y_tree(self.args, self.xml_path)
                get_tree_end_time = time.time()
                self.adb_tree_latency.append(get_tree_end_time - step_t0)

                root = parse_a11y_tree(xml_path=self.xml_path)
                parse_tree_end_time = time.time()
                tree_latency = parse_tree_end_time - get_tree_end_time
                self.tree_latency.append(tree_latency)

                # 2) semantic pick
                sel_t0 = time.time()
                node, sim, node_txt, n_candidates = self._semantic_pick(root)
                selection_latency = time.time() - sel_t0
                self.selection_latency.append(selection_latency)

                if node is None:
                    fb = fallback_action(self.adb)
                    rec = {
                        "step": self.cur_steps + 1,
                        "type": "fallback",
                        "fallback": fb,
                        "n_candidates": n_candidates,
                        "reason": "no_valid_semantic_candidate",
                        "depth": depth + 1,
                        "branch": branches_done + 1,
                        "time_sec": round(time.time() - step_t0, 4),
                    }
                    self.log_step(rec)
                    # fallback 后不建议继续深入，直接结束这条分支并回滚
                    break

                # 3) click
                act_t0 = time.time()
                coordinate, bounds = click_node_center(self.adb, node)
                action_latency = time.time() - act_t0
                self.action_latency.append(action_latency)

                if not coordinate:
                    rec = {
                        "step": self.cur_steps + 1,
                        "type": "fail",
                        "reason": "bounds_parse_failed",
                        "node_bounds": getattr(node, "bounds", None),
                        "node_text": getattr(node, "text", None),
                        "node_desc": getattr(node, "content_desc", None),
                        "node_resource_id": getattr(node, "resource_id", None),
                        "node_class": getattr(node, "class_name", None),
                        "best_sim": float(sim) if sim is not None else None,
                        "n_candidates": n_candidates,
                        "depth": depth + 1,
                        "branch": branches_done + 1,
                        "time_sec": round(time.time() - step_t0, 4),
                    }
                    self.log_step(rec)
                    break

                # ✅ 成功点击：计数 + 去重
                self.cur_steps += 1
                self.clicked_bounds.add(getattr(node, "bounds", ""))

                # 4) screenshot + mark（保持你原来的注入方法）
                self.vis_step += 1
                ui_text = getattr(node, "text", "") or ""
                ui_desc = getattr(node, "content_desc", "") or ""

                shot_t0 = time.time()
                get_screenshot(self.args, self.explore_screenshot_path, image_id=2)
                self.screenshot_latency.append(time.time() - shot_t0)

                out = mark_and_save_explore_click(
                    screenshot_path=self.explore_screenshot_path,
                    save_dir=self.explore_vis_dir,
                    step_idx=self.vis_step,
                    xy=coordinate,
                    bounds=bounds,
                    text=f"{ui_text} {ui_desc} | sim={sim:.3f}",
                )

                total_latency = time.time() - step_t0
                self.total_latency.append(total_latency)

                rec = {
                    "step": self.cur_steps,
                    "type": "click",
                    "best_sim": round(float(sim), 3),
                    "coordinate": list(coordinate),
                    "bounds": getattr(node, "bounds", None),
                    "node_text": ui_text,
                    "node_desc": ui_desc,
                    "node_resource_id": getattr(node, "resource_id", None),
                    "node_class": getattr(node, "class_name", None),
                    "node_match_text": node_txt,
                    "n_candidates": n_candidates,
                    "saved_vis": out,
                    "depth": depth + 1,
                    "branch": branches_done + 1,
                    "time_sec": round(total_latency, 3),
                    "tree_latency": round(tree_latency, 3),
                    "selection_latency": round(selection_latency, 3),
                    "action_latency": round(action_latency, 3),
                }
                self.log_step(rec)
                branch_path.append(rec)

                depth += 1

            # ====== 这条分支结束：立刻 rollback 回起点 ======
            # 这里完全复用你已有的 fast_rollback，不改逻辑
            self.fast_rollback()

            branches_done += 1

            # 可选：给 UI 一点稳定时间
            time.sleep(0.05)

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
            self.thread.join(timeout=1.0)

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
    def fast_rollback(self):
        if self.cur_steps == 0:
            return

        print(f"⚡ Fast rollback: backing {self.cur_steps} steps...")
        i = 0
        for i in range(self.cur_steps):
            start_time = time.time()
            get_screenshot(self.args, self.rollback_screenshot_path, image_id=3)
            is_same = check_same_image(self.rollback_root_path, self.rollback_screenshot_path)
            end_time = time.time()
            check_latency = end_time - start_time
            if is_same:
                break
            else:
                print(f" Fast rollback step {i+1} / {self.cur_steps} in {check_latency * 1000:.3f} ms")
                back(self.adb)
                time.sleep(0.1)
        print(f"⚡ Fast rollback done in {i + 1} steps.")

    # -----------------------------
    # Clues for VLM prompt
    # -----------------------------
    def build_prompt_clues(self, max_items: int = 4):
        return build_prompt_clues(self.history, max_items=max_items)

