import threading
import time
import os

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
    mark_and_save_explore_click
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
        screenshot_path: str = "explore_screenshot.png",
        embed_model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ):
        self.adb = adb_path
        self.args = args
        self.xml_path = xml_path
        self.task_text = task_text

        self.stop_event = threading.Event()
        self.thread = None

        self.explore_vis_dir = explore_vis_dir
        self.explore_screenshot_path = screenshot_path
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

    # -----------------------------
    # Thread control
    # -----------------------------
    def start(self, max_steps: int = 2):
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self.worker,
            args=(max_steps, ),
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)

    # -----------------------------
    # Worker
    # -----------------------------
    def worker(self, max_steps: int):
        start_time = time.time()

        self.cur_steps = 0
        for k in range(max_steps):
            if self.stop_event.is_set():
                return

            step_t0 = time.time()

            # 1) pull latest a11y tree + parse + collect candidates
            
            get_a11y_tree(self.args, self.xml_path)
            get_tree_end_time = time.time()
            get_tree_latency = get_tree_end_time - step_t0
            self.adb_tree_latency.append(get_tree_latency)

            root = parse_a11y_tree(xml_path=self.xml_path)

            parse_tree_end_time = time.time()
            tree_latency = parse_tree_end_time - get_tree_end_time

            # if you already have collect_clickable_nodes inside MobileAgentE.tree
            # you can replace this call with that one, logic is the same
            candidates = collect_clickable_nodes(root)

            selection_end_time = time.time()
            selection_latency = selection_end_time - parse_tree_end_time
            self.selection_latency.append(selection_latency)

            if len(candidates) == 0:
                fb = fallback_action(self.adb)
                rec = {
                    "step": k + 1,
                    "type": "fallback",
                    "fallback": fb,
                    "n_candidates": 0,
                    "time_sec": round(time.time() - step_t0, 4),
                }
                self.log_step(rec)
                print(f"[Explore] step={k + 1}/{max_steps} candidates=0 -> fallback={fb}")
                continue

            # 2) semantic selection
            t0 = time.time()
            node, sim, node_txt = select_node_by_semantic(
                candidates=candidates,
                goal_emb_list=self.goal_emb,
                embed_model=self.embed_model,
                emb_cache=self.emb_cache,
                clicked_bounds=self.clicked_bounds,
            )
            selection_latency = time.time() - t0

            if node is None:
                fb = fallback_action(self.adb)
                rec = {
                    "step": k + 1,
                    "type": "fallback",
                    "fallback": fb,
                    "n_candidates": len(candidates),
                    "reason": "no_valid_semantic_candidate",
                    "time_sec": round(time.time() - step_t0, 4),
                }
                self.log_step(rec)
                print(f"[Explore] step={k+1}/{max_steps} candidates={len(candidates)} -> fallback={fb}")
                continue

            # 3) click
            t0 = time.time()
            coordinate, bounds = click_node_center(self.adb, node)
            action_latency = time.time() - t0

            if not coordinate:
                total_latency = time.time() - step_t0
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
                    "total_latency": round(total_latency, 4),
                    "tree_latency": round(tree_latency, 4),
                    "selection_latency": round(selection_latency, 4),
                    "action_latency": round(action_latency, 4),
                }
                self.log_step(rec)
                print(f"[Explore] step={k+1}/{max_steps} bounds parse failed: {getattr(node,'bounds',None)}")
                continue

            self.cur_steps += 1

            # mark as clicked
            self.clicked_bounds.add(getattr(node, "bounds", ""))

            # 4) screenshot + mark
            self.vis_step += 1
            ui_text = getattr(node, "text", "") or ""
            ui_desc = getattr(node, "content_desc", "") or ""

            get_screenshot(self.args, self.explore_screenshot_path)

            out = mark_and_save_explore_click(
                screenshot_path=self.explore_screenshot_path,
                save_dir=self.explore_vis_dir,
                step_idx=self.vis_step,
                xy=coordinate,
                bounds=bounds,
                text=f"{ui_text} {ui_desc} | sim={sim:.3f}",
            )

            # record + latency list
            total_latency = time.time() - step_t0
            rec = {
                "step": k + 1,
                "type": "click",
                "best_sim": round(float(sim), 3),
                "coordinate": list(coordinate),
                "bounds": getattr(node, "bounds", None),
                "node_text": ui_text,
                "node_desc": ui_desc,
                "node_resource_id": getattr(node, "resource_id", None),
                "node_class": getattr(node, "class_name", None),
                "node_match_text": node_txt,
                "n_candidates": len(candidates),
                "saved_vis": out,
                "time_sec": round(time.time() - step_t0, 3),
                "total_latency": round(total_latency, 3),
                "tree_latency": round(tree_latency, 3),
                "selection_latency": round(selection_latency, 3),
                "action_latency": round(action_latency, 3),
            }
            self.log_step(rec)

            self.total_latency.append(total_latency)
            self.tree_latency.append(tree_latency)
            self.selection_latency.append(selection_latency)
            self.action_latency.append(action_latency)

            print("=" * 90)
            print(f"[Explore] step={k + 1}/{max_steps}  ✅ semantic click")
            print(f"  task_text     : {self.task_text}")
            print(f"  best_sim      : {sim:.4f}")
            print(f"  candidates    : {len(candidates)}")
            print(f"  coordinate    : {coordinate}")
            print(f"  bounds        : {getattr(node, 'bounds', None)}")
            print(f"  node_text     : {ui_text}")
            print(f"  node_desc     : {ui_desc}")
            print(f"  match_text    : {node_txt}")
            print(f"  saved_vis     : {out}")
            print(f"  log_path      : {self.log_path}")
            print(f"  Get Tree Latency: {get_tree_latency * 1000:.3f} ms")
            print(f"  Total latency : {total_latency * 1000:.3f} ms")
            print(f"  Tree latency  : {tree_latency * 1000:.3f} ms")
            print("=" * 90)

        end_time = time.time()
        exploration_latency = end_time - start_time
        print_latency_summary(
            total_latency=self.total_latency,
            get_tree_latency=self.adb_tree_latency,
            parse_tree_latency=self.tree_latency,
            selection_latency=self.selection_latency,
            action_latency=self.action_latency,
            exploration_latency=exploration_latency,
        )

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
        for _ in range(self.cur_steps):
            back(self.adb)
        print("⚡ Fast rollback done.")

    # -----------------------------
    # Clues for VLM prompt
    # -----------------------------
    def build_prompt_clues(self, max_items: int = 4):
        return build_prompt_clues(self.history, max_items=max_items)