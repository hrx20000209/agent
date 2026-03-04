"""Explorer-style element agent for AndroidWorld.

This file keeps the original tool-call parser/action mapping style, and adds
parallel threaded depth-first exploration that runs during model reasoning,
then rolls back before executing the reasoning action.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
import time
from collections import deque
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.agents.gelab_agent import GELAB_SYSTEM_PROMPT
from android_world.agents.gelab_agent import gelab_action_to_json_action
from android_world.agents.gelab_agent import parse_gelab_response
from android_world.agents.explorer_agent_constants import (
    _CHOICE_HINT_TOKENS,
    _DIALOG_HINT_TOKENS,
    _DISMISS_ACTION_KEYWORDS,
    _INFO_ONLY_TEXT_HINTS,
    _KEYBOARD_HINT_TOKENS,
    _LOW_VALUE_CLASS_HINTS,
    _LOW_VALUE_TEXT_HINTS,
    _NAV_HELPFUL_KEYWORDS,
    _RISKY_KEYWORDS,
    _SUBMIT_ACTION_KEYWORDS,
    _TASK_INPUT_KEYWORDS,
    _TASK_SELECT_KEYWORDS,
)
from android_world.agents.explorer_agent_utils import (
    _element_hint_compact_label,
    _extract_task_queries,
    _hash_diff,
    _mae_small,
    _normalize_space,
    _phash_pixels,
    _pixel_delta,
    _safe_int,
    _target_index,
    _to_data_url,
    _to_json_action,
    _tokenize,
    parse_tool_call,
    parse_tool_call_strict,
)
from android_world.env import adb_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import json_action

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pylint: disable=broad-exception-caught
    SentenceTransformer = None

MAX_AGENT_STEPS = 15
EXPLORER_MAI_SYSTEM_PROMPT = """
你是一个手机 GUI-Agent 操作专家。你会收到：任务目标、历史动作、执行反馈、探索线索、当前截图。
请输出“下一步唯一动作”，坐标使用 0-1000 归一化空间（左上角原点，x 向右，y 向下）。

动作空间（GELAB）：
1. CLICK：action:CLICK\tpoint:x,y
2. TYPE：action:TYPE\tvalue:输入文本\tpoint:x,y
3. COMPLETE：action:COMPLETE\treturn:最终回复
4. WAIT：action:WAIT\tvalue:秒数
5. AWAKE：action:AWAKE\tvalue:应用名
6. INFO：action:INFO\tvalue:提问内容
7. ABORT：action:ABORT\tvalue:原因
8. SLIDE：action:SLIDE\tpoint1:x1,y1\tpoint2:x2,y2
9. LONGPRESS：action:LONGPRESS\tpoint:x,y

首选输出格式（推荐）：
<THINK>简短思考</THINK>
explain:本步目的\taction:动作名\t...参数...\tsummary:本步后简短进展

兼容格式（可选）：
<tool_call>
{"name":"mobile_use","arguments":{"action":"click","coordinate":[x,y]}}
</tool_call>

强约束：
- 只输出一个动作，不要输出多个候选。
- 不要把 point/value/summary 拼进 action 字段。
- action 字段只能是纯动作名（如 CLICK 或 click）。
- 若使用 tool_call JSON，坐标必须放在 coordinate 数组，不要写成 "action":"CLICK\\tpoint:..."
- 优先推动任务完成，避免无效重复动作。
""".strip()




@dataclasses.dataclass(frozen=True)
class ExplorerHint:
    index: int
    score: float
    label: str


@dataclasses.dataclass
class CandidateScore:
    index: int
    key: str
    text: str
    score: float
    task_similarity: float
    runtime_similarity: float
    similarity: float
    visits: float
    is_clickable: bool




class ExplorerElementAgent(base_agent.EnvironmentInteractingAgent):
    """AndroidWorld agent with parallel threaded exploration."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        vllm: Any,
        name: str = "ExplorerElementAgent",
        max_history: int = 10,
        max_hints: int = 6,
        no_effect_delta_threshold: float = 1.2,
        force_explore_after_repeats: int = 2,
        explore_max_steps: int = 14,
        explore_max_depth: int = 2,
        explore_leaf_width: int = 4,
        explore_max_branches: int | None = None,
        rollback_backtrack_limit: int = 2,
        explore_action_pause_sec: float = 0.25,
        reasoning_sleep_sec: float = 20.0,
        embed_model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
        verbose_step_logs: bool = True,
        reasoning_preview_chars: int = 180,
        log_full_ui_tree_every_n_steps: int = 0,
        model_coordinate_mode: str = "1000",
        explore_mask_output_dir: str = "explore_k1_masks",
        save_explore_masks: bool = True,
        trace_output_dir: str = "explorer_traces",
        prompt_ui_element_limit: int | None = None,
        enable_parallel_exploration: bool = True,
        explore_warmup_timeout_sec: float = 0.0,
        explore_min_actions_before_reasoning: int = 0,
        explore_budget_boost_per_zero_step: int = 2,
        explore_budget_boost_max: int = 8,
        targeted_exploration: bool = True,
        explore_bootstrap_steps: int = 2,
        explore_periodic_interval: int = 4,
        explore_read_task_step_cap: int = 2,
        explore_stuck_action_repeat: int = 2,
        explore_cooldown_after_safe_mode: int = 1,
        strict_json_reprompt_retries: int = 1,
        text_verify_retry_limit: int = 2,
        structured_edit_disable_explore: bool = True,
    ):
        super().__init__(env, name)
        # Backward-compatible arg retained to avoid breaking older configs.
        _ = prompt_ui_element_limit
        self.vllm = vllm
        self.max_history = max_history
        self.max_hints = max_hints
        self.no_effect_delta_threshold = no_effect_delta_threshold
        self.force_explore_after_repeats = force_explore_after_repeats

        self.explore_max_steps = max(1, int(explore_max_steps))
        self.explore_max_depth = max(1, int(explore_max_depth))
        self.explore_leaf_width = max(1, int(explore_leaf_width))
        self.explore_max_branches = explore_max_branches
        self.rollback_backtrack_limit = max(1, int(rollback_backtrack_limit))
        self.explore_action_pause_sec = max(0.05, float(explore_action_pause_sec))
        # Simulate slow reasoning so parallel exploration can accumulate signal.
        self.reasoning_sleep_sec = max(0.0, float(reasoning_sleep_sec))
        self.embed_model_name = embed_model_name
        self.verbose_step_logs = bool(verbose_step_logs)
        self.reasoning_preview_chars = max(60, int(reasoning_preview_chars))
        self.log_full_ui_tree_every_n_steps = max(0, int(log_full_ui_tree_every_n_steps))
        requested_mode = str(model_coordinate_mode or "1000").strip().lower()
        # Avoid ambiguous absolute-vs-1000 guessing by default.
        if requested_mode == "auto":
            requested_mode = "1000"
        self.model_coordinate_mode = requested_mode
        self.enable_parallel_exploration = bool(enable_parallel_exploration)
        self.save_explore_masks = bool(save_explore_masks)
        self.explore_mask_output_dir = str(explore_mask_output_dir or "explore_k1_masks").strip()
        if self.save_explore_masks and self.explore_mask_output_dir:
            try:
                os.makedirs(self.explore_mask_output_dir, exist_ok=True)
            except Exception:  # pylint: disable=broad-exception-caught
                self.save_explore_masks = False
        self.trace_output_dir = str(trace_output_dir or "explorer_traces").strip()
        if self.trace_output_dir:
            try:
                os.makedirs(self.trace_output_dir, exist_ok=True)
            except Exception:  # pylint: disable=broad-exception-caught
                self.trace_output_dir = ""

        self.actions: list[dict[str, Any]] = []
        self.history: list[str] = []
        self._reasoning_page_records: list[dict[str, Any]] = []
        self._recent_indices: deque[int] = deque(maxlen=50)
        self._last_pixels: np.ndarray | None = None
        self._last_action_text: str = ""
        self._last_action_effect: dict[str, Any] = {}
        self._no_effect_repeat = 0
        self._execution_feedback = ""

        self._ui_lock = threading.Lock()
        self._rollback_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._explore_stop_event = threading.Event()
        self._explore_progress_event = threading.Event()
        self._explore_thread: threading.Thread | None = None
        self._explore_thread_stop_clean: bool = True
        self._explore_action_count = 0
        self._explore_action_count_lock = threading.Lock()
        self._last_explore_action_ts: float = 0.0

        self._explore_iteration_candidates: list[dict[str, Any]] = []
        self._pending_explore_payload: dict[str, Any] | None = None
        self._last_clue_debug: dict[str, Any] = {}

        self._explore_root_hash: int | None = None
        self._explore_root_pixels: np.ndarray | None = None
        self._explore_root_activity: str | None = None
        self._replay_action_history: list[dict[str, Any]] = []
        self._reasoning_action_history: list[dict[str, Any]] = []
        self._branch_action_history: list[dict[str, Any]] = []

        self._clicked_bounds: set[str] = set()
        self._bound_visit_count: dict[str, float] = {}
        self._bound_effect_ema: dict[str, float] = {}
        self._bound_effect_count: dict[str, int] = {}
        self._bound_seen_count: dict[str, int] = {}
        self._bound_skip_count: dict[str, int] = {}
        self._recent_clicked_bounds: deque[str] = deque(maxlen=128)
        self._recent_clicked_regions: deque[str] = deque(maxlen=128)
        self._recent_clicked_window = 12
        self._last_filter_stats: dict[str, Any] = {}

        self._emb_cache: dict[str, np.ndarray] = {}
        self._embed_model = None
        self._embed_ready = False

        self._task_goal: str | None = None
        self._task_start_ts: float | None = None
        self._task_trace_dir: str | None = None
        self._task_step_latencies: list[float] = []
        self.targeted_exploration = bool(targeted_exploration)
        self.explore_bootstrap_steps = max(0, int(explore_bootstrap_steps))
        self.explore_periodic_interval = max(0, int(explore_periodic_interval))
        self.explore_read_task_step_cap = max(0, int(explore_read_task_step_cap))
        self.explore_stuck_action_repeat = max(2, int(explore_stuck_action_repeat))
        self.explore_cooldown_after_safe_mode = max(0, int(explore_cooldown_after_safe_mode))
        self.explore_warmup_timeout_sec = max(0.0, float(explore_warmup_timeout_sec))
        self.explore_min_actions_before_reasoning = max(0, int(explore_min_actions_before_reasoning))
        self.explore_budget_boost_per_zero_step = max(0, int(explore_budget_boost_per_zero_step))
        self.explore_budget_boost_max = max(0, int(explore_budget_boost_max))
        self.strict_json_reprompt_retries = max(0, int(strict_json_reprompt_retries))
        self.text_verify_retry_limit = max(1, int(text_verify_retry_limit))
        self.structured_edit_disable_explore = bool(structured_edit_disable_explore)
        self._explore_trigger_reason: str = ""
        self._explore_cooldown_steps: int = 0
        self._structured_recovery_used: bool = False
        self._title_edit_retry_failures: int = 0
        self.rollback_fail_explore_cooldown_steps: int = 3

    def set_max_steps(self, max_steps: int) -> None:
        super().set_max_steps(min(MAX_AGENT_STEPS, int(max_steps)))

    def _effective_max_steps(self) -> int:
        if self._max_steps is None:
            return MAX_AGENT_STEPS
        return min(MAX_AGENT_STEPS, int(self._max_steps))

    def _emit_log(self, message: str, tag: str = "EXPLORE") -> None:
        if not self.verbose_step_logs:
            return
        stamp = time.strftime("%H:%M:%S")
        with self._log_lock:
            print(f"[{tag} {stamp}] {self._humanize_log_message(message)}")

    def _emit_log_block(self, title: str, content: str, tag: str = "REASON") -> None:
        if not self.verbose_step_logs:
            return
        stamp = time.strftime("%H:%M:%S")
        with self._log_lock:
            print(f"[{tag} {stamp}] {self._humanize_log_message(title)} BEGIN")
            print(content if content else "<EMPTY>")
            print(f"[{tag} {stamp}] {self._humanize_log_message(title)} END")

    @staticmethod
    def _humanize_log_message(message: str) -> str:
        """Convert compact key=value fragments into easier-to-read text."""
        text = _normalize_space(message)
        if not text or "=" not in text:
            return text

        out_tokens: list[str] = []
        for tok in text.split(" "):
            if "=" not in tok:
                out_tokens.append(tok)
                continue
            key, value = tok.split("=", 1)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                out_tokens.append(f"{key.replace('_', ' ')}: {value}")
            else:
                out_tokens.append(tok)
        return " ".join(out_tokens)

    def _step_separator(
        self,
        step_no: int,
        phase: str,
        goal: str | None = None,
        summary: str | None = None,
    ) -> None:
        if not self.verbose_step_logs:
            return
        sep = "=" * 96
        with self._log_lock:
            print(sep)
            if phase == "start":
                print(f"[STEP {step_no:03d} START] Goal: {goal}")
            else:
                end_text = self._humanize_log_message(summary or "")
                print(f"[STEP {step_no:03d} END] {end_text}".rstrip())
            print(sep)

    @staticmethod
    def _safe_task_name(goal: str, max_len: int = 72) -> str:
        value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(goal or "").strip())
        value = re.sub(r"_+", "_", value).strip("._")
        if not value:
            value = "task"
        return value[:max_len]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): ExplorerElementAgent._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ExplorerElementAgent._json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _write_json(self, path: str, payload: dict[str, Any]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._json_safe(payload), f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._emit_log(f"write_json_failed path={path} error={exc}", tag="TRACE")

    def _append_jsonl(self, path: str, payload: dict[str, Any]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._json_safe(payload), ensure_ascii=False))
                f.write("\n")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._emit_log(f"append_jsonl_failed path={path} error={exc}", tag="TRACE")

    @staticmethod
    def _save_png(path: str, pixels: np.ndarray) -> None:
        Image.fromarray(np.array(pixels, copy=True)).save(path)

    @staticmethod
    def _element_brief_label(element: Any, max_len: int = 36) -> str:
        text = _normalize_space(getattr(element, "text", ""))
        desc = _normalize_space(getattr(element, "content_description", ""))
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        )
        cls = _normalize_space(getattr(element, "class_name", ""))
        label = text or desc or rid or cls or "unnamed"
        if len(label) > max(12, int(max_len)):
            label = label[: max(12, int(max_len))].rstrip(" ,.;:") + "..."
        return label

    def _compact_page_record(self, state: Any, max_cues: int = 3) -> dict[str, Any]:
        try:
            page_hash = int(_phash_pixels(state.pixels))
        except Exception:  # pylint: disable=broad-exception-caught
            page_hash = -1

        activity = self._foreground_activity_name()
        cues: list[str] = []
        seen: set[str] = set()
        for idx, element in enumerate(list(getattr(state, "ui_elements", None) or [])):
            if not self._is_valid_element(element):
                continue
            label = self._element_brief_label(element)
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            cues.append(f"{idx}:{label}")
            if len(cues) >= max(1, int(max_cues)):
                break
        return {
            "activity": activity,
            "hash": page_hash,
            "cues": cues,
        }

    @staticmethod
    def _hash_distance(lhs_hash: Any, rhs_hash: Any) -> int | None:
        try:
            lhs = int(lhs_hash)
            rhs = int(rhs_hash)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        if lhs < 0 or rhs < 0:
            return None
        return int(_hash_diff(lhs, rhs))

    def _page_alignment_summary(
        self,
        start_page: dict[str, Any],
        current_page: dict[str, Any],
    ) -> dict[str, Any]:
        diff = self._hash_distance(start_page.get("hash"), current_page.get("hash"))
        start_activity = self._normalize_activity_name(start_page.get("activity"))
        current_activity = self._normalize_activity_name(current_page.get("activity"))
        same_activity = bool(start_activity and current_activity and start_activity == current_activity)
        # Same activity allows a looser hash threshold; cross-activity requires stronger evidence.
        matched = bool(diff is not None and ((same_activity and diff <= 18) or diff <= 10))
        return {
            "matched": matched,
            "hash_diff": diff,
            "same_activity": same_activity,
        }

    @staticmethod
    def _page_brief_text(page: dict[str, Any]) -> str:
        activity = _normalize_space(page.get("activity") or "")
        page_hash = page.get("hash")
        cues = page.get("cues") or []
        cue_text = ", ".join([str(x) for x in cues[:2]]) if cues else "no cues"
        return f"{activity} (hash {page_hash}; {cue_text})"

    def _ensure_task_context(self, goal: str) -> None:
        if self._task_goal == goal and self._task_start_ts is not None:
            return
        self._finalize_task_context(status="switch")
        self._task_goal = str(goal)
        self._task_start_ts = time.time()
        self._task_step_latencies = []
        self._structured_recovery_used = False
        self._title_edit_retry_failures = 0
        self._task_trace_dir = None
        if self.trace_output_dir:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            task_name = self._safe_task_name(goal)
            task_dir = os.path.join(self.trace_output_dir, f"{stamp}_{task_name}")
            try:
                os.makedirs(task_dir, exist_ok=True)
                self._task_trace_dir = task_dir
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._emit_log(f"task_trace_dir_create_failed error={exc}", tag="TRACE")
                self._task_trace_dir = None

    def _finalize_task_context(self, status: str = "completed") -> None:
        if self._task_start_ts is None or self._task_goal is None:
            return
        total_sec = float(max(0.0, time.time() - float(self._task_start_ts)))
        steps = len(self._task_step_latencies)
        avg_sec = float(sum(self._task_step_latencies) / steps) if steps > 0 else 0.0
        self._emit_log(
            (
                f"task_summary goal={self._task_goal} status={status} steps={steps} "
                f"total_latency_sec={total_sec:.3f} avg_step_latency_sec={avg_sec:.3f}"
            ),
            tag="INFO",
        )
        if self._task_trace_dir:
            summary_path = os.path.join(self._task_trace_dir, "task_summary.json")
            self._write_json(
                summary_path,
                {
                    "goal": self._task_goal,
                    "status": status,
                    "steps": steps,
                    "total_latency_sec": total_sec,
                    "avg_step_latency_sec": avg_sec,
                    "step_latencies_sec": [float(x) for x in self._task_step_latencies],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        self._task_goal = None
        self._task_start_ts = None
        self._task_trace_dir = None
        self._task_step_latencies = []
        self._structured_recovery_used = False
        self._title_edit_retry_failures = 0

    def _save_explore_trace(
        self,
        source_step: int,
        branch_id: int,
        depth: int,
        before_pixels: np.ndarray,
        after_pixels: np.ndarray,
        record: dict[str, Any],
    ) -> None:
        if not self._task_trace_dir:
            return
        base = f"step_{int(source_step):03d}_explore_branch_{int(branch_id):02d}_depth_{int(depth):02d}"
        before_path = os.path.join(self._task_trace_dir, f"{base}_before.png")
        after_path = os.path.join(self._task_trace_dir, f"{base}_after.png")
        json_path = os.path.join(self._task_trace_dir, f"{base}.json")
        try:
            self._save_png(before_path, before_pixels)
            self._save_png(after_path, after_pixels)
            payload = dict(record or {})
            payload["before_image"] = before_path
            payload["after_image"] = after_path
            payload["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._write_json(json_path, payload)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._emit_log(f"save_explore_trace_failed step={source_step} error={exc}", tag="TRACE")

    def _save_step_trace(
        self,
        step_no: int,
        start_pixels: np.ndarray,
        end_pixels: np.ndarray,
        payload: dict[str, Any],
    ) -> None:
        if not self._task_trace_dir:
            return
        base = f"step_{int(step_no):03d}"
        start_img = os.path.join(self._task_trace_dir, f"{base}_start.png")
        end_img = os.path.join(self._task_trace_dir, f"{base}_end.png")
        json_file = os.path.join(self._task_trace_dir, f"{base}.json")
        try:
            self._save_png(start_img, start_pixels)
            self._save_png(end_img, end_pixels)
            out = dict(payload or {})
            out["start_image"] = start_img
            out["end_image"] = end_img
            out["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._write_json(json_file, out)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._emit_log(f"save_step_trace_failed step={step_no} error={exc}", tag="TRACE")

    @staticmethod
    def _normalize_activity_name(activity: str | None) -> str:
        return _normalize_space(activity or "").lower()

    @staticmethod
    def _activity_package_name(activity: str | None) -> str:
        value = _normalize_space(activity or "")
        if not value:
            return ""
        if "/" in value:
            return value.split("/", 1)[0].strip().lower()
        return value.strip().lower()

    def _foreground_activity_name(self) -> str:
        try:
            return str(self.env.foreground_activity_name or "").strip()
        except Exception:  # pylint: disable=broad-exception-caught
            return ""

    @staticmethod
    def _clue_text_snippet(text: str, max_chars: int = 160) -> str:
        value = _normalize_space(text)
        if not value:
            return ""
        if len(value) <= max_chars:
            return value
        return value[: max(24, int(max_chars))].rstrip(" ,.;:") + "..."

    @staticmethod
    def _element_short_label(element: Any) -> str:
        return _element_hint_compact_label(element)

    def _state_page_hint(self, state: Any, max_cues: int = 8) -> str:
        try:
            page_hash = _phash_pixels(state.pixels)
        except Exception:  # pylint: disable=broad-exception-caught
            page_hash = -1

        try:
            foreground_activity = self.env.foreground_activity_name
        except Exception:  # pylint: disable=broad-exception-caught
            foreground_activity = ""
        try:
            logical_size = self.env.logical_screen_size
        except Exception:  # pylint: disable=broad-exception-caught
            logical_size = None
        try:
            orientation = self.env.orientation
        except Exception:  # pylint: disable=broad-exception-caught
            orientation = None

        cues = []
        seen = set()
        all_elements = list(getattr(state, "ui_elements", None) or [])
        for idx, element in enumerate(all_elements):
            if not self._is_valid_element(element):
                continue
            label = f"#{idx}: {self._element_short_label(element)}"
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            cues.append(label)
            if len(cues) >= max(1, int(max_cues)):
                break

        cue_text = "; ".join(cues) if cues else "no_semantic_nodes"
        return (
            f"Activity {foreground_activity}; screen {logical_size}; orientation {orientation}; "
            f"{len(all_elements)} UI elements; hash {page_hash}; cues: {cue_text}"
        )

    # -----------------------------
    # Reset / thread lifecycle
    # -----------------------------
    def reset(self, go_home: bool = False) -> None:
        self._finalize_task_context(status="reset")
        self._stop_explorer_thread()
        super().reset(go_home)
        self.actions.clear()
        self.history.clear()
        self._reasoning_page_records.clear()
        self._recent_indices.clear()
        self._last_pixels = None
        self._last_action_text = ""
        self._last_action_effect = {}
        self._no_effect_repeat = 0
        self._execution_feedback = ""

        self._explore_iteration_candidates = []
        self._pending_explore_payload = None
        self._last_clue_debug = {}
        self._explore_root_hash = None
        self._explore_root_pixels = None
        self._explore_root_activity = None
        self._explore_thread_stop_clean = True
        self._last_explore_action_ts = 0.0
        self._replay_action_history = []
        self._reasoning_action_history = []
        self._branch_action_history = []

        self._clicked_bounds.clear()
        self._bound_visit_count.clear()
        self._bound_effect_ema.clear()
        self._bound_effect_count.clear()
        self._bound_seen_count.clear()
        self._bound_skip_count.clear()
        self._recent_clicked_bounds.clear()
        self._recent_clicked_regions.clear()
        self._last_filter_stats = {}
        self._emb_cache.clear()
        self._embed_model = None
        self._embed_ready = False
        self._task_goal = None
        self._task_start_ts = None
        self._task_trace_dir = None
        self._task_step_latencies = []
        self._explore_trigger_reason = ""
        self._explore_cooldown_steps = 0
        self._structured_recovery_used = False
        self._title_edit_retry_failures = 0

    def _start_explorer_thread(
        self,
        goal: str,
        history_tail: list[str],
        clues_text: str,
        source_step: int,
        trigger_reason: str = "unknown",
    ) -> None:
        if not self.enable_parallel_exploration:
            self._clear_explore_root_baseline()
            return
        self._stop_explorer_thread()
        if not self._explore_thread_stop_clean:
            self._clear_explore_root_baseline()
            self._emit_log(
                f"step={source_step} explorer_thread_start_skipped reason=previous_thread_not_stopped",
                tag="EXPLORE",
            )
            return
        self._explore_stop_event.clear()
        self._explore_progress_event.clear()
        with self._explore_action_count_lock:
            self._explore_action_count = 0
        self._explore_iteration_candidates = []
        self._explore_trigger_reason = str(trigger_reason or "unknown")
        self._clicked_bounds = set()
        self._branch_action_history = []
        self._replay_action_history = list(self._reasoning_action_history)

        root_state = self._capture_explore_root_baseline(
            trigger=f"step_{source_step}_{self._explore_trigger_reason}"
        )
        if root_state is None:
            self._emit_log(
                f"step={source_step} explorer_thread_start_skipped reason=root_baseline_capture_failed",
                tag="EXPLORE",
            )
            self._clear_explore_root_baseline()
            return
        self._emit_log(
            f"step={source_step} explorer_thread_started trigger={self._explore_trigger_reason} "
            f"root_page=({self._state_page_hint(root_state)})",
            tag="EXPLORE",
        )

        self._explore_thread = threading.Thread(
            target=self._explore_worker,
            args=(goal, history_tail, clues_text, source_step),
            daemon=True,
        )
        self._explore_thread.start()

    def _clear_explore_root_baseline(self) -> None:
        self._explore_root_hash = None
        self._explore_root_pixels = None
        self._explore_root_activity = None

    def _capture_explore_root_baseline(self, trigger: str = "") -> Any | None:
        try:
            with self._ui_lock:
                root_state = self.env.get_state(wait_to_stabilize=False)
            self._explore_root_pixels = np.array(root_state.pixels, copy=True)
            self._explore_root_hash = _phash_pixels(root_state.pixels)
            self._explore_root_activity = self._foreground_activity_name() or None
            self._emit_log(
                (
                    f"trigger={trigger or 'unspecified'} root_baseline_captured "
                    f"activity={self._explore_root_activity} hash={self._explore_root_hash}"
                ),
                tag="ROLLBACK",
            )
            return root_state
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._clear_explore_root_baseline()
            self._emit_log(
                (
                    f"trigger={trigger or 'unspecified'} root_baseline_capture_failed "
                    f"error={exc}"
                ),
                tag="ROLLBACK",
            )
            return None

    def _compute_explore_budget(self, trigger_reason: str | None = None) -> tuple[int, int, int, int]:
        max_depth = max(1, int(self.explore_max_depth))
        max_steps = max(1, int(self.explore_max_steps))
        depth_topk = max(2, int(self.explore_leaf_width) + 3)
        targeted = bool(getattr(self, "targeted_exploration", False))
        if not targeted:
            return max_depth, depth_topk, max_steps, 0

        reason = str(trigger_reason or getattr(self, "_explore_trigger_reason", "") or "default").strip().lower()
        compact_depth = 1
        compact_topk = max(2, min(depth_topk, 3))
        compact_steps = max(2, min(max_steps, 3))

        if reason in {"bootstrap", "read_only_bootstrap", "periodic_probe"}:
            return compact_depth, compact_topk, compact_steps, 0

        if reason in {
            "no_effect_repeat",
            "repeat_loop",
            "page_loop",
            "fallback_recovery",
            "safe_mode_recovery",
            "parse_uncertain",
        }:
            repeats = max(1, int(getattr(self, "_no_effect_repeat", 0)))
            boost_per = max(0, int(getattr(self, "explore_budget_boost_per_zero_step", 0)))
            boost_cap = max(0, int(getattr(self, "explore_budget_boost_max", 0)))
            boost = min(boost_cap, boost_per * repeats)
            stressed_steps = min(max_steps, max(4, compact_steps + boost))
            stressed_depth = min(max_depth, 2)
            stressed_topk = max(compact_topk, min(depth_topk, max(4, int(self.explore_leaf_width) + 1)))
            return stressed_depth, stressed_topk, stressed_steps, boost

        return compact_depth, compact_topk, compact_steps, 0

    def _has_explore_root_baseline(self) -> bool:
        return bool(self._explore_root_hash is not None and self._explore_root_pixels is not None)

    def _get_explore_action_count(self) -> int:
        with self._explore_action_count_lock:
            return int(self._explore_action_count)

    def _stop_explorer_thread(self) -> list[dict[str, Any]]:
        self._explore_stop_event.set()
        self._explore_progress_event.set()
        clean = True
        if self._explore_thread and self._explore_thread.is_alive():
            self._explore_thread.join(timeout=6.0)
            if self._explore_thread.is_alive():
                self._emit_log("explorer_thread_join_timeout waiting_extra=2s", tag="EXPLORE")
                self._explore_thread.join(timeout=2.0)
            if self._explore_thread.is_alive():
                self._emit_log("explorer_thread_still_alive_after_stop", tag="EXPLORE")
                clean = False
        self._explore_thread_stop_clean = clean
        candidates = list(self._explore_iteration_candidates)
        if clean:
            self._explore_thread = None
        self._explore_iteration_candidates = []
        return candidates

    # -----------------------------
    # Embedding / scoring
    # -----------------------------
    def _ensure_embed_model(self) -> bool:
        if self._embed_ready:
            return self._embed_model is not None
        self._embed_ready = True
        if SentenceTransformer is None:
            return False
        try:
            self._embed_model = SentenceTransformer(self.embed_model_name)
        except Exception:  # pylint: disable=broad-exception-caught
            self._embed_model = None
        return self._embed_model is not None

    def _embed_text(self, text: str) -> np.ndarray | None:
        key = _normalize_space(text)
        if not key:
            return None
        if key in self._emb_cache:
            return self._emb_cache[key]
        if not self._ensure_embed_model():
            return None
        try:
            emb = np.asarray(self._embed_model.encode(key), dtype=np.float32)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        self._emb_cache[key] = emb
        return emb

    @staticmethod
    def _cosine_max(goal_vecs: list[np.ndarray], cand_vec: np.ndarray | None) -> float:
        if cand_vec is None or not goal_vecs:
            return 0.0
        cand_norm = float(np.linalg.norm(cand_vec)) + 1e-8
        best = 0.0
        for vec in goal_vecs:
            vec_norm = float(np.linalg.norm(vec)) + 1e-8
            sim = float(np.dot(vec, cand_vec) / (vec_norm * cand_norm))
            if sim > best:
                best = sim
        return best

    @staticmethod
    def _lexical_similarity(queries: list[str], candidate_text: str) -> float:
        if not queries:
            return 0.0
        c_low = candidate_text.lower()
        c_tokens = set(_tokenize(c_low))
        best = 0.0
        for query in queries:
            q_low = query.lower()
            q_tokens = set(_tokenize(q_low))
            if not q_tokens:
                continue
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            jaccard = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
            contains = 0.25 if q_low and (q_low in c_low or c_low in q_low) else 0.0
            score = min(1.0, 0.6 * overlap + 0.4 * jaccard + contains)
            best = max(best, score)
        return float(best)

    def _semantic_similarity(self, queries: list[str], candidate_text: str) -> float:
        if not queries:
            return 0.0
        if self._ensure_embed_model():
            goal_vecs = []
            for query in queries:
                vec = self._embed_text(query)
                if vec is not None:
                    goal_vecs.append(vec)
            if goal_vecs:
                cand_vec = self._embed_text(candidate_text)
                return self._cosine_max(goal_vecs, cand_vec)
        return self._lexical_similarity(queries, candidate_text)

    def _runtime_queries(self, history_tail: list[str], clues_text: str) -> list[str]:
        queries: list[str] = []
        seen: set[str] = set()
        stop_tokens = {
            "action",
            "branch",
            "candidate",
            "possible",
            "effect",
            "semantic",
            "score",
            "region",
            "widget",
            "android",
            "com",
            "dimowner",
            "audiorecorder",
        }

        def _add_query(value: str, max_len: int = 80) -> None:
            text = _normalize_space(value).strip("[]|,")
            if not text:
                return
            if len(text) > max_len:
                text = text[:max_len].rstrip(" ,.;:") + "..."
            key = text.lower()
            if key in seen:
                return
            seen.add(key)
            queries.append(text)

        # Prefer structured action records over free-form history text.
        for entry in (self.actions or [])[-4:]:
            if not isinstance(entry, dict):
                continue
            tool_call = entry.get("tool_call") or {}
            args = tool_call.get("arguments") if isinstance(tool_call, dict) else {}
            if not isinstance(args, dict):
                continue
            action_name = _normalize_space(args.get("action") or args.get("action_type") or "")
            if action_name:
                _add_query(action_name, max_len=28)
            for key in ("text", "app_name", "button", "direction", "status", "goal_status"):
                value = _normalize_space(args.get(key) or "")
                if value:
                    _add_query(value, max_len=64)
            if len(queries) >= 12:
                return queries[:12]

        if history_tail:
            for item in history_tail[-4:]:
                text = _normalize_space(item)
                if not text:
                    continue
                action_match = re.search(r"action=([a-zA-Z_]+)", text, flags=re.IGNORECASE)
                if not action_match:
                    action_match = re.search(r"\[(?:llm|fallback[^\]]*)\]\s*([a-zA-Z_]+)", text, flags=re.IGNORECASE)
                action_name = _normalize_space(action_match.group(1)) if action_match else ""
                if action_name:
                    _add_query(action_name, max_len=28)
                for key in ("text", "app_name", "button", "direction", "status", "goal_status"):
                    field_match = re.search(rf"{key}=([^|]+)", text, flags=re.IGNORECASE)
                    if not field_match:
                        field_match = re.search(rf"{key}\s*:\s*([^|;]+)", text, flags=re.IGNORECASE)
                    if not field_match:
                        continue
                    value = _normalize_space(field_match.group(1)).strip("[]")
                    if not value:
                        continue
                    _add_query(value, max_len=64)
                if len(queries) >= 12:
                    return queries[:12]

        if clues_text:
            clues = str(clues_text)
            keyword_matches = re.findall(r"candidate_keywords=([^\n]+)", clues, flags=re.IGNORECASE)
            if not keyword_matches:
                keyword_matches = re.findall(r"candidate keywords:\s*([^\n]+)", clues, flags=re.IGNORECASE)
            for keyword_line in keyword_matches[:2]:
                for token in keyword_line.split(","):
                    token = _normalize_space(token).lower()
                    if len(token) < 3 or token in stop_tokens or token.isdigit():
                        continue
                    _add_query(token, max_len=28)
                    if len(queries) >= 12:
                        return queries[:12]
            if not keyword_matches:
                trunk_match = re.search(r"trunk_text=([^,\n]+)", clues, flags=re.IGNORECASE)
                if not trunk_match:
                    trunk_match = re.search(r"text:\s*([^\n]+)", clues, flags=re.IGNORECASE)
                if trunk_match:
                    trunk_text = self._clean_clue_text(trunk_match.group(1))
                    if trunk_text:
                        _add_query(trunk_text, max_len=72)
        return queries[:12]

    @staticmethod
    def _task_wants_settings(goal_queries: list[str], runtime_queries: list[str]) -> bool:
        merged = " ".join([*(goal_queries or []), *(runtime_queries or [])]).lower()
        if not merged:
            return False
        tokens = {
            "setting",
            "settings",
            "setup",
            "configuration",
            "config",
            "sample rate",
            "recording format",
            "quality",
            "bitrate",
            "rename",
            "name format",
            "theme",
        }
        return any(token in merged for token in tokens)

    @staticmethod
    def _task_wants_back_navigation(goal_queries: list[str], runtime_queries: list[str]) -> bool:
        merged = " ".join([*(goal_queries or []), *(runtime_queries or [])]).lower()
        if not merged:
            return False
        tokens = {
            "go back",
            "navigate back",
            "back to",
            "return to",
            "previous screen",
            "上一页",
            "返回",
            "回到",
        }
        return any(token in merged for token in tokens)

    @staticmethod
    def _is_settings_like_text(merged_text: str) -> bool:
        text = _normalize_space(merged_text).lower()
        if not text:
            return False
        tokens = {
            "setup",
            "settings",
            "recording format",
            "sample rate",
            "name format",
            "theme",
            "reset",
            "apply",
            "information",
            "setting_btn_info",
            "txt_information",
        }
        return any(token in text for token in tokens)

    @staticmethod
    def _goal_is_read_only_query(goal: str) -> bool:
        text = _normalize_space(goal).lower()
        if not text:
            return False
        query_markers = {
            "what",
            "which",
            "where",
            "when",
            "how many",
            "do i have",
            "is there",
            "is the",
            "answer",
            "verify",
            "check whether",
            "check if",
        }
        action_markers = {
            "create",
            "add",
            "delete",
            "remove",
            "record",
            "turn ",
            "enable",
            "disable",
            "toggle",
            "set ",
            "rename",
            "move",
            "edit",
            "write",
            "type",
            "send",
        }
        has_query = bool("?" in text or any(token in text for token in query_markers))
        has_action = bool(any(token in text for token in action_markers))
        return bool(has_query and not has_action)

    @staticmethod
    def _goal_has_delete_intent(goal: str) -> bool:
        text = _normalize_space(goal).lower()
        if not text:
            return False
        tokens = {
            "delete",
            "remove",
            "clear",
            "discard",
            "erase",
            "trash",
        }
        return any(token in text for token in tokens)

    @staticmethod
    def _goal_has_share_intent(goal: str) -> bool:
        text = _normalize_space(goal).lower()
        if not text:
            return False
        tokens = {
            "share",
            "send",
            "export",
            "attach",
            "forward",
        }
        return any(token in text for token in tokens)

    @staticmethod
    def _infer_target_app(goal: str) -> tuple[str | None, list[str]]:
        text = _normalize_space(goal).lower()
        if not text:
            return None, []
        mappings = [
            ("joplin", "Joplin", ["net.cozic.joplin", "joplin"]),
            ("broccoli", "Broccoli", ["com.flauschcode.broccoli", "broccoli"]),
            ("simple calendar", "Simple Calendar Pro", ["com.simplemobiletools.calendar.pro", "calendar"]),
            ("calendar pro", "Simple Calendar Pro", ["com.simplemobiletools.calendar.pro", "calendar"]),
            ("audio recorder", "Audio Recorder", ["com.dimowner.audiorecorder", "audiorecorder"]),
        ]
        for marker, app_name, hints in mappings:
            if marker in text:
                return app_name, list(hints)
        return None, []

    @staticmethod
    def _is_launcher_activity(activity: str | None) -> bool:
        value = _normalize_space(activity).lower()
        if not value:
            return False
        return bool("launcher" in value or "nexuslauncher" in value or "quickstep" in value)

    @staticmethod
    def _is_in_target_app(activity: str | None, package_hints: list[str]) -> bool:
        value = _normalize_space(activity).lower()
        if not value or not package_hints:
            return False
        return any(_normalize_space(hint).lower() in value for hint in package_hints if _normalize_space(hint))

    @staticmethod
    def _is_android_chooser_activity(activity: str | None) -> bool:
        value = _normalize_space(activity).lower()
        return bool(value and "chooseractivity" in value)

    @staticmethod
    def _build_open_app_action(app_name: str) -> tuple[json_action.JSONAction, dict[str, Any]]:
        safe_name = _normalize_space(app_name)
        return (
            json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=safe_name),
            {
                "name": "mobile_use",
                "arguments": {
                    "action": "open_app",
                    "text": safe_name,
                    "app_name": safe_name,
                },
            },
        )

    @staticmethod
    def _coord_bucket(value: Any, bucket: int = 10) -> str:
        num = _safe_int(value)
        if num is None:
            return "na"
        step = max(1, int(bucket))
        return str(int(round(float(num) / float(step)) * step))

    @staticmethod
    def _action_signature_from_dict(action_dict: dict[str, Any] | None) -> str:
        data = dict(action_dict or {})
        action_type = str(data.get("action_type") or data.get("action") or "").strip().lower()
        if not action_type:
            return ""
        if action_type in {"click", "tap", "long_press", "input_text", "type"}:
            x = ExplorerElementAgent._coord_bucket(data.get("x"))
            y = ExplorerElementAgent._coord_bucket(data.get("y"))
            idx = _safe_int(data.get("index"))
            text = _normalize_space(data.get("text", ""))
            text_key = text.lower()[:16] if text else ""
            return f"{action_type}@{x}:{y}#{idx}:{text_key}"
        if action_type in {"swipe", "scroll"}:
            direction = _normalize_space(data.get("direction", "")).lower()
            return f"{action_type}:{direction}"
        if action_type in {"open_app", "open"}:
            app_name = _normalize_space(data.get("app_name") or data.get("text") or "").lower()
            return f"{action_type}:{app_name[:20]}"
        if action_type in {"navigate_back", "navigate_home", "keyboard_enter", "wait"}:
            return action_type
        return action_type

    @staticmethod
    def _action_signature_from_action(action: json_action.JSONAction) -> str:
        if action is None:
            return ""
        try:
            action_dict = action.as_dict(skip_none=True)
        except Exception:  # pylint: disable=broad-exception-caught
            return ""
        return ExplorerElementAgent._action_signature_from_dict(action_dict)

    def _has_repeated_recent_action_signature(self, signature: str, repeats: int = 2) -> bool:
        if not signature:
            return False
        need = max(2, int(repeats))
        if len(self.actions) < need:
            return False
        tail = self.actions[-need:]
        for entry in tail:
            if not isinstance(entry, dict):
                return False
            sig = self._action_signature_from_dict(entry.get("action_dict") or {})
            if sig != signature:
                return False
        return True

    def _recent_page_hash_stable(self, repeats: int = 2, max_hash_diff: int = 2) -> bool:
        need = max(2, int(repeats))
        if len(self._reasoning_page_records) < need:
            return False
        hashes: list[int] = []
        for record in self._reasoning_page_records[-need:]:
            if not isinstance(record, dict):
                return False
            end_page = record.get("end_page") or {}
            value = end_page.get("hash")
            try:
                hashes.append(int(value))
            except Exception:  # pylint: disable=broad-exception-caught
                return False
        latest = hashes[-1]
        threshold = max(0, int(max_hash_diff))
        for prev in hashes[:-1]:
            if int(_hash_diff(latest, prev)) > threshold:
                return False
        return True

    def _should_start_exploration(
        self,
        step_no: int,
        goal: str,
        current_activity: str | None = None,
    ) -> tuple[bool, str]:
        if not bool(self.enable_parallel_exploration):
            return False, "disabled"
        if not bool(getattr(self, "targeted_exploration", False)):
            return True, "always_on"
        if bool(getattr(self, "structured_edit_disable_explore", True)) and self._is_structured_edit_activity(
            current_activity,
            goal=goal,
        ):
            return False, "structured_edit_mode"

        cooldown = max(0, int(getattr(self, "_explore_cooldown_steps", 0)))
        if cooldown > 0:
            self._explore_cooldown_steps = max(0, cooldown - 1)
            return False, "cooldown"

        read_only = self._goal_is_read_only_query(goal)
        if int(step_no) <= max(0, int(getattr(self, "explore_bootstrap_steps", 0))):
            return True, "read_only_bootstrap" if read_only else "bootstrap"
        if int(getattr(self, "_no_effect_repeat", 0)) >= 1:
            return True, "no_effect_repeat"

        last_source = ""
        if self.actions and isinstance(self.actions[-1], dict):
            last_source = str(self.actions[-1].get("source") or "").lower()
        if "fallback" in last_source:
            return True, "fallback_recovery"
        if "parse" in last_source:
            return True, "parse_uncertain"
        if "safe_mode" in last_source:
            return True, "safe_mode_recovery"

        last_signature = ""
        if self.actions and isinstance(self.actions[-1], dict):
            last_signature = self._action_signature_from_dict(self.actions[-1].get("action_dict") or {})
        repeat_need = max(2, int(getattr(self, "explore_stuck_action_repeat", 2)))
        if last_signature and self._has_repeated_recent_action_signature(last_signature, repeats=repeat_need):
            if self._recent_page_hash_stable(repeats=2, max_hash_diff=2):
                return True, "page_loop"
            return True, "repeat_loop"

        read_cap = max(0, int(getattr(self, "explore_read_task_step_cap", 0)))
        if read_only and int(step_no) > read_cap:
            _, target_hints = self._infer_target_app(goal)
            if self._is_in_target_app(current_activity, target_hints):
                return False, "read_only_skip"

        periodic = max(0, int(getattr(self, "explore_periodic_interval", 0)))
        if periodic > 0 and int(step_no) % periodic == 0:
            return True, "periodic_probe"
        return False, "not_triggered"

    def _should_apply_loop_guard(self, action: json_action.JSONAction) -> bool:
        if action is None:
            return False
        if action.action_type in {
            json_action.STATUS,
            json_action.ANSWER,
            json_action.WAIT,
            json_action.UNKNOWN,
            json_action.NAVIGATE_BACK,
            json_action.NAVIGATE_HOME,
        }:
            return False
        signature = self._action_signature_from_action(action)
        if not signature:
            return False
        repeats = max(2, int(getattr(self, "explore_stuck_action_repeat", 2)))
        if not self._has_repeated_recent_action_signature(signature, repeats=repeats):
            return False
        if int(getattr(self, "_no_effect_repeat", 0)) >= 1:
            return True
        if len(self.actions) < repeats:
            return False
        tail = self.actions[-repeats:]
        unchanged = 0
        for entry in tail:
            if not isinstance(entry, dict):
                continue
            effect = entry.get("action_effect") or {}
            if not bool(effect.get("changed")):
                unchanged += 1
        return bool(unchanged >= repeats)

    def _should_force_open_target_app(
        self,
        step_no: int,
        goal: str,
        current_activity: str | None,
        action: json_action.JSONAction,
    ) -> tuple[bool, str | None]:
        target_name, target_hints = self._infer_target_app(goal)
        if not target_name or not target_hints:
            return False, None
        if self._is_in_target_app(current_activity, target_hints):
            return False, target_name
        if action.action_type in {json_action.STATUS, json_action.ANSWER, json_action.OPEN_APP}:
            return False, target_name
        if not self._is_launcher_activity(current_activity):
            return False, target_name
        if int(step_no) <= max(3, int(getattr(self, "explore_bootstrap_steps", 0)) + 1):
            return True, target_name
        signature = self._action_signature_from_action(action)
        if signature and self._has_repeated_recent_action_signature(signature, repeats=2):
            return True, target_name
        return False, target_name

    def _should_force_back_from_chooser(
        self,
        goal: str,
        current_activity: str | None,
        action: json_action.JSONAction,
    ) -> bool:
        if not self._is_android_chooser_activity(current_activity):
            return False
        if self._goal_has_share_intent(goal):
            return False
        if not self._goal_has_delete_intent(goal):
            return False
        return action.action_type != json_action.NAVIGATE_BACK

    # -----------------------------
    # Candidate collection / scoring
    # -----------------------------
    @staticmethod
    def _is_interactive(element: Any) -> bool:
        return bool(
            getattr(element, "is_clickable", False)
            or getattr(element, "is_editable", False)
            or getattr(element, "is_long_clickable", False)
            or getattr(element, "is_scrollable", False)
        )

    @staticmethod
    def _is_valid_element(element: Any) -> bool:
        visible = bool(getattr(element, "is_visible", True))
        enabled = getattr(element, "is_enabled", True)
        return visible and enabled is not False

    def _element_key(self, index: int, element: Any) -> str:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is not None:
            return f"{int(bbox.x_min)}:{int(bbox.y_min)}:{int(bbox.x_max)}:{int(bbox.y_max)}"
        rid = _normalize_space(getattr(element, "resource_id", "") or getattr(element, "resource_name", ""))
        cls = _normalize_space(getattr(element, "class_name", ""))
        txt = _normalize_space(getattr(element, "text", ""))[:40]
        return f"idx={index}|rid={rid}|cls={cls}|txt={txt}"

    def _element_text(self, index: int, element: Any) -> tuple[str, bool]:
        text = _normalize_space(getattr(element, "text", ""))
        desc = _normalize_space(getattr(element, "content_description", ""))
        hint = _normalize_space(getattr(element, "hint_text", ""))
        rid = _normalize_space(getattr(element, "resource_id", "") or getattr(element, "resource_name", ""))
        cls = _normalize_space(getattr(element, "class_name", ""))
        has_semantic_text = bool(text or desc or hint)
        joined = " | ".join([x for x in [text, desc, hint, rid, cls] if x])
        if not joined:
            joined = f"[icon_only] {self._element_key(index, element)}"
        return joined, has_semantic_text

    @staticmethod
    def _element_merged_text(element: Any) -> str:
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        hint = _normalize_space(getattr(element, "hint_text", "")).lower()
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        return " ".join([x for x in [text, desc, hint, rid, cls] if x]).strip()

    @staticmethod
    def _query_keywords(goal_queries: list[str], runtime_queries: list[str], limit: int = 24) -> list[str]:
        merged = " ".join([*(goal_queries or []), *(runtime_queries or [])])
        out: list[str] = []
        seen: set[str] = set()
        for token in _tokenize(merged):
            if len(token) < 3 or token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= max(1, int(limit)):
                break
        return out

    @staticmethod
    def _query_overlap_score(query_keywords: list[str], merged_text: str) -> float:
        if not query_keywords or not merged_text:
            return 0.0
        hit = 0
        for keyword in query_keywords:
            if keyword and keyword in merged_text:
                hit += 1
        if hit <= 0:
            return 0.0
        denom = max(2.0, min(6.0, float(len(query_keywords))))
        return min(1.0, float(hit) / denom)

    @staticmethod
    def _is_submit_or_dismiss_control(element: Any) -> bool:
        merged = ExplorerElementAgent._element_merged_text(element)
        if not merged:
            return False
        return any(token in merged for token in (_SUBMIT_ACTION_KEYWORDS | _DISMISS_ACTION_KEYWORDS))

    @staticmethod
    def _is_dialog_secondary_action(element: Any) -> bool:
        merged = ExplorerElementAgent._element_merged_text(element)
        if not merged:
            return False
        secondary_tokens = {
            "detail",
            "details",
            "more",
            "learn more",
            "info",
            "information",
            "help",
            "settings",
            "advanced",
            "详情",
            "更多",
            "帮助",
        }
        return any(token in merged for token in secondary_tokens)

    @staticmethod
    def _is_dialog_primary_ack_action(element: Any) -> bool:
        merged = ExplorerElementAgent._element_merged_text(element)
        if not merged:
            return False
        primary_tokens = {
            "ok",
            "okay",
            "confirm",
            "allow",
            "continue",
            "got it",
            "yes",
            "accept",
            "agree",
            "close",
            "dismiss",
            "确定",
            "允许",
            "继续",
            "同意",
            "好的",
            "关闭",
            "我知道了",
        }
        return any(token in merged for token in primary_tokens)

    @staticmethod
    def _is_keyboard_key_like_element(element: Any) -> bool:
        text = _normalize_space(getattr(element, "text", ""))
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        merged = f"{rid} {cls}".strip()
        if any(token in merged for token in _KEYBOARD_HINT_TOKENS):
            return True
        if len(text) == 1 and re.fullmatch(r"[A-Za-z0-9]", text):
            return True
        return False

    @staticmethod
    def _is_choice_option_like(element: Any) -> bool:
        if bool(getattr(element, "is_checkable", False)):
            return True
        merged = ExplorerElementAgent._element_merged_text(element)
        if not merged:
            return False
        return any(token in merged for token in _CHOICE_HINT_TOKENS)

    @staticmethod
    def _is_date_time_picker_like(element: Any) -> bool:
        merged = ExplorerElementAgent._element_merged_text(element)
        if not merged:
            return False
        picker_tokens = {
            "datepicker",
            "timepicker",
            "calendarview",
            "calendar",
            "month",
            "year",
            "numberpicker",
            "date_picker",
            "time_picker",
            "clockface",
        }
        return any(token in merged for token in picker_tokens)

    @staticmethod
    def _is_system_ui_noise_element(element: Any) -> bool:
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        merged = " ".join([rid, cls, text, desc])
        if "com.android.systemui" in rid or "systemui" in merged:
            return True
        noise_tokens = {
            "wifi signal",
            "phone signal",
            "battery",
            "do not disturb",
            "privacy chip",
            "internet",
            "bluetooth",
            "quick settings",
        }
        if any(token in merged for token in noise_tokens):
            return True
        return False

    @staticmethod
    def _is_meaningless_element(element: Any, intent_flags: dict[str, bool] | None = None) -> bool:
        intent_flags = intent_flags or {"input": False, "select": False, "nav": False}
        merged = ExplorerElementAgent._element_merged_text(element)
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        text = _normalize_space(getattr(element, "text", "")).lower()

        if ExplorerElementAgent._is_keyboard_key_like_element(element):
            return not bool(intent_flags.get("input"))
        if ExplorerElementAgent._is_date_time_picker_like(element):
            return True
        if ExplorerElementAgent._is_system_ui_noise_element(element):
            return True
        # Keep only obvious no-op widgets filtered out.
        if "progressbar" in cls:
            return True
        if "seekbar" in cls and not bool(intent_flags.get("select")):
            return True
        if len(text) == 1 and not bool(intent_flags.get("input")) and not bool(
            getattr(element, "is_clickable", False)
        ):
            return True
        if not bool(getattr(element, "is_clickable", False)) and not bool(
            getattr(element, "is_editable", False)
        ) and not bool(getattr(element, "is_scrollable", False)):
            if any(token in merged for token in {"clock", "status", "notification", "title_text"}):
                return True
        return False

    @staticmethod
    def _keyword_normalized_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

    @staticmethod
    def _is_back_navigation_element(element: Any) -> bool:
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        merged = " ".join([text, desc, rid, cls])
        normalized = ExplorerElementAgent._keyword_normalized_text(merged)
        if any(phrase in normalized for phrase in ("navigate up", "go back", "back")):
            return True
        rid_markers = ("btn_back", "navigate_up", "up_button", "nav_back")
        return any(marker in rid for marker in rid_markers)

    def _screen_mode_flags(self, ui_elements: list[Any]) -> dict[str, Any]:
        interactive_total = 0
        keyboard_hits = 0
        dialog_hits = 0
        choice_hits = 0
        for element in ui_elements or []:
            if not self._is_valid_element(element):
                continue
            if not self._is_interactive(element):
                continue
            interactive_total += 1
            merged = self._element_merged_text(element)
            if merged:
                if any(token in merged for token in _KEYBOARD_HINT_TOKENS):
                    keyboard_hits += 1
                if any(token in merged for token in _DIALOG_HINT_TOKENS):
                    dialog_hits += 1
            if self._is_choice_option_like(element):
                choice_hits += 1
        denom = max(1, interactive_total)
        keyboard = bool(keyboard_hits >= 3 or (keyboard_hits >= 2 and (keyboard_hits / denom) >= 0.20))
        dialog = bool(dialog_hits >= 1 and (dialog_hits / denom) >= 0.08)
        choice = bool(choice_hits >= 4 and (choice_hits / denom) >= 0.30)
        return {
            "keyboard": keyboard,
            "dialog": dialog,
            "choice": choice,
            "interactive_total": interactive_total,
            "keyboard_hits": keyboard_hits,
            "dialog_hits": dialog_hits,
            "choice_hits": choice_hits,
        }

    @staticmethod
    def _is_risky_element(element: Any) -> bool:
        merged = " ".join(
            [
                _normalize_space(getattr(element, "text", "")).lower(),
                _normalize_space(getattr(element, "content_description", "")).lower(),
                _normalize_space(getattr(element, "resource_id", "") or getattr(element, "resource_name", "")).lower(),
                _normalize_space(getattr(element, "class_name", "")).lower(),
            ]
        )
        merged = ExplorerElementAgent._keyword_normalized_text(merged)
        # Keep risky filtering narrow: only destructive operations.
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

    @staticmethod
    def _is_critical_risky_element(element: Any) -> bool:
        """Narrow risk gate used by fallback-relaxed candidate recovery."""
        merged = ExplorerElementAgent._element_merged_text(element)
        merged = ExplorerElementAgent._keyword_normalized_text(merged)
        if not merged:
            return False
        critical_patterns = [
            r"\bdelete\b",
            r"\bremove\b",
            r"\bclear all\b",
            r"\berase\b",
            r"\bwipe\b",
            r"\bdiscard\b",
            r"\buninstall\b",
            r"\bfactory reset\b",
        ]
        return any(re.search(pattern, merged) for pattern in critical_patterns)

    def _element_region_label(self, element: Any) -> str:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            return "unknown"
        cx = float(bbox.x_min + bbox.x_max) / 2.0
        cy = float(bbox.y_min + bbox.y_max) / 2.0
        try:
            width, height = self.env.logical_screen_size
            width = max(1.0, float(width))
            height = max(1.0, float(height))
        except Exception:  # pylint: disable=broad-exception-caught
            width = max(1.0, float(bbox.x_max))
            height = max(1.0, float(bbox.y_max))
        horiz = "left" if cx < width / 3.0 else ("right" if cx > 2.0 * width / 3.0 else "center")
        vert = "top" if cy < height / 3.0 else ("bottom" if cy > 2.0 * height / 3.0 else "middle")
        return f"{vert}-{horiz}"

    @staticmethod
    def _is_low_value_explore_element(element: Any) -> bool:
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        merged = " ".join([text, desc, rid, cls])

        class_low_value = any(token in cls for token in _LOW_VALUE_CLASS_HINTS)
        checkable_low_value = bool(getattr(element, "is_checkable", False))
        statusy_text = any(token in merged for token in _LOW_VALUE_TEXT_HINTS)
        has_semantic = bool(text or desc)
        # Typical pitfall: repeatedly toggling checkable/status controls with weak semantics.
        return bool(class_low_value or checkable_low_value or (statusy_text and not has_semantic))

    @staticmethod
    def _relevance_attenuation(task_score: float, reason_score: float) -> float:
        rel = max(float(task_score), float(reason_score))
        if rel >= 0.70:
            return 0.25
        if rel >= 0.55:
            return 0.50
        if rel >= 0.40:
            return 0.75
        return 1.0

    @staticmethod
    def _intent_flags(goal_queries: list[str], runtime_queries: list[str]) -> dict[str, bool]:
        merged_goal = " ".join(goal_queries or []).lower()
        merged_runtime = " ".join(runtime_queries or []).lower()
        merged_for_action = " ".join([merged_goal, merged_runtime]).strip()
        wants_input = any(keyword in merged_for_action for keyword in _TASK_INPUT_KEYWORDS)
        wants_select = any(keyword in merged_for_action for keyword in _TASK_SELECT_KEYWORDS)
        # Navigation intent should mostly follow the task goal, not noisy runtime clues.
        wants_nav = any(keyword in merged_goal for keyword in _NAV_HELPFUL_KEYWORDS)
        return {
            "input": bool(wants_input),
            "select": bool(wants_select),
            "nav": bool(wants_nav),
        }

    @staticmethod
    def _navigation_helpfulness(element: Any) -> float:
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        merged = " ".join([text, desc, rid, cls])
        bonus = 0.0
        if any(keyword in merged for keyword in _NAV_HELPFUL_KEYWORDS):
            bonus += 0.22
        if "button" in cls or "tab" in cls or "toolbar" in cls:
            bonus += 0.06
        if bool(getattr(element, "is_scrollable", False)):
            bonus += 0.06
        if bool(getattr(element, "is_clickable", False)) and (text or desc or rid):
            bonus += 0.04
        return min(0.34, float(bonus))

    @staticmethod
    def _is_information_only_element(element: Any) -> bool:
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        merged = " ".join([text, desc, rid])
        if text:
            allowed_chars = set("0123456789:./- \t")
            numeric_like = all(ch in allowed_chars for ch in text)
        else:
            numeric_like = False
        has_info_token = any(token in merged for token in _INFO_ONLY_TEXT_HINTS)
        return bool(numeric_like or has_info_token)

    def _collect_candidates(
        self,
        ui_elements: list[Any],
        safe: bool = True,
        intent_flags: dict[str, bool] | None = None,
        query_keywords: list[str] | None = None,
        avoid_keys: set[str] | None = None,
        hard_avoid: bool = False,
        allow_back_navigation: bool = False,
    ) -> tuple[list[tuple[int, Any]], dict[str, Any]]:
        _ = safe
        intent_flags = intent_flags or {"input": False, "select": False, "nav": False}
        query_keywords = list(query_keywords or [])

        stats: dict[str, Any] = {
            "total": len(ui_elements),
            "interactive_total": 0,
            "filter_level": "similarity_only",
            "screen_flags": {},
            "valid_interactive": 0,
            "removed_visited": 0,
            "removed_risky": 0,
            "removed_meaningless": 0,
            "removed_intent_mismatch": 0,
            "kept_back_escape": 0,
            "hard_avoid_fallback": 0,
            "candidates": 0,
        }
        candidates: list[tuple[int, Any]] = []
        for idx, element in enumerate(ui_elements):
            if not self._is_valid_element(element):
                continue
            if not self._is_interactive(element):
                continue
            stats["valid_interactive"] += 1
            candidates.append((idx, element))
        stats["interactive_total"] = stats["valid_interactive"]
        screen_flags = self._screen_mode_flags(ui_elements)
        stats["screen_flags"] = dict(screen_flags or {})
        if bool(screen_flags.get("dialog")) and candidates:
            dialog_primary: list[tuple[int, Any]] = []
            dialog_controls: list[tuple[int, Any]] = []
            for idx, element in candidates:
                if self._is_critical_risky_element(element):
                    continue
                if self._is_dialog_primary_ack_action(element):
                    dialog_primary.append((idx, element))
                    continue
                if (
                    self._is_submit_or_dismiss_control(element)
                    and not self._is_dialog_secondary_action(element)
                ):
                    dialog_controls.append((idx, element))
            if dialog_primary:
                stats["filter_level"] = "dialog_primary_button"
                stats["dialog_candidates"] = len(dialog_primary)
                candidates = dialog_primary
            elif dialog_controls:
                stats["filter_level"] = "dialog_control_button"
                stats["dialog_candidates"] = len(dialog_controls)
                candidates = dialog_controls

        if not candidates:
            fallback_candidates: list[tuple[int, Any]] = []
            for idx, element in enumerate(ui_elements):
                if not self._is_valid_element(element):
                    continue
                if self._is_interactive(element):
                    continue
                if getattr(element, "bbox_pixels", None) is None:
                    continue
                text = _normalize_space(getattr(element, "text", ""))
                desc = _normalize_space(getattr(element, "content_description", ""))
                hint = _normalize_space(getattr(element, "hint_text", ""))
                if not (text or desc or hint):
                    continue
                if self._is_system_ui_noise_element(element):
                    stats["removed_meaningless"] = int(stats.get("removed_meaningless", 0)) + 1
                    continue
                if self._is_information_only_element(element):
                    stats["removed_meaningless"] = int(stats.get("removed_meaningless", 0)) + 1
                    continue
                if self._is_meaningless_element(element, intent_flags=intent_flags):
                    stats["removed_meaningless"] = int(stats.get("removed_meaningless", 0)) + 1
                    continue
                fallback_candidates.append((idx, element))
            if fallback_candidates:
                stats["filter_level"] = "similarity_with_noninteractive_fallback"
                candidates = fallback_candidates
        if len(candidates) > 1 and query_keywords:
            intent_filtered: list[tuple[int, Any]] = []
            for idx, element in candidates:
                merged = self._element_merged_text(element)
                overlap = self._query_overlap_score(query_keywords, merged)
                nav_bonus = self._navigation_helpfulness(element)
                keep_back = bool(allow_back_navigation and self._is_back_navigation_element(element))
                keep_common_control = bool(self._is_submit_or_dismiss_control(element))
                keep = bool(overlap >= 0.12 or nav_bonus >= 0.22 or keep_back or keep_common_control)
                if keep:
                    if keep_back:
                        stats["kept_back_escape"] = int(stats.get("kept_back_escape", 0)) + 1
                    intent_filtered.append((idx, element))
                else:
                    stats["removed_intent_mismatch"] = int(stats.get("removed_intent_mismatch", 0)) + 1
            if intent_filtered:
                candidates = intent_filtered
                stats["filter_level"] = "targeted_similarity"
        if len(candidates) <= 1:
            stats["candidates"] = len(candidates)
            return candidates, stats

        avoid = set(avoid_keys or set())
        if not avoid:
            stats["candidates"] = len(candidates)
            return candidates, stats

        filtered: list[tuple[int, Any]] = []
        for idx, element in candidates:
            key = self._element_key(idx, element)
            if key in avoid:
                stats["removed_visited"] = int(stats.get("removed_visited", 0)) + 1
                if hard_avoid:
                    continue
            filtered.append((idx, element))

        if filtered:
            stats["candidates"] = len(filtered)
            return filtered, stats

        # Keep progress when all candidates are avoided; fall back to original set.
        stats["hard_avoid_fallback"] = 1
        stats["candidates"] = len(candidates)
        return candidates, stats

    def _score_candidate(
        self,
        index: int,
        element: Any,
        goal_queries: list[str],
        runtime_queries: list[str],
        intent_flags: dict[str, bool] | None = None,
        query_keywords: list[str] | None = None,
    ) -> CandidateScore:
        intent_flags = intent_flags or {"input": False, "select": False, "nav": False}
        query_keywords = list(query_keywords or [])
        key = self._element_key(index, element)
        cand_text, _ = self._element_text(index, element)
        task_similarity = float(self._semantic_similarity(goal_queries, cand_text))
        runtime_similarity = float(self._semantic_similarity(runtime_queries, cand_text)) if runtime_queries else 0.0
        merged_goal = " ".join([*(goal_queries or []), *(runtime_queries or [])]).lower()
        delete_intent = bool(any(token in merged_goal for token in {"delete", "remove", "clear", "erase", "trash"}))
        query_overlap = float(self._query_overlap_score(query_keywords, self._element_merged_text(element)))
        similarity = float(max(task_similarity, runtime_similarity, min(1.0, query_overlap * 0.92)))
        visits = float(self._bound_visit_count.get(key, 0.0))
        effect_ema = self._bound_effect_ema.get(key)
        low_effect_penalty = 0.0
        if effect_ema is not None and float(effect_ema) <= float(self.no_effect_delta_threshold) * 1.5:
            low_effect_penalty = 0.10
        repeat_penalty = min(0.35, float(visits) * 0.07)
        score = max(0.0, float(similarity) - repeat_penalty - low_effect_penalty)
        if query_overlap > 0.0:
            score += min(0.18, query_overlap * 0.18)
        if bool(intent_flags.get("nav")):
            score += float(self._navigation_helpfulness(element)) * 0.35
        if self._is_submit_or_dismiss_control(element):
            score += 0.05
        if delete_intent:
            merged_text = self._element_merged_text(element)
            if any(token in merged_text for token in {"delete", "remove", "trash", "bin", "discard"}):
                score += 0.24
            if any(token in merged_text for token in {"share", "export", "send", "chooser"}):
                score -= 0.36
        score = min(1.0, float(score))
        is_clickable = bool(getattr(element, "is_clickable", False))

        return CandidateScore(
            index=index,
            key=key,
            text=cand_text,
            score=float(score),
            task_similarity=float(task_similarity),
            runtime_similarity=float(runtime_similarity),
            similarity=float(similarity),
            visits=float(visits),
            is_clickable=bool(is_clickable),
        )

    def _pick_topk(
        self,
        ui_elements: list[Any],
        goal_queries: list[str],
        runtime_queries: list[str],
        k: int,
        avoid_keys: set[str] | None = None,
        hard_avoid: bool = False,
        intent_flags: dict[str, bool] | None = None,
    ) -> tuple[list[CandidateScore], int]:
        query_keywords = self._query_keywords(goal_queries, runtime_queries, limit=24)
        allow_back_navigation = self._task_wants_back_navigation(goal_queries, runtime_queries)
        candidates, filter_stats = self._collect_candidates(
            ui_elements,
            safe=True,
            intent_flags=intent_flags,
            query_keywords=query_keywords,
            avoid_keys=avoid_keys,
            hard_avoid=hard_avoid,
            allow_back_navigation=allow_back_navigation,
        )
        self._last_filter_stats = filter_stats
        n_candidates = len(candidates)
        if n_candidates == 0:
            return [], 0
        scored = []
        for idx, element in candidates:
            key = self._element_key(idx, element)
            self._bound_seen_count[key] = int(self._bound_seen_count.get(key, 0)) + 1
            try:
                cand_score = self._score_candidate(
                    idx,
                    element,
                    goal_queries,
                    runtime_queries,
                    intent_flags=intent_flags,
                    query_keywords=query_keywords,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._emit_log(
                    f"_pick_topk score_failed index={idx} key={key} error={exc}",
                    tag="EXPLORE",
                )
                continue
            scored.append(cand_score)
        if not scored:
            return [], n_candidates
        scored.sort(key=lambda item: item.score, reverse=True)
        want = max(1, int(k))
        return scored[:want], n_candidates

    def _pick_topk_relaxed(
        self,
        ui_elements: list[Any],
        goal_queries: list[str],
        runtime_queries: list[str],
        k: int,
        intent_flags: dict[str, bool] | None = None,
    ) -> tuple[list[CandidateScore], int]:
        """Fallback picker used when strict similarity-only filtering returns nothing."""
        picked, n_candidates = self._pick_topk(
            ui_elements=ui_elements,
            goal_queries=goal_queries,
            runtime_queries=runtime_queries,
            k=k,
            avoid_keys=None,
            hard_avoid=False,
            intent_flags=intent_flags,
        )
        if picked:
            return picked, n_candidates

        fallback_scored: list[CandidateScore] = []
        for idx, element in enumerate(ui_elements):
            if not self._is_valid_element(element):
                continue
            if getattr(element, "bbox_pixels", None) is None:
                continue
            text = _normalize_space(getattr(element, "text", ""))
            desc = _normalize_space(getattr(element, "content_description", ""))
            rid = _normalize_space(
                getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
            )
            # Keep weakly semantic nodes as a last resort when interactive metadata is sparse.
            if not self._is_interactive(element) and not (text or desc or rid):
                continue
            try:
                cand_score = self._score_candidate(
                    idx,
                    element,
                    goal_queries,
                    runtime_queries,
                    intent_flags=intent_flags,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            fallback_scored.append(cand_score)

        if not fallback_scored:
            return [], 0
        fallback_scored.sort(key=lambda item: item.score, reverse=True)
        want = max(1, int(k))
        return fallback_scored[:want], len(fallback_scored)

    def _decay_visit_counts(self, decay: float = 0.92) -> None:
        if not self._bound_visit_count:
            return
        new_count = {}
        for bound_key, val in self._bound_visit_count.items():
            decayed = float(val) * float(decay)
            if decayed >= 0.15:
                new_count[bound_key] = decayed
        self._bound_visit_count = new_count
        # Keep effect memory aligned with active explored bounds.
        active_keys = set(new_count.keys())
        for key in list(self._bound_effect_ema.keys()):
            if key not in active_keys:
                self._bound_effect_ema.pop(key, None)
                self._bound_effect_count.pop(key, None)
                self._bound_seen_count.pop(key, None)
                self._bound_skip_count.pop(key, None)

    # -----------------------------
    # Exploration worker / rollback
    # -----------------------------
    def _same_root_page(
        self,
        curr_pixels: np.ndarray,
        curr_activity: str | None = None,
        phash_thr: int = 18,
        mae_thr: float = 14.0,
    ) -> tuple[bool, str]:
        if self._explore_root_pixels is None or self._explore_root_hash is None:
            return False, "missing_root"
        root_activity = self._normalize_activity_name(self._explore_root_activity)
        root_pkg = self._activity_package_name(self._explore_root_activity)
        if curr_activity is None:
            curr_activity = self._foreground_activity_name()
        curr_activity_norm = self._normalize_activity_name(curr_activity)
        curr_pkg = self._activity_package_name(curr_activity)
        different_package = bool(
            root_pkg and curr_pkg and curr_pkg != root_pkg
        )
        different_activity = bool(
            root_activity and curr_activity_norm and curr_activity_norm != root_activity
        )
        try:
            curr_hash = _phash_pixels(curr_pixels)
            phash_diff = _hash_diff(self._explore_root_hash, curr_hash)
            if phash_diff <= int(phash_thr):
                if different_package:
                    return False, f"pkg_mismatch_phash:{phash_diff}"
                if different_activity:
                    # Avoid false-positive rollback verification across different screens
                    # that happen to look visually similar.
                    pass
                else:
                    return True, "phash"
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            mae = _mae_small(self._explore_root_pixels, curr_pixels)
            if different_package and mae <= mae_thr:
                return False, f"pkg_mismatch_mae:{mae:.2f}"
            if different_activity:
                strict_thr = min(float(mae_thr), 11.0)
                if mae <= strict_thr:
                    return True, f"activity_mismatch_mae:{mae:.2f}"
                return False, f"activity_mismatch_mae:{mae:.2f}"
            if mae <= mae_thr:
                return True, f"mae:{mae:.2f}"
            return False, f"mae:{mae:.2f}"
        except Exception:  # pylint: disable=broad-exception-caught
            return False, "mae:err"

    def _rollback_to_root(
        self,
        max_depth: int = 2,
        enable_replay: bool = True,
        trigger: str = "",
    ) -> dict[str, Any]:
        result = {
            "trigger": trigger or "unspecified",
            "success": False,
            "mode": "skip_no_root",
            "back_presses": 0,
            "replayed_actions": 0,
            "matched_by": None,
            "replay_open_resets": [],
        }
        if self._explore_root_hash is None:
            self._emit_log(
                f"trigger={result['trigger']} rollback_skipped reason=no_root_hash",
                tag="ROLLBACK",
            )
            return result
        if self._explore_stop_event.is_set() and str(result["trigger"]).startswith("explore_"):
            result["mode"] = "interrupted"
            self._emit_log(
                f"trigger={result['trigger']} rollback_interrupted by stop_event",
                tag="ROLLBACK",
            )
            return result

        rollback_success = False
        with self._rollback_lock:
            for i in range(max_depth):
                if self._explore_stop_event.is_set() and str(result["trigger"]).startswith("explore_"):
                    result["mode"] = "interrupted"
                    return result
                with self._ui_lock:
                    state = self.env.get_state(wait_to_stabilize=False)
                same, same_reason = self._same_root_page(
                    state.pixels,
                    curr_activity=self._foreground_activity_name(),
                )
                if same:
                    rollback_success = True
                    result["success"] = True
                    result["mode"] = "backtrack"
                    result["matched_by"] = same_reason
                    self._emit_log(
                        f"trigger={result['trigger']} rollback_done back_presses={result['back_presses']} "
                        f"matched_by={same_reason}",
                        tag="ROLLBACK",
                    )
                    break

                self._emit_log(
                    f"trigger={result['trigger']} rollback_back#{i + 1}/{max_depth} "
                    f"from_page=({self._state_page_hint(state)})",
                    tag="ROLLBACK",
                )
                with self._ui_lock:
                    self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
                result["back_presses"] += 1
                time.sleep(0.35)

            if not rollback_success and enable_replay:
                if self._explore_stop_event.is_set() and str(result["trigger"]).startswith("explore_"):
                    result["mode"] = "interrupted"
                    return result
                result["mode"] = "home_replay"
                self._emit_log(
                    f"trigger={result['trigger']} rollback_fallback=navigate_home+replay",
                    tag="ROLLBACK",
                )
                with self._ui_lock:
                    self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_HOME))
                time.sleep(0.8)
                replay_actions = self._replay_action_history or self._reasoning_action_history
                for action_dict in replay_actions:
                    try:
                        action = json_action.JSONAction(**action_dict)
                    except Exception:  # pylint: disable=broad-exception-caught
                        continue
                    if action.action_type == json_action.OPEN_APP:
                        reset_info = self._replay_open_app_with_force_stop(action.app_name)
                        result["replay_open_resets"].append(reset_info)
                    else:
                        with self._ui_lock:
                            self.env.execute_action(action)
                    result["replayed_actions"] += 1
                    if action.action_type == json_action.OPEN_APP:
                        time.sleep(1.2)
                    else:
                        time.sleep(0.35)

                with self._ui_lock:
                    final_state = self.env.get_state(wait_to_stabilize=False)
                same, same_reason = self._same_root_page(
                    final_state.pixels,
                    curr_activity=self._foreground_activity_name(),
                )
                result["success"] = bool(same)
                result["matched_by"] = same_reason
                self._emit_log(
                    f"trigger={result['trigger']} rollback_replay_done success={result['success']} "
                    f"replayed={result['replayed_actions']} matched_by={same_reason} "
                    f"final_page=({self._state_page_hint(final_state)})",
                    tag="ROLLBACK",
                )
            elif not rollback_success:
                result["mode"] = "backtrack_failed"
                self._emit_log(
                    f"trigger={result['trigger']} rollback_failed back_presses={result['back_presses']}",
                    tag="ROLLBACK",
                )

        return result

    def _replay_open_app_with_force_stop(self, app_name: str | None) -> dict[str, Any]:
        """Replay helper: force-stop target app before relaunching it."""
        app_text = str(app_name or "").strip()
        info: dict[str, Any] = {
            "app_name": app_text,
            "force_stop_attempted": False,
            "force_stop_ok": False,
            "force_stop_method": None,
            "force_stop_error": None,
            "launch_ok": False,
            "launch_error": None,
        }
        adb_target = getattr(self.env, "controller", self.env)
        if not app_text:
            info["force_stop_error"] = "empty_app_name"
            self._emit_log(
                f"replay_open_app skipped force-stop: empty app name",
                tag="ROLLBACK",
            )
            return info

        # Prefer official helper first.
        try:
            info["force_stop_attempted"] = True
            closed = adb_utils.close_app(app_text, adb_target, timeout_sec=5)
            if closed is not None:
                info["force_stop_ok"] = True
                info["force_stop_method"] = "close_app"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            info["force_stop_error"] = str(exc)

        # Fallback: derive package and issue explicit force-stop.
        if not info["force_stop_ok"]:
            package_name = ""
            try:
                mapped = str(adb_utils.normalize_app_name(app_text) or "").strip()
                if "/" in mapped:
                    mapped = mapped.split("/", 1)[0]
                package_name = mapped
                if "." not in package_name:
                    activity = adb_utils.get_adb_activity(app_text)
                    if activity:
                        package_name = adb_utils.extract_package_name(activity)
            except Exception:  # pylint: disable=broad-exception-caught
                package_name = ""

            if package_name and "." in package_name:
                try:
                    info["force_stop_attempted"] = True
                    adb_utils.issue_generic_request(
                        ["shell", "am", "force-stop", package_name],
                        adb_target,
                        timeout_sec=5,
                    )
                    info["force_stop_ok"] = True
                    info["force_stop_method"] = f"force_stop:{package_name}"
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    if not info["force_stop_error"]:
                        info["force_stop_error"] = str(exc)

        try:
            adb_utils.launch_app(app_text, adb_target)
            info["launch_ok"] = True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            info["launch_error"] = str(exc)
            with self._ui_lock:
                self.env.execute_action(
                    json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=app_text)
                )
            info["launch_ok"] = True

        self._emit_log(
            (
                f"replay_open_app app={app_text} force_stop_ok={info['force_stop_ok']} "
                f"method={info['force_stop_method']} launch_ok={info['launch_ok']}"
            ),
            tag="ROLLBACK",
        )
        return info

    def _ensure_root_before_reasoning(
        self,
        step_no: int,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        if not self._has_explore_root_baseline():
            verify_result = {
                "success": True,
                "verified": True,
                "attempts": 0,
                "details": [],
                "skipped": True,
                "reason": "missing_root_baseline",
                "matched_by": "missing_root_baseline",
            }
            self._emit_log(
                f"step={step_no} rollback_guard_skipped reason=missing_root_baseline",
                tag="ROLLBACK",
            )
            return verify_result

        def _is_root_stable(checks: int = 2, interval_sec: float = 0.12) -> tuple[bool, list[str]]:
            reasons: list[str] = []
            for idx in range(max(1, int(checks))):
                with self._ui_lock:
                    state = self.env.get_state(
                        wait_to_stabilize=bool(idx + 1 == max(1, int(checks)))
                    )
                same_root, same_reason = self._same_root_page(
                    state.pixels,
                    curr_activity=self._foreground_activity_name(),
                )
                reasons.append(str(same_reason))
                if not same_root:
                    return False, reasons
                if idx + 1 < checks:
                    time.sleep(max(0.0, float(interval_sec)))
            return True, reasons

        verify_result: dict[str, Any] = {
            "success": False,
            "verified": False,
            "attempts": 0,
            "details": [],
        }
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            info = self._rollback_to_root(
                max_depth=self.rollback_backtrack_limit,
                enable_replay=True,
                trigger=f"reasoning_step_{step_no}_post_llm_attempt_{attempt}",
            )
            with self._ui_lock:
                state = self.env.get_state(wait_to_stabilize=False)
            same_root, same_reason = self._same_root_page(
                state.pixels,
                curr_activity=self._foreground_activity_name(),
            )
            detail = {
                "attempt": attempt,
                "rollback": info,
                "same_root": bool(same_root),
                "same_reason": same_reason,
            }
            if same_root:
                stable, stable_reasons = _is_root_stable(checks=2, interval_sec=0.12)
                detail["stable"] = bool(stable)
                detail["stable_reasons"] = stable_reasons
                same_root = bool(stable)
            verify_result["details"].append(detail)
            verify_result["attempts"] = attempt
            if same_root:
                verify_result["success"] = True
                verify_result["verified"] = True
                verify_result["matched_by"] = detail.get("stable_reasons") or same_reason
                break
        return verify_result

    @staticmethod
    def _center_and_bounds(element: Any) -> tuple[list[int] | None, list[int] | None]:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            return None, None
        center = [int((bbox.x_min + bbox.x_max) / 2.0), int((bbox.y_min + bbox.y_max) / 2.0)]
        bounds = [int(bbox.x_min), int(bbox.y_min), int(bbox.x_max), int(bbox.y_max)]
        return center, bounds

    def _candidate_action(self, ui_elements: list[Any], cand: CandidateScore) -> json_action.JSONAction:
        element = ui_elements[cand.index]
        center, _ = self._center_and_bounds(element)
        if bool(getattr(element, "is_clickable", False) or getattr(element, "is_long_clickable", False)):
            if center is not None:
                return json_action.JSONAction(action_type=json_action.CLICK, x=center[0], y=center[1])
            return json_action.JSONAction(action_type=json_action.CLICK, index=cand.index)
        if bool(getattr(element, "is_scrollable", False)):
            return json_action.JSONAction(action_type=json_action.SCROLL, direction="down", index=cand.index)
        if center is not None:
            return json_action.JSONAction(action_type=json_action.CLICK, x=center[0], y=center[1])
        return json_action.JSONAction(action_type=json_action.CLICK, index=cand.index)

    def _execute_action_with_coordinate_priority(self, action: json_action.JSONAction) -> str:
        """Execute coordinate actions directly through adb actuation when possible."""
        if action.action_type in {
            json_action.CLICK,
            json_action.LONG_PRESS,
            json_action.INPUT_TEXT,
        } and action.x is not None and action.y is not None:
            actuation.execute_adb_action(
                action,
                [],
                self.env.logical_screen_size,
                self.env.controller,
            )
            return "actuation_coordinate"
        self.env.execute_action(action)
        return "env_execute_action"

    @staticmethod
    def _is_high_risk_interaction_action(action: json_action.JSONAction) -> bool:
        """Actions that can drift UI significantly when exploration did not stop cleanly."""
        return action.action_type in {
            json_action.CLICK,
            json_action.LONG_PRESS,
            json_action.INPUT_TEXT,
            json_action.SCROLL,
            json_action.SWIPE,
            json_action.OPEN_APP,
        }

    @staticmethod
    def _build_safe_mode_back_action() -> tuple[json_action.JSONAction, dict[str, Any]]:
        return (
            json_action.JSONAction(action_type=json_action.NAVIGATE_BACK),
            {
                "name": "mobile_use",
                "arguments": {"action": "system_button", "button": "back"},
            },
        )

    @staticmethod
    def _is_calendar_or_contact_goal(goal: str) -> bool:
        low = _normalize_space(goal).lower()
        return bool(
            any(token in low for token in ("calendar", "event", "schedule"))
            or any(token in low for token in ("contact", "phone number", "address book"))
        )

    def _is_structured_edit_activity(self, activity: str | None, goal: str = "") -> bool:
        value = self._normalize_activity_name(activity)
        if not value:
            return False
        edit_markers = (
            "editeventactivity",
            "eventeditactivity",
            "editactivity",
            "contacteditor",
            "insertcontactactivity",
            "createcontactactivity",
        )
        if any(marker in value for marker in edit_markers):
            return True
        if not self._is_calendar_or_contact_goal(goal):
            return False
        if any(token in value for token in ("calendar", "contact")) and "edit" in value:
            return True
        return False

    @staticmethod
    def _recover_tool_call_from_malformed_response(response_text: str) -> dict[str, Any] | None:
        text = str(response_text or "")
        if not text:
            return None
        match = re.search(
            r"<tool_call>\s*(.*?)(?:</tool_call>|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        payload = match.group(1).strip() if match else text.strip()
        if not payload:
            return None

        def _extract_action_token() -> str:
            patterns = [
                r'"action(?:_type)?"\s*:\s*"([^"]*)',
                r"\baction(?:_type)?\s*[:=]\s*([A-Za-z_]+)",
                (
                    r"\b("
                    r"click|tap|longpress|long_press|type|input_text|inputtext|swipe|scroll|slide|"
                    r"open_app|openapp|open|awake|system_button|systembutton|back|home|enter|"
                    r"answer|info|complete|terminate|status|abort|wait"
                    r")\b"
                ),
            ]
            for pattern in patterns:
                action_match = re.search(pattern, payload, flags=re.IGNORECASE | re.DOTALL)
                if not action_match:
                    continue
                raw_value = str(action_match.group(1) or "").strip()
                if not raw_value:
                    continue
                word_match = re.search(r"[A-Za-z_]+", raw_value)
                if word_match:
                    return word_match.group(0).lower()
                return raw_value.lower()
            return ""

        def _extract_coord() -> list[int] | None:
            patterns = [
                r"(?:point|coordinate|tap_point|xy)\s*[:=]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
                r"(?:point|coordinate|tap_point|xy)\s*['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
                r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
            ]
            for pattern in patterns:
                coord_match = re.search(pattern, payload, flags=re.IGNORECASE)
                if not coord_match:
                    continue
                try:
                    return [
                        int(round(float(coord_match.group(1)))),
                        int(round(float(coord_match.group(2)))),
                    ]
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
            return None

        def _extract_text_value(*keys: str) -> str:
            for key in keys:
                key_patterns = [
                    rf"{re.escape(key)}\s*[:=]\s*\"([^\"]*)\"",
                    rf"{re.escape(key)}\s*[:=]\s*([^\t\r\n,}}]+)",
                ]
                for pattern in key_patterns:
                    value_match = re.search(pattern, payload, flags=re.IGNORECASE | re.DOTALL)
                    if not value_match:
                        continue
                    value = _normalize_space(value_match.group(1))
                    if value:
                        return value
            return ""

        action_token = _extract_action_token()
        if not action_token:
            return None
        normalized_token = re.sub(r"[^a-z_]+", "", action_token)
        alias_map = {
            "tap": "click",
            "click": "click",
            "longpress": "long_press",
            "long_press": "long_press",
            "type": "type",
            "inputtext": "type",
            "input_text": "type",
            "swipe": "swipe",
            "slide": "swipe",
            "scroll": "scroll",
            "openapp": "open_app",
            "open_app": "open_app",
            "open": "open_app",
            "awake": "open_app",
            "systembutton": "system_button",
            "system_button": "system_button",
            "back": "back",
            "home": "home",
            "enter": "enter",
            "answer": "answer",
            "info": "answer",
            "complete": "terminate",
            "terminate": "terminate",
            "status": "terminate",
            "abort": "terminate",
            "wait": "wait",
        }
        action_name = alias_map.get(normalized_token, "")
        if not action_name:
            return None

        coord = _extract_coord()
        args: dict[str, Any] = {"action": action_name}
        if action_name in {"click", "long_press"}:
            if coord is None:
                return None
            args["coordinate"] = coord
        elif action_name == "type":
            args["text"] = _extract_text_value("text", "value", "content")
            if coord is not None:
                args["coordinate"] = coord
        elif action_name in {"swipe", "scroll"}:
            direction = _extract_text_value("direction").lower() or "down"
            args["direction"] = direction
            if coord is not None:
                args["coordinate"] = coord
        elif action_name == "open_app":
            app_name = _extract_text_value("text", "value", "app_name")
            if app_name:
                args["text"] = app_name
                args["app_name"] = app_name
        elif action_name in {"back", "home", "enter"}:
            args["action"] = "system_button"
            args["button"] = action_name
        elif action_name == "answer":
            args["text"] = _extract_text_value("text", "value", "return", "content")
        elif action_name == "terminate":
            status_value = _extract_text_value("status", "goal_status").lower()
            args["status"] = "fail" if status_value in {"fail", "failed", "infeasible"} else "complete"
            final_text = _extract_text_value("text", "value", "return", "content")
            if final_text:
                args["text"] = final_text
        return {"name": "mobile_use", "arguments": args}

    @staticmethod
    def _strict_json_retry_prompt(previous_response: str, parse_error: str) -> str:
        return (
            "格式错误：上一次输出无法解析。\n"
            "请只返回一个动作，并严格使用以下两种格式之一：\n"
            "A) GELAB键值格式（优先）：\n"
            "<THINK>...</THINK>\n"
            "explain:...\taction:CLICK|TYPE|COMPLETE|WAIT|AWAKE|INFO|ABORT|SLIDE|LONGPRESS\t...参数...\tsummary:...\n"
            "B) tool_call JSON 格式：\n"
            "<tool_call>\n"
            "{\"name\":\"mobile_use\",\"arguments\":{\"action\":\"click|type|swipe|open_app|system_button|answer|terminate\",\"coordinate\":[x,y]}}\n"
            "</tool_call>\n"
            "注意：不要把 point/value/summary 拼接到 action 字段里。\n"
            f"解析错误: {parse_error}\n"
            f"你上一次输出:\n{previous_response}"
        )

    def _reprompt_for_strict_json(
        self,
        messages: list[dict[str, Any]],
        response_text: str,
        parse_error: str,
        step_no: int,
    ) -> str:
        reprompt_messages = list(messages)
        reprompt_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._strict_json_retry_prompt(response_text, parse_error),
                    }
                ],
            }
        )
        self._emit_log(
            f"step={step_no} strict_json_reprompt_triggered error={parse_error}",
            tag="REASON",
        )
        retry_response, _, _ = self.vllm.predict_mm("", [], messages=reprompt_messages)
        self._emit_log_block(
            title=f"step={step_no} strict_json_retry_response",
            content=str(retry_response),
            tag="REASON",
        )
        return str(retry_response)

    def _extract_expected_strings_from_goal(self, goal: str) -> dict[str, str]:
        text = _normalize_space(goal)
        low = text.lower()
        expected: dict[str, str] = {}

        title_match = re.search(
            r"(?:title(?:d)?|named|name(?:d)?)\s+['\"]([^'\"]{2,120})['\"]",
            text,
            flags=re.IGNORECASE,
        )
        if not title_match:
            title_match = re.search(
                r"(?:title(?:d)?|named|name(?:d)?)\s+([A-Za-z0-9][^.,;]{2,80})",
                text,
                flags=re.IGNORECASE,
            )
        if title_match:
            expected["title"] = _normalize_space(title_match.group(1))

        date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if date_match:
            expected["date"] = date_match.group(0)

        time_match = re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", low)
        if not time_match:
            time_match = re.search(r"\b\d{1,2}\s*h\b", low)
        if time_match:
            expected["time"] = _normalize_space(time_match.group(0))

        duration_match = re.search(r"\b(?:duration|for)\s+(\d+\s*(?:min|mins|minutes|hour|hours|h))\b", low)
        if duration_match:
            expected["duration"] = _normalize_space(duration_match.group(1))

        desc_match = re.search(r"(?:description|notes?)\s+['\"]([^'\"]{2,160})['\"]", text, flags=re.IGNORECASE)
        if desc_match:
            expected["description"] = _normalize_space(desc_match.group(1))
        return expected

    def _state_flat_text(self, state: Any, limit: int = 80) -> str:
        parts: list[str] = []
        for element in list(getattr(state, "ui_elements", None) or [])[: max(1, int(limit))]:
            merged = self._element_merged_text(element)
            if merged:
                parts.append(merged)
        return " | ".join(parts).lower()

    def _completion_checkpoint(self, goal: str, state: Any) -> dict[str, Any]:
        activity = self._foreground_activity_name()
        expected = self._extract_expected_strings_from_goal(goal)
        flat = self._state_flat_text(state)
        required_matches: dict[str, bool] = {}
        for key, value in expected.items():
            required_matches[key] = bool(_normalize_space(value).lower() in flat)
        save_action = self._build_save_action_from_state(state)
        in_edit = self._is_structured_edit_activity(activity, goal=goal)
        return {
            "enabled": self._is_calendar_or_contact_goal(goal),
            "activity": activity,
            "in_edit_activity": in_edit,
            "expected_fields": expected,
            "required_matches": required_matches,
            "save_visible": bool(save_action is not None),
            "can_finish": bool((not in_edit) and (save_action is None)),
        }

    def _build_save_action_from_state(
        self, state: Any
    ) -> tuple[json_action.JSONAction, dict[str, Any]] | None:
        tokens = {"save", "done", "ok", "confirm", "apply", "update", "保存", "完成", "确定"}
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        for idx, element in enumerate(ui_elements):
            if not self._is_valid_element(element):
                continue
            if not self._is_interactive(element):
                continue
            merged = self._element_merged_text(element)
            if not merged:
                continue
            if self._is_critical_risky_element(element):
                continue
            if not any(token in merged for token in tokens):
                continue
            center = self._safe_center_from_element(element)
            if center is not None:
                return (
                    json_action.JSONAction(action_type=json_action.CLICK, x=center[0], y=center[1]),
                    {
                        "name": "mobile_use",
                        "arguments": {"action": "click", "element_id": idx, "coordinate": [center[0], center[1]]},
                    },
                )
            return (
                json_action.JSONAction(action_type=json_action.CLICK, index=idx),
                {"name": "mobile_use", "arguments": {"action": "click", "element_id": idx}},
            )
        return None

    def _resolve_input_target_snapshot(
        self,
        state: Any,
        action: json_action.JSONAction,
        fallback_ui_elements: list[Any] | None = None,
    ) -> dict[str, Any]:
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        if not ui_elements and fallback_ui_elements is not None:
            ui_elements = list(fallback_ui_elements or [])

        x = _safe_int(getattr(action, "x", None))
        y = _safe_int(getattr(action, "y", None))
        idx = _safe_int(getattr(action, "index", None))
        if idx is not None and not (0 <= idx < len(ui_elements)):
            idx = None

        def _is_editable_index(index: int | None) -> bool:
            if index is None or index < 0 or index >= len(ui_elements):
                return False
            return bool(getattr(ui_elements[index], "is_editable", False))

        if not _is_editable_index(idx):
            candidate = self._index_from_coordinate(ui_elements=ui_elements, x=x, y=y)
            if _is_editable_index(candidate):
                idx = candidate

        if not _is_editable_index(idx) and x is not None and y is not None:
            nearest_idx = None
            nearest_dist = None
            for i, element in enumerate(ui_elements):
                if not bool(getattr(element, "is_editable", False)):
                    continue
                center = self._safe_center_from_element(element)
                if center is None:
                    continue
                dist = float((center[0] - x) ** 2 + (center[1] - y) ** 2)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = i
            if nearest_idx is not None:
                idx = nearest_idx

        if not _is_editable_index(idx):
            for i, element in enumerate(ui_elements):
                if bool(getattr(element, "is_editable", False)):
                    idx = i
                    break

        element = ui_elements[idx] if idx is not None and 0 <= idx < len(ui_elements) else None
        center = self._safe_center_from_element(element) if element is not None else None
        field_text = _normalize_space(getattr(element, "text", "")) if element is not None else ""
        field_desc = _normalize_space(getattr(element, "content_description", "")) if element is not None else ""
        merged = self._element_merged_text(element) if element is not None else ""

        return {
            "index": idx,
            "center": center,
            "text": field_text,
            "desc": field_desc,
            "merged": merged,
            "ui_elements": ui_elements,
        }

    @staticmethod
    def _detect_text_concatenation_or_no_effect(
        old_text: str,
        observed_text: str,
        target_text: str,
    ) -> tuple[bool, str]:
        old_norm = _normalize_space(old_text).lower()
        observed_norm = _normalize_space(observed_text).lower()
        target_norm = _normalize_space(target_text).lower()
        if not target_norm:
            return False, ""
        if not observed_norm:
            return True, "empty_after_input"
        if observed_norm == target_norm:
            return False, ""
        if old_norm and observed_norm == old_norm:
            return True, "no_effect"
        if old_norm and old_norm != target_norm and old_norm in observed_norm and target_norm in observed_norm:
            return True, "contains_old_and_new"
        if observed_norm.count(target_norm) > 1:
            return True, "target_repeated"
        if old_norm and old_norm in observed_norm and len(observed_norm) > max(len(target_norm), len(old_norm)) + 2:
            return True, "length_growth_unexpected"
        return False, ""

    def _click_context_action_token(
        self,
        state: Any,
        tokens: set[str],
        avoid_tokens: set[str] | None = None,
    ) -> dict[str, Any]:
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        avoid_tokens = avoid_tokens or set()
        for idx, element in enumerate(ui_elements):
            if not self._is_interactive(element):
                continue
            merged = self._element_merged_text(element)
            if not merged:
                continue
            if any(token in merged for token in avoid_tokens):
                continue
            if not any(token in merged for token in tokens):
                continue
            center = self._safe_center_from_element(element)
            try:
                with self._ui_lock:
                    if center is not None:
                        self._execute_action_with_coordinate_priority(
                            json_action.JSONAction(action_type=json_action.CLICK, x=center[0], y=center[1])
                        )
                    else:
                        self._execute_action_with_coordinate_priority(
                            json_action.JSONAction(action_type=json_action.CLICK, index=idx)
                        )
                return {"clicked": True, "index": idx, "label": merged}
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return {"clicked": False, "index": idx, "error": str(exc)}
        return {"clicked": False}

    def _issue_delete_keyevents(self, count: int) -> dict[str, Any]:
        adb_target = getattr(self.env, "controller", self.env)
        requested = max(1, min(int(count), 48))
        pressed = 0
        first_error = None
        for _ in range(requested):
            try:
                adb_utils.issue_generic_request(
                    ["shell", "input", "keyevent", "67"],
                    adb_target,
                    timeout_sec=2,
                )
                pressed += 1
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if first_error is None:
                    first_error = str(exc)
                break
            time.sleep(0.02)
        return {"requested": requested, "pressed": pressed, "error": first_error}

    def replace_text_in_focused_field(
        self,
        action: json_action.JSONAction,
        target_text: str,
        target_snapshot: dict[str, Any],
        attempt_no: int = 1,
    ) -> tuple[str | None, Any, dict[str, Any], str | None]:
        expected_text = str(target_text or "")
        snapshot = dict(target_snapshot or {})
        target_idx = _safe_int(snapshot.get("index"))
        center = snapshot.get("center")
        old_text = _normalize_space(snapshot.get("text", ""))
        info: dict[str, Any] = {
            "mode": "replace_text",
            "attempt": int(attempt_no),
            "target_index": target_idx,
            "old_text": old_text,
            "target_text": expected_text,
            "select_all_clicked": False,
        }
        execution_path: str | None = None
        post_state = None
        error: str | None = None

        def _focus_action() -> json_action.JSONAction | None:
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                return json_action.JSONAction(action_type=json_action.CLICK, x=int(center[0]), y=int(center[1]))
            if target_idx is not None:
                return json_action.JSONAction(action_type=json_action.CLICK, index=int(target_idx))
            if action.x is not None and action.y is not None:
                return json_action.JSONAction(action_type=json_action.CLICK, x=int(action.x), y=int(action.y))
            return None

        def _long_press_action() -> json_action.JSONAction | None:
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                return json_action.JSONAction(action_type=json_action.LONG_PRESS, x=int(center[0]), y=int(center[1]))
            if target_idx is not None:
                return json_action.JSONAction(action_type=json_action.LONG_PRESS, index=int(target_idx))
            if action.x is not None and action.y is not None:
                return json_action.JSONAction(action_type=json_action.LONG_PRESS, x=int(action.x), y=int(action.y))
            return None

        try:
            focus_action = _focus_action()
            if focus_action is not None:
                with self._ui_lock:
                    self._execute_action_with_coordinate_priority(focus_action)
                time.sleep(0.08)

            long_action = _long_press_action()
            if long_action is not None:
                with self._ui_lock:
                    self._execute_action_with_coordinate_priority(long_action)
                time.sleep(0.12)

            with self._ui_lock:
                menu_state = self.env.get_state(wait_to_stabilize=False)
            select_res = self._click_context_action_token(
                menu_state,
                tokens={"select all", "全选"},
                avoid_tokens={"delete", "remove", "discard"},
            )
            info["select_all"] = select_res
            info["select_all_clicked"] = bool(select_res.get("clicked"))
            if bool(select_res.get("clicked")):
                time.sleep(0.10)

            delete_count = max(8, len(old_text) + 10)
            delete_info = self._issue_delete_keyevents(delete_count)
            info["delete_keyevents"] = delete_info

            input_action = json_action.JSONAction(
                action_type=json_action.INPUT_TEXT,
                index=(
                    int(target_idx)
                    if target_idx is not None
                    and (action.x is None or action.y is None)
                    else None
                ),
                x=(
                    int(action.x)
                    if action.x is not None
                    else (int(center[0]) if isinstance(center, (list, tuple)) and len(center) >= 2 else None)
                ),
                y=(
                    int(action.y)
                    if action.y is not None
                    else (int(center[1]) if isinstance(center, (list, tuple)) and len(center) >= 2 else None)
                ),
                text=expected_text,
                clear_text=False,
            )
            with self._ui_lock:
                execution_path = self._execute_action_with_coordinate_priority(input_action)
                post_state = self.env.get_state(wait_to_stabilize=False)

            verify = self._verify_input_text_match(post_state, input_action, expected_text)
            info["verify"] = verify
            info["matched"] = bool(verify.get("matched"))
            self._emit_log(
                (
                    f"replace_text attempt={attempt_no} idx={target_idx} "
                    f"old_len={len(old_text)} verify={bool(verify.get('matched'))}"
                ),
                tag="CHECK",
            )
            if not bool(verify.get("matched")):
                error = "replace_text_verify_mismatch"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = str(exc)
            info["error"] = error
        return execution_path, post_state, info, error

    def _verify_input_text_match(
        self,
        state: Any,
        action: json_action.JSONAction,
        expected_text: str,
    ) -> dict[str, Any]:
        expected_norm = _normalize_space(expected_text)
        if not expected_norm:
            return {"matched": True, "expected": "", "observed": []}

        ui_elements = list(getattr(state, "ui_elements", None) or [])
        index = _safe_int(getattr(action, "index", None))
        if index is None:
            index = self._index_from_coordinate(
                ui_elements=ui_elements,
                x=_safe_int(getattr(action, "x", None)),
                y=_safe_int(getattr(action, "y", None)),
            )

        observed: list[str] = []
        candidate_indices: list[int] = []
        if index is not None:
            candidate_indices.extend([index - 1, index, index + 1])
        for idx, element in enumerate(ui_elements):
            if bool(getattr(element, "is_editable", False)):
                candidate_indices.append(idx)

        seen_idx: set[int] = set()
        for idx in candidate_indices:
            if idx in seen_idx or idx < 0 or idx >= len(ui_elements):
                continue
            seen_idx.add(idx)
            element = ui_elements[idx]
            text = _normalize_space(getattr(element, "text", ""))
            desc = _normalize_space(getattr(element, "content_description", ""))
            if text:
                observed.append(text)
            if desc:
                observed.append(desc)
            merged = self._element_merged_text(element)
            if merged and merged not in observed:
                observed.append(merged)

        normalized_observed = [_normalize_space(item) for item in observed if _normalize_space(item)]
        exact_count = sum(1 for item in normalized_observed if item == expected_norm)
        contains_once_count = sum(
            1
            for item in normalized_observed
            if expected_norm in item and item.count(expected_norm) == 1
        )
        primary_observed = ""
        if index is not None and 0 <= index < len(ui_elements):
            element = ui_elements[index]
            primary_observed = (
                _normalize_space(getattr(element, "text", ""))
                or _normalize_space(getattr(element, "content_description", ""))
                or self._element_merged_text(element)
            )
        matched = bool(exact_count > 0 or contains_once_count > 0)
        return {
            "matched": bool(matched),
            "expected": expected_norm,
            "observed": normalized_observed[:12],
            "target_index": index,
            "exact_count": exact_count,
            "contains_once_count": contains_once_count,
            "primary_observed": _normalize_space(primary_observed),
        }

    def _execute_verified_text_input(
        self,
        action: json_action.JSONAction,
        ui_elements: list[Any],
    ) -> tuple[str | None, Any, dict[str, Any], str | None]:
        expected_text = str(getattr(action, "text", "") or "")
        attempts: list[dict[str, Any]] = []
        execution_path: str | None = None
        post_state = None
        error: str | None = None

        if not expected_text:
            try:
                with self._ui_lock:
                    execution_path = self._execute_action_with_coordinate_priority(action)
                    post_state = self.env.get_state(wait_to_stabilize=False)
                return execution_path, post_state, {"attempts": attempts, "skipped": "empty_text"}, None
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return None, None, {"attempts": attempts, "skipped": "empty_text"}, str(exc)

        target_idx = _safe_int(getattr(action, "index", None))
        title_target = False

        attempt_cap = max(1, min(2, int(self.text_verify_retry_limit)))
        for attempt in range(1, attempt_cap + 1):
            attempt_info: dict[str, Any] = {"attempt": attempt}
            try:
                with self._ui_lock:
                    pre_state = self.env.get_state(wait_to_stabilize=False)
                target_snapshot = self._resolve_input_target_snapshot(
                    state=pre_state,
                    action=action,
                    fallback_ui_elements=ui_elements,
                )
                target_idx = _safe_int(target_snapshot.get("index"))
                target_text_before = _normalize_space(target_snapshot.get("text", ""))
                target_merged = _normalize_space(target_snapshot.get("merged", ""))
                attempt_info["target_index"] = target_idx
                attempt_info["before_text"] = target_text_before
                if "title" in target_merged:
                    title_target = True

                if _normalize_space(target_text_before) == _normalize_space(expected_text):
                    attempt_info["skip_reason"] = "already_matched"
                    attempts.append(attempt_info)
                    if title_target:
                        self._title_edit_retry_failures = 0
                    self._emit_log(
                        f"text_edit_skip already_matched idx={target_idx}",
                        tag="CHECK",
                    )
                    return execution_path, pre_state, {"attempts": attempts, "matched": True, "skipped": "already_matched"}, None

                # Never type directly into a non-empty mismatched EditText.
                if target_text_before and _normalize_space(target_text_before) != _normalize_space(expected_text):
                    execution_path, post_state, replace_info, replace_error = self.replace_text_in_focused_field(
                        action=action,
                        target_text=expected_text,
                        target_snapshot=target_snapshot,
                        attempt_no=attempt,
                    )
                    attempt_info["replace"] = replace_info
                    attempts.append(attempt_info)
                    if bool((replace_info.get("verify") or {}).get("matched")):
                        if title_target:
                            self._title_edit_retry_failures = 0
                        return execution_path, post_state, {"attempts": attempts, "matched": True}, None
                    error = replace_error or "replace_text_failed"
                    if title_target:
                        self._title_edit_retry_failures += 1
                    continue

                focus_action = None
                if action.x is not None and action.y is not None:
                    focus_action = json_action.JSONAction(action_type=json_action.CLICK, x=int(action.x), y=int(action.y))
                elif target_idx is not None:
                    focus_action = json_action.JSONAction(action_type=json_action.CLICK, index=int(target_idx))

                with self._ui_lock:
                    if focus_action is not None:
                        self._execute_action_with_coordinate_priority(focus_action)
                    input_action = json_action.JSONAction(
                        action_type=json_action.INPUT_TEXT,
                        index=(
                            int(target_idx)
                            if target_idx is not None and (action.x is None or action.y is None)
                            else None
                        ),
                        x=(int(action.x) if action.x is not None else None),
                        y=(int(action.y) if action.y is not None else None),
                        text=expected_text,
                        clear_text=True,
                    )
                    execution_path = self._execute_action_with_coordinate_priority(input_action)
                    post_state = self.env.get_state(wait_to_stabilize=False)
                verify = self._verify_input_text_match(post_state, input_action, expected_text)
                attempt_info["verify"] = verify
                observed_now = _normalize_space(
                    verify.get("primary_observed")
                    or (verify.get("observed") or [""])[0]
                )
                concat_fail, concat_reason = self._detect_text_concatenation_or_no_effect(
                    old_text=target_text_before,
                    observed_text=observed_now,
                    target_text=expected_text,
                )
                if concat_fail:
                    attempt_info["concat_detected"] = concat_reason
                    replace_exec_path, replace_post_state, replace_info, replace_error = self.replace_text_in_focused_field(
                        action=action,
                        target_text=expected_text,
                        target_snapshot=target_snapshot,
                        attempt_no=attempt,
                    )
                    if replace_exec_path is not None:
                        execution_path = replace_exec_path
                    if replace_post_state is not None:
                        post_state = replace_post_state
                    attempt_info["replace_after_direct"] = replace_info
                    attempts.append(attempt_info)
                    if bool((replace_info.get("verify") or {}).get("matched")):
                        if title_target:
                            self._title_edit_retry_failures = 0
                        return execution_path, post_state, {"attempts": attempts, "matched": True}, None
                    error = replace_error or f"concat_detected:{concat_reason}"
                    if title_target:
                        self._title_edit_retry_failures += 1
                    continue

                attempts.append(attempt_info)
                if bool(verify.get("matched")):
                    if title_target:
                        self._title_edit_retry_failures = 0
                    return execution_path, post_state, {"attempts": attempts, "matched": True}, None
                error = "text_verify_mismatch"
                if title_target:
                    self._title_edit_retry_failures += 1
                    if self._title_edit_retry_failures >= 2:
                        break
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error = str(exc)
                attempt_info["error"] = error
                attempts.append(attempt_info)

        if title_target and self._title_edit_retry_failures >= 2:
            self._execution_feedback = (
                "Title field remains unstable after retries; proceed with other required fields first."
            )
        return execution_path, post_state, {"attempts": attempts, "matched": False, "title_target": title_target}, error

    def _handle_unexpected_delete_dialog(self, goal: str, state: Any) -> tuple[Any, dict[str, Any]]:
        info: dict[str, Any] = {"handled": False}
        if self._goal_has_delete_intent(goal):
            return state, info
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        if not ui_elements:
            return state, info

        merged_page = " ".join(self._element_merged_text(element) for element in ui_elements)
        danger_tokens = {
            "delete",
            "remove",
            "discard",
            "proceed with deletion",
            "are you sure",
            "confirm deletion",
        }
        if not any(token in merged_page for token in danger_tokens):
            return state, info
        flags = self._screen_mode_flags(ui_elements)
        if not bool(flags.get("dialog")):
            return state, info

        cancel_tokens = {
            "cancel",
            "no",
            "keep",
            "not now",
            "back",
            "dismiss",
            "取消",
            "返回",
            "否",
            "保留",
        }
        cancel_idx = None
        cancel_center = None
        for idx, element in enumerate(ui_elements):
            merged = self._element_merged_text(element)
            if not merged:
                continue
            if not any(token in merged for token in cancel_tokens):
                continue
            if not self._is_interactive(element):
                continue
            cancel_idx = idx
            cancel_center = self._safe_center_from_element(element)
            break

        try:
            with self._ui_lock:
                if cancel_center is not None:
                    self._execute_action_with_coordinate_priority(
                        json_action.JSONAction(action_type=json_action.CLICK, x=cancel_center[0], y=cancel_center[1])
                    )
                elif cancel_idx is not None:
                    self._execute_action_with_coordinate_priority(
                        json_action.JSONAction(action_type=json_action.CLICK, index=cancel_idx)
                    )
                else:
                    self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
                recovered_state = self.env.get_state(wait_to_stabilize=False)
            rollback_info = None
            if self._has_explore_root_baseline():
                same_root, _ = self._same_root_page(
                    recovered_state.pixels,
                    curr_activity=self._foreground_activity_name(),
                )
                if not same_root:
                    rollback_info = self._rollback_to_root(
                        max_depth=self.rollback_backtrack_limit,
                        enable_replay=False,
                        trigger="unexpected_delete_dialog_restore",
                    )
                    try:
                        with self._ui_lock:
                            recovered_state = self.env.get_state(wait_to_stabilize=False)
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
            info = {
                "handled": True,
                "reason": "unexpected_delete_dialog_cancelled",
                "cancel_index": cancel_idx,
                "restore_rollback": rollback_info,
            }
            return recovered_state, info
        except Exception as exc:  # pylint: disable=broad-exception-caught
            info = {"handled": False, "error": str(exc)}
            return state, info

    def _attempt_structured_recovery_before_infeasible(
        self,
        goal: str,
        state: Any,
        step_no: int,
    ) -> tuple[json_action.JSONAction, dict[str, Any], dict[str, Any]] | None:
        if self._structured_recovery_used:
            return None
        if not self._is_calendar_or_contact_goal(goal):
            return None

        recovery_meta: dict[str, Any] = {"attempted": True, "step": step_no}
        current_state = state
        current_activity = self._foreground_activity_name()

        if not self._is_structured_edit_activity(current_activity, goal):
            try:
                with self._ui_lock:
                    self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
                    current_state = self.env.get_state(wait_to_stabilize=False)
                recovery_meta["pre_back"] = True
            except Exception as exc:  # pylint: disable=broad-exception-caught
                recovery_meta["pre_back_error"] = str(exc)

        save_pair = self._build_save_action_from_state(current_state)
        self._structured_recovery_used = True
        if save_pair is not None:
            action, tool_call = save_pair
            recovery_meta["recovery_action"] = "save"
            return action, tool_call, recovery_meta

        action, tool_call = self._build_safe_mode_back_action()
        recovery_meta["recovery_action"] = "back"
        return action, tool_call, recovery_meta

    def _action_effect_summary(
        self,
        before_pixels: np.ndarray | None,
        after_pixels: np.ndarray | None,
        before_activity: str | None,
        after_activity: str | None,
    ) -> dict[str, Any]:
        pixel_delta = _pixel_delta(before_pixels, after_pixels)
        hash_diff = None
        try:
            if before_pixels is not None and after_pixels is not None:
                hash_diff = int(_hash_diff(_phash_pixels(before_pixels), _phash_pixels(after_pixels)))
        except Exception:  # pylint: disable=broad-exception-caught
            hash_diff = None

        before_norm = self._normalize_activity_name(before_activity)
        after_norm = self._normalize_activity_name(after_activity)
        activity_changed = bool(before_norm and after_norm and before_norm != after_norm)
        changed = bool(
            activity_changed
            or (hash_diff is not None and hash_diff >= 3)
            or (pixel_delta is not None and pixel_delta > self.no_effect_delta_threshold)
        )
        return {
            "changed": changed,
            "pixel_delta": pixel_delta,
            "hash_diff": hash_diff,
            "activity_changed": activity_changed,
            "before_activity": before_activity,
            "after_activity": after_activity,
        }

    def _click_and_record(
        self,
        cand: CandidateScore,
        ui_elements: list[Any],
        before_pixels: np.ndarray,
        n_candidates: int,
        depth: int,
        branch_id: int,
        source_step: int,
    ) -> dict[str, Any] | None:
        if self._explore_stop_event.is_set():
            return None
        action = self._candidate_action(ui_elements, cand)
        before_hash = None
        try:
            before_hash = _phash_pixels(before_pixels)
        except Exception:  # pylint: disable=broad-exception-caught
            before_hash = None
        if self._explore_stop_event.is_set():
            return None
        try:
            with self._ui_lock:
                self._execute_action_with_coordinate_priority(action)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        with self._explore_action_count_lock:
            self._explore_action_count += 1
        self._last_explore_action_ts = time.time()
        self._explore_progress_event.set()

        time.sleep(self.explore_action_pause_sec)
        with self._ui_lock:
            after_state = self.env.get_state(wait_to_stabilize=False)
        after_hash = None
        try:
            after_hash = _phash_pixels(after_state.pixels)
        except Exception:  # pylint: disable=broad-exception-caught
            after_hash = None
        effect_delta = _pixel_delta(before_pixels, after_state.pixels)
        page_hash_diff = (
            int(_hash_diff(before_hash, after_hash))
            if before_hash is not None and after_hash is not None
            else None
        )

        self._clicked_bounds.add(cand.key)
        self._bound_visit_count[cand.key] = float(self._bound_visit_count.get(cand.key, 0.0)) + 1.0
        self._recent_clicked_bounds.append(cand.key)
        if effect_delta is not None:
            prev_ema = self._bound_effect_ema.get(cand.key)
            if prev_ema is None:
                new_ema = float(effect_delta)
            else:
                new_ema = 0.65 * float(prev_ema) + 0.35 * float(effect_delta)
            self._bound_effect_ema[cand.key] = float(new_ema)
            self._bound_effect_count[cand.key] = int(self._bound_effect_count.get(cand.key, 0)) + 1

        element = ui_elements[cand.index]
        self._recent_clicked_regions.append(self._element_region_label(element))
        center, bounds = self._center_and_bounds(element)
        low_value_hit = self._is_low_value_explore_element(element)
        semantic_rel = float(cand.similarity)
        useful_by_change = bool(
            (effect_delta is not None and effect_delta > (self.no_effect_delta_threshold * 1.8))
            or (page_hash_diff is not None and page_hash_diff >= 6)
        )
        useful_by_semantic = bool(semantic_rel >= 0.35)
        is_useful = bool(useful_by_change or useful_by_semantic)
        if low_value_hit and semantic_rel < 0.22 and not useful_by_change:
            is_useful = False
        score_detail = {
            "score": round(cand.score, 4),
            "task_similarity": round(cand.task_similarity, 4),
            "runtime_similarity": round(cand.runtime_similarity, 4),
            "similarity": round(cand.similarity, 4),
            "visits": round(cand.visits, 3),
            "is_clickable": cand.is_clickable,
            "node_txt": cand.text,
            "effect_delta": None if effect_delta is None else round(float(effect_delta), 3),
            "page_hash_diff": page_hash_diff,
            "effect_ema": None if self._bound_effect_ema.get(cand.key) is None else round(float(self._bound_effect_ema[cand.key]), 3),
        }
        record = {
            "best_sim": round(cand.score, 3),
            "score_detail": score_detail,
            "executed_action": action.as_dict(skip_none=True),
            "coordinate": center,
            "bounds": bounds,
            "node_text": getattr(element, "text", None),
            "node_desc": getattr(element, "content_description", None),
            "node_resource_id": getattr(element, "resource_name", None) or getattr(element, "resource_id", None),
            "node_class": getattr(element, "class_name", None),
            "node_match_text": cand.text,
            "n_candidates": n_candidates,
            "depth": depth,
            "branch": branch_id,
            "hash": _phash_pixels(after_state.pixels),
            "page_hint": self._state_page_hint(after_state),
            "action_type": action.action_type,
            "effect_delta": effect_delta,
            "page_hash_diff": page_hash_diff,
            "is_useful": is_useful,
            "useful_by_change": useful_by_change,
            "useful_by_semantic": useful_by_semantic,
            "low_value_hit": low_value_hit,
            "filter_stats": dict(self._last_filter_stats or {}),
        }
        self._save_explore_trace(
            source_step=source_step,
            branch_id=branch_id,
            depth=depth,
            before_pixels=before_pixels,
            after_pixels=after_state.pixels,
            record=record,
        )
        return record

    def _select_depth_candidate(
        self,
        candidates: list[CandidateScore],
        semantic_low: float = 0.20,
        intent_flags: dict[str, bool] | None = None,
        avoid_keys: set[str] | None = None,
        hard_avoid: bool = False,
    ) -> tuple[CandidateScore | None, int]:
        _ = intent_flags
        if not candidates:
            return None, 0
        avoid = set(avoid_keys or set())
        skipped = 0
        pool: list[CandidateScore] = []
        for cand in candidates:
            if cand.key in avoid and hard_avoid:
                skipped += 1
                skip_map = getattr(self, "_bound_skip_count", None)
                if not isinstance(skip_map, dict):
                    skip_map = {}
                    setattr(self, "_bound_skip_count", skip_map)
                skip_map[cand.key] = int(skip_map.get(cand.key, 0)) + 1
                continue
            pool.append(cand)
        if not pool:
            pool = list(candidates)
        semantic_floor = max(0.0, float(semantic_low))
        semantic_pool = [cand for cand in pool if float(cand.similarity) >= semantic_floor]
        if semantic_pool:
            pool = semantic_pool

        recent_keys = set(list(self._recent_clicked_bounds)[-int(self._recent_clicked_window) :])
        best: CandidateScore | None = None
        best_score = -1e9
        for cand in pool:
            adjusted = float(cand.score)
            if cand.key in avoid and not hard_avoid:
                adjusted -= 0.12
            if cand.key in recent_keys:
                adjusted -= 0.18
            if float(cand.visits) > 0.0:
                adjusted -= min(0.25, float(cand.visits) * 0.05)
            effect_ema = self._bound_effect_ema.get(cand.key)
            if effect_ema is not None and float(effect_ema) <= float(self.no_effect_delta_threshold) * 1.5:
                adjusted -= 0.10
            if best is None or adjusted > best_score:
                best = cand
                best_score = adjusted
        return best, skipped

    def _collect_useless_keys_for_mask(self, ui_elements: list[Any]) -> set[str]:
        keys: set[str] = set()
        thr = float(self.no_effect_delta_threshold)
        for idx, element in enumerate(ui_elements):
            key = self._element_key(idx, element)
            seen = int(self._bound_seen_count.get(key, 0))
            skipped = int(self._bound_skip_count.get(key, 0))
            clicked = float(self._bound_visit_count.get(key, 0.0))
            effect_cnt = int(self._bound_effect_count.get(key, 0))
            effect_ema = self._bound_effect_ema.get(key)
            low_value = self._is_low_value_explore_element(element)

            repeated_no_effect = bool(
                clicked >= 2.0
                and effect_cnt >= 2
                and effect_ema is not None
                and float(effect_ema) <= thr * 1.5
            )
            explored_not_clicked = bool((seen >= 6 and clicked < 1.0) or (seen >= 10 and clicked <= 1.0 and skipped >= 4))
            repeated_low_value = bool(low_value and skipped >= 2 and clicked <= 1.0)
            if repeated_no_effect or explored_not_clicked or repeated_low_value:
                keys.add(key)
        return keys

    def _save_k1_masked_snapshot(
        self,
        source_step: int,
        branch_id: int,
        state_pixels: np.ndarray,
        ui_elements: list[Any],
        masked_keys: set[str],
        selected_key: str | None = None,
    ) -> None:
        if not self.save_explore_masks or not self.explore_mask_output_dir:
            return
        if not masked_keys:
            return
        try:
            image = Image.fromarray(np.array(state_pixels, copy=True))
            draw = ImageDraw.Draw(image, "RGBA")
            masked_count = 0
            for idx, element in enumerate(ui_elements):
                bbox = getattr(element, "bbox_pixels", None)
                if bbox is None:
                    continue
                key = self._element_key(idx, element)
                rect = [int(bbox.x_min), int(bbox.y_min), int(bbox.x_max), int(bbox.y_max)]
                if key in masked_keys:
                    draw.rectangle(rect, fill=(0, 0, 0, 150), outline=(255, 70, 70, 230), width=3)
                    masked_count += 1
                elif selected_key is not None and key == selected_key:
                    draw.rectangle(rect, outline=(60, 220, 60, 255), width=4)
            if masked_count <= 0:
                return
            file_name = f"step_{int(source_step):03d}_branch_{int(branch_id):02d}_k1_masked.png"
            out_path = os.path.join(self.explore_mask_output_dir, file_name)
            image.save(out_path)
            self._emit_log(
                f"step={source_step} branch={branch_id} k1_mask_saved path={out_path} masked={masked_count}",
                tag="EXPLORE",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._emit_log(
                f"step={source_step} branch={branch_id} k1_mask_save_failed error={exc}",
                tag="EXPLORE",
            )

    def _explore_worker(
        self,
        goal: str,
        history_tail: list[str],
        clues_text: str,
        source_step: int,
    ) -> None:
        self._decay_visit_counts(decay=0.92)
        goal_queries = _extract_task_queries(goal)
        runtime_queries = self._runtime_queries(history_tail=history_tail, clues_text=clues_text)
        intent_flags = self._intent_flags(goal_queries, runtime_queries)
        wants_settings = self._task_wants_settings(goal_queries, runtime_queries)
        self._emit_log(
            f"step={source_step} explore_begin goal_queries={goal_queries} runtime_queries={runtime_queries} "
            f"intent_flags={intent_flags} wants_settings={wants_settings}",
            tag="EXPLORE",
        )

        budget_reason = str(getattr(self, "_explore_trigger_reason", "") or "default")
        max_depth, depth_topk, max_steps, budget_boost = self._compute_explore_budget(trigger_reason=budget_reason)
        max_branches = self.explore_max_branches
        if max_branches is None:
            max_branches = max(1, max_steps // max_depth)
        self._emit_log(
            f"step={source_step} explore_budget reason={budget_reason} depth={max_depth} topk={depth_topk} "
            f"max_steps={max_steps} boost={budget_boost}",
            tag="EXPLORE",
        )

        steps_used = 0
        branches_done = 0
        self._explore_iteration_candidates = []
        explored_root_keys: set[str] = set()

        while (
            not self._explore_stop_event.is_set()
            and branches_done < max_branches
            and steps_used < max_steps
        ):
            branch_id = branches_done + 1
            self._emit_log(
                f"step={source_step} branch={branch_id} begin steps_used={steps_used}/{max_steps}",
                tag="EXPLORE",
            )

            with self._ui_lock:
                root_state = self.env.get_state(wait_to_stabilize=False)
            same_root, same_reason = self._same_root_page(
                root_state.pixels,
                curr_activity=self._foreground_activity_name(),
            )
            if not same_root:
                rollback_info = self._rollback_to_root(
                    max_depth=self.rollback_backtrack_limit,
                    enable_replay=True,
                    trigger=f"explore_step_{source_step}_branch_{branch_id}_pre_root",
                )
                self._emit_log(
                    f"step={source_step} branch={branch_id} pre_root_rollback={rollback_info}",
                    tag="EXPLORE",
                )
                with self._ui_lock:
                    root_state = self.env.get_state(wait_to_stabilize=False)
                same_root, same_reason = self._same_root_page(
                    root_state.pixels,
                    curr_activity=self._foreground_activity_name(),
                )
            self._emit_log(
                f"step={source_step} branch={branch_id} root_check same_root={same_root} by={same_reason}",
                tag="EXPLORE",
            )
            if not same_root:
                self._emit_log(
                    f"step={source_step} branch={branch_id} root_check_failed_after_recovery -> skip_branch",
                    tag="EXPLORE",
                )
                branches_done += 1
                time.sleep(0.05)
                continue
            branch_max_depth = max_depth
            branch_depth_topk = depth_topk

            root_candidates, n_candidates = self._pick_topk(
                ui_elements=root_state.ui_elements,
                goal_queries=goal_queries,
                runtime_queries=runtime_queries,
                k=max(branch_depth_topk, max_branches + 2),
                avoid_keys=explored_root_keys,
                hard_avoid=True,
                intent_flags=intent_flags,
            )
            root_filter_stats = dict(self._last_filter_stats or {})
            if root_filter_stats:
                self._emit_log(
                    f"step={source_step} branch={branch_id} root_filter_mode={root_filter_stats.get('filter_level')} "
                    f"interactive_total={root_filter_stats.get('interactive_total')} "
                    f"candidates={root_filter_stats.get('candidates')} "
                    f"removed_meaningless={root_filter_stats.get('removed_meaningless')} "
                    f"removed_intent_mismatch={root_filter_stats.get('removed_intent_mismatch')}",
                    tag="EXPLORE",
                )
            if not root_candidates:
                root_candidates, n_candidates = self._pick_topk_relaxed(
                    ui_elements=root_state.ui_elements,
                    goal_queries=goal_queries,
                    runtime_queries=runtime_queries,
                    k=max(branch_depth_topk, max_branches + 2),
                    intent_flags=intent_flags,
                )
                if root_candidates:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} root_relaxed_candidates={len(root_candidates)}",
                        tag="EXPLORE",
                    )
                    root_filter_stats = dict(self._last_filter_stats or {})
                    if root_filter_stats:
                        self._emit_log(
                            f"step={source_step} branch={branch_id} root_relaxed_filter_mode="
                            f"{root_filter_stats.get('filter_level')}",
                            tag="EXPLORE",
                        )
            if not root_candidates:
                self._emit_log(
                    f"step={source_step} branch={branch_id} no_root_candidates remaining",
                    tag="EXPLORE",
                )
                explored_root_keys.clear()
                branches_done += 1
                time.sleep(0.05)
                continue
            root_cand, skipped_root = self._select_depth_candidate(
                root_candidates,
                semantic_low=0.30,
                intent_flags=intent_flags,
                avoid_keys=explored_root_keys,
                hard_avoid=True,
            )
            if root_cand is None:
                self._emit_log(
                    f"step={source_step} branch={branch_id} root_selection_failed",
                    tag="EXPLORE",
                )
                break
            if skipped_root > 0:
                self._emit_log(
                    f"step={source_step} branch={branch_id} root_candidate_filter_skipped="
                    f"{skipped_root}/{len(root_candidates)}",
                    tag="EXPLORE",
                )
            explored_root_keys.add(root_cand.key)

            useless_keys = self._collect_useless_keys_for_mask(root_state.ui_elements)
            self._save_k1_masked_snapshot(
                source_step=source_step,
                branch_id=branch_id,
                state_pixels=root_state.pixels,
                ui_elements=root_state.ui_elements,
                masked_keys=useless_keys,
                selected_key=root_cand.key,
            )

            branch_candidate = {"branch_id": branch_id, "trunk": None, "leaf_observations": []}
            obs_root = self._click_and_record(
                cand=root_cand,
                ui_elements=root_state.ui_elements,
                before_pixels=root_state.pixels,
                n_candidates=n_candidates,
                depth=1,
                branch_id=branch_id,
                source_step=source_step,
            )
            if obs_root is None:
                self._emit_log(
                    f"step={source_step} branch={branch_id} root_click_failed",
                    tag="EXPLORE",
                )
                rollback_info = self._rollback_to_root(
                    max_depth=self.rollback_backtrack_limit,
                    enable_replay=True,
                    trigger=f"explore_step_{source_step}_branch_{branch_id}_after_root_fail",
                )
                self._emit_log(
                    f"step={source_step} branch={branch_id} rollback_result={rollback_info}",
                    tag="EXPLORE",
                )
                branches_done += 1
                time.sleep(0.05)
                continue

            branch_candidate["trunk"] = obs_root
            steps_used += 1
            branch_path_keys = {root_cand.key}
            self._emit_log(
                f"step={source_step} branch={branch_id} depth=1 root_action={obs_root.get('action_type')} "
                f"node={obs_root.get('node_match_text')} score={obs_root.get('best_sim')} "
                f"to_page=({obs_root.get('page_hint')})",
                tag="EXPLORE",
            )

            current_depth = 1
            while (
                current_depth < branch_max_depth
                and steps_used < max_steps
                and not self._explore_stop_event.is_set()
            ):
                with self._ui_lock:
                    state = self.env.get_state(wait_to_stabilize=False)
                depth_candidates, n_candidates = self._pick_topk(
                    ui_elements=state.ui_elements,
                    goal_queries=goal_queries,
                    runtime_queries=runtime_queries,
                    k=branch_depth_topk,
                    avoid_keys=branch_path_keys,
                    hard_avoid=True,
                    intent_flags=intent_flags,
                )
                if not depth_candidates:
                    depth_candidates, n_candidates = self._pick_topk_relaxed(
                        ui_elements=state.ui_elements,
                        goal_queries=goal_queries,
                        runtime_queries=runtime_queries,
                        k=branch_depth_topk,
                        intent_flags=intent_flags,
                    )
                    if depth_candidates:
                        self._emit_log(
                            f"step={source_step} branch={branch_id} depth={current_depth + 1} "
                            f"relaxed_candidates={len(depth_candidates)}",
                            tag="EXPLORE",
                        )
                if not depth_candidates:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} no_candidates",
                        tag="EXPLORE",
                    )
                    break
                selected_cand, skipped_depth = self._select_depth_candidate(
                    depth_candidates,
                    semantic_low=0.35,
                    intent_flags=intent_flags,
                    avoid_keys=branch_path_keys,
                    hard_avoid=True,
                )
                if selected_cand is None:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} selection_failed",
                        tag="EXPLORE",
                    )
                    break
                if skipped_depth > 0:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} "
                        f"candidate_filter_skipped={skipped_depth}/{len(depth_candidates)}",
                        tag="EXPLORE",
                    )
                obs = self._click_and_record(
                    cand=selected_cand,
                    ui_elements=state.ui_elements,
                    before_pixels=state.pixels,
                    n_candidates=n_candidates,
                    depth=current_depth + 1,
                    branch_id=branch_id,
                    source_step=source_step,
                )
                if obs is None:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} click_failed",
                        tag="EXPLORE",
                    )
                    break
                if bool(obs.get("is_useful")):
                    score_detail = obs.get("score_detail") or {}
                    leaf_sim = float(score_detail.get("similarity", 0.0))
                    leaf_score = float(score_detail.get("score", 0.0))
                    useful_by_change = bool(obs.get("useful_by_change"))
                    match_text = _normalize_space(obs.get("node_match_text") or "")
                    looks_like_settings = self._is_settings_like_text(match_text)
                    min_sim = 0.08
                    min_score = 0.06
                    keep_leaf = bool(
                        (leaf_sim >= min_sim or leaf_score >= min_score)
                        or useful_by_change
                    )
                    if keep_leaf:
                        branch_candidate["leaf_observations"].append(obs)
                    else:
                        self._emit_log(
                            f"step={source_step} branch={branch_id} depth={current_depth + 1} "
                            f"discarded_low_relevance_leaf sim={leaf_sim:.3f} score={leaf_score:.3f} "
                            f"useful_by_change={useful_by_change} settings_like={looks_like_settings}",
                            tag="EXPLORE",
                        )
                else:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} discarded_non_useful "
                        f"node={obs.get('node_match_text')} effect_delta={obs.get('effect_delta')} "
                        f"page_hash_diff={obs.get('page_hash_diff')} low_value={obs.get('low_value_hit')}",
                        tag="EXPLORE",
                    )
                self._emit_log(
                    f"step={source_step} branch={branch_id} depth={current_depth + 1} path_action={obs.get('action_type')} "
                    f"node={obs.get('node_match_text')} score={obs.get('best_sim')} "
                    f"to_page=({obs.get('page_hint')})",
                    tag="EXPLORE",
                )
                branch_path_keys.add(selected_cand.key)
                steps_used += 1
                current_depth += 1

            if self._explore_stop_event.is_set():
                self._emit_log(
                    f"step={source_step} branch={branch_id} interrupted_before_rollback",
                    tag="EXPLORE",
                )
                break

            rollback_info = self._rollback_to_root(
                max_depth=max(1, min(self.rollback_backtrack_limit, current_depth)),
                enable_replay=True,
                trigger=f"explore_step_{source_step}_branch_{branch_id}",
            )
            self._emit_log(
                f"step={source_step} branch={branch_id} rollback_result={rollback_info}",
                tag="EXPLORE",
            )
            if not bool(rollback_info.get("success")):
                self._emit_log(
                    f"step={source_step} branch={branch_id} dropped reason=rollback_not_restored",
                    tag="EXPLORE",
                )
                branches_done += 1
                time.sleep(0.05)
                continue
            trunk_obs = branch_candidate.get("trunk") or {}
            trunk_useful = bool(trunk_obs.get("is_useful"))
            has_useful_leaf = bool(branch_candidate.get("leaf_observations"))
            if trunk_obs and (trunk_useful or has_useful_leaf):
                self._explore_iteration_candidates.append(branch_candidate)
                self._emit_log(
                    f"step={source_step} branch={branch_id} recorded "
                    f"trunk_useful={trunk_useful} leaves={len(branch_candidate.get('leaf_observations', []))}",
                    tag="EXPLORE",
                )
            else:
                self._emit_log(
                    f"step={source_step} branch={branch_id} dropped "
                    f"reason=low_value_or_no_signal trunk_useful={trunk_useful} "
                    f"leaves={len(branch_candidate.get('leaf_observations', []))}",
                    tag="EXPLORE",
                )
            branches_done += 1
            time.sleep(0.05)
        candidate_count = len(self._explore_iteration_candidates)
        self._emit_log(
            f"step={source_step} explore_end branches={branches_done} steps_used={steps_used} "
            f"candidates={candidate_count}",
            tag="EXPLORE",
        )

    # -----------------------------
    # Clues
    # -----------------------------
    @staticmethod
    def _extract_keywords(text: str, limit: int = 6) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\\-]{2,}", (text or "").lower())
        stop = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "using",
            "click",
            "button",
            "open",
            "close",
            "page",
            "screen",
            "app",
            "action",
            "task",
            "branch",
            "observed",
            "likely",
        }
        out = []
        seen = set()
        for token in tokens:
            if token in stop or token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _clean_clue_text(text: str) -> str:
        value = _normalize_space(text)
        if not value:
            return ""
        low = value.lower()
        noisy_patterns = [
            "adb shell",
            "content://",
            "[ ]",
            "- [ ]",
            "yyyy-mm-dd",
            "date only",
            "time only",
        ]
        if any(pattern in low for pattern in noisy_patterns):
            return ""
        if len(value) > 180:
            value = value[:180].rstrip(" ,.;:") + "..."
        return value

    @staticmethod
    def _infer_ui_effect(text: str) -> str:
        low = (text or "").lower()
        mapping = [
            (["more options", "menu", "settings"], "open options/menu"),
            (["search"], "open search"),
            (["record", "resume", "stop", "finish"], "control recording state"),
            (["save", "ok", "confirm", "done"], "confirm/save action"),
            (["new", "create", "add", "folder"], "create new item"),
            (["back", "up", "navigate"], "navigate backward"),
        ]
        for keys, effect in mapping:
            if any(key in low for key in keys):
                return effect
        return "possible next action"

    def _region_from_record(self, rec: dict[str, Any]) -> str:
        bounds = rec.get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            x1, y1, x2, y2 = [float(v) for v in bounds]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            coord = rec.get("coordinate")
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                cx, cy = float(coord[0]), float(coord[1])
            else:
                width, height = self.env.logical_screen_size
                cx, cy = width / 2.0, height / 2.0
        width, height = self.env.logical_screen_size
        horiz = "left" if cx < width / 3 else ("right" if cx > 2 * width / 3 else "center")
        vert = "top" if cy < height / 3 else ("bottom" if cy > 2 * height / 3 else "middle")
        if vert == "middle" and horiz == "center":
            return "center"
        return f"{vert}-{horiz}"

    def build_prompt_clues_from_candidates(
        self,
        candidates: list[dict[str, Any]],
        current_pixels: np.ndarray,
        max_items: int = 4,
        last_reasoning_action: str | None = None,
    ) -> str:
        self._last_clue_debug = {
            "status": "start",
            "n_candidates": len(candidates or []),
            "best_diff": None,
            "best_action_hit": None,
            "confidence": None,
            "n_leaves": 0,
            "n_selected": 0,
        }
        if not candidates:
            self._last_clue_debug["status"] = "empty_candidates"
            return ""

        current_hash = _phash_pixels(current_pixels)
        last_action_text = (last_reasoning_action or "").lower().strip()
        ranked = []
        for cand in candidates:
            trunk = cand.get("trunk") or {}
            trunk_hash = trunk.get("hash")
            if trunk_hash is None:
                continue
            try:
                diff = _hash_diff(current_hash, int(trunk_hash))
            except Exception:  # pylint: disable=broad-exception-caught
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
            self._last_clue_debug["status"] = "no_ranked_candidates"
            return ""

        ranked.sort(key=lambda item: item[0])
        top_ranked = ranked[: max(1, min(3, len(ranked)))]
        _, best_diff, best_action_hit, best = top_ranked[0]
        self._last_clue_debug["best_diff"] = int(best_diff)
        self._last_clue_debug["best_action_hit"] = int(best_action_hit)

        if best_diff <= 12:
            confidence = "high"
        elif best_diff <= 24:
            confidence = "medium"
        elif best_diff <= 36:
            confidence = "low"
        else:
            confidence = "very_low"

        self._last_clue_debug["confidence"] = confidence
        trunk = best.get("trunk") or {}
        observations: list[dict[str, Any]] = []
        branch_ids: list[Any] = []
        for _, _, _, cand in top_ranked:
            bid = cand.get("branch_id")
            if bid is not None:
                branch_ids.append(bid)
            leaves = cand.get("leaf_observations") or []
            if leaves:
                observations.extend(leaves)
            else:
                trunk_only = dict(cand.get("trunk") or {})
                if trunk_only:
                    trunk_only["_from_trunk_only"] = True
                    observations.append(trunk_only)
        self._last_clue_debug["n_leaves"] = len(observations)
        if confidence == "very_low" and int(best_action_hit or 0) <= 0:
            self._last_clue_debug["status"] = "very_low_confidence"

        trunk_text = self._clean_clue_text(
            trunk.get("node_text") or trunk.get("node_desc") or trunk.get("node_match_text") or ""
        )
        trunk_text = trunk_text or str(trunk.get("node_match_text") or trunk.get("node_desc") or "")
        trunk_text = self._clue_text_snippet(trunk_text, max_chars=160)
        lines = [
            "[Parallel Exploration Clues]",
            (
                f"- Matched branches: {branch_ids or [best.get('branch_id')]}; "
                f"page hash diff {best_diff}; overlap with previous action {best_action_hit}; "
                f"confidence {confidence}."
            ),
            (
                f"- Entry action {trunk.get('action_type')} at {self._region_from_record(trunk)} "
                f"(coord {trunk.get('coordinate')}, bounds {trunk.get('bounds')}). "
                f"Text: {trunk_text}. Resource: {trunk.get('node_resource_id')}."
            ),
            "- Possible next actions:",
        ]

        goal_text = _normalize_space(self._task_goal or "")
        goal_queries = _extract_task_queries(goal_text) if goal_text else []
        if not goal_queries:
            goal_queries = _extract_task_queries(" ".join(self.history[-3:]))
        scored: list[tuple[float, float, float, dict[str, Any]]] = []
        min_rel = 0.10
        min_prior = 0.05
        for obs in observations:
            text = self._clean_clue_text(
                obs.get("node_text") or obs.get("node_desc") or obs.get("node_match_text") or ""
            )
            if not text:
                continue
            rel = float(self._semantic_similarity(goal_queries, text)) if goal_queries else 0.0
            prior = float((obs.get("score_detail") or {}).get("score", 0.0))
            if prior <= 0.0:
                prior = float(obs.get("best_sim") or 0.0)
            if rel < min_rel and prior < min_prior:
                continue
            # Prefer semantically relevant leaves, fallback to explore score if semantics are weak.
            rank = rel * 0.75 + prior * 0.25
            scored.append((rank, rel, prior, obs))
        if not scored and observations:
            for obs in observations:
                text = self._clean_clue_text(
                    obs.get("node_text") or obs.get("node_desc") or obs.get("node_match_text") or ""
                )
                if not text:
                    continue
                prior = float((obs.get("score_detail") or {}).get("score", 0.0))
                if prior <= 0.0:
                    prior = float(obs.get("best_sim") or 0.0)
                if prior <= 0.0:
                    continue
                scored.append((prior, 0.0, prior, obs))
        scored.sort(key=lambda item: item[0], reverse=True)

        added = 0
        seen = set()
        k2_texts = []
        for _, rel, prior, leaf in scored:
            text = self._clean_clue_text(
                leaf.get("node_text") or leaf.get("node_desc") or leaf.get("node_match_text") or ""
            )
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            pos = self._region_from_record(leaf)
            effect = self._infer_ui_effect(text)
            text = self._clue_text_snippet(text, max_chars=140)
            lines.append(
                (
                    f"{added + 1}. Branch {leaf.get('branch')}: {leaf.get('action_type')} at {pos} "
                    f"(coord {leaf.get('coordinate')}, bounds {leaf.get('bounds')}). "
                    f"Text: {text}. Likely effect: {effect}. "
                    f"Relevance {rel:.3f}, explore score {prior:.3f}."
                )
            )
            k2_texts.append(text)
            added += 1
            if added >= max_items:
                break

        self._last_clue_debug["n_selected"] = int(added)
        self._last_clue_debug["status"] = "ok" if added > 0 else "no_selected_clues"
        if added <= 0:
            return ""

        keywords = self._extract_keywords(" ".join(k2_texts), limit=6)
        if keywords:
            lines.append(f"- Candidate keywords: {', '.join(keywords)}")
        lines.append("")
        out = "\n".join(lines)
        if len(out) > 1400:
            out = out[:1400].rstrip() + "\n"
        return out

    def get_last_clue_debug_lines(self) -> list[str]:
        d = self._last_clue_debug or {}
        return [
            f"[ClueDebug] Status: {d.get('status')}",
            (
                f"[ClueDebug] Candidates {d.get('n_candidates')}, "
                f"Leaves {d.get('n_leaves')}, Selected {d.get('n_selected')}"
            ),
            (
                f"[ClueDebug] Best diff {d.get('best_diff')}, "
                f"Action overlap {d.get('best_action_hit')}, Confidence {d.get('confidence')}"
            ),
        ]

    # -----------------------------
    # Reasoning prompt helpers
    # -----------------------------
    def _build_hints(self, ui_elements: list[Any]) -> list[ExplorerHint]:
        hints: list[ExplorerHint] = []
        for idx, element in enumerate(ui_elements):
            if not self._is_valid_element(element):
                continue
            if not self._is_interactive(element):
                continue

            score = 0.0
            if idx not in self._recent_indices:
                score += 2.0
            if bool(getattr(element, "is_clickable", False)):
                score += 1.2
            if bool(getattr(element, "is_editable", False)):
                score += 1.2
            if bool(getattr(element, "is_long_clickable", False)):
                score += 0.6
            if bool(getattr(element, "is_scrollable", False)):
                score += 0.4
            if getattr(element, "text", ""):
                score += 1.0
            if getattr(element, "content_description", ""):
                score += 0.8

            hints.append(ExplorerHint(index=idx, score=score, label=_element_hint_compact_label(element)))
        hints.sort(key=lambda hint: hint.score, reverse=True)
        return hints[: self.max_hints]

    def _history_text(self) -> str:
        if not self.history:
            return "None yet."
        tail = self.history[-self.max_history :]
        return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(tail))

    @staticmethod
    def _simplify_history_item(item: str) -> str:
        text = _normalize_space(item)
        if not text:
            return "wait"

        action = "wait"
        action_match = re.search(r"action=([a-zA-Z_]+)", text, flags=re.IGNORECASE)
        if not action_match:
            action_match = re.search(r"\[(?:llm|fallback[^\]]*)\]\s*([a-zA-Z_]+)", text, flags=re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip().lower()

        coord_match = re.search(r"coordinate=\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text)
        if not coord_match:
            coord_match = re.search(r"at\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text, flags=re.IGNORECASE)
        coord_text = ""
        if coord_match:
            coord_text = f"[{coord_match.group(1)}, {coord_match.group(2)}]"

        element_match = re.search(r"element_id=(\d+)", text)
        if not element_match:
            element_match = re.search(r"element\s*#(\d+)", text, flags=re.IGNORECASE)
        element_text = f"#{element_match.group(1)}" if element_match else ""

        text_match = re.search(r"text=([^|]+)", text, flags=re.IGNORECASE)
        if not text_match:
            text_match = re.search(r"text\s*\"([^\"]*)\"", text, flags=re.IGNORECASE)
        arg_text = _normalize_space(text_match.group(1)) if text_match else ""
        if not arg_text:
            app_match = re.search(r"app_name=([^|]+)", text, flags=re.IGNORECASE)
            if not app_match:
                app_match = re.search(r"app\s*\"([^\"]*)\"", text, flags=re.IGNORECASE)
            arg_text = _normalize_space(app_match.group(1)) if app_match else ""
        if len(arg_text) > 28:
            arg_text = arg_text[:28].rstrip(" ,.;:") + "..."

        if action in {"click", "long_press", "tap"}:
            if coord_text:
                return f"{action} {coord_text}"
            if element_text:
                return f"{action} {element_text}"
            return action
        if action in {"type", "input_text"}:
            if coord_text and arg_text:
                return f"type {coord_text} \"{arg_text}\""
            if arg_text:
                return f"type \"{arg_text}\""
            return "type"
        if action in {"swipe", "scroll"}:
            direction_match = re.search(r"direction=([^|,]+)", text, flags=re.IGNORECASE)
            if not direction_match:
                direction_match = re.search(r"direction\s+([a-zA-Z]+)", text, flags=re.IGNORECASE)
            direction = _normalize_space(direction_match.group(1)) if direction_match else ""
            start_match = re.search(
                r"start_coordinate=\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
                text,
                flags=re.IGNORECASE,
            )
            end_match = re.search(
                r"end_coordinate=\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
                text,
                flags=re.IGNORECASE,
            )
            if start_match and end_match:
                return (
                    "swipe "
                    f"[{start_match.group(1)}, {start_match.group(2)}]"
                    "->"
                    f"[{end_match.group(1)}, {end_match.group(2)}]"
                )
            return f"swipe {direction}".strip()
        if action in {"open", "open_app"}:
            return f"open_app \"{arg_text}\"" if arg_text else "open_app"
        if action == "system_button":
            button_match = re.search(r"button=([^|,]+)", text, flags=re.IGNORECASE)
            button = _normalize_space(button_match.group(1)) if button_match else ""
            return f"system_button {button}".strip()
        if action in {"navigate_back", "back"}:
            return "system_button back"
        if action in {"navigate_home", "home"}:
            return "system_button home"
        if action in {"keyboard_enter", "enter"}:
            return "system_button enter"
        if action in {"terminate", "status"}:
            status_match = re.search(r"(?:status|goal_status)=([^|,]+)", text, flags=re.IGNORECASE)
            status = _normalize_space(status_match.group(1)) if status_match else ""
            return f"terminate {status}".strip()
        if action == "answer":
            return "answer"
        return action

    def _simplify_action_entry(self, entry: dict[str, Any], fallback: str = "") -> str:
        if not isinstance(entry, dict):
            return self._simplify_history_item(fallback)

        tool_call = entry.get("tool_call")
        action_dict = entry.get("action_dict")
        action_data = action_dict if isinstance(action_dict, dict) else {}
        args = {}
        if isinstance(tool_call, dict) and isinstance(tool_call.get("arguments"), dict):
            args = dict(tool_call.get("arguments") or {})
        act = str(
            args.get("action")
            or args.get("action_type")
            or action_data.get("action_type")
            or ""
        ).strip().lower()

        if not act:
            return self._simplify_history_item(fallback)

        def _xy_text(x: Any, y: Any) -> str:
            ix = _safe_int(x)
            iy = _safe_int(y)
            if ix is None or iy is None:
                return ""
            return f"[{ix}, {iy}]"

        def _coord_from_args() -> str:
            coord = args.get("coordinate")
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                out = _xy_text(coord[0], coord[1])
                if out:
                    return out
            return ""

        coord_text = _coord_from_args()
        if not coord_text and isinstance(action_dict, dict):
            coord_text = _xy_text(action_data.get("x"), action_data.get("y"))

        idx = _safe_int(args.get("element_id"))
        if idx is None:
            idx = _safe_int(args.get("index"))
        if idx is None and isinstance(action_dict, dict):
            idx = _safe_int(action_data.get("index"))
        idx_text = f"#{idx}" if idx is not None else ""

        text_val = _normalize_space(
            str(
                args.get("text")
                or args.get("app_name")
                or action_data.get("text")
                or action_data.get("app_name")
                or ""
            )
        )
        if len(text_val) > 28:
            text_val = text_val[:28].rstrip(" ,.;:") + "..."

        if act in {"navigate_back", "back"}:
            return "system_button back"
        if act in {"navigate_home", "home"}:
            return "system_button home"
        if act in {"keyboard_enter", "enter"}:
            return "system_button enter"
        if act in {"click", "tap", "double_tap"}:
            if coord_text:
                return f"click {coord_text}"
            return f"click {idx_text}".strip()
        if act == "long_press":
            if coord_text:
                return f"long_press {coord_text}"
            return f"long_press {idx_text}".strip()
        if act in {"type", "input_text"}:
            if coord_text and text_val:
                return f"type {coord_text} \"{text_val}\""
            if idx_text and text_val:
                return f"type {idx_text} \"{text_val}\""
            if text_val:
                return f"type \"{text_val}\""
            return "type"
        if act in {"swipe", "scroll"}:
            direction = _normalize_space(str(args.get("direction") or action_data.get("direction") or ""))
            start = args.get("start_coordinate")
            end = args.get("end_coordinate")
            if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2:
                start_text = _xy_text(start[0], start[1])
                end_text = _xy_text(end[0], end[1])
                if start_text and end_text:
                    return f"swipe {start_text}->{end_text}"
            return f"swipe {direction}".strip() if direction else "swipe"
        if act in {"open", "open_app"}:
            return f"open_app \"{text_val}\"" if text_val else "open_app"
        if act == "system_button":
            button = _normalize_space(str(args.get("button") or ""))
            return f"system_button {button}".strip()
        if act in {"terminate", "status"}:
            status = _normalize_space(str(args.get("status") or args.get("goal_status") or action_data.get("goal_status") or ""))
            return f"terminate {status}".strip()
        if act == "answer":
            return "answer"
        if act == "wait":
            return "wait"
        return self._simplify_history_item(fallback) if fallback else act

    def _history_prompt_text(self, max_items: int = 8) -> str:
        limit = max(1, int(max_items))
        if self.actions:
            tail_actions = self.actions[-limit:]
            tail_history = self.history[-len(tail_actions) :] if self.history else []
            lines = []
            for idx, entry in enumerate(tail_actions, start=1):
                fallback = tail_history[idx - 1] if idx - 1 < len(tail_history) else ""
                lines.append(f"{idx}. {self._simplify_action_entry(entry, fallback=fallback)}")
            return "\n".join(lines)
        if not self.history:
            return "None yet."
        tail = self.history[-limit:]
        return "\n".join(f"{idx}. {self._simplify_history_item(item)}" for idx, item in enumerate(tail, start=1))

    @staticmethod
    def _hints_text(hints: list[ExplorerHint]) -> str:
        if not hints:
            return "None."
        lines = []
        for rank, hint in enumerate(hints, start=1):
            lines.append(f"- #{rank}: element {hint.index}, score {hint.score:.2f}. {hint.label}")
        return "\n".join(lines)

    @staticmethod
    def _safe_center_from_element(element: Any) -> tuple[int, int] | None:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            return None
        return int((bbox.x_min + bbox.x_max) / 2.0), int((bbox.y_min + bbox.y_max) / 2.0)

    @staticmethod
    def _index_from_coordinate(
        ui_elements: list[Any],
        x: int | None,
        y: int | None,
    ) -> int | None:
        if x is None or y is None:
            return None
        hit: tuple[int, float] | None = None
        nearest: tuple[int, float] | None = None
        for idx, element in enumerate(ui_elements):
            bbox = getattr(element, "bbox_pixels", None)
            if bbox is None:
                continue
            x_min = int(bbox.x_min)
            y_min = int(bbox.y_min)
            x_max = int(bbox.x_max)
            y_max = int(bbox.y_max)
            cx = int((x_min + x_max) / 2.0)
            cy = int((y_min + y_max) / 2.0)
            dist = float((cx - x) ** 2 + (cy - y) ** 2)
            if nearest is None or dist < nearest[1]:
                nearest = (idx, dist)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                area = float(max(1, (x_max - x_min) * (y_max - y_min)))
                if hit is None or area < hit[1]:
                    hit = (idx, area)
        if hit is not None:
            return hit[0]
        if nearest is not None:
            return nearest[0]
        return None

    def _fallback_explore(
        self,
        hints: list[ExplorerHint],
        ui_elements: list[Any],
    ) -> tuple[json_action.JSONAction, dict[str, Any]]:
        for hint in hints:
            if hint.index not in self._recent_indices:
                center = None
                if 0 <= hint.index < len(ui_elements):
                    center = self._safe_center_from_element(ui_elements[hint.index])
                if center is not None:
                    return (
                        json_action.JSONAction(action_type=json_action.CLICK, x=center[0], y=center[1]),
                        {
                            "name": "mobile_use",
                            "arguments": {
                                "action": "click",
                                "element_id": hint.index,
                                "coordinate": [center[0], center[1]],
                            },
                        },
                    )
                return (
                    json_action.JSONAction(action_type=json_action.CLICK, index=hint.index),
                    {"name": "mobile_use", "arguments": {"action": "click", "element_id": hint.index}},
                )
        if hints:
            idx = hints[0].index
            center = None
            if 0 <= idx < len(ui_elements):
                center = self._safe_center_from_element(ui_elements[idx])
            if center is not None:
                return (
                    json_action.JSONAction(action_type=json_action.CLICK, x=center[0], y=center[1]),
                    {
                        "name": "mobile_use",
                        "arguments": {
                            "action": "click",
                            "element_id": idx,
                            "coordinate": [center[0], center[1]],
                        },
                    },
                )
            return (
                json_action.JSONAction(action_type=json_action.CLICK, index=idx),
                {"name": "mobile_use", "arguments": {"action": "click", "element_id": idx}},
            )
        return (
            json_action.JSONAction(action_type=json_action.WAIT),
            {"name": "mobile_use", "arguments": {"action": "wait"}},
        )

    @staticmethod
    def _tool_call_text(tool_call: dict[str, Any], source: str) -> str:
        args = dict(tool_call.get("arguments") or {})
        action = str(args.get("action") or args.get("action_type") or "unknown").strip().lower()
        idx = _target_index(args)
        detail_bits: list[str] = []

        coordinate = args.get("coordinate")
        if coordinate is not None:
            detail_bits.append(f"at {coordinate}")
        elif "x" in args or "y" in args:
            detail_bits.append(f"at [{args.get('x')}, {args.get('y')}]")

        if "start_coordinate" in args and "end_coordinate" in args:
            detail_bits.append(
                f"from {args.get('start_coordinate')} to {args.get('end_coordinate')}"
            )
        if "direction" in args:
            detail_bits.append(f"direction {args.get('direction')}")
        if "text" in args:
            detail_bits.append(f'text "{args.get("text")}"')
        if "button" in args:
            detail_bits.append(f"button {args.get('button')}")
        if "status" in args:
            detail_bits.append(f"status {args.get('status')}")
        if "goal_status" in args:
            detail_bits.append(f"goal status {args.get('goal_status')}")
        if "app_name" in args:
            detail_bits.append(f'app "{args.get("app_name")}"')
        if idx is not None:
            detail_bits.append(f"element #{idx}")

        detail = ", ".join([x for x in detail_bits if _normalize_space(x)])
        if detail:
            return f"[{source}] {action} ({detail})"
        return f"[{source}] {action}"

    @staticmethod
    def _normalize_goal_status(goal_status: Any) -> str:
        return str(goal_status or "").strip().lower()

    @classmethod
    def _is_complete_status_action(cls, action: json_action.JSONAction) -> bool:
        if action.action_type != json_action.STATUS:
            return False
        status = cls._normalize_goal_status(getattr(action, "goal_status", ""))
        return status in {"complete", "completed", "done", "success", "task_complete"}

    @classmethod
    def _task_status_from_action(cls, action: json_action.JSONAction) -> str | None:
        if action.action_type != json_action.STATUS:
            return None
        if cls._is_complete_status_action(action):
            return "completed"
        status = cls._normalize_goal_status(getattr(action, "goal_status", ""))
        if status in {"infeasible", "failed", "failure", "impossible"}:
            return "infeasible"
        if not status:
            return "status_unknown"
        return f"status:{status}"

    # -----------------------------
    # Step
    # -----------------------------
    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        self._ensure_task_context(goal)
        if len(self.actions) >= self._effective_max_steps():
            summary = f"Reached the maximum step limit ({self._effective_max_steps()})."
            self._emit_log(summary, tag="INFO")
            self._finalize_task_context(status="infeasible")
            action = json_action.JSONAction(
                action_type=json_action.STATUS,
                goal_status="infeasible",
            )
            tool_call = {
                "name": "mobile_use",
                "arguments": {"action": "terminate", "status": "fail"},
            }
            return base_agent.AgentInteractionResult(
                done=True,
                data={
                    "response": "",
                    "tool_call": tool_call,
                    "action": repr(action),
                    "action_dict": action.as_dict(skip_none=True),
                    "source": "max_step_guard",
                    "parse_error": None,
                    "execution_error": None,
                    "execution_path": None,
                    "coordinate_mode": self.model_coordinate_mode,
                    "goal_status": action.goal_status,
                    "task_status": "infeasible",
                    "task_completed": False,
                    "no_effect_repeat": self._no_effect_repeat,
                    "clues": "",
                    "clue_debug": [],
                    "explore_candidates_count": 0,
                    "explore_action_count": self._get_explore_action_count(),
                    "reasoning_page_record": None,
                    "return_failed_suspected": False,
                    "latency_sec": 0.0,
                    "step_latency_sec": 0.0,
                    "task_elapsed_sec": (
                        float(max(0.0, time.time() - float(self._task_start_ts)))
                        if self._task_start_ts is not None
                        else 0.0
                    ),
                    "avg_step_latency_sec": (
                        float(sum(self._task_step_latencies) / len(self._task_step_latencies))
                        if self._task_step_latencies
                        else 0.0
                    ),
                    "summary": summary,
                },
            )
        step_start_ts = time.time()
        step_start_perf = time.perf_counter()
        # Defensive: ensure no stale explorer thread leaks into next reasoning step.
        self._stop_explorer_thread()
        if not self.enable_parallel_exploration:
            # Prevent stale root baseline from triggering rollback guards when exploration is off.
            self._clear_explore_root_baseline()
        step_no = len(self.actions) + 1
        self._step_separator(step_no=step_no, phase="start", goal=goal)
        state = self.get_post_transition_state()
        step_start_pixels = np.array(state.pixels, copy=True)
        ui_elements = state.ui_elements
        hints = self._build_hints(ui_elements)
        reasoning_start_page = self._compact_page_record(state)
        reasoning_pre_action_page: dict[str, Any] | None = None
        reasoning_end_page: dict[str, Any] | None = None
        return_alignment: dict[str, Any] | None = None
        reasoning_page_record: dict[str, Any] | None = None
        self._emit_log(
            f"step={step_no} current_page=({self._state_page_hint(state)})",
            tag="STEP",
        )
        self._emit_log_block(
            title=f"step={step_no} action_history",
            content=self._history_text(),
            tag="REASON",
        )
        self._emit_log_block(
            title=f"step={step_no} reasoning_hints",
            content=self._hints_text(hints),
            tag="REASON",
        )
        self._emit_log_block(
            title=f"step={step_no} ui_tree_summary",
            content=self._state_page_hint(state, max_cues=12),
            tag="STEP",
        )

        delta = _pixel_delta(self._last_pixels, state.pixels)
        if self._last_action_text and delta is not None and delta <= self.no_effect_delta_threshold:
            self._no_effect_repeat += 1
            self._execution_feedback = (
                f"Previous action had tiny change (pixel_delta={delta:.3f}). "
                "Avoid same element/action."
            )
        elif delta is not None:
            self._no_effect_repeat = 0
            self._execution_feedback = ""

        clues = ""
        if self._pending_explore_payload:
            pending_candidates = self._pending_explore_payload.get("candidates") or []
            source_step = self._pending_explore_payload.get("source_step")
            clue_text = self.build_prompt_clues_from_candidates(
                candidates=pending_candidates,
                current_pixels=state.pixels,
                max_items=4,
                last_reasoning_action=(self.history[-1] if self.history else ""),
            )
            if clue_text:
                clues = (
                    f"[Clue Source] Exploration step {source_step} prepared hints for reasoning step {len(self.actions) + 1}.\n"
                    + clue_text
                )

        self._emit_log_block(
            title=f"step={step_no} clues",
            content=clues or "None.",
            tag="REASON",
        )
        explore_gate_reason = "disabled"
        should_explore = False
        if self.enable_parallel_exploration:
            should_explore, explore_gate_reason = self._should_start_exploration(
                step_no=step_no,
                goal=goal,
                current_activity=reasoning_start_page.get("activity"),
            )
            self._emit_log(
                f"step={step_no} explore_gate should_explore={should_explore} reason={explore_gate_reason}",
                tag="EXPLORE",
            )
        if not should_explore:
            with self._explore_action_count_lock:
                self._explore_action_count = 0
            self._explore_iteration_candidates = []
            self._clear_explore_root_baseline()
        if self.enable_parallel_exploration:
            if should_explore:
                self._start_explorer_thread(
                    goal=goal,
                    history_tail=self.history[-3:],
                    clues_text=clues,
                    source_step=step_no,
                    trigger_reason=explore_gate_reason,
                )

        prompt_history = self._history_prompt_text(max_items=8)
        user_text = (
            f"Task: {goal}\n\n"
            f"History:\n{prompt_history}\n\n"
            f"Execution feedback:\n{self._execution_feedback or 'None.'}\n\n"
            f"Parallel explorer clues:\n{clues or 'None.'}\n\n"
            f"Explorer hints:\n{self._hints_text(hints)}"
        )
        prompt_log_text = (
            "[VLM text input | system]\n"
            + EXPLORER_MAI_SYSTEM_PROMPT
            + "\n\n[VLM text input | user]\n"
            + user_text
        )
        self._emit_log_block(
            title=f"step={step_no} vlm_text_input",
            content=prompt_log_text,
            tag="REASON",
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": EXPLORER_MAI_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": _to_data_url(state.pixels)}},
                ],
            },
        ]

        reasoning_start_perf = time.perf_counter()
        response, _, _ = self.vllm.predict_mm("", [], messages=messages)
        reasoning_elapsed = float(max(0.0, time.perf_counter() - reasoning_start_perf))
        if self.reasoning_sleep_sec > reasoning_elapsed:
            sleep_sec = float(self.reasoning_sleep_sec - reasoning_elapsed)
            self._emit_log(
                (
                    f"step={step_no} simulate_reasoning_sleep={sleep_sec:.3f}s "
                    f"(elapsed={reasoning_elapsed:.3f}s target={self.reasoning_sleep_sec:.3f}s)"
                ),
                tag="REASON",
            )
            time.sleep(sleep_sec)
        self._emit_log_block(
            title=f"step={step_no} llm_response",
            content=str(response),
            tag="REASON",
        )

        source = "llm"
        parse_error = None
        execution_error = None
        execution_path = None
        coord_mode_used = self.model_coordinate_mode
        fallback_index = hints[0].index if hints else None
        strict_retry_count = 0
        strict_reject = False
        completion_checkpoint: dict[str, Any] = {}
        structured_recovery_info: dict[str, Any] | None = None
        text_edit_info: dict[str, Any] = {}

        response_text_for_parse = str(response)
        last_parse_exc: Exception | None = None
        for attempt in range(max(0, int(self.strict_json_reprompt_retries)) + 1):
            try:
                parsed_action = parse_gelab_response(response_text_for_parse)
                self._emit_log_block(
                    title=f"step={step_no} parsed_gelab_action",
                    content=json.dumps(dict(parsed_action), ensure_ascii=False, indent=2),
                    tag="REASON",
                )
                action, tool_call, _ = gelab_action_to_json_action(
                    parsed_action=parsed_action,
                    screen_size=self.env.logical_screen_size,
                )
                self._emit_log_block(
                    title=f"step={step_no} parsed_tool_call",
                    content=json.dumps(tool_call, ensure_ascii=False, indent=2),
                    tag="REASON",
                )
                self._emit_log_block(
                    title=f"step={step_no} normalized_action",
                    content=repr(action),
                    tag="REASON",
                )
                if action.action_type == json_action.UNKNOWN:
                    raise seeact_utils.ParseActionError("unknown action")
                if (
                    action.action_type == json_action.INPUT_TEXT
                    and action.x is None
                    and action.y is None
                    and action.index is None
                ):
                    raise seeact_utils.ParseActionError("input_text missing target (coordinate/index)")
                response = response_text_for_parse
                strict_retry_count = attempt
                parse_error = None
                break
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_parse_exc = exc
                parse_error = str(exc)
                self._emit_log(
                    f"step={step_no} strict_parse_failed attempt={attempt + 1} error={parse_error}",
                    tag="REASON",
                )
                repaired_tool_call = self._recover_tool_call_from_malformed_response(response_text_for_parse)
                if repaired_tool_call is not None:
                    repaired_action = _to_json_action(
                        repaired_tool_call,
                        ui_elements=ui_elements,
                        fallback_index=fallback_index,
                        logical_screen_size=self.env.logical_screen_size,
                        coordinate_mode=self.model_coordinate_mode,
                    )
                    if repaired_action.action_type != json_action.UNKNOWN:
                        tool_call = repaired_tool_call
                        action = repaired_action
                        response = response_text_for_parse
                        strict_retry_count = attempt
                        parse_error = None
                        source = "malformed_tool_call_repair"
                        self._emit_log_block(
                            title=f"step={step_no} malformed_tool_call_repaired",
                            content=json.dumps(tool_call, ensure_ascii=False, indent=2),
                            tag="REASON",
                        )
                        break
                try:
                    # MAI-UI-style robust fallback parser: tolerant to minor JSON/tag defects.
                    tool_call = parse_tool_call(response_text_for_parse)
                    self._emit_log_block(
                        title=f"step={step_no} fallback_parsed_tool_call",
                        content=json.dumps(tool_call, ensure_ascii=False, indent=2),
                        tag="REASON",
                    )
                    action = _to_json_action(
                        tool_call,
                        ui_elements=ui_elements,
                        fallback_index=fallback_index,
                        logical_screen_size=self.env.logical_screen_size,
                        coordinate_mode=self.model_coordinate_mode,
                    )
                    if action.action_type == json_action.UNKNOWN:
                        repaired_tool_call = self._recover_tool_call_from_malformed_response(
                            response_text_for_parse
                        )
                        if repaired_tool_call is not None:
                            repaired_action = _to_json_action(
                                repaired_tool_call,
                                ui_elements=ui_elements,
                                fallback_index=fallback_index,
                                logical_screen_size=self.env.logical_screen_size,
                                coordinate_mode=self.model_coordinate_mode,
                            )
                            if repaired_action.action_type != json_action.UNKNOWN:
                                tool_call = repaired_tool_call
                                action = repaired_action
                                source = "malformed_tool_call_repair"
                                response = response_text_for_parse
                                strict_retry_count = attempt
                                self._emit_log_block(
                                    title=f"step={step_no} malformed_tool_call_repaired",
                                    content=json.dumps(tool_call, ensure_ascii=False, indent=2),
                                    tag="REASON",
                                )
                    if action.action_type != json_action.UNKNOWN:
                        response = response_text_for_parse
                        strict_retry_count = attempt
                        parse_error = None
                        if source != "malformed_tool_call_repair":
                            source = "mai_compatible_parse"
                        self._emit_log_block(
                            title=f"step={step_no} fallback_normalized_action",
                            content=repr(action),
                            tag="REASON",
                        )
                        break
                except Exception as fallback_exc:  # pylint: disable=broad-exception-caught
                    self._emit_log(
                        f"step={step_no} mai_compatible_parse_failed error={fallback_exc}",
                        tag="REASON",
                    )
                    repaired_tool_call = self._recover_tool_call_from_malformed_response(response_text_for_parse)
                    if repaired_tool_call is not None:
                        repaired_action = _to_json_action(
                            repaired_tool_call,
                            ui_elements=ui_elements,
                            fallback_index=fallback_index,
                            logical_screen_size=self.env.logical_screen_size,
                            coordinate_mode=self.model_coordinate_mode,
                        )
                        if repaired_action.action_type != json_action.UNKNOWN:
                            tool_call = repaired_tool_call
                            action = repaired_action
                            response = response_text_for_parse
                            strict_retry_count = attempt
                            parse_error = None
                            source = "malformed_tool_call_repair"
                            self._emit_log_block(
                                title=f"step={step_no} malformed_tool_call_repaired",
                                content=json.dumps(tool_call, ensure_ascii=False, indent=2),
                                tag="REASON",
                            )
                            break
                    try:
                        # Keep strict parser as second fallback for clean <tool_call> JSON.
                        tool_call = parse_tool_call_strict(response_text_for_parse, require_tool_tag=False)
                        action = _to_json_action(
                            tool_call,
                            ui_elements=ui_elements,
                            fallback_index=fallback_index,
                            logical_screen_size=self.env.logical_screen_size,
                            coordinate_mode=self.model_coordinate_mode,
                        )
                        if action.action_type != json_action.UNKNOWN:
                            response = response_text_for_parse
                            strict_retry_count = attempt
                            parse_error = None
                            source = "strict_tool_call_parse"
                            self._emit_log_block(
                                title=f"step={step_no} strict_fallback_normalized_action",
                                content=repr(action),
                                tag="REASON",
                            )
                            break
                    except Exception as strict_fallback_exc:  # pylint: disable=broad-exception-caught
                        self._emit_log(
                            f"step={step_no} strict_fallback_parse_failed error={strict_fallback_exc}",
                            tag="REASON",
                        )
                if attempt < max(0, int(self.strict_json_reprompt_retries)):
                    response_text_for_parse = self._reprompt_for_strict_json(
                        messages=messages,
                        response_text=response_text_for_parse,
                        parse_error=parse_error,
                        step_no=step_no,
                    )
                    continue
                strict_reject = True
                source = "rejected_non_json"
                action = json_action.JSONAction(action_type=json_action.WAIT)
                tool_call = {
                    "name": "mobile_use",
                    "arguments": {
                        "action": "wait",
                        "reason": "strict_json_parse_failed",
                    },
                }
                break

        if strict_reject and last_parse_exc is not None:
            parse_error = str(last_parse_exc)
        if not strict_reject:
            force_open_app, app_name = self._should_force_open_target_app(
                step_no=step_no,
                goal=goal,
                current_activity=reasoning_start_page.get("activity"),
                action=action,
            )
            if force_open_app and app_name:
                source = "target_app_bootstrap"
                action, tool_call = self._build_open_app_action(app_name)
                self._emit_log(
                    f"step={step_no} target_app_bootstrap force_open_app={app_name}",
                    tag="REASON",
                )
            if self._should_force_back_from_chooser(
                goal=goal,
                current_activity=reasoning_start_page.get("activity"),
                action=action,
            ):
                source = "chooser_guard_back"
                action = json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
                tool_call = {
                    "name": "mobile_use",
                    "arguments": {"action": "system_button", "button": "back"},
                }
                self._emit_log(
                    f"step={step_no} chooser_guard applied -> navigate_back",
                    tag="REASON",
                )
        if not strict_reject and self._should_apply_loop_guard(action):
            loop_guard_action, loop_guard_tool = self._fallback_explore(hints, ui_elements)
            prev_sig = self._action_signature_from_action(action)
            alt_sig = self._action_signature_from_action(loop_guard_action)
            if (
                loop_guard_action.action_type not in {json_action.UNKNOWN, json_action.WAIT}
                and alt_sig
                and alt_sig != prev_sig
            ):
                source = "loop_guard_fallback"
                action = loop_guard_action
                tool_call = loop_guard_tool
                self._emit_log(
                    f"step={step_no} loop_guard_applied prev={prev_sig} alt={alt_sig}",
                    tag="REASON",
                )

        explore_candidates = self._stop_explorer_thread()
        explore_action_count = self._get_explore_action_count()
        explore_thread_clean = bool(self._explore_thread_stop_clean)
        if not explore_thread_clean:
            self._emit_log(
                f"step={step_no} explorer_thread_not_stopped_cleanly -> guard_mode",
                tag="STEP",
            )
        rollback_info = self._ensure_root_before_reasoning(
            step_no=step_no,
            max_attempts=2,
        )
        self._emit_log(
            f"step={step_no} post_reasoning_rollback={rollback_info}",
            tag="STEP",
        )
        self._pending_explore_payload = {
            "source_step": len(self.actions) + 1,
            "candidates": explore_candidates,
        }

        safety_mode_reason = None
        if not explore_thread_clean:
            safety_mode_reason = "explorer_thread_not_stopped_cleanly"
            source = f"{source}_thread_guard"
            parse_error = safety_mode_reason
            self._clear_explore_root_baseline()
            cooldown_steps = max(
                int(self.explore_cooldown_after_safe_mode),
                int(self.rollback_fail_explore_cooldown_steps),
            )
            self._explore_cooldown_steps = max(
                int(self._explore_cooldown_steps),
                cooldown_steps,
            )
            self._emit_log(
                f"step={step_no} thread_guard_failed -> enable_safe_mode",
                tag="REASON",
            )
        elif self._has_explore_root_baseline() and not bool(rollback_info.get("verified")):
            safety_mode_reason = "rollback_guard_failed_not_at_root"
            source = f"{source}_rollback_guard"
            parse_error = safety_mode_reason
            cooldown_steps = max(
                int(self.explore_cooldown_after_safe_mode),
                int(self.rollback_fail_explore_cooldown_steps),
            )
            self._explore_cooldown_steps = max(
                int(self._explore_cooldown_steps),
                cooldown_steps,
            )
            self._emit_log(
                f"step={step_no} rollback_guard_failed -> enable_safe_mode",
                tag="REASON",
            )

        force_explore_due_to_uncertainty = bool(
            action.action_type == json_action.WAIT
            or (
                parse_error is not None
                and parse_error
                not in {
                    "rollback_guard_failed_not_at_root",
                    "explorer_thread_not_stopped_cleanly",
                }
            )
        )
        if strict_reject:
            force_explore_due_to_uncertainty = False
        if (
            self._no_effect_repeat >= self.force_explore_after_repeats
            and (explore_action_count > 0 or bool(explore_candidates))
            and force_explore_due_to_uncertainty
            and action.action_type not in {json_action.STATUS, json_action.ANSWER}
        ):
            source = "forced_explore"
            action, tool_call = self._fallback_explore(hints, ui_elements)

        tool_args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
        raw_action_name = str(tool_args.get("action") or tool_args.get("action_type") or "").strip().lower()
        if raw_action_name == "answer":
            answer_text = str(tool_args.get("text") or "").strip()
            if answer_text:
                self.env.interaction_cache = answer_text

        explore_thread_alive = bool(self._explore_thread and self._explore_thread.is_alive())
        last_explore_age_sec = None
        if self._last_explore_action_ts > 0.0:
            last_explore_age_sec = max(0.0, float(time.time() - self._last_explore_action_ts))
        self._emit_log(
            (
                f"step={step_no} pre_action_thread_state alive={explore_thread_alive} "
                f"stop_event={self._explore_stop_event.is_set()} last_explore_age_sec={last_explore_age_sec}"
            ),
            tag="CHECK",
        )

        if safety_mode_reason and self._is_high_risk_interaction_action(action):
            source = f"safe_mode_{safety_mode_reason}"
            action, tool_call = self._build_safe_mode_back_action()
            tool_args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
            self._emit_log(
                f"step={step_no} safe_mode_applied reason={safety_mode_reason} replace_with=back",
                tag="CHECK",
            )

        checkpoint_state = state
        if self._is_calendar_or_contact_goal(goal):
            try:
                with self._ui_lock:
                    checkpoint_state = self.env.get_state(wait_to_stabilize=False)
                completion_checkpoint = self._completion_checkpoint(goal, checkpoint_state)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                completion_checkpoint = {"enabled": True, "error": str(exc)}

            if self._is_complete_status_action(action) and not bool(completion_checkpoint.get("can_finish", True)):
                save_pair = self._build_save_action_from_state(checkpoint_state)
                if save_pair is not None:
                    source = "completion_checkpoint_save"
                    action, tool_call = save_pair
                    tool_args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
                    self._emit_log(
                        (
                            f"step={step_no} completion_checkpoint_blocked_finish "
                            f"save_visible={completion_checkpoint.get('save_visible')} "
                            f"in_edit={completion_checkpoint.get('in_edit_activity')}"
                        ),
                        tag="CHECK",
                    )
                elif bool(completion_checkpoint.get("in_edit_activity")):
                    source = "completion_checkpoint_back"
                    action, tool_call = self._build_safe_mode_back_action()
                    tool_args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
                    self._emit_log(
                        f"step={step_no} completion_checkpoint_force_back_from_edit",
                        tag="CHECK",
                    )

            if self._task_status_from_action(action) == "infeasible":
                recovery_pair = self._attempt_structured_recovery_before_infeasible(
                    goal=goal,
                    state=checkpoint_state,
                    step_no=step_no,
                )
                if recovery_pair is not None:
                    source = "structured_recovery_before_infeasible"
                    action, tool_call, structured_recovery_info = recovery_pair
                    tool_args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
                    self._emit_log(
                        f"step={step_no} infeasible_intercepted structured_recovery={structured_recovery_info}",
                        tag="CHECK",
                    )

        model_coord = tool_args.get("coordinate") if isinstance(tool_args, dict) else None
        parsed_coord = (
            [int(action.x), int(action.y)]
            if getattr(action, "x", None) is not None and getattr(action, "y", None) is not None
            else None
        )
        if model_coord is not None or parsed_coord is not None:
            try:
                screen_size = self.env.logical_screen_size
            except Exception:  # pylint: disable=broad-exception-caught
                screen_size = None
            self._emit_log(
                (
                    f"step={step_no} coord_check model={model_coord} parsed={parsed_coord} "
                    f"screen={screen_size} mode={self.model_coordinate_mode}"
                ),
                tag="CHECK",
            )

        pre_action_pixels = np.array(step_start_pixels, copy=True)
        post_state = None
        action_effect: dict[str, Any] = {}
        action_retry_attempted = False
        action_retry_succeeded = False

        try:
            with self._ui_lock:
                pre_action_state = self.env.get_state(wait_to_stabilize=False)
            pre_action_pixels = np.array(pre_action_state.pixels, copy=True)
            reasoning_pre_action_page = self._compact_page_record(pre_action_state)
        except Exception:  # pylint: disable=broad-exception-caught
            reasoning_pre_action_page = dict(reasoning_start_page)
        return_alignment = self._page_alignment_summary(
            reasoning_start_page,
            reasoning_pre_action_page,
        )

        try:
            if action.action_type == json_action.INPUT_TEXT:
                execution_path, post_state, text_edit_info, execution_error = self._execute_verified_text_input(
                    action=action,
                    ui_elements=ui_elements,
                )
            else:
                with self._ui_lock:
                    execution_path = self._execute_action_with_coordinate_priority(action)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            execution_error = str(exc)
            source = f"{source}_exec_error"

        if post_state is None:
            try:
                with self._ui_lock:
                    post_state = self.env.get_state(wait_to_stabilize=False)
            except Exception:  # pylint: disable=broad-exception-caught
                post_state = None

        after_activity = self._foreground_activity_name()
        delete_dialog_info: dict[str, Any] = {}
        if post_state is not None:
            post_state, delete_dialog_info = self._handle_unexpected_delete_dialog(goal, post_state)
            if bool(delete_dialog_info.get("handled")):
                after_activity = self._foreground_activity_name()
                if execution_error:
                    execution_error = f"{execution_error}; unexpected_delete_dialog_cancelled"
                else:
                    execution_error = "unexpected_delete_dialog_cancelled"
        action_effect = self._action_effect_summary(
            before_pixels=pre_action_pixels,
            after_pixels=(post_state.pixels if post_state is not None else None),
            before_activity=reasoning_pre_action_page.get("activity"),
            after_activity=after_activity,
        )
        if bool(delete_dialog_info.get("handled")):
            action_effect["unstable"] = True

        # One retry for click-like actions when nothing appears to change.
        if (
            not bool(action_effect.get("changed"))
            and action.action_type in {json_action.CLICK, json_action.LONG_PRESS}
            and post_state is not None
        ):
            retry_idx = _safe_int(getattr(action, "index", None))
            if retry_idx is None:
                retry_idx = self._index_from_coordinate(
                    ui_elements=ui_elements,
                    x=_safe_int(getattr(action, "x", None)),
                    y=_safe_int(getattr(action, "y", None)),
                )
            if retry_idx is not None and 0 <= retry_idx < len(ui_elements):
                retry_center = self._safe_center_from_element(ui_elements[retry_idx])
                if retry_center is not None:
                    action_retry_attempted = True
                    retry_action = json_action.JSONAction(
                        action_type=action.action_type,
                        x=retry_center[0],
                        y=retry_center[1],
                    )
                    try:
                        with self._ui_lock:
                            self._execute_action_with_coordinate_priority(retry_action)
                            post_state = self.env.get_state(wait_to_stabilize=False)
                        action_retry_succeeded = True
                        execution_path = f"{execution_path}+retry_center"
                        action_effect = self._action_effect_summary(
                            before_pixels=pre_action_pixels,
                            after_pixels=post_state.pixels,
                            before_activity=reasoning_pre_action_page.get("activity"),
                            after_activity=self._foreground_activity_name(),
                        )
                    except Exception:  # pylint: disable=broad-exception-caught
                        action_retry_succeeded = False

        idx = _target_index(tool_call.get("arguments", {}))
        if idx is None:
            idx = _safe_int(getattr(action, "index", None))
        if idx is None:
            idx = self._index_from_coordinate(
                ui_elements=ui_elements,
                x=_safe_int(getattr(action, "x", None)),
                y=_safe_int(getattr(action, "y", None)),
            )
        if idx is not None and 0 <= idx < len(ui_elements):
            self._recent_indices.append(idx)

        action_text = self._tool_call_text(tool_call, source=source)
        if parse_error:
            action_text += f"; parse error: {parse_error}"
        if execution_error:
            action_text += f"; execution error: {execution_error}"
        if execution_path:
            action_text += f"; executed via {execution_path}"
        if action_effect:
            changed_text = "yes" if bool(action_effect.get("changed")) else "no"
            action_text += (
                f"; effect changed: {changed_text}"
                f" (delta={action_effect.get('pixel_delta')}, hash_diff={action_effect.get('hash_diff')})"
            )
        if action_retry_attempted:
            action_text += f"; retry attempted: {action_retry_succeeded}"
        if strict_retry_count > 0:
            action_text += f"; strict json retries: {strict_retry_count}"
        if strict_reject:
            action_text += "; strict json rejected"
        if text_edit_info:
            action_text += f"; text edit info: {text_edit_info}"
        action_text += f"; coordinate mode: {coord_mode_used}"
        self._emit_log(
            f"step={step_no} Final decision: {action_text}",
            tag="STEP",
        )

        self._last_action_text = action_text
        self._last_action_effect = dict(action_effect or {})
        if action_effect and not bool(action_effect.get("changed")):
            self._execution_feedback = "Last action likely had no visible effect; avoid repeating same target."
        self._last_pixels = np.array(state.pixels, copy=True)
        self.history.append(action_text)

        action_dict = action.as_dict(skip_none=True)
        if action.action_type not in {json_action.STATUS, json_action.ANSWER, json_action.UNKNOWN}:
            self._reasoning_action_history.append(action_dict)

        self.actions.append(
            {
                "response": response,
                "tool_call": tool_call,
                "action": repr(action),
                "action_dict": action_dict,
                "source": source,
                "parse_error": parse_error,
                "execution_error": execution_error,
                "execution_path": execution_path,
                "action_effect": action_effect,
                "action_retry_attempted": action_retry_attempted,
                "action_retry_succeeded": action_retry_succeeded,
                "strict_retry_count": strict_retry_count,
                "strict_reject": strict_reject,
                "text_edit_info": text_edit_info,
                "completion_checkpoint": completion_checkpoint,
                "structured_recovery_info": structured_recovery_info,
                "delete_dialog_info": delete_dialog_info,
                "coordinate_mode": coord_mode_used,
                "clues": clues,
                "clue_debug": self.get_last_clue_debug_lines(),
                "explore_candidates_count": len(explore_candidates),
                "reasoning_page_record": reasoning_page_record,
            }
        )

        task_status = self._task_status_from_action(action)
        done = task_status is not None
        task_completed = bool(task_status == "completed")
        if post_state is None:
            with self._ui_lock:
                post_state = self.env.get_state(wait_to_stabilize=False)
        reasoning_end_page = self._compact_page_record(post_state)
        action_summary = self._simplify_action_entry(
            {"tool_call": tool_call, "action_dict": action_dict},
            fallback=action_text,
        )
        start_to_end_alignment = self._page_alignment_summary(
            reasoning_start_page,
            reasoning_end_page,
        )
        reasoning_page_record = {
            "step": step_no,
            "action": action_summary,
            "source": source,
            "explore_return_verified": bool(rollback_info.get("verified")),
            "pre_action_thread_alive": explore_thread_alive,
            "last_explore_age_sec": last_explore_age_sec,
            "safety_mode_reason": safety_mode_reason,
            "start_page": reasoning_start_page,
            "before_action_page": reasoning_pre_action_page,
            "end_page": reasoning_end_page,
            "start_to_before_action": return_alignment,
            "start_to_end": start_to_end_alignment,
            "return_failed_suspected": bool(
                (not bool(rollback_info.get("verified")))
                or (return_alignment is not None and not bool(return_alignment.get("matched")))
            ),
            "action_effect": action_effect,
            "action_retry_attempted": action_retry_attempted,
            "action_retry_succeeded": action_retry_succeeded,
            "completion_checkpoint": completion_checkpoint,
            "structured_recovery_info": structured_recovery_info,
            "delete_dialog_info": delete_dialog_info,
        }
        if self.actions:
            self.actions[-1]["reasoning_page_record"] = reasoning_page_record
        self._reasoning_page_records.append(reasoning_page_record)
        if self._task_trace_dir:
            self._append_jsonl(
                os.path.join(self._task_trace_dir, "reasoning_page_records.jsonl"),
                reasoning_page_record,
            )
        self._emit_log(
            (
                f"Step {step_no} reasoning record. Action {action_summary}. "
                f"Start page: {self._page_brief_text(reasoning_start_page)}. "
                f"Before action: {self._page_brief_text(reasoning_pre_action_page or {})}. "
                f"Return match: {bool((return_alignment or {}).get('matched'))}, "
                f"rollback verified: {bool(rollback_info.get('verified'))}. "
                f"End page: {self._page_brief_text(reasoning_end_page)}."
            ),
            tag="CHECK",
        )
        step_latency_sec = float(max(0.0, time.perf_counter() - step_start_perf))
        self._task_step_latencies.append(step_latency_sec)
        avg_step_sec = float(sum(self._task_step_latencies) / max(1, len(self._task_step_latencies)))
        task_elapsed_sec = (
            float(max(0.0, time.time() - float(self._task_start_ts)))
            if self._task_start_ts is not None
            else 0.0
        )
        self._emit_log(
            (
                f"step={step_no} latency_sec={step_latency_sec:.3f} "
                f"avg_step_latency_sec={avg_step_sec:.3f} task_elapsed_sec={task_elapsed_sec:.3f}"
            ),
            tag="INFO",
        )
        self._save_step_trace(
            step_no=step_no,
            start_pixels=step_start_pixels,
            end_pixels=post_state.pixels,
            payload={
                "goal": goal,
                "step": step_no,
                "step_started_at": step_start_ts,
                "step_ended_at": time.time(),
                "step_latency_sec": step_latency_sec,
                "avg_step_latency_sec": avg_step_sec,
                "task_elapsed_sec": task_elapsed_sec,
                "reasoning_input": {
                    "history": self._history_text(),
                    "execution_feedback": self._execution_feedback or "None.",
                    "clues": clues or "None.",
                    "hints": self._hints_text(hints),
                    "vlm_user_text": user_text,
                },
                "explore": {
                    "explore_action_count": explore_action_count,
                    "filter_stats": dict(self._last_filter_stats or {}),
                    "candidates_count": len(explore_candidates),
                    "candidates": explore_candidates,
                },
                "llm_response_raw": str(response),
                "rollback_info": rollback_info,
                "reasoning_page_record": reasoning_page_record,
                "decision": {
                    "source": source,
                    "tool_call": tool_call,
                    "action_repr": repr(action),
                    "action_dict": action_dict,
                    "action_text": action_text,
                    "parse_error": parse_error,
                    "execution_error": execution_error,
                    "execution_path": execution_path,
                    "action_effect": action_effect,
                    "action_retry_attempted": action_retry_attempted,
                    "action_retry_succeeded": action_retry_succeeded,
                    "strict_retry_count": strict_retry_count,
                    "strict_reject": strict_reject,
                    "text_edit_info": text_edit_info,
                    "completion_checkpoint": completion_checkpoint,
                    "structured_recovery_info": structured_recovery_info,
                    "delete_dialog_info": delete_dialog_info,
                    "coordinate_mode": coord_mode_used,
                },
                "return": {
                    "done": done,
                    "data": {
                        "response": response,
                        "tool_call": tool_call,
                        "action": repr(action),
                        "action_dict": action_dict,
                        "source": source,
                        "parse_error": parse_error,
                        "execution_error": execution_error,
                        "execution_path": execution_path,
                        "action_effect": action_effect,
                        "action_retry_attempted": action_retry_attempted,
                        "action_retry_succeeded": action_retry_succeeded,
                        "strict_retry_count": strict_retry_count,
                        "strict_reject": strict_reject,
                        "text_edit_info": text_edit_info,
                        "completion_checkpoint": completion_checkpoint,
                        "structured_recovery_info": structured_recovery_info,
                        "delete_dialog_info": delete_dialog_info,
                        "coordinate_mode": coord_mode_used,
                        "goal_status": getattr(action, "goal_status", None),
                        "task_status": task_status,
                        "task_completed": task_completed,
                        "no_effect_repeat": self._no_effect_repeat,
                        "clues": clues,
                        "clue_debug": self.get_last_clue_debug_lines(),
                        "explore_candidates_count": len(explore_candidates),
                        "explore_action_count": explore_action_count,
                        "reasoning_page_record": reasoning_page_record,
                        "return_failed_suspected": bool(
                            (reasoning_page_record or {}).get("return_failed_suspected", False)
                        ),
                        "latency_sec": step_latency_sec,
                        "step_latency_sec": step_latency_sec,
                        "task_elapsed_sec": task_elapsed_sec,
                        "avg_step_latency_sec": avg_step_sec,
                    },
                },
            },
        )
        self._emit_log(
            f"step={step_no} post_action_page=({self._state_page_hint(post_state)})",
            tag="STEP",
        )
        self._step_separator(
            step_no=step_no,
            phase="end",
            summary=(
                f"done={done}, source={source}, action={action.action_type}, "
                f"explore_candidates={len(explore_candidates)}, latency_sec={step_latency_sec:.3f}"
            ),
        )
        if done:
            self._finalize_task_context(status=task_status or "completed")
        return base_agent.AgentInteractionResult(
            done=done,
            data={
                "response": response,
                "tool_call": tool_call,
                "action": repr(action),
                "action_dict": action_dict,
                "source": source,
                "parse_error": parse_error,
                "execution_error": execution_error,
                "execution_path": execution_path,
                "action_effect": action_effect,
                "action_retry_attempted": action_retry_attempted,
                "action_retry_succeeded": action_retry_succeeded,
                "strict_retry_count": strict_retry_count,
                "strict_reject": strict_reject,
                "text_edit_info": text_edit_info,
                "completion_checkpoint": completion_checkpoint,
                "structured_recovery_info": structured_recovery_info,
                "delete_dialog_info": delete_dialog_info,
                "coordinate_mode": coord_mode_used,
                "goal_status": getattr(action, "goal_status", None),
                "task_status": task_status,
                "task_completed": task_completed,
                "no_effect_repeat": self._no_effect_repeat,
                "clues": clues,
                "clue_debug": self.get_last_clue_debug_lines(),
                "explore_candidates_count": len(explore_candidates),
                "explore_action_count": explore_action_count,
                "reasoning_page_record": reasoning_page_record,
                "return_failed_suspected": bool(
                    (reasoning_page_record or {}).get("return_failed_suspected", False)
                ),
                "latency_sec": step_latency_sec,
                "step_latency_sec": step_latency_sec,
                "task_elapsed_sec": task_elapsed_sec,
                "avg_step_latency_sec": avg_step_sec,
            },
        )

    def save_summary(self, path: str = "androidworld_exec_profile_summary.json") -> None:
        payload = {
            "agent": self.name,
            "steps": len(self.actions),
            "history_tail": self.history[-10:],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# For compatibility with existing `mm_agent.ElementTextAgent` naming.
class ElementTextAgent(ExplorerElementAgent):
    pass
