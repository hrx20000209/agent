import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import json
import threading
import argparse
import shutil
import subprocess
import shlex
import re
import hashlib
import copy
from PIL import Image

from MobileAgentE.controller import get_screenshot, get_a11y_tree
from MobileAgentE.api import (
    inference_chat,
    inference_chat_ollama,
    inference_chat_llama_cpp,
)
from MobileAgentE.tree import parse_a11y_tree, print_tree
from MobileAgentE.utils import parse_bounds
from MobileAgentE.agents import OneStepAgent  # ✅ 换成新的 Agent 和 InfoPool
from agents.mai_ui_agent import MAIOneStepAgent
from agents.utils import execute_action
# from Explorer.online_explorer import A11yTreeOnlineExplorer
from Explorer.GoalExplorer import A11yTreeOnlineExplorer
from Explorer.utils import collect_clickable_nodes, ensure_dir, mark_and_save_explore_click, node_to_text, phash
from Explorer.progressive_belief_graph import (
    GenerationGuardedGraph,
    ProgressiveBeliefGraph,
    describe_ui_state,
)
from Explorer.state_graph_information import MatrixAblations, parse_reasoning_prior
from Explorer.graph_distiller import (
    GraphDistiller,
    GraphDistillerConfig,
    GraphMode,
    GraphReasoningGate,
)

########################################
#              CONFIG
########################################
REASONING_MODEL = "qwen-vl-plus"
LOG_DIR = "./logs/single_step_agent"

### LLM ###
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
USAGE_TRACKING_JSONL = None


def log_event(label, message=""):
    ts = time.strftime("%H:%M:%S")
    suffix = f" {message}" if message else ""
    print(f"[{ts}] {label}{suffix}", flush=True)


########################################
#        LLM CALL FUNCTION
########################################
def get_reasoning_response(chat, model=REASONING_MODEL, api_url="http://localhost:8100/v1/chat/completions", max_tokens=200):
    """唯一的 LLM 调用"""
    temperature = 0.0
    return inference_chat_llama_cpp(chat, api_url=api_url, temperature=temperature, max_tokens=max_tokens)
    # 如果你改回 qwen2.5vl / dashscope，就把上面这一行替换成下方分支即可
    # if model == "qwen2.5vl:3b":
    #     return inference_chat_ollama(chat, model=model, temperature=0.0)
    # else:
    # return inference_chat(chat, model, API_URL, API_KEY,
    #                       usage_tracking_jsonl=USAGE_TRACKING_JSONL,
    #                       temperature=temperature)


def _default_adb_path():
    return os.environ.get("ADB") or shutil.which("adb") or "/Users/huangrunxi/Library/Android/sdk/platform-tools/adb"


def _parse_adb_devices(output):
    devices = []
    for line in (output or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device" and parts[0] != "List":
            devices.append(parts[0])
    return devices


def _select_adb_serial(adb_prefix):
    proc = subprocess.run(
        f"{adb_prefix} devices",
        shell=True,
        capture_output=True,
        text=True,
        timeout=8,
    )
    devices = _parse_adb_devices(proc.stdout)
    if not devices:
        raise RuntimeError(f"No adb device found. stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}")
    if len(devices) == 1:
        print(f"[ADB] selected only connected device: {devices[0]}")
        return devices[0]

    tcp_devices = [d for d in devices if re.match(r"^\d+\.\d+\.\d+\.\d+:\d+$", d)]
    usb_devices = [d for d in devices if not d.startswith("adb-") and "._adb" not in d and d not in tcp_devices]
    mdns_devices = [d for d in devices if d.startswith("adb-") or "._adb" in d]
    chosen = (tcp_devices or usb_devices or mdns_devices or devices)[0]
    print(f"[ADB] multiple devices detected={devices}; selected={chosen}. Override with --adb_serial if needed.")
    return chosen


def resolve_adb_prefix(adb_path, adb_serial="", adb_port=5037):
    prefix = str(adb_path or "adb").strip()
    if adb_port and " -P " not in f" {prefix} ":
        prefix = f"{prefix} -P {int(adb_port)}"
    if " -s " not in f" {prefix} ":
        serial = str(adb_serial or "").strip() or _select_adb_serial(prefix)
        prefix = f"{prefix} -s {shlex.quote(serial)}"
    print(f"[ADB] command_prefix={prefix}")
    return prefix


def ensure_adb_ready(adb_prefix):
    proc = subprocess.run(
        f"{adb_prefix} get-state",
        shell=True,
        capture_output=True,
        text=True,
        timeout=8,
    )
    state = (proc.stdout or "").strip()
    if proc.returncode != 0 or state != "device":
        raise RuntimeError(
            f"ADB target is not ready. state={state!r}, stdout={proc.stdout.strip()}, stderr={proc.stderr.strip()}"
        )
    print(f"[ADB] state={state}")


def _demo_keywords(text):
    stop = {
        "the", "and", "for", "with", "from", "this", "that", "open", "click",
        "tap", "press", "button", "screen", "page", "app", "use", "using",
    }
    tokens = re.findall(r"[a-z0-9_\-]{2,}|[\u4e00-\u9fff]", (text or "").lower())
    return [t for t in tokens if t not in stop]


def _demo_node_score(task, node):
    text = node_to_text(node)
    merged = text.lower()
    if not merged:
        return 0.0
    keywords = _demo_keywords(task)
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw and kw in merged)
    score = float(hits) / max(1.0, min(8.0, float(len(keywords))))
    if bool(getattr(node, "clickable", False)):
        score += 0.05
    return score


def _demo_reasoning_response(args):
    if args.demo_reasoning == "wait":
        action = {"name": "mobile_use", "arguments": {"action": "wait"}}
        print("[DemoReasoning] mode=wait action=wait")
    else:
        root = parse_a11y_tree(xml_path=args._current_xml_path)
        nodes = collect_clickable_nodes(root)
        best = None
        best_score = -1.0
        for node in nodes:
            score = _demo_node_score(args.task, node)
            if score > best_score:
                best = node
                best_score = score

        bounds = parse_bounds(getattr(best, "bounds", "")) if best is not None else None
        if bounds:
            x1, y1, x2, y2 = bounds
            width = max(1, int(getattr(args, "_current_width", 1080)))
            height = max(1, int(getattr(args, "_current_height", 2400)))
            cx = int(((x1 + x2) / 2.0) / width * 1000)
            cy = int(((y1 + y2) / 2.0) / height * 1000)
            node_text = node_to_text(best)
            action = {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [cx, cy]}}
            print(
                f"[DemoReasoning] mode=semantic selected score={best_score:.3f} "
                f"coord1000=({cx},{cy}) text={node_text[:120]!r}"
            )
        else:
            action = {"name": "mobile_use", "arguments": {"action": "wait"}}
            print("[DemoReasoning] mode=semantic no clickable candidate; action=wait")

    return (
        "<thinking>\n"
        "Demo reasoning selects one safe next action so the MobileExplorer pipeline can run without a VLM server.\n"
        "</thinking>\n"
        "<tool_call>\n"
        f"{json.dumps(action, ensure_ascii=False)}\n"
        "</tool_call>"
    )


def build_reasoning_func(args):
    if args.demo_reasoning != "off":
        return lambda chat, model=REASONING_MODEL: _demo_reasoning_response(args)
    return lambda chat, model=REASONING_MODEL: get_reasoning_response(
        chat,
        model=model,
        api_url=args.llama_api_url,
        max_tokens=args.max_tokens,
    )


def _coord_to_px(coord, width, height, coord_space="auto"):
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None
    x, y = float(coord[0]), float(coord[1])
    mode = str(coord_space or "auto").strip().lower()
    if mode in {"norm1", "normalized", "0_1"}:
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px, py = int(x * width), int(y * height)
        else:
            px, py = int(x), int(y)
    elif mode in {"norm1000", "1000", "0_1000"}:
        if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
            px, py = int((x / 1000.0) * width), int((y / 1000.0) * height)
        else:
            px, py = int(x), int(y)
    elif mode in {"pixel", "px"}:
        px, py = int(x), int(y)
    else:
        # auto: keep old behavior first to avoid breaking existing pixel outputs.
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px, py = int(x * width), int(y * height)
        else:
            px, py = int(x), int(y)
    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))
    return px, py


def _action_signature(action_obj, width, height):
    if not isinstance(action_obj, dict):
        return "none"
    at = str(action_obj.get("action_type", "")).lower()
    ai = action_obj.get("action_inputs", {}) or {}
    coord_space = str(action_obj.get("coord_space") or ai.get("coord_space") or "auto").lower()
    if at in {"click", "long_press"}:
        pt = _coord_to_px(ai.get("coordinate"), width, height, coord_space=coord_space)
        return f"{at}:{pt}"
    if at in {"swipe", "drag"}:
        s = _coord_to_px(ai.get("start_coordinate") or ai.get("coordinate"), width, height, coord_space=coord_space)
        e = _coord_to_px(ai.get("end_coordinate"), width, height, coord_space=coord_space)
        d = ai.get("direction")
        return f"{at}:s={s}:e={e}:d={d}"
    if at == "open_app":
        return f"open_app:{ai.get('content') or ai.get('value') or ai.get('text')}"
    return at


def _settle_profile(action_type: str):
    action_type = (action_type or "").lower()
    if action_type == "open_app":
        return 1.0, 2.4
    if action_type in {"click", "long_press", "swipe", "drag", "press_back", "press_home"}:
        return 0.45, 1.6
    if action_type in {"type", "press_enter"}:
        return 0.30, 1.0
    return 0.20, 0.8


def wait_ui_settle(args, screenshot_path, scale, action_type):
    """
    Wait for post-action UI to settle by frame similarity, not fixed sleep only.
    """
    base_wait, settle_timeout = _settle_profile(action_type)
    time.sleep(base_wait)

    tmp_path = screenshot_path + ".settle_tmp"
    prev_h = None
    start = time.time()
    while time.time() - start < settle_timeout:
        get_screenshot(args, tmp_path, scale=scale)
        cur_h = phash(tmp_path)
        if prev_h is not None:
            diff = prev_h - cur_h
            if diff <= 3:
                break
        prev_h = cur_h
        time.sleep(0.2)

    if os.path.exists(tmp_path):
        shutil.move(tmp_path, screenshot_path)
    else:
        get_screenshot(args, screenshot_path, scale=scale)


########################################
#         SINGLE-STEP MAIN LOOP
########################################
def run_single_step_agent(args):
    """
    单步 Agent 框架：
    每一轮都只做一次 LLM 调用 -> 输出动作 -> 执行 -> 再截图。
    """

    input_dir = "/sdcard" if args.on_device else "./screenshot"
    screenshot_path = os.path.join(input_dir, "screenshot.png")
    xml_path = os.path.join(input_dir, "a11y.xml")

    os.makedirs(input_dir, exist_ok=True)

    print("### Running Single-Step Agent ###")
    print(f"[Config] task={args.task}")
    print(
        f"[Config] demo_reasoning={args.demo_reasoning}, llama_api_url={args.llama_api_url}, "
        f"coord_space={args.coord_space}, explorer_mode={args.explorer_mode}"
    )
    ensure_adb_ready(args.adb_path)
    reasoning_func = build_reasoning_func(args)

    # Initialize unified agent
    agent = MAIOneStepAgent(args.adb_path, coord_space=args.coord_space)

    belief_graph = ProgressiveBeliefGraph.load(args.graph_path)
    guarded_graph = GenerationGuardedGraph(belief_graph)
    graph_distiller = GraphDistiller(GraphDistillerConfig(
        max_graph_facts=args.max_graph_facts,
        max_graph_context_tokens=args.max_graph_context_tokens,
    ))
    graph_gate = GraphReasoningGate(graph_distiller)
    task_id = hashlib.sha1(args.task.encode("utf-8", errors="ignore")).hexdigest()[:12]
    graph_distill_log = os.path.join("explore_results", "graph_distillation.jsonl")
    graph_metrics_log = os.path.join("explore_results", "graph_metrics.jsonl")

    def persist_graph():
        budget_bytes = 0 if args.graph_memory_budget_mb <= 0 else int(args.graph_memory_budget_mb * 1024 * 1024)
        prune = belief_graph.prune_to_budget(budget_bytes, protected_node_ids=recent_graph_nodes[-4:])
        belief_graph.save(args.graph_path)
        return prune

    ui_lock = threading.Lock()
    stop_event = threading.Event()
    rollback_done_event = threading.Event()

    scale = float(getattr(args, "scale", 1.0) or 1.0)
    get_screenshot(args, screenshot_path, scale=scale)
    width, height = Image.open(screenshot_path).size

    explorer = A11yTreeOnlineExplorer(
        adb_path=args.adb_path,
        args=args,
        xml_path=xml_path,
        explore_vis_dir="explore_results",
        ui_lock=ui_lock,
        stop_event=stop_event,
        rollback_done_event=rollback_done_event,
        width=width,
        height=height,
        explorer_mode=args.explorer_mode,
    )

    clues = None
    pending_explore_payload = None

    steps = []
    history = []
    reasoning_vis_dir = "reasoning_results"
    ensure_dir(reasoning_vis_dir)
    last_action_sig = None
    last_no_effect_repeat = 0
    execution_feedback = ""
    recent_graph_nodes = []
    taken_edge_ids = []
    inference_call_count = 0
    inference_skip_count = 0
    graph_enhanced_count = 0
    graph_unused_count = 0

    perception_latency_list = []
    screenshot_latency_list = []
    a11y_tree_latency_list = []
    planning_latency_list = []
    operation_latency_list = []
    end_to_end_latency_list = []

    for itr in range(1, args.max_itr + 1):
        start_time = time.time()
        print(f"\n================ Iteration {itr} ================\n")

        # --- Perception ---
        get_screenshot(args, screenshot_path, scale=scale)
        width, height = Image.open(screenshot_path).size
        screenshot_time = time.time()
        screenshot_latency = (screenshot_time - start_time) * 1000
        screenshot_latency_list.append(screenshot_latency)

        get_a11y_tree(args, xml_path)
        a11y_tree_time = time.time()
        a11y_tree_latency = (a11y_tree_time - screenshot_time) * 1000
        a11y_tree_latency_list.append(a11y_tree_latency)

        # tree = parse_a11y_tree(xml_path=xml_path)
        # print_tree(tree)

        perception_end_time = time.time()
        perception_latency = (perception_end_time - start_time) * 1000
        perception_latency_list.append(perception_latency)
        print("[Perception] Captured screenshot:", screenshot_path, f"size=({width},{height})")
        args._current_screenshot_path = screenshot_path
        args._current_xml_path = xml_path
        args._current_width = width
        args._current_height = height

        # Freeze the read generation before any step-i exploration writes occur.
        graph_snapshot = guarded_graph.begin_step()
        current_root = parse_a11y_tree(xml_path=xml_path)
        current_elements = collect_clickable_nodes(current_root)
        current_state = describe_ui_state(current_root, len(current_elements))
        snapshot_node_id = graph_snapshot.match_state(current_state)
        live_node_id = belief_graph.observe_state(current_state)
        if snapshot_node_id:
            recent_graph_nodes.append(snapshot_node_id)
            recent_graph_nodes = recent_graph_nodes[-8:]
        information_need = parse_reasoning_prior(" ".join(history[-3:]), args.task)
        explorer.width, explorer.height = width, height
        explorer.prepare_graph_iteration(
            belief_graph=belief_graph,
            graph_snapshot=graph_snapshot,
            current_node_id=snapshot_node_id,
            live_current_node_id=live_node_id,
            current_state=current_state,
            information_need=information_need,
            step=itr,
            task_id=task_id,
            exploration_policy=args.exploration_policy,
            matrix_ablations=MatrixAblations(
                exact_history=not args.disable_exact_history,
                contextual_history=not args.disable_contextual_history,
                information_need=not args.disable_information_need,
                cost=not args.disable_cost,
                recovery_history=not args.disable_recovery_history,
            ),
            recent_nodes=recent_graph_nodes,
        )

        gate_decision = graph_gate.decide(
            current_node_id=snapshot_node_id,
            graph_snapshot=graph_snapshot,
            information_need=information_need,
            taken_edges=taken_edge_ids,
            recent_nodes=recent_graph_nodes[:-1],
            graph_reasoning=("off" if args.graph_reasoning == "briefing" else args.graph_reasoning),
            token_budget=args.max_graph_context_tokens,
        )
        clues = None
        # Old briefing remains available as an ablation and still reads only i-1 data.
        if args.graph_reasoning == "briefing" and pending_explore_payload:
            source_itr = pending_explore_payload.get("source_itr")
            pending_explore_candidates = pending_explore_payload.get("candidates") or []
            clues = explorer.build_prompt_clues_from_candidates(
                candidates=pending_explore_candidates,
                current_screenshot_path=screenshot_path,
                max_items=4,
                last_reasoning_action=(history[-1] if history else ""),
            )
            if clues:
                clues = (
                    f"[Clue Source] exploration_iteration={source_itr} -> reasoning_iteration={itr}\n"
                    + clues
                )
            print(
                f"[HintGeneration] source_itr={source_itr}, "
                f"branches={len(pending_explore_candidates)}, "
                f"debug={explorer.last_clue_debug}"
            )
            if clues:
                print("[HintGeneration] injected_clues_begin")
                print(clues[:1600])
                print("[HintGeneration] injected_clues_end")
        elif gate_decision.mode == GraphMode.GRAPH_ENHANCED_INFERENCE:
            clues = gate_decision.distillation.context
            graph_enhanced_count += 1
        elif gate_decision.mode == GraphMode.NORMAL_INFERENCE:
            graph_unused_count += 1

        distill_log = gate_decision.distillation.to_log_dict()
        distill_log.update({
            "task_id": task_id,
            "step": itr,
            "node_id": snapshot_node_id,
            "snapshot_generation": graph_snapshot.generation,
            "graph_mode": gate_decision.mode.value,
            "gate_reason": gate_decision.reason,
            "whether_context_was_injected": bool(clues),
        })
        ensure_dir(os.path.dirname(graph_distill_log))
        with open(graph_distill_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(distill_log, ensure_ascii=False) + "\n")

        if execution_feedback:
            fb_block = f"[Execution Feedback from previous step]\n{execution_feedback}\n"
            clues = (clues + "\n" + fb_block) if clues else fb_block

        explorer.set_runtime_focus(history_tail=history[-3:], clues_text=clues)

        action_obj = None
        rollback_info = {"ok": True, "reason": "not_started"}
        explorer_started = False
        skip_edge_id = None
        if gate_decision.mode == GraphMode.SKIP_INFERENCE and gate_decision.reusable_edge is not None:
            action_obj = copy.deepcopy(gate_decision.reusable_edge.action)
            skip_edge_id = gate_decision.reusable_edge.edge_id
            inference_skip_count += 1
            log_event("[GraphGate]", f"skip inference edge={skip_edge_id}")
        else:
            log_event(
                "[Pipeline]",
                f"start real ADB exploration while reasoning (policy={args.exploration_policy}, "
                f"max_steps={args.explore_max_steps}, depth={args.explore_max_depth}, "
                f"leaf_width={args.explore_leaf_width})",
            )
            explorer.start(
                max_steps=args.explore_max_steps,
                max_depth=args.explore_max_depth,
                leaf_width=args.explore_leaf_width,
                time_budget_sec=(args.explore_time_budget_sec if args.explore_time_budget_sec > 0 else None),
            )
            explorer_started = True
            try:
                log_event("[Reasoning]", f"start VLM inference graph_mode={gate_decision.mode.value}")
                inference_call_count += 1
                action_obj = agent.run_step(
                    args.task,
                    screenshot_path,
                    width, height,
                    history=history,
                    llm_api_func=reasoning_func,
                    clues=clues,
                    scale=scale
                )
                log_event("[Reasoning]", "end VLM inference")
            except Exception as exc:
                print(f"[Reasoning] exception={type(exc).__name__}: {exc}")
                action_obj = {
                    "action_type": "wait",
                    "action_inputs": {"seconds": 1},
                    "coord_space": args.coord_space,
                    "raw": {"reasoning_error": str(exc)},
                }
            finally:
                rollback_done_event.clear()
                explorer.stop(
                    min_steps=args.explore_min_steps,
                    min_runtime_sec=args.explore_min_runtime_sec,
                    max_wait_sec=args.explore_stop_wait_sec,
                )
                if explorer.thread is None or not explorer.thread.is_alive():
                    rollback_info = explorer.fast_rollback(step=itr)
                    print(f"[Rollback] final_before_execution={rollback_info}")
                if not rollback_done_event.wait(timeout=args.rollback_wait_timeout_sec):
                    print(f"[Rollback] timeout waiting for rollback_done_event after {args.rollback_wait_timeout_sec:.1f}s")
                    rollback_info = {"ok": False, "reason": "timeout"}

        pending_explore_payload = {
            "source_itr": itr,
            "candidates": explorer.consume_iteration_candidates() if explorer_started else [],
        }
        branch_count = len(pending_explore_payload["candidates"])
        leaf_count = sum(len((c or {}).get("leaf_observations") or []) for c in pending_explore_payload["candidates"])
        print(f"[ExplorationMemory] source_itr={itr}, branches={branch_count}, leaf_observations={leaf_count}")

        planning_end_time = time.time()
        planning_latency = (planning_end_time - perception_end_time) * 1000
        planning_latency_list.append(planning_latency)
        print("[Reasoning] Parsed action:", action_obj)
        rollback_verified = bool((rollback_info or {}).get("ok", False))
        if not rollback_verified:
            print(
                "[Rollback] state not verified after exploration; "
                "skip executing reasoning action to avoid exploration-induced drift."
            )
            execution_feedback = (
                f"rollback_not_verified reason={(rollback_info or {}).get('reason')}; "
                "refresh current page and avoid relying on stale exploration branches"
            )
            history.append(f"skip_execution_unverified_rollback {rollback_info}")
            operation_latency_list.append(0.0)
            step_latency = (time.time() - start_time) * 1000
            end_to_end_latency_list.append(step_latency)
            steps.append({
                "step": itr,
                "operation": "skip_unverified_rollback",
                "rollback": rollback_info,
                "planned_action": action_obj,
            })
            print(f"Step latency: {step_latency:.3f} ms")
            persist_graph()
            continue

        device_width = max(1, int(round(width * scale)))
        device_height = max(1, int(round(height * scale)))
        aligned_edge_id = skip_edge_id or belief_graph.find_edge_for_action(
            live_node_id, action_obj, device_width, device_height
        )
        future_action_rank = explorer.rank_of_action(action_obj) if explorer_started else None
        if aligned_edge_id and skip_edge_id is None:
            belief_graph.record_inference_alignment(aligned_edge_id)
            taken_edge_ids.append(aligned_edge_id)
            taken_edge_ids = taken_edge_ids[-16:]
        with open(graph_metrics_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "record_type": "future_action_alignment",
                "task_id": task_id,
                "step": itr,
                "node_id": live_node_id,
                "aligned_edge_id": aligned_edge_id,
                "rank_of_future_real_action": future_action_rank,
                "top1_future_action_hit": future_action_rank == 1,
                "topK_future_action_coverage": bool(future_action_rank and future_action_rank <= args.explore_leaf_width),
                "graph_history_used": bool(snapshot_node_id),
            }, ensure_ascii=False) + "\n")

        # --- Finish condition ---
        if action_obj:
            action_type = action_obj.get("action_type", "")
            if isinstance(action_type, str) and action_type.lower() in ["finish", "finished", "terminate", "done", "exit", "stop"]:
                print("✅ Task finished by model (by action_type).")
                persist_graph()
                break

        # --- Execution ---
        action_type = (action_obj or {}).get("action_type", "unknown")
        action_sig = _action_signature(action_obj, width, height)
        with ui_lock:
            print(f"[INFO] Action executing...")
            get_screenshot(args, screenshot_path, scale=scale)
            before_hash = phash(screenshot_path)
            executed_action = execute_action(
                action_obj,
                width,
                height,
                args.adb_path,
                coord_scale=scale,
            )
        wait_ui_settle(args, screenshot_path, scale, action_type)
        after_hash = phash(screenshot_path)
        screen_diff = before_hash - after_hash
        # Immediate successor verification updates the same edge statistics used
        # by future exploration ranking and future skip decisions.
        get_a11y_tree(args, xml_path)
        successor_root = parse_a11y_tree(xml_path=xml_path)
        successor_elements = collect_clickable_nodes(successor_root)
        successor_state = describe_ui_state(successor_root, len(successor_elements))
        actual_destination_id = belief_graph.observe_state(successor_state)
        edge_before_verify = belief_graph.edges.get(aligned_edge_id) if aligned_edge_id else None
        expected_destination_id = edge_before_verify.destination_node_id if edge_before_verify else None
        successor_matched = bool(
            aligned_edge_id
            and expected_destination_id
            and expected_destination_id == actual_destination_id
        )
        if aligned_edge_id:
            belief_graph.record_execution_verification(
                aligned_edge_id,
                matched=successor_matched,
                actual_destination=actual_destination_id if successor_matched else None,
            )
        skip_mismatch = False
        if skip_edge_id:
            belief_graph.record_skip_result(skip_edge_id, successor_matched)
            if successor_matched:
                taken_edge_ids.append(skip_edge_id)
                recent_graph_nodes.append(actual_destination_id)
            else:
                skip_mismatch = True
        no_effect = screen_diff <= 3
        if no_effect:
            if action_sig == last_action_sig:
                last_no_effect_repeat += 1
            else:
                last_no_effect_repeat = 1
            execution_feedback = (
                f"last_action={executed_action} had tiny screen change(diff={screen_diff}); "
                f"avoid repeating same action/coordinate on unchanged page"
            )
            if last_no_effect_repeat >= 2:
                execution_feedback += f"; repeated_no_effect_count={last_no_effect_repeat}"
        else:
            last_no_effect_repeat = 0
            execution_feedback = ""
        if skip_mismatch:
            execution_feedback = (
                f"graph_skip_successor_mismatch edge={skip_edge_id}; stop graph skipping and defer to model"
            )
        last_action_sig = action_sig

        # Save a debug frame for reasoning action (same visual style as exploration).
        width, height = Image.open(screenshot_path).size
        action_inputs = (action_obj or {}).get("action_inputs", {}) or {}
        coord_space = str((action_obj or {}).get("coord_space") or action_inputs.get("coord_space") or "auto").lower()

        marker_xy = None
        marker_bounds = None
        coord = action_inputs.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            marker_xy = _coord_to_px(coord, width, height, coord_space=coord_space)

        if action_type in ["swipe", "drag"]:
            sc = action_inputs.get("start_coordinate") or action_inputs.get("coordinate")
            ec = action_inputs.get("end_coordinate")
            if isinstance(sc, (list, tuple)) and len(sc) == 2:
                marker_xy = _coord_to_px(sc, width, height, coord_space=coord_space)
            if isinstance(ec, (list, tuple)) and len(ec) == 2 and marker_xy is not None:
                ex, ey = _coord_to_px(ec, width, height, coord_space=coord_space)
                marker_bounds = (
                    min(marker_xy[0], ex),
                    min(marker_xy[1], ey),
                    max(marker_xy[0], ex),
                    max(marker_xy[1], ey),
                )

        mark_and_save_explore_click(
            screenshot_path=screenshot_path,
            save_dir=reasoning_vis_dir,
            step_idx=itr,
            xy=marker_xy,
            bounds=marker_bounds,
            text=f"reasoning_action={action_type} | inputs={action_inputs} | executed={executed_action}",
            extra_lines=[
                f"iteration={itr}",
                f"task={args.task}",
                f"screen_diff={screen_diff}",
                f"no_effect={no_effect}",
                f"graph_mode={gate_decision.mode.value}",
                f"graph_edge_id={aligned_edge_id}",
                f"successor_matched={successor_matched}",
                f"future_action_rank={future_action_rank}",
                f"history_tail={history[-4:] if history else []}",
            ],
            bottom_lines=(
                ["[Injected Clues for This Reasoning Step]"]
                + explorer.get_last_clue_debug_lines()
                + [ln[:140] for ln in (clues or "None").splitlines()[:14]]
            ),
        )

        if executed_action is not None:
            if no_effect:
                history.append(f"{executed_action} [NO_EFFECT diff={screen_diff}]")
            else:
                history.append(str(executed_action))
            explorer.action_history.append(action_obj)

        steps.append({
            "step": itr,
            "operation": "execution",
            "executed_action": executed_action,
            "graph_mode": gate_decision.mode.value,
            "graph_edge_id": aligned_edge_id,
            "successor_matched": successor_matched,
            "rank_of_future_real_action": future_action_rank,
        })

        persist_graph()

        operation_end_time = time.time()
        operation_latency = (operation_end_time - planning_end_time) * 1000
        operation_latency_list.append(operation_latency)
        print("[Execution] Action done:", executed_action)

        end_time = time.time()
        step_latency = (end_time - start_time) * 1000
        end_to_end_latency_list.append(step_latency)
        print(f"Perception latency: {perception_latency:.3f} ms, "
              f"Screenshot latency: {screenshot_latency:.3f} ms, A11Y Tree latency: {a11y_tree_latency:.3f} ms, "
              f"Planning latency: {planning_latency:.3f} ms, "
              f"Operation latency: {operation_latency:.3f} ms",)
        print(f"Step latency: {step_latency:.3f} ms",)


    safe_avg = lambda values: (sum(values) / len(values)) if values else 0.0
    avg_perception_latency = safe_avg(perception_latency_list)
    avg_screenshot_latency = safe_avg(screenshot_latency_list)
    avg_a11y_tree_latency = safe_avg(a11y_tree_latency_list)
    avg_planning_latency = safe_avg(planning_latency_list)
    avg_operation_latency = safe_avg(operation_latency_list)
    avg_end_to_end_latency = safe_avg(end_to_end_latency_list)

    print("\n=== Finished all iterations ===")
    print(f"Perception latency: {avg_perception_latency:.3f} ms, "
          f"Screenshot latency: {avg_screenshot_latency:.3f} ms, A11Y Tree latency: {avg_a11y_tree_latency:.3f} ms, "
          f"Planning Latency: {avg_planning_latency:.3f} ms, "
          f"Operation Latency: {avg_operation_latency:.3f} ms, "
          f"End-to-end latency: {avg_end_to_end_latency:.3f} ms")
    print(
        f"Graph nodes={len(belief_graph.nodes)}, edges={len(belief_graph.edges)}, "
        f"generation={belief_graph.generation}, inference_calls={inference_call_count}, "
        f"skips={inference_skip_count}, enhanced={graph_enhanced_count}, unused={graph_unused_count}"
    )
    persist_graph()

    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    task = "Set an alarm for 8:00 AM."
    # task = "Search for attractions in Los Angeles in Trip App and open the first attraction."
    # task = "Run the stopwatch"
    parser.add_argument("--task", type=str, default=task,
                        help="Set a clock at 8:00")
    parser.add_argument("--max_itr", type=int, default=10,
                        help="Maximum iterations for the agent")
    parser.add_argument("--adb_path", type=str, default=_default_adb_path(), help="ADB executable path or prefix.")
    parser.add_argument("--adb_serial", type=str, default="", help="ADB serial to target when multiple devices are connected.")
    parser.add_argument("--adb_port", type=int, default=5037, help="ADB server port.")
    parser.add_argument("--adb_cmd_timeout", type=float, default=8.0, help="Timeout for short ADB commands.")
    parser.add_argument("--adb_retries", type=int, default=1, help="Retry count for short ADB commands.")
    parser.add_argument("--screenshot_path", type=str, default="./screenshot/screenshot.png", help="Screenshot path.")
    parser.add_argument("--on_device", action="store_true", help="Run on-device or on server.")
    parser.add_argument("--scale", type=float, default=1.0, help="Screenshot downscale factor (>1 means smaller image).")
    parser.add_argument(
        "--llama_api_url",
        type=str,
        default="http://localhost:8081/v1/chat/completions",
        help="OpenAI-compatible llama.cpp/VLM chat completions endpoint.",
    )
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum reasoning output tokens.")
    parser.add_argument(
        "--demo_reasoning",
        type=str,
        default="off",
        choices=["off", "semantic", "wait"],
        help="Use only for demos without a running VLM server; default off uses the real VLM endpoint.",
    )
    parser.add_argument(
        "--explorer_mode",
        type=str,
        default="task",
        choices=["collect_demo", "task"],
        help="collect_demo prioritizes coverage/traces; task prioritizes strict task relevance.",
    )
    parser.add_argument("--explore_max_steps", type=int, default=6, help="Max exploration actions per reasoning window.")
    parser.add_argument("--explore_max_depth", type=int, default=2, help="Depth-bound for each exploration branch.")
    parser.add_argument("--explore_leaf_width", type=int, default=3, help="Leaf candidates to probe per branch.")
    parser.add_argument(
        "--exploration_policy",
        choices=["information_need", "graph_matrix"],
        default="graph_matrix",
        help="Candidate ranking ablation: legacy information-need scorer or predictive graph matrix.",
    )
    parser.add_argument(
        "--graph_reasoning",
        choices=["off", "briefing", "distill", "skip_only", "distill_and_skip"],
        default="distill_and_skip",
        help="How the previous-generation graph is consumed before VLM inference.",
    )
    parser.add_argument("--graph_path", default="./explore_results/progressive_belief_graph.json")
    parser.add_argument(
        "--graph_memory_budget_mb",
        type=float,
        default=0.0,
        help="Approximate graph-cache budget in MiB; 0 keeps the full graph.",
    )
    parser.add_argument("--max_graph_facts", type=int, default=3)
    parser.add_argument("--max_graph_context_tokens", type=int, default=64)
    parser.add_argument("--disable_exact_history", action="store_true")
    parser.add_argument("--disable_contextual_history", action="store_true")
    parser.add_argument("--disable_information_need", action="store_true")
    parser.add_argument("--disable_cost", action="store_true")
    parser.add_argument("--disable_recovery_history", action="store_true")
    parser.add_argument(
        "--explore_min_steps",
        type=int,
        default=1,
        help="Minimum XML-tree exploration clicks to wait for before stopping the explorer after reasoning.",
    )
    parser.add_argument(
        "--explore_min_runtime_sec",
        type=float,
        default=0.0,
        help="Minimum explorer runtime before stopping it after reasoning.",
    )
    parser.add_argument(
        "--explore_stop_wait_sec",
        type=float,
        default=10.0,
        help="Maximum extra wait for the explorer to reach --explore_min_steps/--explore_min_runtime_sec.",
    )
    parser.add_argument(
        "--explore_time_budget_sec",
        type=float,
        default=0.0,
        help="Optional exploration time cap per reasoning step; 0 means stop when reasoning returns.",
    )
    parser.add_argument(
        "--rollback_wait_timeout_sec",
        type=float,
        default=60.0,
        help="Max wait for exploration rollback before executing the reasoning action.",
    )
    parser.add_argument(
        "--coord_space",
        type=str,
        default="norm1000",
        choices=["auto", "pixel", "norm1", "norm1000"],
        help="How to interpret model coordinates for execution and debug markers.",
    )
    args = parser.parse_args()
    args.adb_path = resolve_adb_prefix(args.adb_path, adb_serial=args.adb_serial, adb_port=args.adb_port)

    run_single_step_agent(args)
