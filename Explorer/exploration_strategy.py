import time
import threading

from MobileAgentE.controller import get_a11y_tree, get_screenshot, back
from MobileAgentE.tree import parse_a11y_tree

from Explorer.utils import (
    collect_clickable_nodes,
    select_node_by_semantic,
    click_node_center,
    fallback_action,
    mark_and_save_explore_click,
)

def semantic_pick(root, goal_emb, embed_model, emb_cache, clicked_bounds):
    candidates = collect_clickable_nodes(root)
    if len(candidates) == 0:
        return None, None, None, 0

    node, sim, node_txt = select_node_by_semantic(
        candidates=candidates,
        goal_emb_list=goal_emb,
        embed_model=embed_model,
        emb_cache=emb_cache,
        clicked_bounds=clicked_bounds,
    )
    return node, sim, node_txt, len(candidates)


def pick_topk_nodes(root, k, goal_emb, embed_model, emb_cache, clicked_bounds):
    """
    只选，不点。
    Return: list[(node, sim, node_txt)], n_candidates
    """
    candidates = collect_clickable_nodes(root)
    n_candidates = len(candidates)
    if n_candidates == 0:
        return [], 0

    picked = []
    local_blocked = set()

    for _ in range(min(k, n_candidates)):
        node, sim, node_txt = select_node_by_semantic(
            candidates=candidates,
            goal_emb_list=goal_emb,
            embed_model=embed_model,
            emb_cache=emb_cache,
            clicked_bounds=clicked_bounds | local_blocked,
        )
        if node is None:
            break
        local_blocked.add(getattr(node, "bounds", ""))
        picked.append((node, float(sim), node_txt))

    return picked, n_candidates


def record_click_action_norm(explorer, coordinate_pixel, action_list):
    """
    给 rollback-fallback 的 replay 用：存归一化坐标，避免分辨率/缩放不一致。
    """
    if explorer.width is None or explorer.height is None:
        return
    x, y = coordinate_pixel
    x_norm = float(x) / float(explorer.width)
    y_norm = float(y) / float(explorer.height)
    action_list.append({
        "action_type": "click",
        "action_inputs": {"coordinate": [x_norm, y_norm]}
    })


def click_and_record(explorer, node, sim, node_txt, n_candidates, depth, branch_id, branch_actions):
    """
    单次点击 + 截图标注 + log +（可选）把动作写入 branch_actions。
    """
    step_t0 = time.time()

    act_t0 = time.time()
    with explorer.ui_lock:
        coordinate, bounds = click_node_center(explorer.adb, node)
    explorer.action_latency.append(time.time() - act_t0)

    if not coordinate:
        explorer.log_step({
            "step": explorer.cur_steps + 1,
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
        return False

    explorer.cur_steps += 1
    explorer.clicked_bounds.add(getattr(node, "bounds", ""))

    # 分支动作（给 rollback 失败时 replay 用，只记录树干动作）
    if branch_actions is not None:
        record_click_action_norm(explorer, coordinate, branch_actions)

    explorer.vis_step += 1
    ui_text = getattr(node, "text", "") or ""
    ui_desc = getattr(node, "content_desc", "") or ""

    shot_t0 = time.time()
    get_screenshot(explorer.args, explorer.explore_screenshot_path)
    explorer.screenshot_latency.append(time.time() - shot_t0)

    out = mark_and_save_explore_click(
        screenshot_path=explorer.explore_screenshot_path,
        save_dir=explorer.explore_vis_dir,
        step_idx=explorer.vis_step,
        xy=coordinate,
        bounds=bounds,
        text=f"{ui_text} {ui_desc} | sim={float(sim):.3f}",
    )

    total_latency = time.time() - step_t0
    explorer.total_latency.append(total_latency)

    explorer.log_step({
        "step": explorer.cur_steps,
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
        "depth": depth,
        "branch": branch_id,
        "time_sec": round(total_latency, 3),
    })
    return True


def run_exploration_policy(explorer, max_steps, max_depth, leaf_width, max_branches=None):
    """
    树干：每层只走 top1，直到 depth = max_depth-1
    叶子层：在父节点处取 top leaf_width，逐个点叶子；叶子之间用 back 回到父节点
    每个分支结束：调用 explorer.fast_rollback() 回根（由 explorer 自己实现）
    """
    if max_depth <= 0:
        max_depth = 1
    if leaf_width <= 0:
        leaf_width = 1
    if max_branches is None:
        max_branches = max(1, max_steps // max_depth)

    explorer.cur_steps = 0
    explorer.clicked_bounds = set()

    branches_done = 0

    while branches_done < max_branches and explorer.cur_steps < max_steps:
        if explorer.stop_event.is_set():
            break

        branch_id = branches_done + 1

        # 只记录树干动作，用于 rollback 失败时 home + replay（你的 fast_rollback 里会用）
        branch_actions = []

        depth = 0
        trunk_ok = True

        # -------- 树干：走到叶子父节点 --------
        while depth < max_depth - 1 and explorer.cur_steps < max_steps:
            if explorer.stop_event.is_set():
                trunk_ok = False
                break

            t0 = time.time()
            get_a11y_tree(explorer.args, explorer.xml_path)
            t1 = time.time()
            explorer.adb_tree_latency.append(t1 - t0)

            root = parse_a11y_tree(xml_path=explorer.xml_path)
            t2 = time.time()
            explorer.tree_latency.append(t2 - t1)

            s0 = time.time()
            node, sim, node_txt, n_candidates = semantic_pick(
                root=root,
                goal_emb=explorer.goal_emb,
                embed_model=explorer.embed_model,
                emb_cache=explorer.emb_cache,
                clicked_bounds=explorer.clicked_bounds,
            )
            explorer.selection_latency.append(time.time() - s0)

            if node is None:
                fb = fallback_action(explorer.adb)
                explorer.log_step({
                    "step": explorer.cur_steps + 1,
                    "type": "fallback",
                    "fallback": fb,
                    "n_candidates": n_candidates,
                    "reason": "no_valid_semantic_candidate",
                    "depth": depth + 1,
                    "branch": branch_id,
                })
                trunk_ok = False
                break

            ok = click_and_record(
                explorer=explorer,
                node=node,
                sim=float(sim) if sim is not None else 0.0,
                node_txt=node_txt,
                n_candidates=n_candidates,
                depth=depth + 1,
                branch_id=branch_id,
                branch_actions=branch_actions,   # ✅ 树干动作进入 replay 列表
            )
            if not ok:
                trunk_ok = False
                break

            depth += 1

        # -------- 树干失败：回根换下一枝 --------
        explorer.action_history = branch_actions
        if not trunk_ok or explorer.stop_event.is_set():
            explorer.fast_rollback(max_depth=max_depth, step=branch_id)
            branches_done += 1
            time.sleep(0.05)
            continue

        # -------- 叶子层：取 top leaf_width，逐个点 --------
        if explorer.cur_steps < max_steps and not explorer.stop_event.is_set():
            get_a11y_tree(explorer.args, explorer.xml_path)
            root = parse_a11y_tree(xml_path=explorer.xml_path)

            s0 = time.time()
            leaf_nodes, n_candidates = pick_topk_nodes(
                root=root,
                k=leaf_width,
                goal_emb=explorer.goal_emb,
                embed_model=explorer.embed_model,
                emb_cache=explorer.emb_cache,
                clicked_bounds=explorer.clicked_bounds,
            )
            explorer.selection_latency.append(time.time() - s0)

            for leaf_node, leaf_sim, leaf_txt in leaf_nodes:
                if explorer.cur_steps >= max_steps or explorer.stop_event.is_set():
                    break

                # 叶子动作不进 branch_actions（你之前也提到叶子不该进 replay）
                click_and_record(
                    explorer=explorer,
                    node=leaf_node,
                    sim=leaf_sim,
                    node_txt=leaf_txt,
                    n_candidates=n_candidates,
                    depth=max_depth,
                    branch_id=branch_id,
                    branch_actions=None,          # ✅ 叶子不进 replay
                )

                # 尽量回父节点，准备点下一个叶子
                back(explorer.adb)
                time.sleep(0.4)

        # -------- 分支结束：回根 --------
        explorer.fast_rollback(max_depth=max_depth, step=branch_id)
        branches_done += 1
        time.sleep(0.05)
