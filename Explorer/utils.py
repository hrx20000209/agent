import os
import re
import json
import time
import random
import imagehash

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer
from MobileAgentE.utils import parse_bounds
from MobileAgentE.controller import tap, back, home


def mark_and_save_explore_click(
    screenshot_path: str,
    save_dir: str,
    step_idx: int,
    xy: tuple,
    bounds: tuple = None,
    text: str = "",
):
    """
    Draw a large visible marker at (x,y) and optional bounds rectangle.
    Save to: {save_dir}/explore_{step_idx:03d}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    img = Image.open(screenshot_path).convert("RGBA")
    W, H = img.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x, y = xy

    # === scale based on image size ===
    # For 1080x2400, marker should be big enough to see.
    scale = max(1.0, W / 1080.0)

    r = int(30 * scale)
    cross = int(50 * scale)
    marker_w = int(8 * scale)
    rect_w = int(8 * scale)

    # === marker circle ===
    draw.ellipse(
        (x - r, y - r, x + r, y + r),
        outline=(255, 0, 0, 255),
        width=marker_w,
    )

    # === crosshair ===
    draw.line((x - cross, y, x + cross, y), fill=(255, 0, 0, 255), width=marker_w)
    draw.line((x, y - cross, x, y + cross), fill=(255, 0, 0, 255), width=marker_w)

    # === bounds rectangle ===
    if bounds is not None:
        x1, y1, x2, y2 = bounds
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(255, 255, 0, 255),
            width=rect_w,
        )

    # === big label bar ===
    label_lines = [
        f"Explore #{step_idx:03d}",
        f"xy=({x},{y})",
    ]
    if bounds is not None:
        x1, y1, x2, y2 = bounds
        label_lines.append(f"bounds=[{x1},{y1}][{x2},{y2}]")
    if text:
        label_lines.append(f'text="{text[:60]}"')

    label = " | ".join(label_lines)

    # choose font size
    font_size = int(44 * scale)  # big
    try:
        # common font path on many Linux systems
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # label background (semi-transparent)
    pad = int(18 * scale)
    x0, y0 = pad, pad

    # compute text box size
    bbox = draw.textbbox((x0, y0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    bg_w = tw + pad * 2
    bg_h = th + pad * 2

    draw.rounded_rectangle(
        (x0 - pad, y0 - pad, x0 - pad + bg_w, y0 - pad + bg_h),
        radius=int(18 * scale),
        fill=(0, 0, 0, 170),  # semi-transparent black
        outline=(255, 255, 255, 220),
        width=max(2, int(3 * scale)),
    )
    draw.text((x0, y0), label, fill=(255, 255, 255, 255), font=font)

    # merge overlay
    out_img = Image.alpha_composite(img, overlay).convert("RGB")

    out_path = os.path.join(save_dir, f"explore_{step_idx:03d}.png")
    out_img.save(out_path)
    return out_path


# very small, common stopword list for GUI tasks
_STOPWORDS = {
    # generic verbs / instructions
    "open","launch","start","go","goto","navigate","enter","type","tap","click","press",
    "select","choose","find","search","look","scroll","swipe","hold","long","back","home",
    "add","create","make","edit","delete","remove","save","send","share","upload","download",
    "turn","enable","disable","set","change","update",
    # function words
    "a","an","the","to","for","in","on","at","of","and","or","with","into","from","as",
    "then","next","after","before","now","please","just","once","when",
    "your","my","me","you","it","this","that",
    # common UI boilerplate
    "button","icon","menu","page","screen","app","application",
}

def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    # keep letters/numbers/space/quotes/hyphen
    s = re.sub(r"[^a-z0-9\s\-\_\"\'\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_task_queries(task_text: str):
    """
    Return a small list of 'content' queries from the task:
    - filtered tokens
    - quoted phrases
    - simple entity-like chunks (numbers, title-like words)
    """
    raw = task_text or ""
    norm = _normalize_text(raw)

    # 1) quoted phrases as high priority (e.g., "OpenAI", 'John Smith')
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', raw)
    quoted_phrases = []
    for a, b in quoted:
        q = (a or b or "").strip()
        if q:
            quoted_phrases.append(q)

    # 2) token filtering
    tokens = [t for t in norm.split(" ") if t and t not in _STOPWORDS]

    # 3) keep numbers / version-like tokens
    numbers = [t for t in tokens if re.search(r"\d", t)]

    # 4) build queries: prefer short phrases (join top tokens)
    #    keep at most 6 tokens to avoid drifting back to long sentences
    content_tokens = [t for t in tokens if t not in numbers]
    content_tokens = content_tokens[:6]

    queries = []
    # quoted phrases first
    for q in quoted_phrases:
        queries.append(q)

    if content_tokens:
        queries.append(" ".join(content_tokens))

    # add individual tokens too (helps short UI labels)
    for t in content_tokens[:5]:
        queries.append(t)

    for n in numbers[:3]:
        queries.append(n)

    # dedup while keeping order
    seen = set()
    out = []
    for q in queries:
        qn = q.strip().lower()
        if qn and qn not in seen:
            seen.add(qn)
            out.append(q.strip())

    return out

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_exploration_clues(history, max_items=4):
    clicks = [r for r in history if r.get("type") == "click"]
    clicks = [r for r in clicks if r.get("best_sim", 0.0) > 0.35]

    uniq = {}
    for r in clicks:
        key = (r.get("node_text", ""), r.get("node_desc", ""), r.get("node_resource_id", ""))
        if key not in uniq or r["best_sim"] > uniq[key]["best_sim"]:
            uniq[key] = r

    items = sorted(uniq.values(), key=lambda x: x["best_sim"], reverse=True)

    lines = ["[Environment Clues from Parallel Exploration]"]
    for r in items[:max_items]:
        name = r.get("node_text") or r.get("node_desc") or ""
        if name.strip():
            lines.append(f"• {name}")

    return "\n".join(lines)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def embed_text(text, embed_model, emb_cache):
    text = (text or "").strip()
    if not text:
        return None
    if text in emb_cache:
        return emb_cache[text]
    emb = embed_model.encode([text])[0].astype(np.float32)
    emb_cache[text] = emb
    return emb


def cosine_sim(goal_emb_list, cand_emb):
    if cand_emb is None or goal_emb_list is None:
        return 0.0

    b = np.asarray(cand_emb, dtype=np.float32)

    # goal_emb_list is a list of vectors
    best = 0.0
    for g in goal_emb_list:
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


def node_to_text(node):
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


def collect_clickable_nodes(root):
    candidates = []

    def dfs(n):
        if n is None:
            return

        bounds = getattr(n, "bounds", None)
        text = getattr(n, "text", "") or ""
        clickable = getattr(n, "clickable", None)
        enabled = getattr(n, "enabled", None)

        if bounds and isinstance(bounds, str) and "[" in bounds and "]" in bounds:
            ok = False
            if clickable is True:
                ok = True
            elif text.strip():
                ok = True

            if enabled is False:
                ok = False

            if ok:
                candidates.append(n)

        for c in getattr(n, "children", None) or []:
            dfs(c)

    dfs(root)
    return candidates


def select_node_by_semantic(
    candidates,
    goal_emb_list,
    embed_model,
    emb_cache,
    clicked_bounds,
):
    best_node = None
    best_score = -1.0
    best_txt = ""

    for n in candidates:
        bounds = getattr(n, "bounds", "") or ""
        if bounds in clicked_bounds:
            continue

        cand_txt = node_to_text(n)
        if not cand_txt:
            continue

        cand_emb = embed_text(cand_txt, embed_model, emb_cache)
        score = cosine_sim(goal_emb_list, cand_emb)

        if getattr(n, "clickable", False):
            score += 0.05

        if score > best_score:
            best_score = score
            best_node = n
            best_txt = cand_txt

    return best_node, best_score, best_txt


def click_node_center(adb_path, node):
    b = parse_bounds(node.bounds)
    if b is None:
        return None, None

    x1, y1, x2, y2 = b
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)

    tap(adb_path, x, y)
    return (x, y), b


def fallback_action(adb_path):
    # keep your original behavior: random back/home
    if random.random() < 0.5:
        back(adb_path)
        return "back"
    home(adb_path)
    return "home"


def build_prompt_clues(history, max_items=4):
    if not history:
        return ""

    clicks = [r for r in history if r.get("type") == "click"]
    clicks = [r for r in clicks if r.get("best_sim", 0.0) > 0.35]
    if not clicks:
        return ""

    uniq = {}
    for r in clicks:
        key = (
            r.get("node_text", ""),
            r.get("node_desc", ""),
            r.get("node_resource_id", ""),
        )
        if key not in uniq or r["best_sim"] > uniq[key]["best_sim"]:
            uniq[key] = r

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
        name = (r.get("node_text") or r.get("node_desc") or "").strip()
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


def print_latency_summary(total_latency, get_tree_latency, parse_tree_latency, selection_latency, action_latency, screenshot_latency, exploration_latency):
    if not total_latency:
        return

    print("\n" + "=" * 90)
    print("📊 Exploration latency summary")
    print(f"Avg Step         : {sum(total_latency) / len(total_latency) * 1000:.3f} ms")
    print(f"Avg Tree ADB     : {sum(get_tree_latency) / len(get_tree_latency) * 1000:.3f} ms")
    print(f"Avg Tree parse   : {sum(parse_tree_latency) / len(parse_tree_latency) * 1000:.3f} ms")
    print(f"Avg Selection    : {sum(selection_latency) / len(selection_latency) * 1000:.3f} ms")
    print(f"Avg Screenshot   : {sum(screenshot_latency) / len(screenshot_latency) * 1000:.3f} ms")
    print(f"Avg Action       : {sum(action_latency) / len(action_latency) * 1000:.3f} ms")
    print(f"Total Exploration: {exploration_latency * 1000:.3f} ms")
    print("=" * 90 + "\n")


def phash(img_path):
    img = Image.open(img_path).convert("L")
    return imagehash.phash(img)  # 64-bit hash


def check_same_image(img1, img2, threshold=8):
    """
    threshold 越小越严格，常用范围 5~12
    """
    h1 = phash(img1)
    h2 = phash(img2)
    diff = h1 - h2  # Hamming distance
    print(f"pHash diff = {diff}")
    return diff <= threshold