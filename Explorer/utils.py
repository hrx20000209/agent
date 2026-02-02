from PIL import Image, ImageDraw, ImageFont
import os
import re

def _mark_and_save_explore_click(
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
