import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import numpy as np
import time

embed_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")


class Node:
    """Accessibility tree node with typed attributes"""
    def __init__(
        self,
        index=None,
        text=None,
        resource_id=None,
        class_name=None,
        package=None,
        content_desc=None,
        checkable=False,
        checked=False,
        clickable=False,
        focusable=False,
        focused=False,
        scrollable=False,
        long_clickable=False,
        password=False,
        selected=False,
        bounds=None,
        drawing_order=None,
        hint=None,
        uid=None,
        children=None,
    ):
        self.index = index
        self.text = text
        self.resource_id = resource_id
        self.class_name = class_name
        self.package = package
        self.content_desc = content_desc
        self.checkable = checkable
        self.checked = checked
        self.clickable = clickable
        self.focusable = focusable
        self.focused = focused
        self.scrollable = scrollable
        self.long_clickable = long_clickable
        self.password = password
        self.selected = selected
        self.bounds = bounds
        self.drawing_order = drawing_order
        self.hint = hint
        self.uid = uid
        self.children = children or []

    def to_dict(self):
        """Convert to JSON-style dict recursively"""
        return {
            "uid": self.uid,
            "index": self.index,
            "text": self.text,
            "resource_id": self.resource_id,
            "class": self.class_name,
            "package": self.package,
            "content_desc": self.content_desc,
            "checkable": self.checkable,
            "checked": self.checked,
            "clickable": self.clickable,
            "focusable": self.focusable,
            "focused": self.focused,
            "scrollable": self.scrollable,
            "long_clickable": self.long_clickable,
            "password": self.password,
            "selected": self.selected,
            "bounds": self.bounds,
            "drawing_order": self.drawing_order,
            "hint": self.hint,
            "children": [child.to_dict() for child in self.children],
        }

    def __repr__(self):
        desc = (
            f"{self.class_name or 'Unknown'} "
            f"text='{self.text or ''}' "
            f"desc='{self.content_desc or ''}' "
            f"clickable={self.clickable}"
        )
        return f"<Node {desc}>"


def parse_bool(s):
    return s == "true"


def _keep_leaf(node: Node) -> bool:
    """
    Keep only useful leaf nodes.
    Useful means:
    - actionable (clickable/focusable/scrollable/long_clickable/checkable)
    OR
    - has strong semantic (text/content_desc)
    """
    if (
        node.clickable
        or node.long_clickable
        or node.focusable
        or node.scrollable
        or node.checkable
    ):
        return True

    if node.text and node.text.strip():
        return True

    if node.content_desc and node.content_desc.strip():
        return True

    return False


def parse_node_collect_leaves(element):
    """
    Parse XML node and collect only useful leaf nodes.
    Return: list[Node]
    """
    attrib = element.attrib

    children_elems = element.findall("node")
    if children_elems:
        leaves = []
        for ch in children_elems:
            leaves.extend(parse_node_collect_leaves(ch))
        return leaves

    # leaf element -> build Node
    node = Node(
        index=attrib.get("index"),
        text=attrib.get("text"),
        resource_id=attrib.get("resource-id"),
        class_name=attrib.get("class"),
        package=attrib.get("package"),
        content_desc=attrib.get("content-desc"),
        checkable=parse_bool(attrib.get("checkable", "false")),
        checked=parse_bool(attrib.get("checked", "false")),
        clickable=parse_bool(attrib.get("clickable", "false")),
        focusable=parse_bool(attrib.get("focusable", "false")),
        focused=parse_bool(attrib.get("focused", "false")),
        scrollable=parse_bool(attrib.get("scrollable", "false")),
        long_clickable=parse_bool(attrib.get("long-clickable", "false")),
        password=parse_bool(attrib.get("password", "false")),
        selected=parse_bool(attrib.get("selected", "false")),
        bounds=attrib.get("bounds"),
        drawing_order=attrib.get("drawing-order"),
        hint=attrib.get("hint"),
        children=[],
    )

    return [node] if _keep_leaf(node) else []


def parse_a11y_tree(xml_path):
    """
    Parse uiautomator dump file into a pruned Node tree:
    - keep only useful leaf nodes
    - assign each leaf a uid (E0001, E0002, ...)
    """
    et_start_time = time.time()
    tree = ET.parse(xml_path)
    et_end_time = time.time()
    et_latency = (et_end_time - et_start_time) * 1000
    hierarchy = tree.getroot()
    hierarchy_end_time = time.time()
    hierarchy_latency = (hierarchy_end_time - et_end_time) * 1000
    first = hierarchy.find("node")
    first_end_time = time.time()
    first_latency = (first_end_time - hierarchy_end_time) * 1000

    leaves = parse_node_collect_leaves(first)

    leaves_end_time = time.time()
    leaves_latency = (leaves_end_time - first_end_time) * 1000

    for i, n in enumerate(leaves):
        n.uid = f"E{i+1:04d}"

    print(f"ET Latency:         {et_latency:.3f} ms\n"
          f"Hierarchy Latency:  {hierarchy_latency:.3f} ms\n"
          f"First Latency:      {first_latency:.3f} ms\n"
          f"Leaves Latency:     {leaves_latency:.3f} ms\n")

    # virtual root
    return Node(class_name="LEAF_ROOT", uid="ROOT", children=leaves)


def print_tree(node, level=0, max_depth=None, max_children=None):
    if max_depth is not None and level > max_depth:
        return

    indent = "  " * level

    # 关键属性
    brief = (
        f"uid={node.uid} "
        f"text='{node.text or ''}' "
        f"desc='{node.content_desc or ''}' "
        f"clickable={node.clickable} "
        f"focusable={node.focusable} "
        f"scrollable={node.scrollable} "
        f"selected={node.selected} "
        f"bounds={node.bounds}"
    )

    print(f"{indent}- {brief}")

    kids = node.children
    if max_children is not None:
        kids = kids[:max_children]
    for c in kids:
        print_tree(c, level + 1, max_depth=max_depth, max_children=max_children)
    if max_children is not None and len(node.children) > max_children:
        print(f"{indent}  ...(+{len(node.children)-max_children} more)")


def find_app_icon(tree, app_name):
    """
    在 XML tree 中查找与 app_name 匹配的节点。
    匹配策略：
      1) 完整匹配 text 或 content-desc
      2) 部分匹配（lowercase contains）
    """
    if not app_name:
        return None

    name = app_name.lower()
    results = []

    def dfs(node):
        t = (node.text or "").lower()
        c = (node.content_desc or "").lower()

        # 完整匹配
        if t == name or c == name:
            results.append(node)
            return

        # 模糊匹配（例如 "note" 匹配 "Notes"）
        if name in t or name in c:
            results.append(node)

        for child in node.children:
            dfs(child)

    dfs(tree)

    if results:
        return results[0]   # 只取第一个最相似的

    return None


_embedding_cache = {}

def embed_text(text: str):
    """高性能 embedding + 缓存，避免重复计算"""
    if text in _embedding_cache:
        return _embedding_cache[text]

    emb = embed_model.encode([text])[0]  # shape (384,)
    emb = emb.astype(np.float32)

    _embedding_cache[text] = emb
    return emb


def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_app_icon_embedding(tree, app_name: str, threshold=0.25):
    """
    使用 paraphrase-MiniLM-L6-v2 做语义匹配。
    threshold: 最低接受相似度，避免匹配到错误图标。
    """
    if not app_name:
        return None

    query_emb = embed_text(app_name)
    candidates = []

    def dfs(node):
        texts = []
        if node.text:
            texts.append(node.text.strip())
        if node.content_desc:
            texts.append(node.content_desc.strip())

        for t in texts:
            emb = embed_text(t)
            sim = cosine_sim(query_emb, emb)
            candidates.append((sim, node, t))

        for child in node.children:
            dfs(child)

    dfs(tree)

    if not candidates:
        return None

    # 排序
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_sim, best_node, best_text = candidates[0]

    print(f"[EmbeddingMatcher] Best match: '{best_text}' (sim={best_sim:.3f})")

    if best_sim < threshold:
        print("[EmbeddingMatcher] similarity too low → no reliable match.")
        return None

    return best_node


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

        if bounds and "[" in bounds and "]" in bounds:
            ok = False
            if clickable is True:
                ok = True
            elif text.strip():
                ok = True
            if enabled is False:
                ok = False

            if ok:
                candidates.append(n)

        for c in getattr(n, "children", []) or []:
            dfs(c)

    dfs(root)
    return candidates


if __name__ == "__main__":
    start_time = time.time()
    tree = parse_a11y_tree("../screenshot/ui.xml")
    end_time = time.time()
    end_to_end_latency = (end_time - start_time) * 1000
    print(f"Latency: {end_to_end_latency:.3f} ms")
    print_tree(tree)
