# xml_tree.py

from xml.etree.ElementTree import Element, SubElement, tostring
import re


# 解析单个节点 block 中的 bounds
_BOUNDS_RE = re.compile(
    r"bounds_in_screen \{\s*left: (\d+)\s*top: (\d+)\s*right: (\d+)\s*bottom: (\d+)",
    re.S,
)


def _parse_single_node_block(block: str):
    """
    从一段 'nodes { ... }' 文本里解析出一个 node dict。
    """
    node = {
        "id": None,
        "class": "",
        "pkg": "",
        "text": "",
        "content_desc": "",
        "rid": "",
        "window_id": "0",
        "clickable": False,
        "visible": False,
        "bounds": "0,0,0,0",
        "children": [],
    }

    # bounds_in_screen
    m = _BOUNDS_RE.search(block)
    if m:
        node["bounds"] = f"{m.group(1)},{m.group(2)},{m.group(3)},{m.group(4)}"

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("unique_id:"):
            node["id"] = line.split(":", 1)[1].strip()

        elif line.startswith("class_name:"):
            node["class"] = line.split(":", 1)[1].strip().strip('"')

        elif line.startswith("package_name:"):
            node["pkg"] = line.split(":", 1)[1].strip().strip('"')

        elif line.startswith("text:"):
            node["text"] = line.split(":", 1)[1].strip().strip('"')

        elif line.startswith("content_description:"):
            node["content_desc"] = line.split(":", 1)[1].strip().strip('"')

        elif line.startswith("view_id_resource_name:"):
            node["rid"] = line.split(":", 1)[1].strip().strip('"')

        elif line.startswith("window_id:"):
            node["window_id"] = line.split(":", 1)[1].strip()

        elif line.startswith("is_clickable:"):
            node["clickable"] = line.endswith("true")

        elif line.startswith("is_visible_to_user:"):
            node["visible"] = line.endswith("true")

        elif line.startswith("child_ids:"):
            cid = line.split(":", 1)[1].strip()
            if cid.isdigit():
                node["children"].append(cid)

    return node


def _parse_androidcontrol_text_forest(text: str):
    """
    把整段 accessibility tree 文本解析成 node 列表。
    text 是类似你贴出来的那种：
        nodes {
          unique_id: 1
          ...
        }
        nodes {
          unique_id: 2
          ...
        }
    """
    # 按 "nodes {" 切分
    parts = text.split("nodes {")
    if len(parts) <= 1:
        return []

    blocks = ["nodes {" + p for p in parts[1:]]
    nodes = []
    for b in blocks:
        nodes.append(_parse_single_node_block(b))
    return nodes


def _nodes_to_simple_xml(nodes):
    """
    把解析出来的 node dict 列表转成极简 XML 字符串。
    只保留 bounds/text/content_desc/rid/children。
    """
    if not nodes:
        return "<forest />"

    forest = Element("forest")

    # 按 window 分组
    win2nodes = {}
    for n in nodes:
        win_id = n["window_id"]
        win2nodes.setdefault(win_id, []).append(n)

    for win_id, ns in win2nodes.items():
        w = SubElement(forest, "window", {"id": str(win_id)})

        # id → node
        id2node = {n["id"]: n for n in ns if n["id"] is not None}

        # 找 root 节点
        all_children = set()
        for n in ns:
            for cid in n["children"]:
                all_children.add(cid)

        roots = [n for n in ns if n["id"] not in all_children]
        if not roots:
            roots = [n for n in ns if n["id"] is not None]

        def add_node(parent_xml, n):
            attrs = {
                "bounds": n["bounds"],
            }

            if n["text"]:
                attrs["text"] = n["text"]
            if n["content_desc"]:
                attrs["content_desc"] = n["content_desc"]
            if n["rid"]:
                attrs["rid"] = n["rid"]

            elem = SubElement(parent_xml, "node", attrs)

            for cid in n["children"]:
                child = id2node.get(cid)
                if child is not None:
                    add_node(elem, child)

        for r in roots:
            add_node(w, r)

    return tostring(forest, encoding="unicode")


def convert_forest_any_to_xml(raw):
    """
    现在只关心 AndroidControl 的 str 格式。
    raw:
      - str: AndroidControl dump 出来的 accessibility_trees 文本
      - 其他类型直接返回空树
    """
    # case 1: AndroidControl 文本
    if isinstance(raw, str):
        nodes = _parse_androidcontrol_text_forest(raw)
        return _nodes_to_simple_xml(nodes)

    # 其他情况（bytes 等）先不管
    return "<forest />"
