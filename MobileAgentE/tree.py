import xml.etree.ElementTree as ET


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
        enabled=True,
        focusable=False,
        focused=False,
        scrollable=False,
        long_clickable=False,
        password=False,
        selected=False,
        bounds=None,
        drawing_order=None,
        hint=None,
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
        self.enabled = enabled
        self.focusable = focusable
        self.focused = focused
        self.scrollable = scrollable
        self.long_clickable = long_clickable
        self.password = password
        self.selected = selected
        self.bounds = bounds
        self.drawing_order = drawing_order
        self.hint = hint
        self.children = children or []

    def to_dict(self):
        """Convert to JSON-style dict recursively"""
        return {
            "index": self.index,
            "text": self.text,
            "resource_id": self.resource_id,
            "class": self.class_name,
            "package": self.package,
            "content_desc": self.content_desc,
            "checkable": self.checkable,
            "checked": self.checked,
            "clickable": self.clickable,
            "enabled": self.enabled,
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
        desc = f"{self.class_name or 'Unknown'} text='{self.text or ''}' desc='{self.content_desc or ''}' clickable={self.clickable}"
        return f"<Node {desc}>"


def parse_bool(s):
    return s == "true"


def parse_node(element):
    """Recursively parse XML <node>"""
    attrib = element.attrib
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
        enabled=parse_bool(attrib.get("enabled", "true")),
        focusable=parse_bool(attrib.get("focusable", "false")),
        focused=parse_bool(attrib.get("focused", "false")),
        scrollable=parse_bool(attrib.get("scrollable", "false")),
        long_clickable=parse_bool(attrib.get("long-clickable", "false")),
        password=parse_bool(attrib.get("password", "false")),
        selected=parse_bool(attrib.get("selected", "false")),
        bounds=attrib.get("bounds"),
        drawing_order=attrib.get("drawing-order"),
        hint=attrib.get("hint"),
        children=[parse_node(child) for child in element.findall("node")],
    )
    return node


def parse_a11y_tree(xml_path):
    """Parse uiautomator dump file into Node tree"""
    tree = ET.parse(xml_path)
    hierarchy = tree.getroot()
    first = hierarchy.find("node")
    return parse_node(first)


def print_tree(node, level=0, max_depth=None, max_children=None):
    if max_depth is not None and level > max_depth:
        return

    indent = "  " * level

    # 关键属性
    brief = (
        f"{node.class_name} "
        f"text='{node.text or ''}' "
        f"desc='{node.content_desc or ''}' "
        f"clickable={node.clickable} "
        f"enabled={node.enabled} "
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


if __name__ == "__main__":
    tree = parse_a11y_tree("../screenshot/a11y.xml")
    print_tree(tree)
