"""Progressive, generation-guarded belief graph for MobileExplorer.

The graph is deliberately deterministic and dependency-free.  Exploration and
execution write to the live graph; reasoning consumes an immutable snapshot
captured before the current exploration window starts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _rate(success: int, failure: int) -> Optional[float]:
    total = int(success) + int(failure)
    return None if total <= 0 else float(success) / float(total)


def _mean(total: float, count: int) -> Optional[float]:
    return None if int(count) <= 0 else float(total) / float(count)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", (text or "").lower())


@dataclass(frozen=True)
class UIStateDescriptor:
    signature: str
    labels: Tuple[str, ...] = ()
    package: str = ""
    coarse_context: str = "generic"
    candidate_element_count: int = 0


def describe_ui_state(root: Any, candidate_element_count: int = 0) -> UIStateDescriptor:
    """Build a stable-enough structural state key from an accessibility tree."""
    structural: List[str] = []
    labels: List[str] = []
    packages: Dict[str, int] = {}

    def walk(node: Any) -> None:
        if node is None:
            return
        rid = str(getattr(node, "resource_id", "") or "").strip().lower()
        cls = str(getattr(node, "class_name", "") or "").strip().lower()
        role = cls.rsplit(".", 1)[-1] if cls else "node"
        bounds = str(getattr(node, "bounds", "") or "")
        package = str(getattr(node, "package", "") or "").strip().lower()
        text = str(getattr(node, "text", "") or getattr(node, "content_desc", "") or "").strip()
        if package:
            packages[package] = packages.get(package, 0) + 1
        # Resource-id and geometry carry identity; text is only a fallback.
        clickable = bool(getattr(node, "clickable", False))
        identity_text = rid.rsplit("/", 1)[-1] if rid else (" ".join(_tokens(text)[:3]) if clickable else "structural")
        structural.append(f"{package}|{role}|{identity_text}|{bounds}")
        if text and len(labels) < 80:
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned and cleaned not in labels:
                labels.append(cleaned[:80])
        for child in getattr(node, "children", None) or []:
            walk(child)

    walk(root)
    package = max(packages, key=packages.get) if packages else ""
    role_counts: Dict[str, int] = {}
    for item in structural:
        role = item.split("|", 2)[1]
        role_counts[role] = role_counts.get(role, 0) + 1
    coarse = ",".join(f"{k}:{v}" for k, v in sorted(role_counts.items())[:8]) or "generic"
    payload = "\n".join(sorted(structural))
    signature = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:24]
    return UIStateDescriptor(
        signature=signature,
        labels=tuple(labels),
        package=package,
        coarse_context=coarse,
        candidate_element_count=max(0, int(candidate_element_count)),
    )


@dataclass
class GraphNode:
    node_id: str
    signature: str
    package: str = ""
    coarse_context: str = "generic"
    discovered_labels: List[str] = field(default_factory=list)
    visit_count: int = 0
    decision_entropy: float = 1.0
    candidate_element_count: int = 0
    explored_element_identities: Set[str] = field(default_factory=set)
    outgoing_edge_ids: List[str] = field(default_factory=list)
    graph_generation: int = 0
    last_updated_generation: int = 0

    @property
    def exploration_coverage(self) -> float:
        denom = max(1, int(self.candidate_element_count))
        return _clip(len(self.explored_element_identities) / float(denom))


@dataclass
class GraphEdge:
    edge_id: str
    source_node_id: str
    element_identity: str
    action_type: str
    role: str = ""
    probe_type: str = "click"
    coarse_context: str = "generic"
    information_need_type: str = "generic"
    action: Dict[str, Any] = field(default_factory=dict)
    bounds: Optional[Tuple[int, int, int, int]] = None
    destination_node_id: Optional[str] = None
    discovered_labels: List[str] = field(default_factory=list)
    edge_status: str = "SPECULATIVE"
    confidence: float = 0.20
    probe_count: int = 0
    inference_alignment_count: int = 0
    execution_hit_count: int = 0
    execution_miss_count: int = 0
    skip_attempt_count: int = 0
    skip_success_count: int = 0
    cumulative_realized_IG: float = 0.0
    realized_IG_count: int = 0
    rollback_success_count: int = 0
    rollback_failure_count: int = 0
    deep_recovery_count: int = 0
    cumulative_exploration_cost: float = 0.0
    exploration_cost_count: int = 0
    known_inverse_level: int = 0
    cross_package_history: bool = False
    risk_level: float = 0.0
    last_updated_generation: int = 0

    @property
    def alignment_rate(self) -> Optional[float]:
        return None if self.probe_count <= 0 else _clip(self.inference_alignment_count / float(self.probe_count))

    @property
    def execution_hit_rate(self) -> Optional[float]:
        return _rate(self.execution_hit_count, self.execution_miss_count)

    @property
    def skip_success_rate(self) -> Optional[float]:
        failures = max(0, self.skip_attempt_count - self.skip_success_count)
        return _rate(self.skip_success_count, failures)

    @property
    def mean_realized_information_gain(self) -> Optional[float]:
        return _mean(self.cumulative_realized_IG, self.realized_IG_count)

    @property
    def rollback_success_rate(self) -> Optional[float]:
        return _rate(self.rollback_success_count, self.rollback_failure_count)

    @property
    def mean_exploration_cost(self) -> Optional[float]:
        return _mean(self.cumulative_exploration_cost, self.exploration_cost_count)

    def recompute_belief(self) -> None:
        exec_rate = self.execution_hit_rate
        rollback_rate = self.rollback_success_rate
        align_rate = self.alignment_rate
        skip_rate = self.skip_success_rate
        evidence = min(1.0, math.log1p(self.probe_count + self.execution_hit_count) / math.log(6.0))
        confidence = 0.18 + 0.16 * evidence
        confidence += 0.28 * (exec_rate if exec_rate is not None else 0.45)
        confidence += 0.16 * (align_rate if align_rate is not None else 0.35)
        confidence += 0.14 * (rollback_rate if rollback_rate is not None else 0.70)
        confidence += 0.08 * (skip_rate if skip_rate is not None else 0.0)
        confidence -= 0.30 * _clip(self.risk_level)
        self.confidence = _clip(confidence)

        if self.execution_miss_count >= 2 and self.execution_miss_count > self.execution_hit_count:
            self.edge_status = "INVALID"
        elif rollback_rate is not None and self.rollback_failure_count >= 2 and rollback_rate < 0.5:
            self.edge_status = "BLOCKED"
        elif self.skip_success_count >= 1 or self.execution_hit_count >= 2:
            self.edge_status = "REUSABLE"
        elif self.execution_hit_count >= 1:
            self.edge_status = "VERIFIED"
        elif self.inference_alignment_count >= 1:
            self.edge_status = "INFERENCE_ALIGNED"
        elif self.probe_count >= 1:
            self.edge_status = "OBSERVED"
        else:
            self.edge_status = "SPECULATIVE"


@dataclass(frozen=True)
class ContextualEdgeStatistics:
    probe_count: int = 0
    alignment_rate: Optional[float] = None
    execution_hit_rate: Optional[float] = None
    mean_realized_IG: Optional[float] = None
    rollback_success_rate: Optional[float] = None
    mean_exploration_cost: Optional[float] = None


class GraphSnapshot:
    """Immutable view used by one reasoning/exploration decision generation."""

    def __init__(self, generation: int, nodes: Dict[str, GraphNode], edges: Dict[str, GraphEdge]):
        self.generation = int(generation)
        self.nodes = copy.deepcopy(nodes)
        self.edges = copy.deepcopy(edges)

    def node(self, node_id: Optional[str]) -> Optional[GraphNode]:
        return self.nodes.get(str(node_id)) if node_id else None

    def outgoing_edges(self, node_id: Optional[str], include_invalid: bool = False) -> List[GraphEdge]:
        node = self.node(node_id)
        if node is None:
            return []
        edges = [self.edges[eid] for eid in node.outgoing_edge_ids if eid in self.edges]
        if not include_invalid:
            edges = [e for e in edges if e.edge_status not in {"INVALID", "BLOCKED"}]
        return edges

    def match_state(self, descriptor: UIStateDescriptor) -> Optional[str]:
        for node_id, node in self.nodes.items():
            if node.signature == descriptor.signature:
                return node_id
        return None

    def exact_edge(self, node_id: Optional[str], element_identity: str, action_type: str) -> Optional[GraphEdge]:
        for edge in self.outgoing_edges(node_id, include_invalid=True):
            if edge.element_identity == element_identity and edge.action_type == action_type:
                return edge
        return None

    def contextual_statistics(
        self, probe_type: str, role: str, coarse_context: str, information_need_type: str
    ) -> ContextualEdgeStatistics:
        matches = [
            e for e in self.edges.values()
            if e.probe_type == probe_type
            and e.role == role
            and e.coarse_context == coarse_context
            and e.information_need_type == information_need_type
        ]
        if not matches:
            return ContextualEdgeStatistics()
        probes = sum(e.probe_count for e in matches)
        aligns = sum(e.inference_alignment_count for e in matches)
        hits = sum(e.execution_hit_count for e in matches)
        misses = sum(e.execution_miss_count for e in matches)
        rb_ok = sum(e.rollback_success_count for e in matches)
        rb_bad = sum(e.rollback_failure_count for e in matches)
        ig_count = sum(e.realized_IG_count for e in matches)
        cost_count = sum(e.exploration_cost_count for e in matches)
        return ContextualEdgeStatistics(
            probe_count=probes,
            alignment_rate=None if probes <= 0 else aligns / float(probes),
            execution_hit_rate=_rate(hits, misses),
            mean_realized_IG=_mean(sum(e.cumulative_realized_IG for e in matches), ig_count),
            rollback_success_rate=_rate(rb_ok, rb_bad),
            mean_exploration_cost=_mean(sum(e.cumulative_exploration_cost for e in matches), cost_count),
        )

    def reusable_edge(
        self,
        node_id: Optional[str],
        recent_nodes: Sequence[str] = (),
        min_confidence: float = 0.82,
        max_age_generations: int = 40,
        max_entropy: float = 0.35,
    ) -> Optional[GraphEdge]:
        node = self.node(node_id)
        if node is None or float(node.decision_entropy) > float(max_entropy):
            return None
        recent = set(recent_nodes or [])
        candidates = []
        for edge in self.outgoing_edges(node_id):
            if edge.edge_status not in {"REUSABLE", "VERIFIED"}:
                continue
            if edge.risk_level > 0.25 or edge.confidence < min_confidence:
                continue
            if self.generation - edge.last_updated_generation > max_age_generations:
                continue
            if edge.destination_node_id and edge.destination_node_id in recent:
                continue
            rb = edge.rollback_success_rate
            if rb is not None and rb < 0.8:
                continue
            candidates.append(edge)
        return max(candidates, key=lambda e: e.confidence, default=None)

    def reusable_path(self, node_id: Optional[str], max_hops: int = 2, recent_nodes: Sequence[str] = ()) -> List[GraphEdge]:
        out: List[GraphEdge] = []
        current = node_id
        seen = set(recent_nodes or [])
        for _ in range(max(0, int(max_hops))):
            edge = self.reusable_edge(current, recent_nodes=tuple(seen))
            if edge is None:
                break
            out.append(edge)
            if not edge.destination_node_id or edge.destination_node_id in seen:
                break
            seen.add(edge.destination_node_id)
            current = edge.destination_node_id
        return out


class ProgressiveBeliefGraph:
    def __init__(self) -> None:
        self.generation = 0
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.blocked_recovery_contexts: Set[str] = set()
        self._signature_to_node: Dict[str, str] = {}
        self._edge_key_to_id: Dict[Tuple[str, str, str], str] = {}
        self._lock = threading.RLock()

    def _advance(self) -> int:
        self.generation += 1
        return self.generation

    def snapshot(self) -> GraphSnapshot:
        with self._lock:
            return GraphSnapshot(self.generation, self.nodes, self.edges)

    def observe_state(self, descriptor: UIStateDescriptor) -> str:
        with self._lock:
            node_id = self._signature_to_node.get(descriptor.signature)
            gen = self._advance()
            if node_id is None:
                node_id = f"n_{descriptor.signature}"
                self._signature_to_node[descriptor.signature] = node_id
                self.nodes[node_id] = GraphNode(node_id=node_id, signature=descriptor.signature)
            node = self.nodes[node_id]
            node.package = descriptor.package or node.package
            node.coarse_context = descriptor.coarse_context or node.coarse_context
            node.visit_count += 1
            node.candidate_element_count = max(node.candidate_element_count, descriptor.candidate_element_count)
            for label in descriptor.labels:
                if label not in node.discovered_labels and len(node.discovered_labels) < 120:
                    node.discovered_labels.append(label)
            node.graph_generation = gen
            node.last_updated_generation = gen
            self._recompute_node_entropy(node_id)
            return node_id

    def record_probe(
        self,
        source_node_id: str,
        element_identity: str,
        action_type: str,
        role: str,
        probe_type: str,
        coarse_context: str,
        information_need_type: str,
        action: Dict[str, Any],
        destination_node_id: Optional[str],
        discovered_labels: Iterable[str],
        exploration_cost: float,
        realized_information_gain: Optional[float],
        bounds: Optional[Tuple[int, int, int, int]] = None,
        risk_level: float = 0.0,
    ) -> str:
        with self._lock:
            key = (source_node_id, element_identity, action_type)
            edge_id = self._edge_key_to_id.get(key)
            if edge_id is None:
                digest = hashlib.sha1("|".join(key).encode()).hexdigest()[:16]
                edge_id = f"e_{digest}"
                self._edge_key_to_id[key] = edge_id
                self.edges[edge_id] = GraphEdge(
                    edge_id=edge_id,
                    source_node_id=source_node_id,
                    element_identity=element_identity,
                    action_type=action_type,
                )
                if source_node_id in self.nodes:
                    self.nodes[source_node_id].outgoing_edge_ids.append(edge_id)
            edge = self.edges[edge_id]
            gen = self._advance()
            edge.role = role
            edge.probe_type = probe_type
            edge.coarse_context = coarse_context
            edge.information_need_type = information_need_type
            edge.action = copy.deepcopy(action)
            edge.bounds = bounds
            edge.destination_node_id = destination_node_id or edge.destination_node_id
            src_node = self.nodes.get(source_node_id)
            dst_node = self.nodes.get(destination_node_id) if destination_node_id else None
            if src_node and dst_node and src_node.package and dst_node.package and src_node.package != dst_node.package:
                edge.cross_package_history = True
            edge.risk_level = max(edge.risk_level, _clip(risk_level))
            edge.probe_count += 1
            edge.cumulative_exploration_cost += max(0.0, float(exploration_cost))
            edge.exploration_cost_count += 1
            if realized_information_gain is not None:
                edge.cumulative_realized_IG += float(realized_information_gain)
                edge.realized_IG_count += 1
            for label in discovered_labels or []:
                clean = re.sub(r"\s+", " ", str(label)).strip()
                if clean and clean not in edge.discovered_labels and len(edge.discovered_labels) < 80:
                    edge.discovered_labels.append(clean[:80])
            edge.last_updated_generation = gen
            if source_node_id in self.nodes:
                node = self.nodes[source_node_id]
                node.explored_element_identities.add(element_identity)
                node.last_updated_generation = gen
            edge.recompute_belief()
            self._recompute_node_entropy(source_node_id)
            return edge_id

    def record_inference_alignment(self, edge_id: Optional[str]) -> None:
        self._record(edge_id, lambda e: setattr(e, "inference_alignment_count", e.inference_alignment_count + 1))

    def record_realized_information_gain(self, edge_id: Optional[str], value: float) -> None:
        def update(edge: GraphEdge) -> None:
            edge.cumulative_realized_IG += float(value)
            edge.realized_IG_count += 1
        self._record(edge_id, update)

    def record_execution_verification(self, edge_id: Optional[str], matched: bool, actual_destination: Optional[str] = None) -> None:
        def update(edge: GraphEdge) -> None:
            if matched:
                edge.execution_hit_count += 1
                if actual_destination:
                    edge.destination_node_id = actual_destination
            else:
                edge.execution_miss_count += 1
        self._record(edge_id, update)

    def record_skip_result(self, edge_id: Optional[str], success: bool) -> None:
        def update(edge: GraphEdge) -> None:
            edge.skip_attempt_count += 1
            if success:
                edge.skip_success_count += 1
        self._record(edge_id, update)

    def record_rollback_result(self, edge_id: Optional[str], success: bool, deep_recovery: bool = False) -> None:
        def update(edge: GraphEdge) -> None:
            if success:
                edge.rollback_success_count += 1
            else:
                edge.rollback_failure_count += 1
                self.blocked_recovery_contexts.add(f"{edge.source_node_id}:{edge.element_identity}")
            if deep_recovery:
                edge.deep_recovery_count += 1
        self._record(edge_id, update)

    def _record(self, edge_id: Optional[str], updater) -> None:
        if not edge_id:
            return
        with self._lock:
            edge = self.edges.get(edge_id)
            if edge is None:
                return
            updater(edge)
            edge.last_updated_generation = self._advance()
            edge.recompute_belief()
            self._recompute_node_entropy(edge.source_node_id)

    def _recompute_node_entropy(self, node_id: str) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        valid = [self.edges[eid] for eid in node.outgoing_edge_ids if eid in self.edges and self.edges[eid].edge_status not in {"INVALID", "BLOCKED"}]
        if not valid:
            node.decision_entropy = 1.0
            return
        weights = [max(0.01, e.confidence) for e in valid]
        total = sum(weights)
        probs = [w / total for w in weights]
        if len(probs) == 1:
            distribution_entropy = 0.0
        else:
            distribution_entropy = -sum(p * math.log(p) for p in probs) / math.log(len(probs))
        maturity_uncertainty = 1.0 - min(1.0, sum(e.probe_count for e in valid) / max(2.0, 2.0 * len(valid)))
        node.decision_entropy = _clip(0.70 * distribution_entropy + 0.30 * maturity_uncertainty)

    def find_edge_for_action(self, node_id: Optional[str], action: Dict[str, Any], width: int, height: int) -> Optional[str]:
        if not node_id or not action:
            return None
        action_type = str(action.get("action_type", "")).lower()
        inputs = action.get("action_inputs") or {}
        coord = inputs.get("coordinate")
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                return None
            candidates = [self.edges[eid] for eid in node.outgoing_edge_ids if eid in self.edges and self.edges[eid].action_type == action_type]
            if action_type != "click" or not isinstance(coord, (list, tuple)) or len(coord) != 2:
                return max(candidates, key=lambda e: e.confidence, default=None).edge_id if candidates else None
            x, y = float(coord[0]), float(coord[1])
            mode = str(action.get("coord_space") or inputs.get("coord_space") or "norm1000").lower()
            if mode in {"norm1000", "1000", "0_1000"}:
                x, y = x / 1000.0 * width, y / 1000.0 * height
            elif mode in {"norm1", "normalized", "0_1"}:
                x, y = x * width, y * height
            def distance(edge: GraphEdge) -> float:
                if not edge.bounds:
                    return 1e9
                x1, y1, x2, y2 = edge.bounds
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return 0.0
                return abs(x - (x1 + x2) / 2.0) + abs(y - (y1 + y2) / 2.0)
            best = min(candidates, key=distance, default=None)
            tolerance = 0.18 * max(width, height)
            return best.edge_id if best is not None and distance(best) <= tolerance else None

    def save(self, path: str) -> None:
        if not path:
            return
        with self._lock:
            payload = {
                "generation": self.generation,
                "nodes": [self._node_to_dict(n) for n in self.nodes.values()],
                "edges": [asdict(e) for e in self.edges.values()],
                "blocked_recovery_contexts": sorted(self.blocked_recovery_contexts),
            }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def approximate_serialized_bytes(self) -> int:
        """Stable cache-size proxy used by the graph-memory evaluation."""
        with self._lock:
            payload = {
                "generation": self.generation,
                "nodes": [self._node_to_dict(n) for n in self.nodes.values()],
                "edges": [asdict(e) for e in self.edges.values()],
                "blocked_recovery_contexts": sorted(self.blocked_recovery_contexts),
            }
            return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def prune_to_budget(self, max_bytes: int, protected_node_ids: Sequence[str] = ()) -> Dict[str, int]:
        """Enforce an approximate graph cache budget with value-aware LRU pruning.

        Verified/reusable and recent edges survive longest.  The method never
        removes protected nodes (normally the current/recent path).
        """
        budget = max(0, int(max_bytes))
        if budget <= 0:
            return {"pruned_edges": 0, "pruned_nodes": 0, "bytes": self.approximate_serialized_bytes()}
        protected = set(protected_node_ids or [])
        removed_edges = 0
        removed_nodes = 0
        with self._lock:
            while self.approximate_serialized_bytes() > budget and self.edges:
                def retention(edge: GraphEdge) -> Tuple[float, int]:
                    status_bonus = {
                        "REUSABLE": 4.0, "VERIFIED": 3.0, "INFERENCE_ALIGNED": 2.0,
                        "OBSERVED": 1.0, "SPECULATIVE": 0.0, "INVALID": -1.0, "BLOCKED": -2.0,
                    }.get(edge.edge_status, 0.0)
                    protected_bonus = 10.0 if edge.source_node_id in protected or edge.destination_node_id in protected else 0.0
                    return (protected_bonus + status_bonus + edge.confidence, edge.last_updated_generation)
                removable = list(self.edges.values())
                if not removable:
                    break
                victim = min(removable, key=retention)
                self.edges.pop(victim.edge_id, None)
                self._edge_key_to_id.pop((victim.source_node_id, victim.element_identity, victim.action_type), None)
                source = self.nodes.get(victim.source_node_id)
                if source and victim.edge_id in source.outgoing_edge_ids:
                    source.outgoing_edge_ids.remove(victim.edge_id)
                    self._recompute_node_entropy(source.node_id)
                removed_edges += 1

            referenced = {e.source_node_id for e in self.edges.values()} | {
                e.destination_node_id for e in self.edges.values() if e.destination_node_id
            }
            orphan_ids = [
                node_id for node_id in self.nodes
                if node_id not in protected and node_id not in referenced
            ]
            orphan_ids.sort(key=lambda nid: self.nodes[nid].last_updated_generation)
            for node_id in orphan_ids:
                if self.approximate_serialized_bytes() <= budget:
                    break
                node = self.nodes.pop(node_id)
                self._signature_to_node.pop(node.signature, None)
                removed_nodes += 1
        return {
            "pruned_edges": removed_edges,
            "pruned_nodes": removed_nodes,
            "bytes": self.approximate_serialized_bytes(),
        }

    @classmethod
    def load(cls, path: str) -> "ProgressiveBeliefGraph":
        graph = cls()
        if not path or not os.path.exists(path):
            return graph
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        graph.generation = int(payload.get("generation", 0))
        for raw in payload.get("nodes", []):
            raw = dict(raw)
            raw["explored_element_identities"] = set(raw.get("explored_element_identities") or [])
            node = GraphNode(**raw)
            graph.nodes[node.node_id] = node
            graph._signature_to_node[node.signature] = node.node_id
        for raw in payload.get("edges", []):
            raw = dict(raw)
            if raw.get("bounds") is not None:
                raw["bounds"] = tuple(raw["bounds"])
            edge = GraphEdge(**raw)
            graph.edges[edge.edge_id] = edge
            graph._edge_key_to_id[(edge.source_node_id, edge.element_identity, edge.action_type)] = edge.edge_id
        graph.blocked_recovery_contexts = set(payload.get("blocked_recovery_contexts") or [])
        return graph

    @staticmethod
    def _node_to_dict(node: GraphNode) -> Dict[str, Any]:
        raw = asdict(node)
        raw["explored_element_identities"] = sorted(node.explored_element_identities)
        return raw


class GenerationGuardedGraph:
    """Small facade that makes the read-generation rule explicit."""

    def __init__(self, graph: ProgressiveBeliefGraph):
        self.graph = graph

    def begin_step(self) -> GraphSnapshot:
        return self.graph.snapshot()

    @staticmethod
    def assert_snapshot(snapshot: GraphSnapshot, expected_generation: int) -> None:
        if snapshot.generation != int(expected_generation):
            raise RuntimeError(
                f"Graph snapshot generation changed: expected={expected_generation}, actual={snapshot.generation}"
            )
