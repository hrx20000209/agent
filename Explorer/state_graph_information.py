"""Candidate matrix and interpretable predictive exploration scorer."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from MobileAgentE.utils import parse_bounds
from Explorer.progressive_belief_graph import GraphEdge, GraphSnapshot, UIStateDescriptor


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _tokens(text: str) -> List[str]:
    text = (text or "").lower()
    found = re.findall(r"[a-z0-9_\-]{2,}|[\u4e00-\u9fff]{1,}", text)
    return list(dict.fromkeys(found))


def _match(query: Iterable[str], text: str) -> float:
    q = [x for x in query if x]
    if not q:
        return 0.0
    low = (text or "").lower()
    hits = sum(1 for token in q if token in low)
    return _clip(hits / max(1.0, min(5.0, float(len(q)))))


@dataclass(frozen=True)
class InformationNeed:
    need_type: str = "generic"
    targets: Tuple[str, ...] = ()
    expected_affordances: Tuple[str, ...] = ()
    unresolved_information: Tuple[str, ...] = ()
    candidate_action_types: Tuple[str, ...] = ("click",)
    risk_terms: Tuple[str, ...] = ()
    summary: str = "task-relevant next UI evidence"


def parse_reasoning_prior(last_model_output: str, goal: str) -> InformationNeed:
    """Deterministic replacement for an LLM-based prior parser."""
    merged = f"{goal or ''} {last_model_output or ''}".strip()
    tokens = _tokens(merged)
    stop = {
        "the", "and", "for", "with", "from", "this", "that", "open", "click", "tap",
        "press", "button", "screen", "page", "app", "use", "using", "action", "history",
    }
    targets = tuple(t for t in tokens if t not in stop)[:16]
    low = merged.lower()
    affordances: List[str] = []
    action_types: List[str] = ["click"]
    if any(k in low for k in ("search", "find", "查找", "搜索")):
        affordances.extend(["search", "input", "field"])
    if any(k in low for k in ("type", "enter", "input", "填写", "输入")):
        affordances.extend(["edittext", "input", "field"])
        action_types.append("type")
    if any(k in low for k in ("select", "choose", "set", "选择", "设置")):
        affordances.extend(["option", "radio", "checkbox", "switch", "picker"])
    if any(k in low for k in ("back", "return", "返回")):
        affordances.extend(["back", "navigate up"])
        action_types.append("press_back")
    need_type = "navigation"
    if any(k in low for k in ("phone", "address", "price", "rating", "号码", "地址", "价格")):
        need_type = "fact_lookup"
    elif "type" in action_types:
        need_type = "input"
    elif any(x in affordances for x in ("option", "radio", "checkbox", "switch", "picker")):
        need_type = "selection"
    unresolved = tuple(targets[:8])
    risk_terms = ("delete", "remove", "purchase", "send", "删除", "购买", "发送")
    summary_tokens = unresolved[:5]
    summary = " ".join(summary_tokens) if summary_tokens else "task-relevant next UI evidence"
    return InformationNeed(
        need_type=need_type,
        targets=targets,
        expected_affordances=tuple(dict.fromkeys(affordances)),
        unresolved_information=unresolved,
        candidate_action_types=tuple(dict.fromkeys(action_types)),
        risk_terms=risk_terms,
        summary=summary,
    )


def element_identity(node: Any, width: int, height: int, action_type: str = "click") -> str:
    """Stable identity using structure/role/position; text is only a fallback."""
    rid = str(getattr(node, "resource_id", "") or "").strip().lower()
    cls = str(getattr(node, "class_name", "") or "").strip().lower()
    role = cls.rsplit(".", 1)[-1] if cls else "node"
    desc = str(getattr(node, "content_desc", "") or "").strip().lower()
    text = str(getattr(node, "text", "") or "").strip().lower()
    bounds = parse_bounds(str(getattr(node, "bounds", "") or ""))
    if bounds:
        x1, y1, x2, y2 = bounds
        cx = (x1 + x2) / 2.0 / max(1.0, float(width))
        cy = (y1 + y2) / 2.0 / max(1.0, float(height))
        position_bucket = f"{min(4, int(cx * 5))}:{min(7, int(cy * 8))}"
    else:
        position_bucket = "na"
    stable_label = rid.rsplit("/", 1)[-1] if rid else (desc or " ".join(_tokens(text)[:3]))
    payload = f"{action_type}|{role}|{stable_label}|{position_bucket}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:20]


@dataclass
class CandidateInformationRow:
    candidate_id: str
    element_identity: str
    text: str
    content_desc: str
    role: str
    probe_type: str
    normalized_position: Tuple[float, float]
    bounds: Optional[Tuple[int, int, int, int]]
    clickable: bool
    scrollable: bool
    enabled: bool
    selected: Optional[bool]
    checked: Optional[bool]
    nearby_context: str
    current_node_id: Optional[str]
    visit_count: int
    decision_entropy: float
    outgoing_edge_count: int
    valid_outgoing_edge_count: int
    explored_element_count: int
    candidate_element_count: int
    exploration_coverage: float
    graph_generation: int
    has_exact_history: bool
    probe_count: Optional[int]
    edge_status: Optional[str]
    confidence: Optional[float]
    inference_alignment_count: Optional[int]
    execution_hit_count: Optional[int]
    execution_miss_count: Optional[int]
    alignment_rate: Optional[float]
    execution_hit_rate: Optional[float]
    skip_attempt_count: Optional[int]
    skip_success_count: Optional[int]
    skip_success_rate: Optional[float]
    mean_realized_information_gain: Optional[float]
    rollback_success_count: Optional[int]
    rollback_failure_count: Optional[int]
    rollback_success_rate: Optional[float]
    mean_exploration_cost: Optional[float]
    known_inverse_level: Optional[int]
    destination_known: bool
    destination_node_id: Optional[str]
    destination_visit_count: Optional[int]
    destination_entropy: Optional[float]
    destination_out_degree: Optional[int]
    destination_valid_out_degree: Optional[int]
    destination_subtree_size: Optional[int]
    destination_known_label_count: Optional[int]
    destination_in_recent_path: Optional[bool]
    discovered_labels: Tuple[str, ...]
    contextual_probe_count: int
    contextual_alignment_rate: Optional[float]
    contextual_execution_hit_rate: Optional[float]
    contextual_mean_realized_IG: Optional[float]
    contextual_rollback_success_rate: Optional[float]
    target_match: float
    expected_affordance_match: float
    unresolved_information_match: float
    candidate_action_type_match: float
    risk_conflict: float
    blocked_element: bool
    blocked_recovery_context: bool
    historical_rollback_success_rate: Optional[float]
    historical_deep_recovery_rate: Optional[float]
    cross_package_history: Optional[bool]
    risk_level: float
    estimated_recoverability: float
    expected_probe_latency: float
    expected_rollback_latency: float
    expected_total_exploration_cost: float
    ui_novelty_score: float
    feasible: bool
    exact_edge_id: Optional[str] = None
    path_probability: float = 0.0
    expected_information_gain: float = 0.0
    predictive_value: float = 0.0
    final_exploration_score: float = 0.0
    selected_for_probe: bool = False

    def to_log_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatrixAblations:
    exact_history: bool = True
    contextual_history: bool = True
    information_need: bool = True
    cost: bool = True
    recovery_history: bool = True


class StateGraphInformationMatrix:
    def __init__(self, generic_probe_cost: float = 0.45, generic_rollback_cost: float = 0.55):
        self.generic_probe_cost = float(generic_probe_cost)
        self.generic_rollback_cost = float(generic_rollback_cost)

    def build(
        self,
        current_state: UIStateDescriptor,
        current_node_id: Optional[str],
        candidate_elements: Sequence[Any],
        graph_snapshot: GraphSnapshot,
        information_need: InformationNeed,
        recovery_history: Optional[Dict[str, Any]],
        recent_nodes: Sequence[str],
        width: int,
        height: int,
        blocked_recovery_contexts: Iterable[str] = (),
        ablations: MatrixAblations = MatrixAblations(),
    ) -> List[CandidateInformationRow]:
        node = graph_snapshot.node(current_node_id)
        outgoing = graph_snapshot.outgoing_edges(current_node_id, include_invalid=True)
        valid_outgoing = [e for e in outgoing if e.edge_status not in {"INVALID", "BLOCKED"}]
        blocked_contexts = set(blocked_recovery_contexts or [])
        rows: List[CandidateInformationRow] = []
        for index, element in enumerate(candidate_elements):
            action_type = "click"
            identity = element_identity(element, width, height, action_type)
            edge = graph_snapshot.exact_edge(current_node_id, identity, action_type) if ablations.exact_history else None
            cls = str(getattr(element, "class_name", "") or "")
            role = cls.rsplit(".", 1)[-1].lower() if cls else "node"
            text = str(getattr(element, "text", "") or "")
            desc = str(getattr(element, "content_desc", "") or "")
            merged = " ".join(x for x in (text, desc, cls, str(getattr(element, "resource_id", "") or "")) if x)
            bounds = parse_bounds(str(getattr(element, "bounds", "") or ""))
            if bounds:
                x1, y1, x2, y2 = bounds
                pos = (((x1 + x2) / 2.0) / max(1, width), ((y1 + y2) / 2.0) / max(1, height))
            else:
                pos = (0.5, 0.5)

            contextual = graph_snapshot.contextual_statistics(
                action_type, role, current_state.coarse_context, information_need.need_type
            ) if ablations.contextual_history else None
            destination = graph_snapshot.node(edge.destination_node_id) if edge and edge.destination_node_id else None
            destination_edges = graph_snapshot.outgoing_edges(destination.node_id, include_invalid=True) if destination else []
            destination_valid = [e for e in destination_edges if e.edge_status not in {"INVALID", "BLOCKED"}]
            subtree_size = self._subtree_size(graph_snapshot, destination.node_id, depth=2) if destination else None

            target_match = _match(information_need.targets, merged) if ablations.information_need else 0.0
            affordance_match = _match(information_need.expected_affordances, merged) if ablations.information_need else 0.0
            unresolved_match = _match(information_need.unresolved_information, merged) if ablations.information_need else 0.0
            action_match = 1.0 if action_type in information_need.candidate_action_types else 0.0
            risk_conflict = 1.0 if any(term and term.lower() in merged.lower() for term in information_need.risk_terms) else 0.0
            edge_rb_rate = edge.rollback_success_rate if edge else None
            contextual_rb = contextual.rollback_success_rate if contextual else None
            recoverability = self._estimate_recoverability(edge, edge_rb_rate, contextual_rb, risk_conflict, ablations)
            blocked_key = f"{current_node_id}:{identity}"
            blocked_element = bool(edge and edge.edge_status in {"INVALID", "BLOCKED"})
            blocked_recovery = blocked_key in blocked_contexts
            risk_level = max(risk_conflict, float(edge.risk_level if edge else 0.0))
            feasible = bool(
                getattr(element, "enabled", True) is not False
                and not blocked_element
                and not blocked_recovery
                and risk_level < 0.55
                and recoverability >= 0.55
            )
            probe_cost = self._probe_cost(edge, contextual.mean_exploration_cost if contextual else None, role)
            rollback_cost = self._rollback_cost(edge, contextual.mean_exploration_cost if contextual else None)
            total_cost = max(0.05, probe_cost + rollback_cost) if ablations.cost else 1.0
            history_probes = edge.probe_count if edge else 0
            ui_novelty = 1.0 / (1.0 + float(history_probes))
            deep_rate = None
            if edge and (edge.rollback_success_count + edge.rollback_failure_count) > 0:
                deep_rate = edge.deep_recovery_count / float(edge.rollback_success_count + edge.rollback_failure_count)

            rows.append(CandidateInformationRow(
                candidate_id=f"{graph_snapshot.generation}:{current_node_id or 'new'}:{identity}:{index}",
                element_identity=identity,
                text=text,
                content_desc=desc,
                role=role,
                probe_type=action_type,
                normalized_position=(_clip(pos[0]), _clip(pos[1])),
                bounds=tuple(bounds) if bounds else None,
                clickable=bool(getattr(element, "clickable", False)),
                scrollable=bool(getattr(element, "scrollable", False)),
                enabled=bool(getattr(element, "enabled", True)),
                selected=getattr(element, "selected", None),
                checked=getattr(element, "checked", None),
                nearby_context=merged[:240],
                current_node_id=current_node_id,
                visit_count=node.visit_count if node else 0,
                decision_entropy=node.decision_entropy if node else 1.0,
                outgoing_edge_count=len(outgoing),
                valid_outgoing_edge_count=len(valid_outgoing),
                explored_element_count=len(node.explored_element_identities) if node else 0,
                candidate_element_count=max(len(candidate_elements), node.candidate_element_count if node else 0),
                exploration_coverage=node.exploration_coverage if node else 0.0,
                graph_generation=graph_snapshot.generation,
                has_exact_history=edge is not None,
                probe_count=edge.probe_count if edge else None,
                edge_status=edge.edge_status if edge else None,
                confidence=edge.confidence if edge else None,
                inference_alignment_count=edge.inference_alignment_count if edge else None,
                execution_hit_count=edge.execution_hit_count if edge else None,
                execution_miss_count=edge.execution_miss_count if edge else None,
                alignment_rate=edge.alignment_rate if edge else None,
                execution_hit_rate=edge.execution_hit_rate if edge else None,
                skip_attempt_count=edge.skip_attempt_count if edge else None,
                skip_success_count=edge.skip_success_count if edge else None,
                skip_success_rate=edge.skip_success_rate if edge else None,
                mean_realized_information_gain=edge.mean_realized_information_gain if edge else None,
                rollback_success_count=edge.rollback_success_count if edge else None,
                rollback_failure_count=edge.rollback_failure_count if edge else None,
                rollback_success_rate=edge_rb_rate,
                mean_exploration_cost=edge.mean_exploration_cost if edge else None,
                known_inverse_level=edge.known_inverse_level if edge else None,
                destination_known=destination is not None,
                destination_node_id=destination.node_id if destination else None,
                destination_visit_count=destination.visit_count if destination else None,
                destination_entropy=destination.decision_entropy if destination else None,
                destination_out_degree=len(destination_edges) if destination else None,
                destination_valid_out_degree=len(destination_valid) if destination else None,
                destination_subtree_size=subtree_size,
                destination_known_label_count=len(destination.discovered_labels) if destination else None,
                destination_in_recent_path=(destination.node_id in set(recent_nodes)) if destination else None,
                discovered_labels=tuple(edge.discovered_labels if edge else ()),
                contextual_probe_count=contextual.probe_count if contextual else 0,
                contextual_alignment_rate=contextual.alignment_rate if contextual else None,
                contextual_execution_hit_rate=contextual.execution_hit_rate if contextual else None,
                contextual_mean_realized_IG=contextual.mean_realized_IG if contextual else None,
                contextual_rollback_success_rate=contextual_rb,
                target_match=target_match,
                expected_affordance_match=affordance_match,
                unresolved_information_match=unresolved_match,
                candidate_action_type_match=action_match,
                risk_conflict=risk_conflict,
                blocked_element=blocked_element,
                blocked_recovery_context=blocked_recovery,
                historical_rollback_success_rate=edge_rb_rate,
                historical_deep_recovery_rate=deep_rate,
                cross_package_history=edge.cross_package_history if edge else None,
                risk_level=risk_level,
                estimated_recoverability=recoverability,
                expected_probe_latency=probe_cost,
                expected_rollback_latency=rollback_cost,
                expected_total_exploration_cost=total_cost,
                ui_novelty_score=ui_novelty,
                feasible=feasible,
                exact_edge_id=edge.edge_id if edge else None,
            ))
        return rows

    @staticmethod
    def _subtree_size(snapshot: GraphSnapshot, node_id: str, depth: int) -> int:
        seen = {node_id}
        frontier = [node_id]
        for _ in range(max(0, int(depth))):
            nxt: List[str] = []
            for current in frontier:
                for edge in snapshot.outgoing_edges(current):
                    if edge.destination_node_id and edge.destination_node_id not in seen:
                        seen.add(edge.destination_node_id)
                        nxt.append(edge.destination_node_id)
            frontier = nxt
        return max(0, len(seen) - 1)

    def _estimate_recoverability(self, edge, exact_rate, contextual_rate, risk, ablations) -> float:
        if not ablations.recovery_history:
            return _clip(0.85 - 0.45 * risk)
        base = exact_rate if exact_rate is not None else contextual_rate
        if base is None:
            base = 0.82
        inverse_penalty = 0.06 * float(edge.known_inverse_level if edge else 0)
        cross_penalty = 0.12 if edge and edge.cross_package_history else 0.0
        return _clip(base - inverse_penalty - cross_penalty - 0.50 * risk)

    def _probe_cost(self, edge, contextual_cost: Optional[float], role: str) -> float:
        if edge and edge.mean_exploration_cost is not None:
            return max(0.05, 0.48 * edge.mean_exploration_cost)
        if contextual_cost is not None:
            return max(0.05, 0.48 * contextual_cost)
        role_factor = 1.25 if any(k in role for k in ("web", "list", "scroll")) else 1.0
        return self.generic_probe_cost * role_factor

    def _rollback_cost(self, edge, contextual_cost: Optional[float]) -> float:
        if edge and edge.mean_exploration_cost is not None:
            return max(0.05, 0.52 * edge.mean_exploration_cost)
        if contextual_cost is not None:
            return max(0.05, 0.52 * contextual_cost)
        return self.generic_rollback_cost


@dataclass(frozen=True)
class PredictiveScorerConfig:
    path_need_weight: float = 0.42
    path_exact_alignment_weight: float = 0.22
    path_execution_weight: float = 0.18
    path_context_weight: float = 0.12
    path_history_utility_weight: float = 0.06
    ig_need_weight: float = 0.38
    ig_uncertainty_weight: float = 0.22
    ig_novelty_weight: float = 0.18
    ig_history_weight: float = 0.14
    ig_destination_weight: float = 0.08
    recoverability_threshold: float = 0.55
    cost_floor: float = 0.10


class PredictiveElementScorer:
    def __init__(self, config: PredictiveScorerConfig = PredictiveScorerConfig()):
        self.config = config

    def score(self, rows: Sequence[CandidateInformationRow]) -> List[CandidateInformationRow]:
        for row in rows:
            if not row.feasible or row.estimated_recoverability < self.config.recoverability_threshold:
                row.path_probability = 0.0
                row.expected_information_gain = 0.0
                row.predictive_value = 0.0
                row.final_exploration_score = 0.0
                continue
            need = _clip(
                0.42 * row.target_match
                + 0.25 * row.expected_affordance_match
                + 0.25 * row.unresolved_information_match
                + 0.08 * row.candidate_action_type_match
                - 0.50 * row.risk_conflict
            )
            alignment = row.alignment_rate if row.alignment_rate is not None else row.contextual_alignment_rate
            execution = row.execution_hit_rate if row.execution_hit_rate is not None else row.contextual_execution_hit_rate
            contextual = row.contextual_alignment_rate
            history_utility = row.mean_realized_information_gain
            if history_utility is None:
                history_utility = row.contextual_mean_realized_IG
            path = (
                self.config.path_need_weight * need
                + self.config.path_exact_alignment_weight * (alignment if alignment is not None else 0.35)
                + self.config.path_execution_weight * (execution if execution is not None else 0.40)
                + self.config.path_context_weight * (contextual if contextual is not None else 0.30)
                + self.config.path_history_utility_weight * _clip(history_utility or 0.0)
            )
            if not row.has_exact_history:
                path += 0.07 * row.ui_novelty_score
            destination_value = 0.0
            if row.destination_known:
                destination_value = _clip(
                    0.45 * min(1.0, float(row.destination_known_label_count or 0) / 6.0)
                    + 0.30 * min(1.0, float(row.destination_subtree_size or 0) / 4.0)
                    + 0.25 * (1.0 - float(row.destination_entropy or 1.0))
                )
            expected_ig = (
                self.config.ig_need_weight * need
                + self.config.ig_uncertainty_weight * row.decision_entropy
                + self.config.ig_novelty_weight * row.ui_novelty_score
                + self.config.ig_history_weight * _clip(history_utility or 0.0)
                + self.config.ig_destination_weight * destination_value
            )
            # When coverage is already high, novelty/remaining uncertainty matters more.
            expected_ig += 0.10 * row.exploration_coverage * row.ui_novelty_score
            path = _clip(path)
            expected_ig = _clip(expected_ig)
            predictive_value = path * expected_ig
            cost = max(self.config.cost_floor, row.expected_total_exploration_cost)
            utility = predictive_value * row.estimated_recoverability / cost
            row.path_probability = path
            row.expected_information_gain = expected_ig
            row.predictive_value = predictive_value
            row.final_exploration_score = _clip(utility)
        return sorted(rows, key=lambda r: r.final_exploration_score, reverse=True)


class SmallModelPredictiveEstimator(PredictiveElementScorer):
    """Future extension point; intentionally uses the interpretable scorer today."""
