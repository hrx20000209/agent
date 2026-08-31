"""Deterministic local graph distillation and three-way reasoning gate."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from Explorer.progressive_belief_graph import GraphEdge, GraphSnapshot
from Explorer.state_graph_information import InformationNeed


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9_\-]{2,}|[\u4e00-\u9fff]{1,}", (text or "").lower()))


class GraphMode(str, Enum):
    NORMAL_INFERENCE = "normal"
    GRAPH_ENHANCED_INFERENCE = "enhanced"
    SKIP_INFERENCE = "skipped"


@dataclass
class GraphFact:
    fact_type: str
    action_label: str
    source_edge_id: str
    evidence_labels: Tuple[str, ...]
    edge_status: str
    certainty: str
    need_match: float
    historical_utility: float
    freshness: float
    already_taken: bool
    risk_level: float
    utility_score: float


@dataclass
class DistillationResult:
    context: str = ""
    facts: List[GraphFact] = field(default_factory=list)
    raw_fact_count: int = 0
    candidate_fact_count: int = 0
    selected_fact_count: int = 0
    selected_fact_types: List[str] = field(default_factory=list)
    selected_edge_ids: List[str] = field(default_factory=list)
    graph_context_chars: int = 0
    estimated_graph_context_tokens: int = 0
    injected: bool = False

    def to_log_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["facts"] = [asdict(f) for f in self.facts]
        return raw


@dataclass(frozen=True)
class GraphDistillerConfig:
    max_graph_facts: int = 3
    max_graph_context_tokens: int = 64
    minimum_fact_utility: float = 0.10
    freshness_generations: int = 40


class GraphDistiller:
    def __init__(self, config: GraphDistillerConfig = GraphDistillerConfig()):
        self.config = config

    def distill(
        self,
        current_node_id: Optional[str],
        graph_snapshot: GraphSnapshot,
        information_need: InformationNeed,
        taken_edges: Sequence[str],
        recent_nodes: Sequence[str],
        token_budget: Optional[int] = None,
    ) -> DistillationResult:
        budget = max(8, int(token_budget or self.config.max_graph_context_tokens))
        edges = graph_snapshot.outgoing_edges(current_node_id, include_invalid=False)
        facts = [
            self._edge_to_fact(edge, graph_snapshot, information_need, set(taken_edges or []), set(recent_nodes or []))
            for edge in edges
        ]
        candidates = [f for f in facts if f.utility_score >= self.config.minimum_fact_utility and f.risk_level < 0.55]
        candidates.sort(key=lambda f: f.utility_score, reverse=True)

        selected: List[GraphFact] = []
        lines = ["[Memory]", f"Need: {information_need.summary}."]
        for fact in candidates:
            if len(selected) >= self.config.max_graph_facts:
                break
            line = self._fact_line(fact)
            proposed = lines + [line]
            if self._estimate_tokens("\n".join(proposed)) > budget:
                continue
            selected.append(fact)
            lines.append(line)
        context = "\n".join(lines) if selected else ""
        return DistillationResult(
            context=context,
            facts=selected,
            raw_fact_count=len(facts),
            candidate_fact_count=len(candidates),
            selected_fact_count=len(selected),
            selected_fact_types=[f.fact_type for f in selected],
            selected_edge_ids=[f.source_edge_id for f in selected],
            graph_context_chars=len(context),
            estimated_graph_context_tokens=self._estimate_tokens(context),
            injected=bool(context),
        )

    def _edge_to_fact(
        self,
        edge: GraphEdge,
        snapshot: GraphSnapshot,
        need: InformationNeed,
        taken: Set[str],
        recent_nodes: Set[str],
    ) -> GraphFact:
        evidence = tuple(edge.discovered_labels[:5])
        action_label = self._action_label(edge)
        edge_words = _tokens(" ".join([action_label, *evidence]))
        need_words = set(need.targets) | set(need.unresolved_information) | set(need.expected_affordances)
        need_match = 0.0 if not need_words else _clip(len(edge_words & need_words) / max(1.0, min(5.0, len(need_words))))
        certainty_value, certainty = self._certainty(edge)
        hist_utility = edge.mean_realized_information_gain
        if hist_utility is None:
            execution = edge.execution_hit_rate
            alignment = edge.alignment_rate
            hist_utility = 0.55 * (execution if execution is not None else 0.35) + 0.45 * (alignment if alignment is not None else 0.30)
        age = max(0, snapshot.generation - edge.last_updated_generation)
        freshness = _clip(1.0 - age / max(1.0, float(self.config.freshness_generations)))
        already_taken = edge.edge_id in taken or bool(edge.destination_node_id and edge.destination_node_id in recent_nodes)
        utility = need_match * certainty_value * _clip(hist_utility) * freshness
        # A verified transition with useful labels remains weakly useful even when
        # lexical overlap is sparse (common for icon-only controls).
        if evidence and edge.edge_status in {"VERIFIED", "REUSABLE", "INFERENCE_ALIGNED"}:
            utility = max(utility, 0.12 * certainty_value * freshness)
        if already_taken:
            utility *= 0.45
        fact_type = "OBSERVED"
        if edge.edge_status in {"VERIFIED", "REUSABLE"}:
            fact_type = "VERIFIED"
        elif already_taken:
            fact_type = "DONE"
        elif not evidence:
            fact_type = "NO_RELEVANT_EVIDENCE"
        return GraphFact(
            fact_type=fact_type,
            action_label=action_label,
            source_edge_id=edge.edge_id,
            evidence_labels=evidence,
            edge_status=edge.edge_status,
            certainty=certainty,
            need_match=need_match,
            historical_utility=_clip(hist_utility),
            freshness=freshness,
            already_taken=already_taken,
            risk_level=_clip(edge.risk_level),
            utility_score=_clip(utility),
        )

    @staticmethod
    def _certainty(edge: GraphEdge) -> Tuple[float, str]:
        if edge.edge_status in {"REUSABLE", "VERIFIED"}:
            return max(0.78, edge.confidence), "Verified"
        if edge.edge_status == "INFERENCE_ALIGNED":
            return max(0.58, edge.confidence), "Observed"
        return max(0.30, edge.confidence), "Tentative"

    @staticmethod
    def _action_label(edge: GraphEdge) -> str:
        action = edge.action or {}
        inputs = action.get("action_inputs") or {}
        label = inputs.get("label") or inputs.get("content")
        if label:
            return str(label)[:48]
        return f"{edge.role or edge.action_type} at {GraphDistiller._region(edge.bounds)}"

    @staticmethod
    def _region(bounds) -> str:
        if not bounds:
            return "current screen"
        x1, y1, x2, y2 = bounds
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        horizontal = "left" if cx < 360 else ("right" if cx > 720 else "center")
        vertical = "top" if cy < 800 else ("bottom" if cy > 1600 else "middle")
        return f"{vertical}-{horizontal}"

    @staticmethod
    def _fact_line(fact: GraphFact) -> str:
        labels = ", ".join(fact.evidence_labels[:4])
        if fact.fact_type == "DONE":
            return f"Done: {fact.action_label} was already traversed."
        if fact.fact_type == "NO_RELEVANT_EVIDENCE":
            return f"Observed: no task-relevant evidence was observed after {fact.action_label}."
        prefix = "Verified" if fact.certainty == "Verified" else "Observed"
        if labels:
            return f"{prefix}: {fact.action_label} led to {{{labels}}}."
        return f"{prefix}: {fact.action_label} produced a screen transition."

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        ascii_words = len(re.findall(r"[A-Za-z0-9_\-]+", text))
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        punctuation = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text))
        return max(1, ascii_words + cjk + punctuation // 3)


@dataclass(frozen=True)
class GraphGateConfig:
    skip_confidence: float = 0.82
    skip_max_entropy: float = 0.35
    skip_max_age_generations: int = 40


@dataclass
class GraphGateDecision:
    mode: GraphMode
    reusable_edge: Optional[GraphEdge] = None
    distillation: DistillationResult = field(default_factory=DistillationResult)
    reason: str = ""


class GraphReasoningGate:
    def __init__(self, distiller: GraphDistiller, config: GraphGateConfig = GraphGateConfig()):
        self.distiller = distiller
        self.config = config

    def decide(
        self,
        current_node_id: Optional[str],
        graph_snapshot: GraphSnapshot,
        information_need: InformationNeed,
        taken_edges: Sequence[str],
        recent_nodes: Sequence[str],
        graph_reasoning: str,
        token_budget: int,
    ) -> GraphGateDecision:
        mode_flag = str(graph_reasoning or "off").lower()
        if mode_flag == "off":
            return GraphGateDecision(GraphMode.NORMAL_INFERENCE, reason="graph_reasoning_off")
        edge = None
        if mode_flag in {"skip_only", "distill_and_skip"}:
            edge = graph_snapshot.reusable_edge(
                current_node_id,
                recent_nodes=recent_nodes,
                min_confidence=self.config.skip_confidence,
                max_age_generations=self.config.skip_max_age_generations,
                max_entropy=self.config.skip_max_entropy,
            )
            if edge is not None:
                return GraphGateDecision(GraphMode.SKIP_INFERENCE, reusable_edge=edge, reason="safe_reusable_edge")
        if mode_flag in {"distill", "distill_and_skip", "briefing"}:
            result = self.distiller.distill(
                current_node_id=current_node_id,
                graph_snapshot=graph_snapshot,
                information_need=information_need,
                taken_edges=taken_edges,
                recent_nodes=recent_nodes,
                token_budget=token_budget,
            )
            if result.context:
                return GraphGateDecision(
                    GraphMode.GRAPH_ENHANCED_INFERENCE,
                    distillation=result,
                    reason="useful_local_graph_facts",
                )
        return GraphGateDecision(GraphMode.NORMAL_INFERENCE, reason="no_useful_graph_evidence")

