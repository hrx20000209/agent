import os
import tempfile
import unittest

from Explorer.graph_distiller import GraphDistiller, GraphMode, GraphReasoningGate
from Explorer.progressive_belief_graph import ProgressiveBeliefGraph, UIStateDescriptor
from Explorer.state_graph_information import (
    PredictiveElementScorer,
    StateGraphInformationMatrix,
    parse_reasoning_prior,
)


class FakeElement:
    def __init__(self, text="Details", rid="app:id/details", bounds="[0,0][200,100]", cls="android.widget.Button"):
        self.text = text
        self.content_desc = ""
        self.resource_id = rid
        self.bounds = bounds
        self.class_name = cls
        self.clickable = True
        self.scrollable = False
        self.enabled = True
        self.selected = False
        self.checked = False


class ProgressiveGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = ProgressiveBeliefGraph()
        self.src_state = UIStateDescriptor("src", ("Hotel", "Details"), "pkg", "button:2", 2)
        self.dst_state = UIStateDescriptor("dst", ("Phone", "Address"), "pkg", "text:2", 1)
        self.src = self.graph.observe_state(self.src_state)
        self.dst = self.graph.observe_state(self.dst_state)

    def add_edge(self):
        return self.graph.record_probe(
            self.src, "identity", "click", "button", "click", "button:2", "fact_lookup",
            {"action_type": "click", "coord_space": "norm1000", "action_inputs": {"coordinate": [100, 50], "label": "Details"}},
            self.dst, ("Phone", "Address"), 0.5, 0.4, (0, 0, 200, 100), 0.0,
        )

    def test_snapshot_is_generation_guarded(self):
        snap = self.graph.snapshot()
        edge = self.add_edge()
        self.assertIsNone(snap.exact_edge(self.src, "identity", "click"))
        self.assertIsNotNone(self.graph.snapshot().exact_edge(self.src, "identity", "click"))
        self.assertGreater(self.graph.generation, snap.generation)

    def test_central_statistics_make_edge_reusable(self):
        edge_id = self.add_edge()
        self.graph.record_probe(
            self.src, "identity", "click", "button", "click", "button:2", "fact_lookup",
            self.graph.edges[edge_id].action, self.dst, ("Phone",), 0.4, 0.3, (0, 0, 200, 100), 0.0,
        )
        for _ in range(2):
            self.graph.record_inference_alignment(edge_id)
            self.graph.record_execution_verification(edge_id, True, self.dst)
            self.graph.record_rollback_result(edge_id, True)
        edge = self.graph.edges[edge_id]
        self.assertIn(edge.edge_status, {"VERIFIED", "REUSABLE"})
        self.assertGreaterEqual(edge.confidence, 0.82)
        self.assertEqual(edge.execution_hit_rate, 1.0)

    def test_matrix_masks_missing_exact_history(self):
        snap = self.graph.snapshot()
        need = parse_reasoning_prior("", "Find the phone number")
        rows = StateGraphInformationMatrix().build(
            self.src_state, self.src, [FakeElement()], snap, need, None, [], 1080, 2400
        )
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0].has_exact_history)
        self.assertIsNone(rows[0].probe_count)
        self.assertIsNone(rows[0].execution_hit_rate)

    def test_safety_is_hard_gate(self):
        snap = self.graph.snapshot()
        need = parse_reasoning_prior("", "Delete the item")
        rows = StateGraphInformationMatrix().build(
            self.src_state, self.src, [FakeElement(text="Delete")], snap, need, None, [], 1080, 2400
        )
        ranked = PredictiveElementScorer().score(rows)
        self.assertFalse(ranked[0].feasible)
        self.assertEqual(ranked[0].final_exploration_score, 0.0)

    def test_distiller_and_skip_gate(self):
        edge_id = self.add_edge()
        for _ in range(3):
            self.graph.record_inference_alignment(edge_id)
            self.graph.record_execution_verification(edge_id, True, self.dst)
            self.graph.record_rollback_result(edge_id, True)
        snap = self.graph.snapshot()
        need = parse_reasoning_prior("", "Find the phone number")
        result = GraphDistiller().distill(self.src, snap, need, [], [], 64)
        self.assertIn("[Memory]", result.context)
        self.assertNotIn("confidence=", result.context)
        decision = GraphReasoningGate(GraphDistiller()).decide(
            self.src, snap, need, [], [], "distill_and_skip", 64
        )
        self.assertEqual(decision.mode, GraphMode.SKIP_INFERENCE)

    def test_roundtrip_and_budget(self):
        self.add_edge()
        self.assertGreater(self.graph.approximate_python_bytes(), 0)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "graph.json")
            self.graph.save(path)
            loaded = ProgressiveBeliefGraph.load(path)
            self.assertEqual(len(loaded.nodes), len(self.graph.nodes))
            self.assertEqual(len(loaded.edges), len(self.graph.edges))
        result = self.graph.prune_to_budget(1, protected_node_ids=[self.src])
        self.assertIn("bytes", result)

    def test_python_heap_budget_is_enforced(self):
        self.add_edge()
        before = self.graph.approximate_python_bytes()
        result = self.graph.prune_to_budget(
            max(1, before // 2),
            protected_node_ids=[self.src],
            size_metric="python",
        )
        self.assertEqual(result["size_metric"], "python")
        self.assertGreaterEqual(result["pruned_edges"], 1)
        self.assertLess(self.graph.approximate_python_bytes(), before)

    def test_unknown_budget_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            self.graph.prune_to_budget(1024, size_metric="rss")


if __name__ == "__main__":
    unittest.main()
