import unittest
from unittest.mock import AsyncMock, patch

from agent.citation_graph import generate_graph_review


def graph(candidates):
    return {
        "stats": {"node_count": 1, "edge_count": 0},
        "nodes": [{"id": 1, "label": "Paper"}],
        "edges": [],
        "future_work_candidates": candidates,
    }


class CitationGraphReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_is_not_generated_without_source_evidence(self):
        with (
            patch("agent.citation_graph.build_citation_graph", return_value=graph([])),
            patch("agent.citation_graph.client.require_configuration"),
            patch("agent.citation_graph.client.complete", new_callable=AsyncMock) as complete,
        ):
            result = await generate_graph_review([1])

        self.assertEqual(result["status"], "insufficient_evidence")
        complete.assert_not_awaited()

    async def test_review_with_invalid_evidence_reference_is_not_shown(self):
        candidates = [{"title": "Paper", "page": 1, "section": "Limitations", "text": "Future work."}]
        with (
            patch("agent.citation_graph.build_citation_graph", return_value=graph(candidates)),
            patch("agent.citation_graph.client.require_configuration"),
            patch(
                "agent.citation_graph.client.complete",
                new_callable=AsyncMock,
                return_value="The paper supports this claim [F2].",
            ),
        ):
            result = await generate_graph_review([1])

        self.assertEqual(result["status"], "citation_validation_failed")
        self.assertEqual(result["citation_check"]["invalid"], ["F2"])
        self.assertNotIn("supports this claim", result["review"])

    async def test_review_with_valid_evidence_reference_is_shown(self):
        candidates = [{"title": "Paper", "page": 1, "section": "Limitations", "text": "Future work."}]
        with (
            patch("agent.citation_graph.build_citation_graph", return_value=graph(candidates)),
            patch("agent.citation_graph.client.require_configuration"),
            patch(
                "agent.citation_graph.client.complete",
                new_callable=AsyncMock,
                return_value="Further validation is needed [F1].",
            ),
        ):
            result = await generate_graph_review([1])

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["citation_check"]["valid"])


if __name__ == "__main__":
    unittest.main()
