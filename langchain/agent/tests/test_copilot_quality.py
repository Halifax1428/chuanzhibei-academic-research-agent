import unittest
from unittest.mock import AsyncMock, PropertyMock, patch

from agent import copilot_service


def evidence(document_id=1):
    return {
        "document_id": document_id,
        "filename": f"paper-{document_id}.pdf",
        "title": f"Paper {document_id}",
        "version": "v1",
        "page": 2,
        "section": "Method",
        "score": 0.9,
        "content": "A sparse attention method reduces computation for long sequences.",
    }


class CopilotQualityTests(unittest.IsolatedAsyncioTestCase):
    def test_repeated_categories_are_aggregated_for_bar_chart(self):
        columns, rows = copilot_service._parse_csv(
            "results.csv", "method,accuracy\nA,0.8\nA,0.9\nB,0.7\n"
        )
        summary = copilot_service._profile_csv("results.csv", columns, rows)
        labels, x_values, values = copilot_service._chart_points(summary, rows)

        self.assertEqual(summary["chart"]["type"], "bar")
        self.assertEqual(labels, ["A", "B"])
        self.assertAlmostEqual(values[0], 0.85)
        self.assertEqual(values[1], 0.7)
        self.assertEqual(x_values, [0.0, 1.0])

    def test_scatter_points_keep_the_numeric_x_values(self):
        columns, rows = copilot_service._parse_csv(
            "results.csv", "tokens,latency\n128,9.2\n512,6.1\n1024,4.4\n"
        )
        summary = copilot_service._profile_csv("results.csv", columns, rows)
        labels, x_values, values = copilot_service._chart_points(summary, rows)
        script = copilot_service._matplotlib_script(summary)

        self.assertEqual(summary["chart"]["type"], "scatter")
        self.assertEqual(x_values, [128.0, 512.0, 1024.0])
        self.assertEqual(values, [9.2, 6.1, 4.4])
        self.assertIn("number(row.get(X_COLUMN))", script)
        compile(script, "generated_csv_plot.py", "exec")
        self.assertEqual(len(labels), 3)

    def test_csv_rejects_duplicate_headers_and_ragged_rows(self):
        with self.assertRaisesRegex(ValueError, "重名列"):
            copilot_service._parse_csv("bad.csv", "score, score\n1,2\n")
        with self.assertRaisesRegex(ValueError, "多余字段"):
            copilot_service._parse_csv("bad.csv", "score\n1,2\n")

    async def test_csv_returns_deterministic_chart_without_model_api(self):
        with patch.object(type(copilot_service.client), "configured", new_callable=PropertyMock, return_value=False):
            result = await copilot_service.analyze_csv(
                "results.csv", "step,accuracy\n1,0.7\n2,0.8\n3,0.85\n"
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["ai_status"], "configuration_required")
        self.assertIn("<svg", result["chart_svg"])
        compile(result["matplotlib_script"], "generated_plot.py", "exec")

    async def test_framework_hides_draft_when_verification_fails(self):
        answer = "Method X improves long-context efficiency [E1] [D1].\n\nReferences\n[D1] Fake title"
        with (
            patch("agent.copilot_service.client.require_configuration"),
            patch("agent.copilot_service.hybrid_retrieve", new_callable=AsyncMock,
                  return_value=[evidence()]),
            patch("agent.copilot_service.get_document", return_value={
                "id": 1, "title": "Verified title", "authors": ["Author"], "version": "v1", "filename": "paper.pdf"
            }),
            patch("agent.copilot_service.client.complete", new_callable=AsyncMock, return_value=answer),
            patch("agent.copilot_service.verify_answer", new_callable=AsyncMock, return_value={
                "status": "verified", "supported": False, "issues": ["unsupported claim"]
            }),
        ):
            result = await copilot_service.generate_framework("An idea", [1, 2])

        self.assertEqual(result["status"], "verification_failed")
        self.assertNotIn("Method X", result["framework"])

    async def test_framework_uses_only_evidenced_documents_and_canonical_references(self):
        answer = "Method X reduces computation [E1] [D1].\n\nReferences\n[D1] Made-up title"
        with (
            patch("agent.copilot_service.client.require_configuration"),
            patch("agent.copilot_service.hybrid_retrieve", new_callable=AsyncMock,
                  return_value=[evidence(1)]),
            patch("agent.copilot_service.get_document", return_value={
                "id": 1, "title": "Verified title", "authors": ["Ada Lovelace"], "version": "v1", "filename": "paper.pdf"
            }) as get_document,
            patch("agent.copilot_service.client.complete", new_callable=AsyncMock, return_value=answer),
            patch("agent.copilot_service.verify_answer", new_callable=AsyncMock, return_value={
                "status": "verified", "supported": True, "issues": []
            }),
        ):
            result = await copilot_service.generate_framework("An idea", [1, 2])

        self.assertEqual(result["status"], "completed")
        get_document.assert_called_once_with(1)
        self.assertIn("[D1] Ada Lovelace. Verified title", result["framework"])
        self.assertNotIn("Made-up title", result["framework"])

    def test_all_diagram_formats_escape_labels_and_python_is_valid(self):
        graph = {
            "nodes": [{"id": "1", "label": 'Title_with_"quoted" [x] & 50%'}],
            "edges": [],
            "stats": {"node_count": 1, "edge_count": 0},
        }
        with patch("agent.copilot_service.build_citation_graph", return_value=graph):
            mermaid = copilot_service.generate_diagram(None, "mermaid")["script"]
            graphviz = copilot_service.generate_diagram(None, "graphviz")["script"]
            tikz = copilot_service.generate_diagram(None, "tikz")["script"]
            matplotlib = copilot_service.generate_diagram(None, "matplotlib")["script"]

        self.assertIn("&quot;", mermaid)
        self.assertIn("\\\"quoted\\\"", graphviz)
        self.assertIn(r"\_", tikz)
        compile(matplotlib, "generated_graph.py", "exec")


if __name__ == "__main__":
    unittest.main()
