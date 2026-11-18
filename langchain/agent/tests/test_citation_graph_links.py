import unittest
from contextlib import nullcontext
from unittest.mock import patch

from agent.citation_graph import _future_work_candidates, build_citation_graph


class CitationGraphLinkTests(unittest.TestCase):
    def test_graph_matches_exact_arxiv_id_without_title_overlap(self):
        documents = [
            {"id": 1, "title": "A Title With No Matching Terms", "version": "v7", "status": "indexed",
             "metadata": {"arxiv_id": "2301.12345"}},
            {"id": 2, "title": "A Citing Paper", "version": "unknown", "status": "indexed",
             "metadata": {}},
        ]
        with (
            patch("agent.citation_graph.list_documents", return_value={"documents": documents}),
            patch("agent.citation_graph._reference_rows", side_effect=[[], [
                {"ref_index": 1, "raw_text": "Prior work, arXiv:2301.12345v7"}
            ]]),
            patch("agent.citation_graph._document_text", return_value=""),
            patch("agent.citation_graph._future_work_candidates", return_value=[]),
        ):
            graph = build_citation_graph()

        self.assertEqual(len(graph["edges"]), 1)
        self.assertEqual(graph["edges"][0]["target"], "1")
        self.assertEqual(graph["edges"][0]["evidence"][0]["match_type"], "arxiv_id")

    def test_future_work_surfaces_citing_paper_excerpt_as_unverified_lead(self):
        documents = [
            {"id": 1, "title": "Source paper"},
            {"id": 2, "title": "Potential follow-up"},
        ]

        class FakeConnection:
            def execute(self, query, params):
                if "document_id IN" in query:
                    rows = [
                        {
                            "document_id": 1,
                            "page": 5,
                            "section": "Future Work",
                            "content": "Future work should improve sparse attention routing for long context sequences.",
                        },
                        {
                            "document_id": 2,
                            "page": 4,
                            "section": "Method",
                            "content": "We study sparse attention routing for long context sequences in this work.",
                        },
                    ]
                else:
                    rows = []
                return type("FakeResult", (), {"fetchall": lambda self: rows})()

        with patch("agent.citation_graph.db.connection", return_value=nullcontext(FakeConnection())):
            candidates = _future_work_candidates(
                documents,
                [{"source": "2", "target": "1"}],
            )

        self.assertEqual(candidates[0]["follow_up_papers"][0]["document_id"], 2)
        self.assertIn("需核验", candidates[0]["status"])


if __name__ == "__main__":
    unittest.main()
