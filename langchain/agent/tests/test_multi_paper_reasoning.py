import unittest
from unittest.mock import AsyncMock, patch

from agent.agent_service import compare_documents


def evidence(document_id, title):
    return {
        "document_id": document_id,
        "filename": f"paper-{document_id}.pdf",
        "title": title,
        "version": "v1",
        "page": 3,
        "section": "Method",
        "score": 0.8,
        "content": f"Evidence from {title}.",
        "retrieval_mode": "keyword",
    }


class MultiPaperReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def test_comparison_refuses_when_only_one_selected_paper_has_evidence(self):
        with (
            patch("agent.agent_service.client.require_configuration"),
            patch("agent.agent_service.hybrid_retrieve", new_callable=AsyncMock,
                  return_value=[evidence(1, "Paper One")]),
            patch("agent.agent_service.client.complete", new_callable=AsyncMock) as complete,
        ):
            result = await compare_documents("Compare the methods", [1, 2])

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["document_coverage"]["missing_document_ids"], [2])
        complete.assert_not_awaited()

    async def test_comparison_includes_paper_identity_and_verifies_answer(self):
        answer = "Paper One uses method A [E1]. Paper Two uses method B [E2]."
        verification = '{"supported":true,"issues":[],"unsupported_claims":[]}'
        with (
            patch("agent.agent_service.client.require_configuration"),
            patch("agent.agent_service.hybrid_retrieve", new_callable=AsyncMock,
                  return_value=[evidence(1, "Paper One"), evidence(2, "Paper Two")]),
            patch("agent.agent_service.get_document", side_effect=[
                {"id": 1, "title": "Paper One", "version": "v1"},
                {"id": 2, "title": "Paper Two", "version": "v1"},
            ]),
            patch("agent.agent_service.client.complete", new_callable=AsyncMock,
                  side_effect=[answer, verification]) as complete,
        ):
            result = await compare_documents("Compare the methods", [1, 2])

        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["document_coverage"]["covered_document_ids"], [1, 2])
        prompt = complete.await_args_list[0].args[0][1]["content"]
        self.assertIn("标题=Paper One", prompt)
        self.assertIn("页码=3", prompt)
        self.assertTrue(result["verification"]["supported"])

    async def test_comparison_hides_unverified_answer(self):
        answer = "Paper One uses method A [E1]. Paper Two uses method B [E2]."
        verification = '{"supported":false,"issues":["unsupported"],"unsupported_claims":[]}'
        with (
            patch("agent.agent_service.client.require_configuration"),
            patch("agent.agent_service.hybrid_retrieve", new_callable=AsyncMock,
                  return_value=[evidence(1, "Paper One"), evidence(2, "Paper Two")]),
            patch("agent.agent_service.get_document", side_effect=[
                {"id": 1, "title": "Paper One", "version": "v1"},
                {"id": 2, "title": "Paper Two", "version": "v1"},
            ]),
            patch("agent.agent_service.client.complete", new_callable=AsyncMock,
                  side_effect=[answer, verification]),
        ):
            result = await compare_documents("Compare the methods", [1, 2])

        self.assertEqual(result["status"], "verification_failed")
        self.assertNotIn("method A", result["answer"])

    async def test_comparison_requires_evidence_for_every_selected_paper(self):
        with (
            patch("agent.agent_service.client.require_configuration"),
            patch("agent.agent_service.hybrid_retrieve", new_callable=AsyncMock,
                  return_value=[evidence(1, "Paper One"), evidence(2, "Paper Two")]),
            patch("agent.agent_service.client.complete", new_callable=AsyncMock) as complete,
        ):
            result = await compare_documents("Compare the methods", [1, 2, 3])

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["document_coverage"]["missing_document_ids"], [3])
        complete.assert_not_awaited()

    async def test_comparison_rejects_low_score_for_a_selected_paper(self):
        weak = {**evidence(2, "Paper Two"), "score": 0.04}
        with (
            patch("agent.agent_service.client.require_configuration"),
            patch("agent.agent_service.hybrid_retrieve", new_callable=AsyncMock,
                  return_value=[evidence(1, "Paper One"), weak]),
            patch("agent.agent_service.client.complete", new_callable=AsyncMock) as complete,
        ):
            result = await compare_documents("Compare the methods", [1, 2])

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["document_coverage"]["missing_document_ids"], [2])
        complete.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
