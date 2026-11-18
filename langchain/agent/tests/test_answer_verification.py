import unittest
from unittest.mock import AsyncMock, patch

from agent.agent_service import answer_question, verify_answer


class AnswerVerificationTests(unittest.IsolatedAsyncioTestCase):
    evidence = [
        {
            "document_id": 1,
            "filename": "paper.pdf",
            "page": 1,
            "section": "Introduction",
            "score": 0.9,
            "content": "The paper proposes a top-k attention method.",
        }
    ]

    async def run_answer(self, answer, verification):
        with (
            patch("agent.agent_service.client.require_configuration"),
            patch("agent.agent_service.hybrid_retrieve", new_callable=AsyncMock, return_value=self.evidence),
            patch("agent.agent_service.client.complete", new_callable=AsyncMock, return_value=answer),
            patch("agent.agent_service.verify_answer", new_callable=AsyncMock, return_value=verification),
        ):
            return await answer_question("What method is proposed?")

    async def test_rejected_answer_is_not_presented_as_supported(self):
        result = await self.run_answer(
            "Unsupported claim [E1]",
            {"status": "verified", "supported": False, "issues": ["Claim exceeds evidence"]},
        )

        self.assertEqual(result["status"], "verification_failed")
        self.assertNotIn("Unsupported claim", result["answer"])
        self.assertEqual(result["verification"]["issues"], ["Claim exceeds evidence"])

    async def test_unavailable_verifier_fails_closed(self):
        result = await self.run_answer(
            "Claim [E1]",
            {"status": "unavailable", "supported": False, "issues": ["timeout"]},
        )

        self.assertEqual(result["status"], "verification_unavailable")
        self.assertNotIn("Claim [E1]", result["answer"])

    async def test_answer_without_evidence_marker_is_not_supported(self):
        result = await self.run_answer(
            "Claim without citation",
            {"status": "verified", "supported": True, "issues": []},
        )

        self.assertEqual(result["status"], "citation_validation_failed")

    async def test_verified_cited_answer_is_supported(self):
        result = await self.run_answer(
            "The paper proposes top-k attention [E1].",
            {"status": "verified", "supported": True, "issues": []},
        )

        self.assertEqual(result["status"], "supported")
        self.assertIn("top-k attention", result["answer"])

    async def test_string_false_from_verifier_is_not_treated_as_true(self):
        with (
            patch("agent.agent_service.client.require_configuration"),
            patch("agent.agent_service.client.complete", new_callable=AsyncMock,
                  return_value='{"supported":"false","issues":[]}'),
        ):
            result = await verify_answer("Question?", "Claim [E1]", self.evidence)

        self.assertFalse(result["supported"])


if __name__ == "__main__":
    unittest.main()
