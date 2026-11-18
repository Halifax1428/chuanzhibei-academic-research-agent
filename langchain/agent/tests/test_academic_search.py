import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch

from agent import academic_search
from agent.main import tracking_refresh_interval


class FakeResponse:
    def __init__(self, status_code, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            request = httpx.Request("GET", academic_search.ARXIV_API_URL)
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)


class FakeAsyncClient:
    response = None

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, *args, **kwargs):
        return self.response


class AcademicSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limit_has_a_clear_retry_message(self):
        FakeAsyncClient.response = FakeResponse(429, headers={"Retry-After": "8"})
        academic_search._arxiv_last_request = 0
        with patch.object(academic_search, "ARXIV_REQUEST_INTERVAL", 0), patch.object(
            academic_search.httpx, "AsyncClient", FakeAsyncClient
        ):
            with self.assertRaisesRegex(academic_search.AcademicSearchError, "约 8 秒后重试"):
                await academic_search.search_arxiv("test", max_results=1)

    async def test_requests_wait_for_the_configured_interval(self):
        FakeAsyncClient.response = FakeResponse(
            200,
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>',
        )
        academic_search._arxiv_last_request = time.monotonic() - 1
        sleep = AsyncMock()
        with (
            patch.object(academic_search, "ARXIV_REQUEST_INTERVAL", 3),
            patch.object(academic_search, "_arxiv_request_lock", asyncio.Lock()),
            patch.object(academic_search.asyncio, "sleep", sleep),
            patch.object(academic_search.httpx, "AsyncClient", FakeAsyncClient),
            patch.object(academic_search, "_persist_papers"),
        ):
            result = await academic_search.search_arxiv("test", max_results=1)

        self.assertEqual(result["count"], 0)
        sleep.assert_awaited_once()
        self.assertGreater(sleep.await_args.args[0], 0)

    async def test_recommendations_request_relevance_ranked_candidates(self):
        with (
            patch(
                "agent.academic_search.search_arxiv",
                new_callable=AsyncMock,
                return_value={"papers": []},
            ) as search,
            patch("agent.academic_search.list_documents", return_value={"documents": []}),
        ):
            result = await academic_search.recommend_papers("efficient attention", limit=3)

        self.assertEqual(result["recommendations"], [])
        self.assertEqual(search.await_args.kwargs["sort_by"], "relevance")

    async def test_recommendation_feedback_changes_ranking_and_dismisses_items(self):
        papers = [
            {"external_id": "attention-1", "title": "Attention one", "summary": "attention paper", "github_urls": [], "updated_at": None},
            {"external_id": "attention-2", "title": "Attention two", "summary": "attention paper", "github_urls": [], "updated_at": None},
            {"external_id": "attention-3", "title": "Attention three", "summary": "attention paper", "github_urls": [], "updated_at": None},
        ]
        with (
            patch("agent.academic_search.search_arxiv", new_callable=AsyncMock,
                  return_value={"papers": papers}),
            patch("agent.academic_search.list_documents", return_value={"documents": []}),
            patch("agent.academic_search._feedback_actions", return_value={
                "attention-1": "dislike", "attention-2": "like", "attention-3": "dismiss"
            }),
        ):
            result = await academic_search.recommend_papers("attention", limit=3)

        recommendations = result["recommendations"]
        self.assertEqual([item["external_id"] for item in recommendations], ["attention-2", "attention-1"])
        self.assertIn("感兴趣", recommendations[0]["recommendation_reason"])
        self.assertNotIn("attention-3", [item["external_id"] for item in recommendations])

    async def test_scheduled_refresh_skips_disabled_and_reports_failures(self):
        with (
            patch("agent.academic_search.list_tracked_queries", return_value=[
                {"id": 1, "enabled": 1}, {"id": 2, "enabled": 0}, {"id": 3, "enabled": 1}
            ]),
            patch("agent.academic_search.refresh_tracked_query", new_callable=AsyncMock,
                  side_effect=[{"track": {"id": 1}}, RuntimeError("rate limited")]) as refresh,
        ):
            result = await academic_search.refresh_enabled_tracked_queries()

        self.assertEqual([item["track"]["id"] for item in result["refreshed"]], [1])
        self.assertEqual(result["failed"][0]["track_id"], 3)
        self.assertEqual(refresh.await_count, 2)

    def test_tracking_interval_has_a_safe_minimum_and_default(self):
        with patch.dict("os.environ", {"TRACK_REFRESH_INTERVAL_SECONDS": "5"}):
            self.assertEqual(tracking_refresh_interval(), 60)
        with patch.dict("os.environ", {"TRACK_REFRESH_INTERVAL_SECONDS": "invalid"}):
            self.assertEqual(tracking_refresh_interval(), 86400)


if __name__ == "__main__":
    unittest.main()
