import unittest
from unittest.mock import AsyncMock, patch

from agent.agent_service import translate_document


class DocumentTranslationContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_empty_page_selection(self):
        with patch("agent.agent_service.get_document", return_value={"id": 1, "page_count": 3}):
            with self.assertRaisesRegex(ValueError, "至少选择一个页面"):
                await translate_document(1, [], "English")

    async def test_rejects_duplicate_pages_before_translating(self):
        page_translator = AsyncMock()
        with (
            patch("agent.agent_service.get_document", return_value={"id": 1, "page_count": 3}),
            patch("agent.agent_service.translate_document_page", page_translator),
        ):
            with self.assertRaisesRegex(ValueError, "重复页面"):
                await translate_document(1, [1, 1], "English")
        page_translator.assert_not_awaited()

    async def test_rejects_out_of_range_pages_before_translating(self):
        page_translator = AsyncMock()
        with (
            patch("agent.agent_service.get_document", return_value={"id": 1, "page_count": 3}),
            patch("agent.agent_service.translate_document_page", page_translator),
        ):
            with self.assertRaisesRegex(ValueError, "有效范围为 1–3"):
                await translate_document(1, [0, 4], "English")
        page_translator.assert_not_awaited()

    async def test_rejects_more_than_one_hundred_pages(self):
        page_translator = AsyncMock()
        with (
            patch("agent.agent_service.get_document", return_value={"id": 1, "page_count": 101}),
            patch("agent.agent_service.translate_document_page", page_translator),
        ):
            with self.assertRaisesRegex(ValueError, "最多翻译 100 页"):
                await translate_document(1, list(range(1, 102)), "English")
        page_translator.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
