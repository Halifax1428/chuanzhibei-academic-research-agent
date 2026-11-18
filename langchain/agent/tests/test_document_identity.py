import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import config, db, document_service


class DocumentIdentityTests(unittest.TestCase):
    def test_extracts_arxiv_id_and_version_from_filename(self):
        identity = document_service.extract_arxiv_identity("2301.12345v7.pdf")

        self.assertEqual(identity, {"arxiv_id": "2301.12345", "arxiv_version": "v7"})

    def test_extracts_legacy_arxiv_id(self):
        identity = document_service.extract_arxiv_identity("hep-th/9901001v2")

        self.assertEqual(identity, {"arxiv_id": "hep-th/9901001", "arxiv_version": "v2"})

    def test_recommends_newer_arxiv_version(self):
        recommendation = document_service._duplicate_recommendation("v7", "v1", True)

        self.assertEqual(recommendation["action"], "keep_new")
        self.assertIn("v7", recommendation["reason"])

    def test_recommends_preserving_newer_existing_version(self):
        recommendation = document_service._duplicate_recommendation("v1", "v7", True)

        self.assertEqual(recommendation["action"], "keep_old")
        self.assertIn("v7", recommendation["reason"])

    def test_unversioned_paper_requires_manual_review(self):
        recommendation = document_service._duplicate_recommendation(None, "unknown", True)

        self.assertEqual(recommendation["action"], "keep_both")
        self.assertIn("人工核对", recommendation["reason"])

    def test_ingest_detects_arxiv_revision_in_isolated_database(self):
        content_v1 = (
            "# Efficient Attention for Long Sequences\n\n"
            "Authors: Ada Lovelace and Alan Turing\n\n"
            "Abstract\nWe introduce a sparse attention method for long sequences.\n\n"
            "1 Introduction\nThe method reduces computation while preserving context.\n"
        ).encode()
        content_v7 = content_v1.replace(
            b"reduces computation", b"reduces quadratic computation"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(config, "DATA_DIR", root),
                patch.object(db, "DATABASE_PATH", root / "test.sqlite3"),
                patch.object(document_service, "UPLOAD_DIR", root / "uploads"),
            ):
                db.init_db()
                document_service.ingest_file("2301.12345v1.md", content_v1)
                result = document_service.ingest_file("2301.12345v7.md", content_v7)

        self.assertTrue(result["duplicate"])
        self.assertEqual(result["document"]["version"], "v7")
        self.assertEqual(result["version_recommendation"]["action"], "keep_new")
        self.assertEqual(result["matched_document"]["version"], "v1")

    def test_unrelated_titles_do_not_match(self):
        score = document_service._title_similarity(
            "Efficient Attention for Long Sequences",
            "A Survey of Medical Image Segmentation",
        )

        self.assertLess(score, 0.5)

    def test_scanned_pdf_reports_missing_tesseract_actionably(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.pdf"
            path.touch()
            tesseract = types.SimpleNamespace(pytesseract=types.SimpleNamespace(tesseract_cmd="tesseract"))
            with (
                patch.dict(sys.modules, {"pypdfium2": types.ModuleType("pypdfium2"), "pytesseract": tesseract}),
                patch("agent.document_service.shutil.which", return_value=None),
            ):
                with self.assertRaisesRegex(RuntimeError, "TESSERACT_CMD"):
                    document_service._ocr_pdf_page(path, 0)

    def test_layout_reflow_marks_wide_column_gaps(self):
        restored = document_service._layout_text("Left column      Right column\nnext left        next right")

        self.assertIn("Left column | Right column", restored)
        self.assertIn("next left | next right", restored)

    @unittest.skipUnless(
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").is_file()
        or shutil.which("tesseract"),
        "Tesseract OCR is not installed",
    )
    def test_scanned_pdf_is_ocr_processed_with_installed_language_packs(self):
        from PIL import Image, ImageDraw, ImageFont

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (1800, 500), "white")
            draw = ImageDraw.Draw(image)
            draw.text((80, 180), "Research Agent OCR 2026", fill="black",
                      font=ImageFont.truetype("arial.ttf", 90))
            path = root / "scan.pdf"
            image.save(path, "PDF", resolution=150)

            with patch.dict(os.environ, {"TESSERACT_LANGUAGES": "eng"}):
                pages, extraction_mode, ocr_status = document_service.extract_pdf_pages(path)

        self.assertEqual(extraction_mode, "ocr")
        self.assertEqual(ocr_status, "completed")
        self.assertIn("Research Agent OCR 2026", pages[0])


if __name__ == "__main__":
    unittest.main()
