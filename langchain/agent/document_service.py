from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from . import db
from .config import UPLOAD_DIR, ensure_runtime_dirs, get_embedding_model
from .model_client import ModelRequestError, OpenAICompatibleClient


embedding_client = OpenAICompatibleClient()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_version(filename: str) -> str:
    match = re.search(r"(?:^|[._-])v(\d+)(?:[._-]|$)", filename.lower())
    return f"v{match.group(1)}" if match else "unknown"


ARXIV_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])((?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})(?:v\d+)?)(?![A-Za-z0-9])",
    flags=re.I,
)


def extract_arxiv_identity(*texts: str) -> dict[str, str | None]:
    for text in texts:
        match = ARXIV_ID_PATTERN.search(text or "")
        if not match:
            continue
        full_id = match.group(1)
        version_match = re.search(r"v(\d+)$", full_id, flags=re.I)
        return {
            "arxiv_id": full_id[:version_match.start()].lower() if version_match else full_id.lower(),
            "arxiv_version": f"v{version_match.group(1)}" if version_match else None,
        }
    return {"arxiv_id": None, "arxiv_version": None}


def _header_lines(pages: list[str]) -> list[str]:
    lines = []
    for line in pages[0].splitlines() if pages else []:
        candidate = re.sub(r"\s+", " ", line).strip()
        if not candidate:
            continue
        if re.match(r"^(abstract|摘要)\b", candidate, flags=re.I):
            break
        lines.append(candidate)
    return lines


def _looks_like_author_line(line: str) -> bool:
    name_tokens = re.findall(r"\b[A-Z][A-Za-z.'-]+\b", line)
    return len(name_tokens) >= 2 and ("," in line or re.search(r"\band\b", line, flags=re.I))


def detect_title(filename: str, pages: list[str]) -> str:
    header = _header_lines(pages)
    for index, line in enumerate(header):
        if line.startswith("#"):
            return line.lstrip("# ")[:200]
        if _looks_like_author_line(line):
            title_lines = header[:index]
            if title_lines:
                title = " ".join(title_lines)
                return re.sub(r"Top-k(?=[A-Z])", "Top-k ", title)[:200]
    if header:
        title = " ".join(header[:2])
        return re.sub(r"Top-k(?=[A-Z])", "Top-k ", title)[:200]
    stem = Path(filename).stem
    return re.sub(r"[_-]+", " ", re.sub(r"v\d+$", "", stem, flags=re.I)).strip() or stem


def heading_level(line: str) -> tuple[int, str] | None:
    clean = re.sub(r"\s+", " ", line).strip()
    markdown = re.match(r"^(#{1,6})\s+(.+)$", clean)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()[:200]
    numbered = re.match(r"^(\d+(?:\.\d+)*)(?:[.)])?\s+(.{2,200})$", clean)
    if numbered:
        return numbered.group(1).count(".") + 1, numbered.group(2).strip()
    if re.match(
        r"^(abstract|摘要|introduction|引言|related work|background|method|methods|methodology|"
        r"experiments?|results?|discussion|conclusion|limitations?|references|参考文献)$",
        clean,
        flags=re.I,
    ):
        return 1, clean
    return None


def extract_headings(pages: list[str]) -> list[dict[str, Any]]:
    headings = []
    for page_number, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            parsed = heading_level(line)
            if parsed:
                level, heading = parsed
                headings.append({"page": page_number, "level": level, "heading": heading})
    return headings[:1000]


def extract_abstract(pages: list[str]) -> str:
    text = "\n".join(pages[:4])
    match = re.search(
        r"(?:^|\n)\s*(?:abstract|摘要)\s*[:：]?\s*\n?(.*?)(?=\n\s*(?:keywords?|关键词|1\s*[.)]?\s*introduction|introduction|引言)\b)",
        text,
        flags=re.I | re.S,
    )
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()[:12000]


def extract_authors(pages: list[str], title: str) -> list[str]:
    lines = _header_lines(pages)
    for line in lines:
        if _looks_like_author_line(line):
            authors = [part.strip() for part in re.split(r",|\s+and\s+", line, flags=re.I) if part.strip()]
            if len(authors) >= 2:
                return authors[:20]
    candidates = []
    for line in lines:
        if not line or line.lower() == title.lower():
            continue
        if re.search(r"abstract|摘要|introduction|引言|university|institute|department|@|http", line, re.I):
            if candidates and re.search(r"abstract|摘要|introduction|引言", line, re.I):
                break
            continue
        words = re.findall(r"[A-Za-z][A-Za-z.'-]*|[\u4e00-\u9fff]{2,4}", line)
        if 1 <= len(words) <= 20 and len(line) <= 220 and not re.search(r"[.!?。！？]$", line):
            candidates.append(line)
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique[:10]


def extract_visual_markers(pages: list[str], pattern: str, kind: str) -> list[dict[str, Any]]:
    markers = []
    compiled = re.compile(pattern, flags=re.I)
    for page_number, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            clean = re.sub(r"\s+", " ", line).strip()
            if compiled.search(clean):
                markers.append({"page": page_number, "text": clean[:500], "type": kind})
    return markers[:500]


def extract_structured_metadata(pages: list[str], filename: str) -> dict[str, Any]:
    title = detect_title(filename, pages)
    headings = extract_headings(pages)
    return {
        "title": title,
        "authors": extract_authors(pages, title),
        "abstract": extract_abstract(pages),
        "headings": headings,
        "tables": extract_visual_markers(pages, r"(?:^|\s)(?:table|tab\.?|表)\s*[\d一二三四五六七八九十]+", "table"),
        "figures": extract_visual_markers(pages, r"(?:^|\s)(?:figure|fig\.?|图)\s*[\d一二三四五六七八九十]+", "figure"),
        "equations": extract_visual_markers(pages, r"(?:=|≈|∈|∏|∫|∑).{3,}", "equation_candidate"),
    }


def _ocr_pdf_page(path: Path, page_number: int) -> str:
    try:
        import pypdfium2
        import pytesseract
    except ImportError as error:
        raise RuntimeError(
            "该 PDF 含扫描页。扫描件 OCR 需要安装 pytesseract，并在系统中安装 Tesseract OCR。"
        ) from error

    executable = os.getenv("TESSERACT_CMD", "").strip()
    if not executable:
        candidates = [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        ]
        executable = next((str(item) for item in candidates if item.is_file()), "tesseract")
    pytesseract.pytesseract.tesseract_cmd = executable
    if not shutil.which(pytesseract.pytesseract.tesseract_cmd):
        raise RuntimeError(
            "该 PDF 含扫描页，但未找到 Tesseract OCR。请安装 Tesseract，或设置 TESSERACT_CMD 后重试。"
        )

    languages = os.getenv("TESSERACT_LANGUAGES", "eng+chi_sim").strip() or "eng"
    data_dir = os.getenv("TESSERACT_DATA_DIR", "").strip()
    if not data_dir:
        candidate = Path(os.getenv("LOCALAPPDATA", "")) / "Tesseract-OCR" / "tessdata"
        data_dir = str(candidate) if candidate.is_dir() else ""
    if data_dir:
        installed_languages = {
            item.stem for item in Path(data_dir).glob("*.traineddata") if item.is_file()
        }
    else:
        installed_languages = set(pytesseract.get_languages(config=""))
    missing_languages = set(languages.split("+")) - installed_languages
    if missing_languages:
        raise RuntimeError(
            "Tesseract 已安装，但缺少语言包：" + ", ".join(sorted(missing_languages))
            + "。请安装对应 traineddata，或通过 TESSERACT_LANGUAGES 选择已安装语言。"
        )
    document = pypdfium2.PdfDocument(str(path))
    page = document[page_number]
    bitmap = None
    previous_tessdata_prefix = os.environ.get("TESSDATA_PREFIX")
    try:
        bitmap = page.render(scale=2.2)
        image = bitmap.to_pil()
        if data_dir:
            os.environ["TESSDATA_PREFIX"] = data_dir
        try:
            return pytesseract.image_to_string(
                image, lang=languages, config="--psm 3"
            ).strip()
        except pytesseract.TesseractError as error:
            raise RuntimeError(
                f"第 {page_number + 1} 页 OCR 执行失败，请检查 Tesseract 语言包和扫描质量。"
            ) from error
    finally:
        if data_dir:
            if previous_tessdata_prefix is None:
                os.environ.pop("TESSDATA_PREFIX", None)
            else:
                os.environ["TESSDATA_PREFIX"] = previous_tessdata_prefix
        if bitmap is not None:
            bitmap.close()
        page.close()
        document.close()


def _layout_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.sub(r" {6,}", "  |  ", line.strip())
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_pdf_visual_metadata(path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        import pdfplumber
    except ImportError:
        return {"tables": [], "images": []}

    tables: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_index, rows in enumerate(page.extract_tables()[:8], start=1):
                cleaned = [
                    [(cell or "").strip()[:500] for cell in row[:20]]
                    for row in rows[:60]
                ]
                if any(any(cell for cell in row) for row in cleaned):
                    tables.append({"page": page_number, "table_index": table_index, "rows": cleaned})
            for image_index, image in enumerate(page.images[:40], start=1):
                images.append({
                    "page": page_number,
                    "image_index": image_index,
                    "bbox": [round(float(image[key]), 1) for key in ("x0", "top", "x1", "bottom")],
                })
    return {"tables": tables[:100], "images": images[:500]}


def extract_pdf_pages(path: Path) -> tuple[list[str], str, str]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("解析 PDF 需要安装 pypdf。") from error
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for index, page in enumerate(pdf.pages[:len(pages)]):
                restored = _layout_text(page.extract_text(layout=True) or "")
                if restored and len(restored) >= min(40, len(pages[index])):
                    pages[index] = restored
    except ImportError:
        pass
    needs_ocr = [index for index, text in enumerate(pages) if len(text) < 40]
    if not needs_ocr:
        return pages, "text", "not_required"

    for index in needs_ocr:
        pages[index] = _ocr_pdf_page(path, index)
    if not any(pages):
        raise ValueError("PDF 文本层和 OCR 均未提取到文字，请检查扫描质量或 OCR 语言包。")
    return pages, "text+ocr" if len(needs_ocr) < len(pages) else "ocr", "completed"


def extract_pages(path: Path) -> list[str]:
    if path.suffix.lower() in {".md", ".txt"}:
        return [path.read_text(encoding="utf-8", errors="ignore")]
    if path.suffix.lower() != ".pdf":
        raise ValueError("当前骨架支持 PDF、Markdown 和 TXT 文件。")
    pages, _, _ = extract_pdf_pages(path)
    if not any(pages):
        raise ValueError("PDF 文本层和 OCR 均未提取到文字。")
    return pages


def split_page(page_text: str, page_number: int) -> list[tuple[str, str]]:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return []
    chunks: list[tuple[str, str]] = []
    section = f"第 {page_number} 页"
    buffer: list[str] = []
    for line in lines:
        if re.match(r"^(#{1,6}|\d+(?:\.\d+)*[.)]?)\s+", line) and buffer:
            chunks.append((section, "\n".join(buffer)))
            buffer = []
        if line.startswith("#"):
            section = line.lstrip("# ").strip()[:120] or section
        buffer.append(line)
        if len(" ".join(buffer)) >= 1800:
            chunks.append((section, "\n".join(buffer)))
            buffer = []
    if buffer:
        chunks.append((section, "\n".join(buffer)))
    return chunks


def similarity(left: str, right: str) -> float:
    left = normalize_text(left)
    right = normalize_text(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_words = set(re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", left))
    right_words = set(re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", right))
    jaccard = len(left_words & right_words) / max(1, len(left_words | right_words))
    sequence = SequenceMatcher(None, left[:12000], right[:12000]).ratio()
    return 0.55 * sequence + 0.45 * jaccard


def _document_text(document_id: int) -> str:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT content FROM chunks WHERE document_id = ? ORDER BY page, id", (document_id,)
        ).fetchall()
    return "\n".join(row["content"] for row in rows)


def _title_similarity(left: str, right: str) -> float:
    normalize_title = lambda value: re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())
    return SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio()


def _token_similarity(left: str, right: str) -> float:
    left_words = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", left.lower()))
    right_words = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", right.lower()))
    return len(left_words & right_words) / max(1, len(left_words | right_words))


def _version_number(version: str | None) -> int | None:
    match = re.fullmatch(r"v(\d+)", version or "", flags=re.I)
    return int(match.group(1)) if match else None


def _duplicate_recommendation(
    incoming_version: str | None, existing_version: str | None, same_arxiv_id: bool
) -> dict[str, str]:
    incoming_number = _version_number(incoming_version)
    existing_number = _version_number(existing_version)
    if same_arxiv_id and incoming_number is not None and existing_number is not None:
        if incoming_number > existing_number:
            return {"action": "keep_new", "reason": f"同一 arXiv 论文，当前版本 {incoming_version} 高于资料库版本 {existing_version}。"}
        if incoming_number < existing_number:
            return {"action": "keep_old", "reason": f"资料库已有较新版本 {existing_version}，本次版本为 {incoming_version}。"}
        return {"action": "keep_both", "reason": f"两份文件都标记为 {incoming_version}；建议先核对文件内容再决定。"}
    if same_arxiv_id:
        return {"action": "keep_both", "reason": "识别为同一 arXiv 论文，但至少一份文件没有明确版本号；建议人工核对后再保留。"}
    return {"action": "keep_both", "reason": "标题或摘要相似，但没有可靠的论文标识证明版本关系；建议人工核对。"}


def find_duplicate(
    text: str, incoming: dict[str, Any] | None = None
) -> tuple[dict[str, Any] | None, float]:
    incoming = incoming or {}
    incoming_title = str(incoming.get("title", ""))
    incoming_abstract = str(incoming.get("abstract", ""))
    incoming_identity = extract_arxiv_identity(
        str(incoming.get("arxiv_id", "")), str(incoming.get("filename", "")), text[:20000]
    )
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE status NOT IN ('rejected_duplicate', 'superseded') ORDER BY id"
        ).fetchall()
    best: dict[str, Any] | None = None
    best_score = 0.0
    for row in rows:
        existing = db.document_dict(row)
        metadata = existing.get("metadata", {})
        existing_identity = extract_arxiv_identity(
            str(metadata.get("arxiv_id", "")), existing.get("filename", "")
        )
        same_arxiv_id = bool(
            incoming_identity["arxiv_id"]
            and incoming_identity["arxiv_id"] == existing_identity["arxiv_id"]
        )
        title_score = _title_similarity(incoming_title, existing.get("title", ""))
        abstract_score = _token_similarity(incoming_abstract, existing.get("abstract_text", ""))
        body_score = similarity(text[:20000], _document_text(row["id"])[:20000])
        score = 1.0 if same_arxiv_id else max(title_score, 0.75 * title_score + 0.25 * abstract_score, body_score)
        duplicate_match = same_arxiv_id or title_score >= 0.94 or (
            title_score >= 0.88 and abstract_score >= 0.62
        ) or body_score >= 0.88
        if duplicate_match:
            existing["duplicate_match"] = {
                "same_arxiv_id": same_arxiv_id,
                "title_similarity": round(title_score, 4),
                "abstract_similarity": round(abstract_score, 4),
                "body_similarity": round(body_score, 4),
                "recommendation": _duplicate_recommendation(
                    incoming_identity["arxiv_version"] or incoming.get("version"),
                    existing_identity["arxiv_version"] or existing.get("version"),
                    same_arxiv_id,
                ),
            }
        if duplicate_match and score > best_score:
            best = existing
            best_score = score
    return best, best_score if best else 0.0


def ingest_file(filename: str, content: bytes) -> dict[str, Any]:
    ensure_runtime_dirs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    sha256 = hashlib.sha256(content).hexdigest()
    with db.connection() as conn:
        existing = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        if existing:
            return {"document": db.document_dict(existing), "duplicate": True, "similarity": 1.0}

    target = UPLOAD_DIR / f"{sha256[:16]}_{safe_name}"
    created_target = not target.exists()
    target.write_bytes(content)
    try:
        if target.suffix.lower() == ".pdf":
            pages, extraction_mode, ocr_status = extract_pdf_pages(target)
            visual_metadata = extract_pdf_visual_metadata(target)
        else:
            pages, extraction_mode, ocr_status = extract_pages(target), "text", "not_required"
            visual_metadata = {"tables": [], "images": []}
    except Exception:
        if created_target:
            target.unlink(missing_ok=True)
        raise
    full_text = "\n".join(pages)
    structured = extract_structured_metadata(pages, safe_name)
    if visual_metadata["tables"]:
        structured["tables"] = visual_metadata["tables"]
    structured["figures"].extend(visual_metadata["images"])
    title = structured["title"]
    identity = extract_arxiv_identity(safe_name, full_text[:20000])
    version = identity["arxiv_version"] or detect_version(safe_name)
    duplicate, score = find_duplicate(
        full_text,
        {
            "title": title,
            "abstract": structured["abstract"],
            "authors": structured["authors"],
            "filename": safe_name,
            "version": version,
            **identity,
        },
    )
    status = "duplicate_candidate" if duplicate else "indexed"
    metadata = {
        "path": str(target),
        "source_filename": safe_name,
        "arxiv_id": identity["arxiv_id"],
        "arxiv_version": identity["arxiv_version"],
        "duplicate_match": duplicate.get("duplicate_match") if duplicate else None,
        "parser": "pypdf+pdfplumber" if target.suffix.lower() == ".pdf" else "plain_text",
        "equation_candidates": len(structured["equations"]),
        "extracted_table_count": len(visual_metadata["tables"]),
        "image_count": len(visual_metadata["images"]),
        "structure_version": "module-1.0",
    }
    with db.connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents(filename, title, version, sha256, page_count, text_length,
                                  status, duplicate_of, metadata_json, authors_json, abstract_text,
                                  headings_json, tables_json, figures_json, extraction_mode, ocr_status,
                                  embedding_status, embedding_model, embedding_error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                safe_name,
                title,
                version,
                sha256,
                len(pages),
                len(full_text),
                status,
                duplicate["id"] if duplicate else None,
                json.dumps(metadata, ensure_ascii=False),
                json.dumps(structured["authors"], ensure_ascii=False),
                structured["abstract"],
                json.dumps(structured["headings"], ensure_ascii=False),
                json.dumps(structured["tables"], ensure_ascii=False),
                json.dumps(structured["figures"], ensure_ascii=False),
                extraction_mode,
                ocr_status,
                "pending",
                get_embedding_model(),
                "",
                db.utc_now(),
            ),
        )
        document_id = cursor.lastrowid
        for page_number, page_text in enumerate(pages, start=1):
            conn.execute(
                """
                INSERT INTO document_pages(document_id, page_number, content, extraction_mode, char_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, page_number, page_text, extraction_mode, len(page_text)),
            )
            for heading in structured["headings"]:
                if heading["page"] == page_number:
                    conn.execute(
                        """
                        INSERT INTO document_sections(document_id, page_number, level, heading)
                        VALUES (?, ?, ?, ?)
                        """,
                        (document_id, page_number, heading["level"], heading["heading"]),
                    )
            for section, chunk in split_page(page_text, page_number):
                conn.execute(
                    "INSERT INTO chunks(document_id, page, section, content) VALUES (?, ?, ?, ?)",
                    (document_id, page_number, section, chunk),
                )
        for index, reference in enumerate(extract_references(full_text), start=1):
            conn.execute(
                "INSERT INTO paper_references(document_id, ref_index, raw_text) VALUES (?, ?, ?)",
                (document_id, index, reference),
            )
        for page_number, page_text in enumerate(pages, start=1):
            for mention in extract_citation_mentions(page_text, page_number):
                conn.execute(
                    """
                    INSERT INTO citation_mentions(document_id, page_number, ref_index, marker, context)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, mention["page"], mention["ref_index"], mention["marker"], mention["context"]),
                )
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    result = {
        "document": db.document_dict(row),
        "duplicate": status == "duplicate_candidate",
        "similarity": round(score, 4),
        "structure": structured,
    }
    if duplicate:
        result["matched_document"] = duplicate
        result["version_recommendation"] = duplicate["duplicate_match"]["recommendation"]
    return result


DOCUMENT_SORT_FIELDS = {
    "created_at": "created_at",
    "title": "title COLLATE NOCASE",
    "version": "version COLLATE NOCASE",
    "page_count": "page_count",
    "text_length": "text_length",
    "status": "status COLLATE NOCASE",
    "embedding_status": "embedding_status COLLATE NOCASE",
}


def list_documents(
    *,
    query: str = "",
    status: str | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    if sort_by not in DOCUMENT_SORT_FIELDS:
        raise ValueError(f"sort_by 必须是：{', '.join(DOCUMENT_SORT_FIELDS)}。")
    if order not in {"asc", "desc"}:
        raise ValueError("order 必须是 asc 或 desc。")
    if page < 1:
        raise ValueError("page 必须大于等于 1。")
    if page_size < 1 or page_size > 100:
        raise ValueError("page_size 必须在 1 到 100 之间。")
    filters = []
    params: list[Any] = []
    clean_query = query.strip()
    if clean_query:
        filters.append("(title LIKE ? OR filename LIKE ? OR version LIKE ?)")
        like_query = f"%{clean_query}%"
        params.extend([like_query, like_query, like_query])
    if status:
        filters.append("status = ?")
        params.append(status)
    where = f" WHERE {' AND '.join(filters)}" if filters else ""
    order_sql = "DESC" if order == "desc" else "ASC"
    offset = (page - 1) * page_size
    with db.connection() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS count FROM documents{where}", params).fetchone()["count"]
        rows = conn.execute(
            f"SELECT * FROM documents{where} ORDER BY {DOCUMENT_SORT_FIELDS[sort_by]} {order_sql}, id DESC LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        ).fetchall()
    return {
        "documents": [db.document_dict(row) for row in rows],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size,
        },
        "query": clean_query,
        "status": status,
        "sort": {"by": sort_by, "order": order},
    }


def get_document(document_id: int) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return db.document_dict(row) if row else None


async def index_document_embeddings(document_id: int) -> dict[str, Any]:
    """Create persistent embeddings for one document's chunks."""
    document = get_document(document_id)
    if not document:
        raise ValueError("文献不存在。")
    with db.connection() as conn:
        chunks = conn.execute(
            "SELECT id, content FROM chunks WHERE document_id = ? ORDER BY id",
            (document_id,),
        ).fetchall()
    if not chunks:
        raise ValueError("文献没有可建立索引的文本分片。")
    if not embedding_client.embedding_configured:
        message = "未配置 OPENAI_API_KEY，暂不能建立 Embedding 索引。"
        with db.connection() as conn:
            conn.execute(
                "UPDATE documents SET embedding_status = 'pending', embedding_error = ? WHERE id = ?",
                (message, document_id),
            )
        return {"document_id": document_id, "status": "pending", "indexed_chunks": 0, "error": message}

    with db.connection() as conn:
        conn.execute(
            "UPDATE documents SET embedding_status = 'processing', embedding_error = '' WHERE id = ?",
            (document_id,),
        )
    try:
        vectors: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors.extend(await embedding_client.embed_texts([row["content"] for row in batch]))
    except ModelRequestError as error:
        with db.connection() as conn:
            conn.execute(
                "UPDATE documents SET embedding_status = 'error', embedding_error = ? WHERE id = ?",
                (str(error), document_id),
            )
        return {"document_id": document_id, "status": "error", "indexed_chunks": 0, "error": str(error)}

    with db.connection() as conn:
        conn.execute("DELETE FROM chunk_embeddings WHERE document_id = ?", (document_id,))
        for row, vector in zip(chunks, vectors):
            conn.execute(
                """
                INSERT INTO chunk_embeddings(chunk_id, document_id, model, dimensions, vector_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    document_id,
                    embedding_client.embedding_model,
                    len(vector),
                    json.dumps(vector, ensure_ascii=False),
                    db.utc_now(),
                ),
            )
        conn.execute(
            """
            UPDATE documents SET embedding_status = 'indexed', embedding_model = ?, embedding_error = ''
            WHERE id = ?
            """,
            (embedding_client.embedding_model, document_id),
        )
    return {
        "document_id": document_id,
        "status": "indexed",
        "indexed_chunks": len(vectors),
        "embedding_model": embedding_client.embedding_model,
        "dimensions": len(vectors[0]) if vectors else 0,
    }


async def index_all_document_embeddings() -> list[dict[str, Any]]:
    catalog = list_documents(page_size=100)
    return [await index_document_embeddings(document["id"]) for document in catalog["documents"]]


def get_document_pages(document_id: int) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT page_number, content, extraction_mode, char_count
            FROM document_pages WHERE document_id = ? ORDER BY page_number
            """,
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_document_page(document_id: int, page_number: int) -> dict[str, Any] | None:
    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT page_number, content, extraction_mode, char_count
            FROM document_pages WHERE document_id = ? AND page_number = ?
            """,
            (document_id, page_number),
        ).fetchone()
    return dict(row) if row else None


def get_document_structure(document_id: int) -> dict[str, Any] | None:
    document = get_document(document_id)
    if not document:
        return None
    with db.connection() as conn:
        sections = conn.execute(
            """
            SELECT page_number, level, heading
            FROM document_sections WHERE document_id = ? ORDER BY page_number, id
            """,
            (document_id,),
        ).fetchall()
        references = conn.execute(
            """
            SELECT ref_index, raw_text
            FROM paper_references WHERE document_id = ? ORDER BY ref_index
            """,
            (document_id,),
        ).fetchall()
    return {
        "document_id": document_id,
        "title": document["title"],
        "authors": document.get("authors", []),
        "abstract": document.get("abstract_text", ""),
        "headings": document.get("headings", []),
        "sections": [dict(row) for row in sections],
        "tables": document.get("tables", []),
        "figures": document.get("figures", []),
        "references": [dict(row) for row in references],
        "extraction_mode": document.get("extraction_mode", "text"),
        "ocr_status": document.get("ocr_status", "not_required"),
    }


def get_document_references(document_id: int) -> list[dict[str, Any]] | None:
    if not get_document(document_id):
        return None
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT ref_index, raw_text FROM paper_references WHERE document_id = ? ORDER BY ref_index",
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_citation_links(document_id: int) -> list[dict[str, Any]] | None:
    if not get_document(document_id):
        return None
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT m.page_number, m.ref_index, m.marker, m.context, r.raw_text
            FROM citation_mentions m
            LEFT JOIN paper_references r
              ON r.document_id = m.document_id AND r.ref_index = m.ref_index
            WHERE m.document_id = ?
            ORDER BY m.page_number, m.id
            """,
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def rebuild_citation_links(document_id: int) -> int:
    if not get_document(document_id):
        raise ValueError("文献不存在。")
    pages = get_document_pages(document_id)
    with db.connection() as conn:
        conn.execute("DELETE FROM citation_mentions WHERE document_id = ?", (document_id,))
        count = 0
        for page in pages:
            for mention in extract_citation_mentions(page["content"], page["page_number"]):
                conn.execute(
                    """
                    INSERT INTO citation_mentions(document_id, page_number, ref_index, marker, context)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, mention["page"], mention["ref_index"], mention["marker"], mention["context"]),
                )
                count += 1
    return count


def get_reference_mentions(document_id: int, ref_index: int) -> list[dict[str, Any]] | None:
    if not get_document(document_id):
        return None
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT page_number, ref_index, marker, context
            FROM citation_mentions WHERE document_id = ? AND ref_index = ?
            ORDER BY page_number, id
            """,
            (document_id, ref_index),
        ).fetchall()
    return [dict(row) for row in rows]


def get_translation_cache(
    document_id: int,
    page_number: int | None,
    source_hash: str,
    target_language: str,
    model: str,
) -> str | None:
    with db.connection() as conn:
        row = conn.execute(
            """
            SELECT translated_text FROM translation_cache
            WHERE document_id = ? AND page_number IS ? AND source_hash = ?
              AND target_language = ? AND model = ?
            """,
            (document_id, page_number, source_hash, target_language, model),
        ).fetchone()
    return row["translated_text"] if row else None


def save_translation_cache(
    document_id: int,
    page_number: int | None,
    source_text: str,
    target_language: str,
    model: str,
    translated_text: str,
) -> None:
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO translation_cache(
                document_id, page_number, source_hash, source_text,
                target_language, model, translated_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, page_number, source_hash, target_language, model)
            DO UPDATE SET translated_text = excluded.translated_text, created_at = excluded.created_at
            """,
            (document_id, page_number, source_hash, source_text, target_language, model, translated_text, db.utc_now()),
        )


def get_document_path(document_id: int) -> Path | None:
    document = get_document(document_id)
    if not document:
        return None
    raw_path = document.get("metadata", {}).get("path")
    if not raw_path:
        return None
    path = Path(raw_path).resolve()
    upload_root = UPLOAD_DIR.resolve()
    try:
        path.relative_to(upload_root)
    except ValueError:
        return None
    return path if path.is_file() else None


def resolve_duplicate(document_id: int, action: str) -> dict[str, Any] | None:
    document = get_document(document_id)
    if not document:
        return None
    matched_id = document.get("duplicate_of")
    with db.connection() as conn:
        if action == "keep_both":
            conn.execute(
                "UPDATE documents SET status = 'indexed', duplicate_of = NULL WHERE id = ?",
                (document_id,),
            )
        elif action == "keep_old":
            conn.execute(
                "UPDATE documents SET status = 'rejected_duplicate' WHERE id = ?",
                (document_id,),
            )
        elif action == "keep_new":
            conn.execute(
                "UPDATE documents SET status = 'indexed', duplicate_of = NULL WHERE id = ?",
                (document_id,),
            )
            if matched_id:
                conn.execute(
                    "UPDATE documents SET status = 'superseded' WHERE id = ?",
                    (matched_id,),
                )
        else:
            raise ValueError("action 必须是 keep_old、keep_new 或 keep_both。")
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return db.document_dict(row)


def extract_references(text: str) -> list[str]:
    references = []
    in_references = False
    for line in text.splitlines():
        clean = line.strip()
        if re.match(r"^(references|参考文献)\s*$", clean, flags=re.I):
            in_references = True
            continue
        if in_references and re.match(r"^(\[\d+\]|\d+[.)])\s+", clean):
            references.append(clean[:1000])
    return references[:500]


def extract_citation_mentions(page_text: str, page_number: int) -> list[dict[str, Any]]:
    mentions = []
    in_references = False
    for line in page_text.splitlines():
        clean = line.strip()
        if re.match(r"^(references|参考文献)\s*$", clean, flags=re.I):
            in_references = True
            continue
        if in_references:
            continue
        for match in re.finditer(r"\[\s*([0-9]+(?:\s*[,;、]\s*[0-9]+|\s*[-–]\s*[0-9]+)*)\s*\]", clean):
            numbers: list[int] = []
            for raw_number in re.split(r"[,;、]", match.group(1)):
                range_parts = re.split(r"\s*[-–]\s*", raw_number.strip())
                try:
                    start = int(range_parts[0])
                    end = int(range_parts[-1])
                except ValueError:
                    continue
                if end >= start and end - start <= 50:
                    numbers.extend(range(start, end + 1))
            for ref_index in sorted(set(numbers)):
                mentions.append(
                    {
                        "page": page_number,
                        "ref_index": ref_index,
                        "marker": match.group(0),
                        "context": clean[:1000],
                    }
                )
    return mentions[:1000]


def retrieve(
    question: str,
    document_ids: list[int] | None = None,
    limit: int = 6,
    sections: list[str] | None = None,
) -> list[dict[str, Any]]:
    tokens = set(re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", question.lower()))
    if not tokens:
        return []
    with db.connection() as conn:
        query = "SELECT c.*, d.filename, d.title, d.version FROM chunks c JOIN documents d ON d.id = c.document_id"
        params: list[Any] = []
        filters = []
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            filters.append(f"c.document_id IN ({placeholders})")
            params.extend(document_ids)
        if sections:
            placeholders = ",".join("?" for _ in sections)
            filters.append(f"c.section IN ({placeholders})")
            params.extend(sections)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        rows = conn.execute(query, params).fetchall()
    results = []
    for row in rows:
        content = row["content"]
        content_tokens = set(re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", content.lower()))
        overlap = len(tokens & content_tokens)
        phrase_bonus = 0.35 if question.lower() in content.lower() else 0.0
        score = overlap / max(1, len(tokens)) + phrase_bonus
        if score > 0:
            results.append(
                {
                    "chunk_id": row["id"],
                    "document_id": row["document_id"],
                    "filename": row["filename"],
                    "title": row["title"],
                    "version": row["version"],
                    "page": row["page"],
                    "section": row["section"],
                    "content": content,
                    "score": round(score, 4),
                }
            )
    return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


async def hybrid_retrieve(
    question: str,
    document_ids: list[int] | None = None,
    limit: int = 6,
    sections: list[str] | None = None,
    max_per_document: int = 4,
) -> list[dict[str, Any]]:
    lexical = retrieve(question, document_ids, limit=max(limit * 3, 12), sections=sections)
    lexical_by_chunk = {item["chunk_id"]: item for item in lexical}
    vector_by_chunk: dict[int, dict[str, Any]] = {}
    retrieval_mode = "keyword"
    retrieval_warning = ""

    if embedding_client.embedding_configured:
        try:
            query_vector = (await embedding_client.embed_texts([question]))[0]
            with db.connection() as conn:
                query = """
                    SELECT e.chunk_id, e.document_id, e.vector_json, c.page, c.section, c.content,
                           d.filename, d.title, d.version
                    FROM chunk_embeddings e
                    JOIN chunks c ON c.id = e.chunk_id
                    JOIN documents d ON d.id = e.document_id
                """
                params: list[Any] = []
                filters = []
                if document_ids:
                    placeholders = ",".join("?" for _ in document_ids)
                    filters.append(f"e.document_id IN ({placeholders})")
                    params.extend(document_ids)
                if sections:
                    placeholders = ",".join("?" for _ in sections)
                    filters.append(f"c.section IN ({placeholders})")
                    params.extend(sections)
                if filters:
                    query += " WHERE " + " AND ".join(filters)
                rows = conn.execute(query, params).fetchall()
            for row in rows:
                vector_score = _cosine_similarity(query_vector, json.loads(row["vector_json"]))
                vector_by_chunk[row["chunk_id"]] = {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "filename": row["filename"],
                    "title": row["title"],
                    "version": row["version"],
                    "page": row["page"],
                    "section": row["section"],
                    "content": row["content"],
                    "vector_score": round(vector_score, 6),
                }
            retrieval_mode = "hybrid" if vector_by_chunk else "keyword"
        except (ModelRequestError, json.JSONDecodeError, TypeError, ValueError) as error:
            retrieval_warning = str(error)

    merged: dict[int, dict[str, Any]] = {}
    for chunk_id, item in lexical_by_chunk.items():
        merged[chunk_id] = {**item, "keyword_score": item["score"], "vector_score": 0.0}
    for chunk_id, item in vector_by_chunk.items():
        merged.setdefault(chunk_id, {**item, "keyword_score": 0.0})
        merged[chunk_id].update({"vector_score": item["vector_score"]})

    question_lower = question.lower()
    results = []
    for item in merged.values():
        keyword_score = min(1.0, max(0.0, float(item.get("keyword_score", 0.0))))
        vector_score = max(0.0, float(item.get("vector_score", 0.0)))
        if retrieval_mode == "hybrid":
            score = 0.35 * keyword_score + 0.65 * vector_score
        else:
            score = keyword_score
        if question_lower in item["content"].lower():
            score += 0.1
        if item["title"].lower() in question_lower:
            score += 0.05
        item["score"] = round(score, 6)
        item["retrieval_mode"] = retrieval_mode
        if retrieval_warning:
            item["retrieval_warning"] = retrieval_warning
        results.append(item)
    ranked = sorted(results, key=lambda item: item["score"], reverse=True)
    diversified = []
    counts: dict[int, int] = {}
    for item in ranked:
        document_id = item["document_id"]
        if counts.get(document_id, 0) >= max_per_document:
            continue
        diversified.append(item)
        counts[document_id] = counts.get(document_id, 0) + 1
        if len(diversified) >= limit:
            break
    return diversified


def citation_payload(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"E{index}",
            "document_id": item["document_id"],
            "filename": item["filename"],
            "page": item["page"],
            "section": item["section"],
            "score": item["score"],
            "keyword_score": item.get("keyword_score", item["score"]),
            "vector_score": item.get("vector_score", 0.0),
            "retrieval_mode": item.get("retrieval_mode", "keyword"),
            "quote": item["content"][:900],
        }
        for index, item in enumerate(evidence, start=1)
    ]
