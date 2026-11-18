from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree

import httpx

from . import db
from .document_service import ingest_file, list_documents


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_SOURCE = "arxiv"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
GITHUB_PATTERN = re.compile(r"https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+", flags=re.I)
ARXIV_REQUEST_INTERVAL = 3.0
_arxiv_request_lock = asyncio.Lock()
_arxiv_last_request = 0.0


class AcademicSearchError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_arxiv_id(value: str) -> str:
    clean = value.strip()
    clean = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", clean, flags=re.I)
    clean = clean.removesuffix(".pdf")
    return clean.strip("/")


def _text(element: ElementTree.Element | None) -> str:
    return " ".join((element.text or "").split()) if element is not None else ""


def _parse_entry(entry: ElementTree.Element, query: str) -> dict[str, Any]:
    entry_url = _text(entry.find("atom:id", ATOM_NS))
    external_id = normalize_arxiv_id(entry_url.rsplit("/", 1)[-1])
    title = _text(entry.find("atom:title", ATOM_NS))
    summary = _text(entry.find("atom:summary", ATOM_NS))
    authors = [_text(author.find("atom:name", ATOM_NS)) for author in entry.findall("atom:author", ATOM_NS)]
    categories = [item.attrib.get("term", "") for item in entry.findall("atom:category", ATOM_NS)]
    links = [item.attrib for item in entry.findall("atom:link", ATOM_NS)]
    pdf_url = next((item.get("href") for item in links if item.get("title") == "pdf"), None)
    abs_url = entry_url or f"https://arxiv.org/abs/{external_id}"
    github_urls = sorted(set(GITHUB_PATTERN.findall(f"{title} {summary}")))
    return {
        "source": ARXIV_SOURCE,
        "external_id": external_id,
        "title": title,
        "authors": [name for name in authors if name],
        "summary": summary,
        "published_at": _text(entry.find("atom:published", ATOM_NS)),
        "updated_at": _text(entry.find("atom:updated", ATOM_NS)),
        "pdf_url": pdf_url or f"https://arxiv.org/pdf/{external_id}.pdf",
        "abs_url": abs_url,
        "categories": categories,
        "github_urls": github_urls,
        "query": query,
        "raw": {"entry_url": entry_url, "links": links},
    }


def _persist_papers(papers: list[dict[str, Any]]) -> None:
    db.init_db()
    with db.connection() as conn:
        for paper in papers:
            conn.execute(
                """
                INSERT INTO external_papers(
                    source, external_id, title, authors_json, summary, published_at, updated_at,
                    pdf_url, abs_url, categories_json, github_urls_json, query, raw_json, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, external_id) DO UPDATE SET
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    summary = excluded.summary,
                    published_at = excluded.published_at,
                    updated_at = excluded.updated_at,
                    pdf_url = excluded.pdf_url,
                    abs_url = excluded.abs_url,
                    categories_json = excluded.categories_json,
                    github_urls_json = excluded.github_urls_json,
                    query = excluded.query,
                    raw_json = excluded.raw_json,
                    fetched_at = excluded.fetched_at
                """,
                (
                    paper["source"],
                    paper["external_id"],
                    paper["title"],
                    json.dumps(paper["authors"], ensure_ascii=False),
                    paper["summary"],
                    paper["published_at"],
                    paper["updated_at"],
                    paper["pdf_url"],
                    paper["abs_url"],
                    json.dumps(paper["categories"], ensure_ascii=False),
                    json.dumps(paper["github_urls"], ensure_ascii=False),
                    paper["query"],
                    json.dumps(paper["raw"], ensure_ascii=False),
                    utc_now(),
                ),
            )


def _external_dict(row: Any) -> dict[str, Any]:
    item = dict(row)
    for field in ("authors_json", "categories_json", "github_urls_json"):
        item[field.removesuffix("_json")] = json.loads(item.pop(field) or "[]")
    item["raw"] = json.loads(item.pop("raw_json") or "{}")
    return item


async def search_arxiv(
    query: str = "",
    *,
    arxiv_id: str | None = None,
    start: int = 0,
    max_results: int = 10,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> dict[str, Any]:
    clean_query = query.strip()
    if not clean_query and not arxiv_id:
        raise ValueError("query 或 arxiv_id 至少填写一个。")
    if start < 0 or start > 10000:
        raise ValueError("start 必须在 0 到 10000 之间。")
    if max_results < 1 or max_results > 50:
        raise ValueError("max_results 必须在 1 到 50 之间。")
    if sort_by not in {"relevance", "lastUpdatedDate", "submittedDate"}:
        raise ValueError("sort_by 必须是 relevance、lastUpdatedDate 或 submittedDate。")
    if sort_order not in {"ascending", "descending"}:
        raise ValueError("sort_order 必须是 ascending 或 descending。")
    params = {
        "search_query": f"id:{normalize_arxiv_id(arxiv_id)}" if arxiv_id else f"all:{clean_query}",
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    global _arxiv_last_request
    try:
        async with _arxiv_request_lock:
            delay = ARXIV_REQUEST_INTERVAL - (time.monotonic() - _arxiv_last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
                    response = await client.get(ARXIV_API_URL, params=params)
            finally:
                _arxiv_last_request = time.monotonic()
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_message = f"约 {retry_after} 秒后" if retry_after and retry_after.isdigit() else "稍后"
            raise AcademicSearchError(f"arXiv 请求频率受限，请{wait_message}重试。")
        response.raise_for_status()
    except AcademicSearchError:
        raise
    except httpx.HTTPStatusError as error:
        raise AcademicSearchError(f"arXiv 检索失败（HTTP {error.response.status_code}）。请稍后重试。") from error
    except httpx.HTTPError as error:
        raise AcademicSearchError("arXiv 暂时无法连接，请检查网络后重试。") from error
    try:
        root = ElementTree.fromstring(response.text)
    except ElementTree.ParseError as error:
        raise AcademicSearchError("arXiv 返回内容不是合法 XML。") from error
    papers = [_parse_entry(entry, clean_query) for entry in root.findall("atom:entry", ATOM_NS)]
    _persist_papers(papers)
    return {
        "source": ARXIV_SOURCE,
        "query": clean_query,
        "arxiv_id": normalize_arxiv_id(arxiv_id) if arxiv_id else None,
        "start": start,
        "max_results": max_results,
        "count": len(papers),
        "papers": papers,
    }


async def download_arxiv_paper(external_id: str, pdf_url: str | None = None) -> dict[str, Any]:
    clean_id = normalize_arxiv_id(external_id)
    if not clean_id:
        raise ValueError("external_id 不能为空。")
    db.init_db()
    with db.connection() as conn:
        cached = conn.execute(
            "SELECT title FROM external_papers WHERE source = ? AND external_id = ?",
            (ARXIV_SOURCE, clean_id),
        ).fetchone()
    paper_title = cached["title"].strip() if cached and cached["title"] else ""
    if not paper_title:
        metadata = await search_arxiv("", arxiv_id=clean_id, max_results=1)
        if metadata["papers"]:
            paper_title = metadata["papers"][0]["title"].strip()
    url = pdf_url or f"https://arxiv.org/pdf/{clean_id}.pdf"
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("pdf_url 必须是 HTTP(S) 地址。")
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
    except httpx.HTTPError as error:
        raise AcademicSearchError(f"论文下载失败：{error}") from error
    if len(content) > 80 * 1024 * 1024:
        raise ValueError("论文文件超过 80 MB，已拒绝下载。")
    if not content.startswith(b"%PDF"):
        raise AcademicSearchError("下载内容不是 PDF 文件。")
    result = ingest_file(f"arxiv_{clean_id.replace('/', '_')}.pdf", content)
    if paper_title and not result.get("duplicate"):
        with db.connection() as conn:
            conn.execute("UPDATE documents SET title = ? WHERE id = ?", (paper_title, result["document"]["id"]))
        result["document"]["title"] = paper_title
    result["source"] = ARXIV_SOURCE
    result["external_id"] = clean_id
    result["download_url"] = url
    return result


def list_tracked_queries() -> list[dict[str, Any]]:
    db.init_db()
    with db.connection() as conn:
        rows = conn.execute("SELECT * FROM tracked_queries ORDER BY enabled DESC, created_at DESC").fetchall()
    return [dict(row) for row in rows]


def create_tracked_query(query: str, max_results: int = 10) -> dict[str, Any]:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("追踪主题不能为空。")
    if max_results < 1 or max_results > 50:
        raise ValueError("max_results 必须在 1 到 50 之间。")
    db.init_db()
    with db.connection() as conn:
        cursor = conn.execute(
            "INSERT INTO tracked_queries(query, max_results, enabled, created_at) VALUES (?, ?, 1, ?)",
            (clean_query, max_results, utc_now()),
        )
        row = conn.execute("SELECT * FROM tracked_queries WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def delete_tracked_query(track_id: int) -> bool:
    db.init_db()
    with db.connection() as conn:
        cursor = conn.execute("DELETE FROM tracked_queries WHERE id = ?", (track_id,))
    return cursor.rowcount > 0


async def refresh_tracked_query(track_id: int) -> dict[str, Any]:
    db.init_db()
    with db.connection() as conn:
        row = conn.execute("SELECT * FROM tracked_queries WHERE id = ?", (track_id,)).fetchone()
    if not row:
        raise ValueError("追踪主题不存在。")
    result = await search_arxiv(row["query"], max_results=row["max_results"])
    with db.connection() as conn:
        conn.execute("UPDATE tracked_queries SET last_run_at = ? WHERE id = ?", (utc_now(), track_id))
    return {"track": dict(row), "search": result}


async def refresh_enabled_tracked_queries() -> dict[str, Any]:
    refreshed = []
    failed = []
    for track in list_tracked_queries():
        if not track.get("enabled"):
            continue
        try:
            refreshed.append(await refresh_tracked_query(int(track["id"])))
        except Exception as error:
            failed.append({"track_id": track["id"], "error": str(error)})
    return {"refreshed": refreshed, "failed": failed}


def _query_terms(query: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9+.#-]{1,}|[\u4e00-\u9fff]", query.lower()))


def _recency_score(date_text: str | None) -> float:
    if not date_text:
        return 0.0
    try:
        parsed = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
        days = max(0, (datetime.now(timezone.utc) - parsed).days)
        return max(0.0, 1.0 - min(days, 730) / 730)
    except ValueError:
        return 0.0


def _feedback_actions() -> dict[str, str]:
    db.init_db()
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT external_id, action FROM recommendation_feedback
            WHERE source = ? ORDER BY created_at DESC, id DESC
            """,
            (ARXIV_SOURCE,),
        ).fetchall()
    actions: dict[str, str] = {}
    for row in rows:
        actions.setdefault(row["external_id"], row["action"])
    return actions


def _feedback_adjustment(action: str | None) -> tuple[float, str | None]:
    adjustments = {
        "like": (0.25, "符合你标记的感兴趣方向"),
        "save": (0.2, "你已收藏该论文"),
        "dislike": (-0.5, "你曾标记为不感兴趣"),
        "dismiss": (-1.0, "你已将该论文移出推荐"),
    }
    return adjustments.get(action or "", (0.0, None))


async def recommend_papers(query: str, limit: int = 10) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("推荐主题不能为空。")
    if limit < 1 or limit > 30:
        raise ValueError("limit 必须在 1 到 30 之间。")
    search = await search_arxiv(
        query,
        max_results=min(50, max(limit * 2, 10)),
        sort_by="relevance",
    )
    query_terms = _query_terms(query)
    local_catalog = list_documents(page_size=100)["documents"]
    local_terms = _query_terms(" ".join(item["title"] for item in local_catalog))
    feedback = _feedback_actions()
    ranked = []
    for paper in search["papers"]:
        action = feedback.get(paper["external_id"])
        if action == "dismiss":
            continue
        text = f"{paper['title']} {paper['summary']}".lower()
        overlap = sum(term in text for term in query_terms)
        local_overlap = sum(term in text for term in local_terms)
        github_bonus = 0.2 if paper["github_urls"] else 0.0
        feedback_score, feedback_reason = _feedback_adjustment(action)
        score = overlap / max(1, len(query_terms)) + 0.15 * min(1.0, local_overlap / max(1, len(local_terms))) + github_bonus + 0.15 * _recency_score(paper["updated_at"]) + feedback_score
        reason = [f"与主题关键词匹配 {overlap}/{max(1, len(query_terms))}"]
        if paper["github_urls"]:
            reason.append("摘要中发现 GitHub 代码线索")
        if _recency_score(paper["updated_at"]) > 0.7:
            reason.append("更新时间较近")
        if feedback_reason:
            reason.append(feedback_reason)
        ranked.append({**paper, "feedback_action": action, "recommendation_score": round(score, 4), "recommendation_reason": "；".join(reason)})
    ranked.sort(key=lambda item: item["recommendation_score"], reverse=True)
    return {"query": query, "count": min(limit, len(ranked)), "recommendations": ranked[:limit]}


def save_feedback(source: str, external_id: str, action: str, note: str = "") -> dict[str, Any]:
    if action not in {"like", "dislike", "save", "dismiss"}:
        raise ValueError("action 必须是 like、dislike、save 或 dismiss。")
    db.init_db()
    with db.connection() as conn:
        cursor = conn.execute(
            "INSERT INTO recommendation_feedback(source, external_id, action, note, created_at) VALUES (?, ?, ?, ?, ?)",
            (source, external_id, action, note[:500], utc_now()),
        )
    return {"id": cursor.lastrowid, "source": source, "external_id": external_id, "action": action, "note": note[:500]}
