from __future__ import annotations

import asyncio
import html
import logging
import os
from contextlib import suppress
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .agent_service import (
    answer_question,
    compare_documents,
    generate_writing,
    recommend,
    render_cached_translation_html,
    translate_document,
    translate_document_page,
    translate_text,
    visualize_csv,
)
from .citation_graph import build_citation_graph, generate_graph_review
from .copilot_service import analyze_csv, generate_diagram, generate_framework
from .config import STATIC_DIR, ensure_runtime_dirs, get_api_key, get_embedding_model, get_model
from .document_service import (
    citation_payload,
    get_document,
    get_document_page,
    get_document_pages,
    get_document_path,
    get_document_references,
    get_document_structure,
    get_citation_links,
    get_reference_mentions,
    ingest_file,
    index_all_document_embeddings,
    index_document_embeddings,
    hybrid_retrieve,
    list_documents,
    resolve_duplicate,
    rebuild_citation_links,
)
from .evaluation import evaluate_retrieval
from .academic_search import (
    AcademicSearchError,
    create_tracked_query,
    delete_tracked_query,
    download_arxiv_paper,
    list_tracked_queries,
    recommend_papers as recommend_external_papers,
    refresh_tracked_query,
    refresh_enabled_tracked_queries,
    save_feedback,
    search_arxiv,
)
from .model_client import ModelConfigurationError, ModelRequestError
from .python_sandbox import SandboxError, execute_analysis


app = FastAPI(title="科研助手 Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logger = logging.getLogger(__name__)
tracking_scheduler_task: asyncio.Task | None = None
MAX_DOCUMENT_UPLOAD_BYTES = 80 * 1024 * 1024


def tracking_refresh_interval() -> int:
    try:
        return max(60, int(os.getenv("TRACK_REFRESH_INTERVAL_SECONDS", "86400")))
    except ValueError:
        return 86400


async def tracking_scheduler() -> None:
    while True:
        await asyncio.sleep(tracking_refresh_interval())
        result = await refresh_enabled_tracked_queries()
        if result["failed"]:
            logger.warning("Scheduled research-topic refresh failures: %s", result["failed"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[int] | None = None
    sections: list[str] | None = None
    query_rewrite: bool = False


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[int] | None = None
    limit: int = Field(default=6, ge=1, le=20)
    sections: list[str] | None = None


class CompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[int] = Field(min_length=2, max_length=20)
    query_rewrite: bool = False


class WriteRequest(BaseModel):
    idea: str = Field(min_length=1, max_length=6000)
    document_ids: list[int] | None = None


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    target_language: str = "English"


class TranslatePageRequest(BaseModel):
    target_language: str = "English"


class TranslateDocumentRequest(BaseModel):
    pages: list[int] | None = None
    target_language: str = "English"


class RecommendRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class GraphRequest(BaseModel):
    document_ids: list[int] | None = None


class GraphReviewRequest(BaseModel):
    document_ids: list[int] | None = None


class CopilotFrameworkRequest(BaseModel):
    idea: str = Field(min_length=1, max_length=6000)
    document_ids: list[int] | None = None


class DiagramRequest(BaseModel):
    document_ids: list[int] | None = None
    format: Literal["mermaid", "graphviz", "tikz", "matplotlib"] = "mermaid"


class PythonAnalysisRequest(BaseModel):
    code: str = Field(min_length=1, max_length=8000)
    variables: dict[str, Any] = Field(default_factory=dict)


class DuplicateDecisionRequest(BaseModel):
    action: Literal["keep_old", "keep_new", "keep_both"]


class RetrievalEvaluationRequest(BaseModel):
    cases: list[dict[str, Any]] = Field(min_length=1, max_length=200)


class ArxivSearchRequest(BaseModel):
    query: str = Field(default="", max_length=300)
    arxiv_id: str | None = Field(default=None, max_length=100)
    start: int = Field(default=0, ge=0, le=10000)
    max_results: int = Field(default=10, ge=1, le=50)
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "submittedDate"
    sort_order: Literal["ascending", "descending"] = "descending"


class ArxivDownloadRequest(BaseModel):
    external_id: str = Field(min_length=1, max_length=100)
    pdf_url: str | None = Field(default=None, max_length=500)


class TrackedQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=10, ge=1, le=50)


class RecommendationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=10, ge=1, le=30)


class RecommendationFeedbackRequest(BaseModel):
    source: str = Field(min_length=1, max_length=40)
    external_id: str = Field(min_length=1, max_length=100)
    action: Literal["like", "dislike", "save", "dismiss"]
    note: str = Field(default="", max_length=500)


def model_error(error: Exception) -> HTTPException:
    if isinstance(error, ModelConfigurationError):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, (ModelRequestError, AcademicSearchError)):
        return HTTPException(status_code=502, detail=str(error))
    if isinstance(error, ValueError):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail=f"服务内部错误：{error}")


@app.on_event("startup")
async def startup() -> None:
    global tracking_scheduler_task
    ensure_runtime_dirs()
    db.init_db()
    if tracking_scheduler_task is None or tracking_scheduler_task.done():
        tracking_scheduler_task = asyncio.create_task(tracking_scheduler())


@app.on_event("shutdown")
async def shutdown() -> None:
    global tracking_scheduler_task
    if tracking_scheduler_task:
        tracking_scheduler_task.cancel()
        with suppress(asyncio.CancelledError):
            await tracking_scheduler_task
        tracking_scheduler_task = None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_configured": bool(get_api_key() and get_model()),
        "embedding_configured": bool(get_api_key() and get_embedding_model()),
        "embedding_model": get_embedding_model(),
        "tracking_scheduler": "running" if tracking_scheduler_task and not tracking_scheduler_task.done() else "stopped",
        "tracking_refresh_interval_seconds": tracking_refresh_interval(),
        "message": "模型 API 配置已填写；健康检查未验证远端连通性" if get_api_key() and get_model() else "请先在 .env 配置 OPENAI_API_KEY 和 OPENAI_MODEL",
        "documents": list_documents(page_size=1)["pagination"]["total"],
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名。")
    try:
        content = await file.read(MAX_DOCUMENT_UPLOAD_BYTES + 1)
        if len(content) > MAX_DOCUMENT_UPLOAD_BYTES:
            raise ValueError("单个 PDF 或文档不能超过 80 MB。")
        if not content:
            raise ValueError("上传文件为空。")
        result = ingest_file(file.filename, content)
        if result.get("duplicate"):
            result["embedding"] = {"status": result["document"].get("embedding_status", "pending"), "already_indexed": True}
        else:
            result["embedding"] = await index_document_embeddings(result["document"]["id"])
        return result
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/documents/upload-batch")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个文件。")
    results = []
    for file in files[:50]:
        if not file.filename:
            results.append({"filename": "", "status": "failed", "error": "缺少文件名。"})
            continue
        try:
            content = await file.read(MAX_DOCUMENT_UPLOAD_BYTES + 1)
            if len(content) > MAX_DOCUMENT_UPLOAD_BYTES:
                raise ValueError("单个 PDF 或文档不能超过 80 MB。")
            if not content:
                raise ValueError("上传文件为空。")
            result = ingest_file(file.filename, content)
            if result.get("duplicate"):
                result["embedding"] = {"status": result["document"].get("embedding_status", "pending"), "already_indexed": True}
            else:
                result["embedding"] = await index_document_embeddings(result["document"]["id"])
            results.append({"filename": file.filename, "status": "ok", "result": result})
        except Exception as error:
            results.append({"filename": file.filename, "status": "failed", "error": str(error)})
    return {"count": len(results), "results": results}


@app.post("/api/documents/{document_id}/index")
async def index_document(document_id: int) -> dict[str, Any]:
    try:
        return await index_document_embeddings(document_id)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/documents/index-all")
async def index_all_documents() -> dict[str, Any]:
    try:
        results = await index_all_document_embeddings()
        return {"count": len(results), "results": results}
    except Exception as error:
        raise model_error(error) from error


@app.get("/api/documents")
def documents(
    query: str = Query(default="", max_length=200, description="按标题、文件名或版本搜索"),
    status: str | None = Query(default=None, max_length=40),
    sort_by: Literal[
        "created_at",
        "title",
        "version",
        "page_count",
        "text_length",
        "status",
        "embedding_status",
    ] = Query(default="created_at"),
    order: Literal["asc", "desc"] = Query(default="desc"),
    page: int = Query(default=1, ge=1, le=100000),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    try:
        return list_documents(
            query=query,
            status=status,
            sort_by=sort_by,
            order=order,
            page=page,
            page_size=page_size,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/documents/{document_id}")
def document_detail(document_id: int) -> dict[str, Any]:
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文献不存在。")
    return {"document": document, "structure": get_document_structure(document_id)}


@app.get("/api/documents/{document_id}/pages")
def document_pages(document_id: int) -> dict[str, Any]:
    if not get_document(document_id):
        raise HTTPException(status_code=404, detail="文献不存在。")
    return {"document_id": document_id, "pages": get_document_pages(document_id)}


@app.get("/api/documents/{document_id}/pages/{page_number}")
def document_page(document_id: int, page_number: int) -> dict[str, Any]:
    page = get_document_page(document_id, page_number)
    if not page:
        raise HTTPException(status_code=404, detail="文献页面不存在。")
    return {"document_id": document_id, "page": page}


@app.get("/api/documents/{document_id}/references")
def document_references(document_id: int) -> dict[str, Any]:
    references = get_document_references(document_id)
    if references is None:
        raise HTTPException(status_code=404, detail="文献不存在。")
    return {"document_id": document_id, "references": references}


@app.get("/api/documents/{document_id}/citations")
def document_citations(document_id: int) -> dict[str, Any]:
    try:
        rebuild_citation_links(document_id)
        citations = get_citation_links(document_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"document_id": document_id, "citations": citations}


@app.get("/api/documents/{document_id}/references/{ref_index}/mentions")
def reference_mentions(document_id: int, ref_index: int) -> dict[str, Any]:
    try:
        rebuild_citation_links(document_id)
        mentions = get_reference_mentions(document_id, ref_index)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"document_id": document_id, "ref_index": ref_index, "mentions": mentions}


@app.post("/api/documents/{document_id}/pages/{page_number}/translate")
async def translate_document_page_endpoint(
    document_id: int,
    page_number: int,
    request: TranslatePageRequest,
) -> dict[str, Any]:
    try:
        return await translate_document_page(document_id, page_number, request.target_language)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/documents/{document_id}/translate")
async def translate_document_endpoint(
    document_id: int,
    request: TranslateDocumentRequest,
) -> dict[str, Any]:
    try:
        return await translate_document(document_id, request.pages, request.target_language)
    except Exception as error:
        raise model_error(error) from error


@app.get("/api/documents/{document_id}/translation.html", response_class=HTMLResponse)
def translated_document_html(document_id: int, target_language: str = "English") -> HTMLResponse:
    try:
        return HTMLResponse(render_cached_translation_html(document_id, target_language))
    except ValueError as error:
        message = html.escape(str(error))
        return HTMLResponse(
            "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>译文尚未准备好</title><body style='font:16px/1.7 system-ui,sans-serif;"
            "max-width:720px;margin:12vh auto;padding:24px;color:#334155'>"
            f"<h1 style='font-size:22px;color:#1e3a8a'>译文尚未准备好</h1><p>{message}</p>"
            "<p>请回到阅读器翻译所需页面，或先完成全文翻译，再打开缓存 HTML。</p>"
            "</body></html>",
            status_code=400,
        )
    except Exception as error:
        raise model_error(error) from error


@app.get("/api/documents/{document_id}/file")
def document_file(document_id: int) -> FileResponse:
    document = get_document(document_id)
    path = get_document_path(document_id)
    if not document or not path:
        raise HTTPException(status_code=404, detail="文献文件不存在。")
    return FileResponse(path, filename=document["filename"])


@app.get("/api/documents/{document_id}/preview")
def document_preview(document_id: int) -> FileResponse:
    """Serve the source document inline so browsers can render and page through PDFs."""
    document = get_document(document_id)
    path = get_document_path(document_id)
    if not document or not path:
        raise HTTPException(status_code=404, detail="文献文件不存在。")
    return FileResponse(path, media_type="application/pdf", content_disposition_type="inline")


@app.post("/api/documents/{document_id}/duplicate-decision")
def duplicate_decision(document_id: int, request: DuplicateDecisionRequest) -> dict[str, Any]:
    try:
        document = resolve_duplicate(document_id, request.action)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not document:
        raise HTTPException(status_code=404, detail="文献不存在。")
    return {"document": document, "action": request.action}


@app.post("/api/agent/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    try:
        return await answer_question(
            request.question,
            request.document_ids,
            request.sections,
            request.query_rewrite,
        )
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/retrieval/preview")
async def retrieval_preview(request: RetrievalRequest) -> dict[str, Any]:
    try:
        evidence = await hybrid_retrieve(
            request.question,
            request.document_ids,
            request.limit,
            request.sections,
        )
        top_score = evidence[0]["score"] if evidence else 0.0
        passed = bool(evidence and top_score >= 0.08)
        return {
            "status": "supported" if passed else "insufficient_evidence",
            "retrieval_mode": evidence[0].get("retrieval_mode", "keyword") if evidence else "keyword",
            "evidence_gate": {"passed": passed, "threshold": 0.08, "top_score": top_score},
            "evidence": citation_payload(evidence),
        }
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/evaluate/retrieval")
async def evaluate_retrieval_endpoint(request: RetrievalEvaluationRequest) -> dict[str, Any]:
    try:
        return await evaluate_retrieval(request.cases)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/search/arxiv")
async def arxiv_search(request: ArxivSearchRequest) -> dict[str, Any]:
    try:
        return await search_arxiv(
            request.query,
            arxiv_id=request.arxiv_id,
            start=request.start,
            max_results=request.max_results,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
        )
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/search/arxiv/download")
async def arxiv_download(request: ArxivDownloadRequest) -> dict[str, Any]:
    try:
        return await download_arxiv_paper(request.external_id, request.pdf_url)
    except Exception as error:
        raise model_error(error) from error


@app.get("/api/tracks")
def tracks() -> dict[str, Any]:
    return {"tracks": list_tracked_queries()}


@app.post("/api/tracks")
def track_create(request: TrackedQueryRequest) -> dict[str, Any]:
    try:
        return {"track": create_tracked_query(request.query, request.max_results)}
    except Exception as error:
        raise model_error(error) from error


@app.delete("/api/tracks/{track_id}")
def track_delete(track_id: int) -> dict[str, Any]:
    if not delete_tracked_query(track_id):
        raise HTTPException(status_code=404, detail="追踪主题不存在。")
    return {"deleted": True, "track_id": track_id}


@app.post("/api/tracks/{track_id}/refresh")
async def track_refresh(track_id: int) -> dict[str, Any]:
    try:
        return await refresh_tracked_query(track_id)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/recommendations")
async def recommendations(request: RecommendationRequest) -> dict[str, Any]:
    try:
        return await recommend_external_papers(request.query, request.limit)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/recommendations/feedback")
def recommendation_feedback(request: RecommendationFeedbackRequest) -> dict[str, Any]:
    try:
        return {"feedback": save_feedback(request.source, request.external_id, request.action, request.note)}
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/agent/compare")
async def compare(request: CompareRequest) -> dict[str, Any]:
    try:
        return await compare_documents(request.question, request.document_ids, request.query_rewrite)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/agent/write")
async def write(request: WriteRequest) -> dict[str, Any]:
    try:
        return await generate_writing(request.idea, request.document_ids)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/copilot/framework")
async def copilot_framework(request: CopilotFrameworkRequest) -> dict[str, Any]:
    try:
        return await generate_framework(request.idea, request.document_ids)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/agent/translate")
async def translate(request: TranslateRequest) -> dict[str, Any]:
    try:
        return await translate_text(request.text, request.target_language)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/agent/recommend")
async def recommend_papers(request: RecommendRequest) -> dict[str, Any]:
    try:
        return await recommend(request.query)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/graph")
def graph(request: GraphRequest) -> dict[str, Any]:
    return build_citation_graph(request.document_ids)


@app.post("/api/graph/review")
async def graph_review(request: GraphReviewRequest) -> dict[str, Any]:
    try:
        return await generate_graph_review(request.document_ids)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/copilot/diagram")
def copilot_diagram(request: DiagramRequest) -> dict[str, Any]:
    try:
        return generate_diagram(request.document_ids, request.format)
    except Exception as error:
        raise model_error(error) from error


@app.post("/api/tools/python")
def python_analysis(request: PythonAnalysisRequest) -> dict[str, Any]:
    try:
        return execute_analysis(request.code, request.variables)
    except SandboxError as error:
        raise model_error(error) from error


@app.post("/api/visualize/csv")
async def visualize(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        text = (await file.read()).decode("utf-8-sig", errors="replace")
        return await analyze_csv(file.filename or "upload.csv", text)
    except Exception as error:
        raise model_error(error) from error
