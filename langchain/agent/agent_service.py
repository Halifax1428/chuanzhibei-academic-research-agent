from __future__ import annotations

import json
import hashlib
import html
import re
from typing import Any

from . import db
from .document_service import (
    citation_payload,
    get_document,
    get_document_page,
    get_translation_cache,
    hybrid_retrieve,
    list_documents,
    retrieve,
    save_translation_cache,
)
from .model_client import ModelConfigurationError, OpenAICompatibleClient


client = OpenAICompatibleClient()


def evidence_block(evidence: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[E{index}] 文献ID={item['document_id']}，标题={item.get('title', '')}，"
        f"文件={item['filename']}，版本={item.get('version', 'unknown')}，"
        f"页码={item['page']}，章节={item['section']}\n{item['content']}"
        for index, item in enumerate(evidence, start=1)
    )


def validate_answer_citations(answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    referenced = sorted({int(value) for value in re.findall(r"\[E(\d+)\]", answer)})
    available = list(range(1, len(evidence) + 1))
    invalid = [value for value in referenced if value not in available]
    return {
        "valid": bool(referenced) and not invalid,
        "referenced": [f"E{value}" for value in referenced],
        "invalid": [f"E{value}" for value in invalid],
        "available": [f"E{value}" for value in available],
    }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


async def verify_answer(
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt = (
        "核验下面的科研问答是否完全被证据支持。不要引入外部知识。"
        "返回 JSON，不要 Markdown，字段必须为 supported(boolean)、issues(array of strings)、"
        "unsupported_claims(array of strings)。如果无法判断，supported 设为 false。\n\n"
        f"问题：{question}\n回答：{answer}\n证据：\n{evidence_block(evidence)}"
    )
    try:
        raw = await client.complete(
            [
                {"role": "system", "content": "你是严格的学术事实核验器。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        parsed = _parse_json_object(raw)
        if not parsed:
            return {"status": "unparseable", "supported": False, "issues": ["核验模型没有返回合法 JSON。"]}
        supported = parsed.get("supported") is True
        issues = parsed.get("issues", [])
        unsupported_claims = parsed.get("unsupported_claims", [])
        return {
            "status": "verified",
            "supported": supported,
            "issues": issues if isinstance(issues, list) else ["核验模型返回了无效的 issues 字段。"],
            "unsupported_claims": unsupported_claims if isinstance(unsupported_claims, list) else [],
        }
    except Exception as error:
        return {"status": "unavailable", "supported": False, "issues": [str(error)]}


async def rewrite_search_query(question: str) -> dict[str, str]:
    """Turn a natural-language question into a compact retrieval query."""
    prompt = (
        "将下面的科研问题改写成适合论文检索的关键词查询。保留专有名词、公式变量、"
        "数据集、指标、模型名和数值；只输出一行关键词，不要解释，不要回答问题。\n"
        f"问题：{question}"
    )
    try:
        query = (await client.complete(
            [{"role": "system", "content": "你是论文检索查询改写器。"}, {"role": "user", "content": prompt}],
            temperature=0.0,
        )).strip()
        query = re.sub(r"\s+", " ", query).strip("` \n")
        return {"query": query or question, "status": "rewritten" if query else "fallback"}
    except Exception as error:
        return {"query": question, "status": "fallback", "warning": str(error)}


async def answer_question(
    question: str,
    document_ids: list[int] | None = None,
    sections: list[str] | None = None,
    enable_query_rewrite: bool = False,
) -> dict[str, Any]:
    client.require_configuration()
    rewritten = await rewrite_search_query(question) if enable_query_rewrite else {"query": question, "status": "disabled"}
    search_query = rewritten["query"]
    evidence = await hybrid_retrieve(search_query, document_ids, sections=sections)
    if not evidence or evidence[0]["score"] < 0.08:
        return {
            "status": "insufficient_evidence",
            "answer": "当前文献库没有检索到足够证据，无法确认。",
            "citations": [],
            "retrieval_mode": evidence[0].get("retrieval_mode", "keyword") if evidence else "keyword",
            "evidence_gate": {"passed": False, "threshold": 0.08, "top_score": evidence[0]["score"] if evidence else 0},
            "search_query": search_query,
            "query_rewrite": rewritten,
        }
    prompt = (
        "你是严谨的科研论文助手。只能依据下方证据回答问题，不得使用证据之外的知识补全。"
        "如果证据不足，必须明确说无法确认。答案中的事实后面使用 [E1] 这样的证据编号。\n\n"
        f"问题：{question}\n\n证据：\n{evidence_block(evidence)}"
    )
    answer = await client.complete(
        [
            {"role": "system", "content": "你重视可核查性，回答简洁，保留公式和关键数值。"},
            {"role": "user", "content": prompt},
        ]
    )
    citation_check = validate_answer_citations(answer, evidence)
    verification = await verify_answer(question, answer, evidence)
    result = {
        "status": "supported",
        "answer": answer,
        "citations": citation_payload(evidence),
        "retrieval_mode": evidence[0].get("retrieval_mode", "keyword"),
        "evidence_gate": {"passed": True, "threshold": 0.08, "top_score": evidence[0]["score"]},
        "citation_check": citation_check,
        "verification": verification,
        "search_query": search_query,
        "query_rewrite": rewritten,
    }
    if verification.get("supported") is not True or not citation_check["valid"]:
        if verification.get("status") == "unavailable":
            status = "verification_unavailable"
            message = "核验服务暂不可用，暂不展示未经核验的结论。"
        elif verification.get("supported") is not True:
            status = "verification_failed"
            message = "生成的回答未通过证据核验，暂不提供结论。请查看核验问题并回到原文确认。"
        else:
            status = "citation_validation_failed"
            message = "回答未能关联到有效证据，暂不提供结论。请缩小问题范围后重试。"
        result.update(status=status, answer=message)
    return result


async def compare_documents(
    question: str,
    document_ids: list[int],
    enable_query_rewrite: bool = False,
) -> dict[str, Any]:
    client.require_configuration()
    if len(document_ids) < 2:
        raise ValueError("跨文献对比至少需要选择两篇文献。")
    rewritten = await rewrite_search_query(question) if enable_query_rewrite else {"query": question, "status": "disabled"}
    document_ids = list(dict.fromkeys(document_ids))
    if len(document_ids) < 2:
        raise ValueError("跨文献对比至少需要选择两篇不同的文献。")
    evidence = await hybrid_retrieve(
        rewritten["query"], document_ids, limit=max(12, len(document_ids)), max_per_document=1
    )
    evidence_by_document: dict[int, float] = {}
    for item in evidence:
        document_id = int(item["document_id"])
        evidence_by_document[document_id] = max(evidence_by_document.get(document_id, 0), item["score"])
    covered_ids = sorted(document_id for document_id, score in evidence_by_document.items() if score >= 0.08)
    missing_ids = sorted(set(document_ids) - set(covered_ids))
    if not evidence or missing_ids:
        return {
            "status": "insufficient_evidence",
            "answer": "每篇指定文献都需要分别检索到达到门槛的原文证据；当前有文献证据缺失或过弱，无法进行可靠对比。",
            "citations": citation_payload(evidence),
            "retrieval_mode": evidence[0].get("retrieval_mode", "keyword") if evidence else "keyword",
            "evidence_gate": {"passed": False, "threshold": 0.08, "document_scores": evidence_by_document},
            "document_coverage": {
                "requested_document_ids": document_ids,
                "covered_document_ids": covered_ids,
                "missing_document_ids": missing_ids,
            },
            "search_query": rewritten["query"],
            "query_rewrite": rewritten,
        }
    documents = [get_document(document_id) for document_id in document_ids]
    document_labels = [
        f"ID={item['id']}，标题={item['title']}，版本={item['version']}"
        for item in documents if item
    ]
    prompt = (
        "你是科研论文对比助手。只根据证据回答，不要编造缺失信息。先按指定文献分别说明证据，"
        "再给出相同点、差异和证据不足处。每个关键事实必须引用直接支持它的 [E数字]；"
        "若证据没有回答某项，就明确写未在当前证据中找到，不要推断。\n\n"
        f"对比问题：{question}\n指定文献：\n" + "\n".join(document_labels)
        + f"\n\n证据：\n{evidence_block(evidence)}"
    )
    answer = await client.complete([{"role": "system", "content": "输出结构清晰的 Markdown。"}, {"role": "user", "content": prompt}])
    citation_check = validate_answer_citations(answer, evidence)
    verification = await verify_answer(question, answer, evidence)
    if not citation_check["valid"] or verification.get("supported") is not True:
        if verification.get("status") == "unavailable":
            status = "verification_unavailable"
            message = "对比结果的证据核验服务不可用，暂不展示未经核验的内容。"
        elif verification.get("supported") is not True:
            status = "verification_failed"
            message = "对比结果未通过证据核验，暂不展示结论。"
        else:
            status = "citation_validation_failed"
            message = "对比结果未能关联到有效证据，暂不展示。请调整问题后重试。"
        return {
            "status": status,
            "answer": message,
            "citations": citation_payload(evidence),
            "citation_check": citation_check,
            "verification": verification,
            "document_coverage": {
                "requested_document_ids": document_ids,
                "covered_document_ids": covered_ids,
                "missing_document_ids": missing_ids,
            },
            "evidence_gate": {"passed": True, "threshold": 0.08, "top_score": evidence[0]["score"]},
        }
    return {
        "status": "supported",
        "answer": answer,
        "citations": citation_payload(evidence),
        "retrieval_mode": evidence[0].get("retrieval_mode", "keyword"),
        "evidence_gate": {"passed": True, "threshold": 0.08, "top_score": evidence[0]["score"]},
        "citation_check": citation_check,
        "verification": verification,
        "document_coverage": {
            "requested_document_ids": document_ids,
            "covered_document_ids": covered_ids,
            "missing_document_ids": missing_ids,
        },
        "search_query": rewritten["query"],
        "query_rewrite": rewritten,
    }


async def generate_writing(idea: str, document_ids: list[int] | None = None) -> dict[str, Any]:
    client.require_configuration()
    evidence = await hybrid_retrieve(idea, document_ids, limit=10)
    prompt = (
        "根据研究想法和给定论文证据，生成学术论文框架，包含 Abstract、Introduction、Related Work、"
        "Method、Experiments、Limitations。只能把证据中的论文放入 References，并为引用标记 [E数字]。"
        "没有证据支持的内容要标为待验证。\n\n研究想法：" + idea + "\n\n证据：" + evidence_block(evidence)
    )
    answer = await client.complete([{"role": "system", "content": "你是学术写作助手，不捏造参考文献。"}, {"role": "user", "content": prompt}])
    citation_check = validate_answer_citations(answer, evidence)
    if not citation_check["valid"]:
        return {
            "status": "citation_validation_failed",
            "answer": "写作草稿未能关联到有效证据，暂不展示。请补充文献或调整研究想法后重试。",
            "citations": citation_payload(evidence),
            "citation_check": citation_check,
        }
    return {
        "status": "completed",
        "answer": answer,
        "citations": citation_payload(evidence),
        "citation_check": citation_check,
    }


def _protect_translation_regions(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}
    patterns = [
        r"```[\s\S]*?```",
        r"\$\$[\s\S]*?\$\$",
        r"\\\[[\s\S]*?\\\]",
        r"\\\([\s\S]*?\\\)",
        r"(?<!\$)\$[^$\n]+\$(?!\$)",
    ]
    combined = re.compile("|".join(f"({pattern})" for pattern in patterns))

    def replace(match: re.Match[str]) -> str:
        key = f"__ACADEMIC_REGION_{len(protected)}__"
        protected[key] = match.group(0)
        return key

    return combined.sub(replace, text), protected


def _restore_translation_regions(text: str, protected: dict[str, str]) -> str:
    for key, original in protected.items():
        text = text.replace(key, original)
    return text


async def translate_preserving_markup(text: str, target_language: str = "English") -> str:
    protected_text, protected = _protect_translation_regions(text)
    answer = await client.complete(
        [
            {
                "role": "system",
                "content": "你是学术翻译助手。只翻译自然语言；必须原样保留 __ACADEMIC_REGION_数字__ 占位符、引用编号和段落结构。",
            },
            {"role": "user", "content": f"将下面内容翻译为{target_language}，只输出译文：\n\n{protected_text}"},
        ],
        temperature=0.0,
    )
    return _restore_translation_regions(answer, protected)


async def translate_text(text: str, target_language: str = "English") -> dict[str, Any]:
    client.require_configuration()
    answer = await translate_preserving_markup(text, target_language)
    return {"status": "completed", "translation": answer}


async def translate_document_page(
    document_id: int,
    page_number: int,
    target_language: str = "English",
) -> dict[str, Any]:
    client.require_configuration()
    page = get_document_page(document_id, page_number)
    if not page:
        raise ValueError("文献页面不存在。")
    source_text = page["content"]
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    cached = get_translation_cache(document_id, page_number, source_hash, target_language, client.model)
    if cached is not None:
        return {
            "status": "completed",
            "cached": True,
            "document_id": document_id,
            "page_number": page_number,
            "source": source_text,
            "translation": cached,
            "target_language": target_language,
        }
    translation = await translate_preserving_markup(source_text, target_language)
    save_translation_cache(document_id, page_number, source_text, target_language, client.model, translation)
    return {
        "status": "completed",
        "cached": False,
        "document_id": document_id,
        "page_number": page_number,
        "source": source_text,
        "translation": translation,
        "target_language": target_language,
    }


async def translate_document(
    document_id: int,
    pages: list[int] | None = None,
    target_language: str = "English",
) -> dict[str, Any]:
    document = get_document(document_id)
    if not document:
        raise ValueError("文献不存在。")
    page_numbers = pages if pages is not None else list(range(1, document["page_count"] + 1))
    if not page_numbers:
        raise ValueError("至少选择一个页面进行翻译。")
    if len(page_numbers) > 100:
        raise ValueError("单次最多翻译 100 页，请分批提交。")
    if len(set(page_numbers)) != len(page_numbers):
        raise ValueError("页码列表包含重复页面，请检查后重试。")
    invalid_pages = [page for page in page_numbers if page < 1 or page > document["page_count"]]
    if invalid_pages:
        raise ValueError(
            f"页码超出范围：{', '.join(map(str, invalid_pages[:10]))}；"
            f"有效范围为 1–{document['page_count']}。"
        )
    translated_pages = [await translate_document_page(document_id, page, target_language) for page in page_numbers]
    markdown = "\n\n".join(
        f"## Page {item['page_number']}\n\n### Original\n\n{item['source']}\n\n### {target_language}\n\n{item['translation']}"
        for item in translated_pages
    )
    return {
        "status": "completed",
        "document_id": document_id,
        "target_language": target_language,
        "pages": translated_pages,
        "markdown": f"# {document['title']}\n\n{markdown}",
    }


def render_cached_translation_html(document_id: int, target_language: str = "English") -> str:
    document = get_document(document_id)
    if not document:
        raise ValueError("文献不存在。")
    sections = []
    for page_number in range(1, document["page_count"] + 1):
        page = get_document_page(document_id, page_number)
        if not page:
            continue
        source = page["content"]
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        translation = get_translation_cache(document_id, page_number, source_hash, target_language, client.model)
        if translation is None:
            raise ValueError(f"第 {page_number} 页尚未翻译，请先调用页级或全文翻译接口。")
        sections.append(
            f'<section class="page" id="page-{page_number}"><h2>Page {page_number}</h2>'
            f'<div class="columns"><pre>{html.escape(source)}</pre><pre>{html.escape(translation)}</pre></div></section>'
        )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>'
        + html.escape(document["title"])
        + '</title><style>body{font-family:Arial,sans-serif;max-width:1200px;margin:24px auto;'
        'color:#172033}.page{border-bottom:1px solid #cbd5e1;padding:16px 0}.columns{display:grid;'
        'grid-template-columns:1fr 1fr;gap:20px}.columns pre{white-space:pre-wrap;line-height:1.6;'
        'background:#f8fafc;padding:14px;border-radius:8px}@media(max-width:800px){.columns{grid-template-columns:1fr}}'
        '</style></head><body><h1>'
        + html.escape(document["title"])
        + f'</h1><p>Target language: {html.escape(target_language)}</p>'
        + "".join(sections)
        + "</body></html>"
    )


async def recommend(query: str) -> dict[str, Any]:
    client.require_configuration()
    documents = list_documents(page_size=100)["documents"]
    if not documents:
        return {"status": "empty", "recommendations": []}
    evidence = await hybrid_retrieve(query, [item["id"] for item in documents], limit=12)
    catalog = "\n".join(f"ID={item['id']} 标题={item['title']} 版本={item['version']} 文件={item['filename']}" for item in documents)
    prompt = f"根据用户需求，从候选论文中推荐相关文献，给出推荐理由。只使用候选列表，不要编造论文。\n需求：{query}\n候选：\n{catalog}"
    answer = await client.complete([{"role": "system", "content": "输出简洁的 Markdown 推荐结果。"}, {"role": "user", "content": prompt}])
    return {"status": "completed", "answer": answer, "citations": citation_payload(evidence)}


def csv_summary(filename: str, text: str) -> dict[str, Any]:
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("CSV 没有数据行或缺少表头。")
    columns = list(rows[0])
    numeric = {}
    for column in columns:
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError):
                pass
        if values:
            numeric[column] = {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}
    return {"filename": filename, "rows": len(rows), "columns": columns, "numeric_summary": numeric, "preview": rows[:10]}


async def visualize_csv(filename: str, text: str) -> dict[str, Any]:
    client.require_configuration()
    summary = csv_summary(filename, text)
    prompt = "为以下 CSV 摘要选择合适的学术图表，返回图表类型、x/y字段、绘图理由和 Figure Caption。不要编造数据。\n" + json.dumps(summary, ensure_ascii=False)
    answer = await client.complete([{"role": "system", "content": "你是学术数据可视化助手。"}, {"role": "user", "content": prompt}])
    return {"status": "completed", "summary": summary, "recommendation": answer}
