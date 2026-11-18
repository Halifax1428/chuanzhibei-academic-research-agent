from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from . import db
from .document_service import extract_arxiv_identity, list_documents
from .model_client import OpenAICompatibleClient


client = OpenAICompatibleClient()
_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9+.#-]{2,}|[\u4e00-\u9fff]")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "using", "based", "into",
    "via", "are", "is", "of", "to", "in", "on", "a", "an", "等", "的", "与", "和",
}
_FUTURE_SECTION = re.compile(
    r"future|limitation|discussion|conclusion|outlook|open problem|未来|局限|讨论|结论|展望|不足",
    flags=re.I,
)


def _tokens(text: str) -> set[str]:
    return {token for token in _WORD_PATTERN.findall(text.lower()) if token not in _STOPWORDS}


def _compact(text: str, limit: int = 900) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[:limit].rstrip() + "…"


def _selected_documents(document_ids: list[int] | None) -> list[dict[str, Any]]:
    documents = [
        item for item in list_documents(page_size=100)["documents"]
        if item.get("status") == "indexed"
    ]
    if document_ids:
        selected = set(document_ids)
        documents = [item for item in documents if item["id"] in selected]
        missing = selected - {item["id"] for item in documents}
        if missing:
            raise ValueError(f"文献不存在：{', '.join(str(item) for item in sorted(missing))}")
    return documents


def _reference_rows(document_id: int) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT ref_index, raw_text FROM paper_references WHERE document_id = ? ORDER BY ref_index",
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _document_text(document_id: int) -> str:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT content FROM chunks WHERE document_id = ? ORDER BY page, id",
            (document_id,),
        ).fetchall()
    return "\n".join(row["content"] for row in rows)


def _reference_score(reference: str, title: str, arxiv_id: str | None = None) -> float:
    reference_text = reference.lower()
    if arxiv_id:
        canonical_id = extract_arxiv_identity(arxiv_id)["arxiv_id"]
        reference_id = extract_arxiv_identity(reference)["arxiv_id"]
        if canonical_id and canonical_id == reference_id:
            return 1.0
    title_text = title.lower().strip()
    if len(title_text) >= 12 and title_text in reference_text:
        return 1.0
    title_tokens = _tokens(title)
    if len(title_tokens) < 2:
        return 0.0
    matched = sum(token in reference_text for token in title_tokens)
    ratio = matched / len(title_tokens)
    if matched >= 2 and ratio >= 0.55:
        return ratio
    return 0.0


def _fallback_title_mentions(source_id: int, target: dict[str, Any]) -> list[dict[str, Any]]:
    text = _document_text(source_id).lower()
    title_tokens = _tokens(target["title"])
    if len(title_tokens) < 2:
        return []
    matched = sum(token in text for token in title_tokens)
    if matched < max(2, math.ceil(len(title_tokens) * 0.75)):
        return []
    return [{"ref_index": None, "raw_text": f"正文标题线索：{target['title']}", "score": matched / len(title_tokens)}]


def _pagerank(nodes: list[str], edges: list[dict[str, Any]]) -> dict[str, float]:
    if not nodes:
        return {}
    damping = 0.85
    scores = {node: 1 / len(nodes) for node in nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        outgoing[edge["source"]].append(edge["target"])
    for _ in range(30):
        next_scores = {node: (1 - damping) / len(nodes) for node in nodes}
        dangling = sum(scores[node] for node in nodes if not outgoing[node])
        for node in nodes:
            next_scores[node] += damping * dangling / len(nodes)
        for source, targets in outgoing.items():
            if not targets:
                continue
            share = damping * scores[source] / len(targets)
            for target in targets:
                next_scores[target] += share
        scores = next_scores
    return scores


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 1.0 if high else 0.0 for key in values}
    return {key: round((value - low) / (high - low), 4) for key, value in values.items()}


def _future_work_candidates(
    documents: list[dict[str, Any]], edges: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    document_by_id = {item["id"]: item for item in documents}
    incoming: dict[int, list[int]] = defaultdict(list)
    for edge in edges or []:
        incoming[int(edge["target"])].append(int(edge["source"]))
    chunks_by_document: dict[int, list[Any]] = defaultdict(list)
    if documents:
        placeholders = ",".join("?" for _ in documents)
        with db.connection() as conn:
            all_chunks = conn.execute(
                f"""SELECT document_id, page, section, content FROM chunks
                    WHERE document_id IN ({placeholders}) ORDER BY document_id, page, id""",
                [item["id"] for item in documents],
            ).fetchall()
        for chunk in all_chunks:
            chunks_by_document[int(chunk["document_id"])].append(chunk)
    for document in documents:
        rows = chunks_by_document.get(document["id"], [])
        matched_rows = [row for row in rows if _FUTURE_SECTION.search(f"{row['section']} {row['content']}")]
        for row in matched_rows[:4]:
            idea_tokens = _tokens(row["content"]) - {
                "future", "work", "paper", "study", "research", "we", "our", "will", "could",
                "further", "more", "also", "need", "needs", "remain", "explore", "explored",
            }
            follow_ups = []
            for citing_id in incoming.get(document["id"], []):
                follow_up_document = document_by_id.get(citing_id)
                if not follow_up_document or not idea_tokens:
                    continue
                for chunk in chunks_by_document.get(citing_id, []):
                    chunk_tokens = _tokens(chunk["content"])
                    overlap = idea_tokens & chunk_tokens
                    ratio = len(overlap) / len(idea_tokens)
                    if len(overlap) >= 3 and ratio >= 0.18:
                        follow_ups.append(
                            {
                                "document_id": citing_id,
                                "title": follow_up_document["title"],
                                "page": chunk["page"],
                                "matched_terms": sorted(overlap)[:12],
                                "match_score": round(ratio, 4),
                                "text": _compact(chunk["content"], 420),
                            }
                        )
            follow_ups.sort(key=lambda item: item["match_score"], reverse=True)
            candidates.append(
                {
                    "document_id": document["id"],
                    "title": document["title"],
                    "page": row["page"],
                    "section": row["section"],
                    "text": _compact(row["content"]),
                    "status": "找到潜在关联论文，需核验是否解决该方向" if follow_ups else "当前资料库未找到足够关联证据",
                    "follow_up_papers": follow_ups[:3],
                }
            )
    return candidates[:40]


def build_citation_graph(document_ids: list[int] | None = None) -> dict[str, Any]:
    documents = _selected_documents(document_ids)
    nodes = [{"id": str(item["id"]), "label": item["title"], "version": item["version"]} for item in documents]
    document_by_id = {item["id"]: item for item in documents}
    edges_by_pair: dict[tuple[int, int], dict[str, Any]] = {}

    for source in documents:
        references = _reference_rows(source["id"])
        for target in documents:
            if source["id"] == target["id"]:
                continue
            matches = []
            target_identity = (target.get("metadata") or {}).get("arxiv_id")
            for reference in references:
                score = _reference_score(reference["raw_text"], target["title"], target_identity)
                if score:
                    reference_identity = extract_arxiv_identity(reference["raw_text"])["arxiv_id"]
                    matches.append({
                        **reference,
                        "score": round(score, 4),
                        "match_type": "arxiv_id" if target_identity and reference_identity == target_identity else "title",
                    })
            if not matches and not references:
                matches = _fallback_title_mentions(source["id"], target)
            if matches:
                edges_by_pair[(source["id"], target["id"])] = {
                    "source": str(source["id"]),
                    "target": str(target["id"]),
                    "weight": len(matches),
                    "evidence": matches[:3],
                }

    edges = list(edges_by_pair.values())
    node_ids = [node["id"] for node in nodes]
    indegree = {node_id: 0 for node_id in node_ids}
    outdegree = {node_id: 0 for node_id in node_ids}
    weighted_indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        indegree[edge["target"]] += 1
        outdegree[edge["source"]] += 1
        weighted_indegree[edge["target"]] += edge["weight"]

    rank = _pagerank(node_ids, edges)
    rank_norm = _normalize(rank)
    citation_norm = _normalize({key: float(value) for key, value in weighted_indegree.items()})
    for node in nodes:
        node_id = node["id"]
        foundation_score = round(0.6 * rank_norm.get(node_id, 0.0) + 0.4 * citation_norm.get(node_id, 0.0), 4)
        node.update(
            {
                "in_degree": indegree[node_id],
                "out_degree": outdegree[node_id],
                "weighted_in_degree": weighted_indegree[node_id],
                "pagerank": round(rank.get(node_id, 0.0), 6),
                "foundation_score": foundation_score,
            }
        )

    ranked_foundation = sorted(nodes, key=lambda item: (item["foundation_score"], item["in_degree"]), reverse=True)
    foundation_ids = {item["id"] for item in ranked_foundation[: max(1, math.ceil(len(nodes) * 0.2))] if item["in_degree"] > 0}
    peripheral_ids = {
        item["id"] for item in nodes if item["out_degree"] > 0 and item["in_degree"] == 0
    }
    for node in nodes:
        if node["id"] in foundation_ids:
            node["role"] = "基石论文"
        elif node["id"] in peripheral_ids:
            node["role"] = "边缘衍生论文"
        else:
            node["role"] = "普通节点"

    future_work = _future_work_candidates(documents, edges)
    top_foundation = [item["id"] for item in ranked_foundation if item["id"] in foundation_ids]
    top_peripheral = [item["id"] for item in nodes if item["id"] in peripheral_ids]
    return {
        "nodes": nodes,
        "edges": edges,
        "future_work_candidates": future_work,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "isolated_node_count": sum(1 for node in nodes if not node["in_degree"] and not node["out_degree"]),
            "density": round(len(edges) / max(1, len(nodes) * max(1, len(nodes) - 1)), 4),
            "foundation_nodes": top_foundation,
            "peripheral_nodes": top_peripheral,
        },
        "algorithm": "优先精确匹配 arXiv 主编号，其次匹配参考文献标题；PageRank 与入度联合排序。Future Work 关联只依据库内引用边和文本词项重合，不能单独证明某方向已解决。",
        "documents": [
            {"id": item["id"], "title": item["title"], "created_at": item.get("created_at")}
            for item in document_by_id.values()
        ],
    }


def _review_evidence(graph: dict[str, Any]) -> str:
    sections = []
    for index, item in enumerate(graph["future_work_candidates"], start=1):
        follow_ups = "\n".join(
            f"潜在关联论文（仅词项与引用关系线索，不能视为已解决）：{paper['title']}，"
            f"第 {paper['page']} 页，匹配词={', '.join(paper['matched_terms'])}\n{paper['text']}"
            for paper in item.get("follow_up_papers", [])
        ) or "当前资料库没有找到足够的关联论文证据。"
        sections.append(
            f"[F{index}] 来源：{item['title']}，第 {item['page']} 页，章节={item['section']}\n"
            f"未来方向原文：{item['text']}\n{follow_ups}"
        )
    return "\n\n".join(sections) or "没有提取到明确的 Future Work/Limitations 片段。"


async def generate_graph_review(document_ids: list[int] | None = None) -> dict[str, Any]:
    graph = build_citation_graph(document_ids)
    candidates = graph["future_work_candidates"]
    if not candidates:
        return {
            "status": "insufficient_evidence",
            "review": "所选文献中没有提取到明确的 Future Work 或局限性原文片段，无法生成有证据支撑的综述。",
            "graph": graph,
            "evidence_count": 0,
            "citation_check": {"valid": False, "referenced": [], "invalid": [], "available": []},
            "evidence_rule": "综述中的 [F数字] 只能引用返回的 Future Work 原文片段。",
        }
    client.require_configuration()
    prompt = (
        "你是严谨的学术综述助手。只允许使用提供的图谱统计、论文标题和原文证据。"
        "生成一份简短中文综述，包含：1.领域演进脉络；2.基石论文与边缘衍生论文；"
        "3.论文之间的引用关系解释；4.Future Work 候选。"
        "不得补写证据中不存在的作者、年份、方法、指标或引用。"
        "每个具体判断后附 [F数字]；逐条标记‘可能已被后续工作覆盖’、‘仍待探索’或‘证据不足’。"
        "仅在后续论文原文明示完成了该方向时才可判定可能已覆盖；词项重合或引用关系本身不足以证明解决。\n\n"
        f"图谱统计：{graph['stats']}\n节点：{graph['nodes']}\n边：{graph['edges']}\n"
        f"Future Work 原文证据：\n{_review_evidence(graph)}"
    )
    review = await client.complete(
        [
            {"role": "system", "content": "你是有引用纪律的科研综述助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    referenced = sorted({int(value) for value in re.findall(r"\[F(\d+)\]", review)})
    available = list(range(1, len(candidates) + 1))
    invalid = [value for value in referenced if value not in available]
    citation_check = {
        "valid": bool(referenced) and not invalid,
        "referenced": [f"F{value}" for value in referenced],
        "invalid": [f"F{value}" for value in invalid],
        "available": [f"F{value}" for value in available],
    }
    if not citation_check["valid"]:
        return {
            "status": "citation_validation_failed",
            "review": "综述未能关联到有效的 Future Work 原文证据，暂不展示。",
            "graph": graph,
            "evidence_count": len(candidates),
            "citation_check": citation_check,
            "evidence_rule": "综述中的 [F数字] 只能引用返回的 Future Work 原文片段。",
        }
    return {
        "status": "completed",
        "review": review,
        "graph": graph,
        "evidence_count": len(candidates),
        "citation_check": citation_check,
        "evidence_rule": "综述中的 [F数字] 只能引用返回的 Future Work 原文片段；无 API 时不会生成综述。",
    }
