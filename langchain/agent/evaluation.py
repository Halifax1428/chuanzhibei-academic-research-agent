from __future__ import annotations

from typing import Any

from .document_service import hybrid_retrieve


async def evaluate_retrieval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for index, case in enumerate(cases, start=1):
        question = str(case.get("question", "")).strip()
        expected_terms = [str(term).lower() for term in case.get("expected_terms", []) if str(term).strip()]
        document_ids = case.get("document_ids")
        if not question:
            results.append({"id": index, "status": "invalid", "error": "question 不能为空"})
            continue
        evidence = await hybrid_retrieve(question, document_ids, limit=int(case.get("limit", 6)))
        evidence_text = "\n".join(item["content"].lower() for item in evidence)
        term_hits = {term: term in evidence_text for term in expected_terms}
        expected_documents = set(document_ids or case.get("expected_document_ids", []))
        retrieved_documents = {item["document_id"] for item in evidence}
        document_hit = not expected_documents or bool(expected_documents & retrieved_documents)
        term_hit = not expected_terms or all(term_hits.values())
        passed = bool(evidence) and term_hit and document_hit
        results.append(
            {
                "id": index,
                "question": question,
                "passed": passed,
                "top_score": evidence[0]["score"] if evidence else 0.0,
                "retrieval_mode": evidence[0].get("retrieval_mode", "keyword") if evidence else "keyword",
                "term_hits": term_hits,
                "retrieved_documents": sorted(retrieved_documents),
                "evidence": evidence,
            }
        )
    valid = [item for item in results if item.get("status") != "invalid"]
    passed = sum(bool(item.get("passed")) for item in valid)
    return {
        "total": len(results),
        "valid": len(valid),
        "passed": passed,
        "hit_rate": round(passed / len(valid), 4) if valid else 0.0,
        "results": results,
    }
