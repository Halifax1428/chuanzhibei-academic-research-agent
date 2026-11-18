from __future__ import annotations

import csv
import html
import io
import json
import math
import re
from collections import Counter
from typing import Any, Literal

from .agent_service import evidence_block, validate_answer_citations, verify_answer
from .citation_graph import build_citation_graph
from .document_service import citation_payload, get_document, hybrid_retrieve
from .model_client import OpenAICompatibleClient


client = OpenAICompatibleClient()
DiagramFormat = Literal["mermaid", "graphviz", "tikz", "matplotlib"]


def _mermaid_label(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("|", "&#124;")
        .replace("\n", "<br/>")
    )


def _graphviz_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _tikz_label(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "#": r"\#",
        "$": r"\$", "%": r"\%", "&": r"\&", "_": r"\_",
        "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(character, r"\\" if character == "\n" else character) for character in value)


def _safe_float(value: str) -> float | None:
    clean = value.strip().replace(",", "")
    if not clean:
        return None
    if clean.endswith("%"):
        try:
            result = float(clean[:-1]) / 100
            return result if math.isfinite(result) else None
        except ValueError:
            return None
    try:
        result = float(clean)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def _compact(value: str, limit: int = 32) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _parse_csv(filename: str, text: str) -> tuple[list[str], list[dict[str, str]]]:
    if len(text.encode("utf-8")) > 20 * 1024 * 1024:
        raise ValueError("CSV 文件超过 20 MB，已拒绝分析。")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not any(field.strip() for field in reader.fieldnames):
        raise ValueError("CSV 缺少有效表头。")
    columns = [field.strip() for field in reader.fieldnames]
    if any(not column for column in columns):
        raise ValueError("CSV 表头不能包含空列名。")
    if len(set(columns)) != len(columns):
        raise ValueError("CSV 表头存在重名列，请先重命名后再分析。")
    rows = []
    for raw in reader:
        if raw.get(None):
            raise ValueError("CSV 存在多余字段，请检查各行列数是否一致。")
        rows.append({column: (raw.get(column) or "").strip() for column in columns})
        if len(rows) > 10000:
            raise ValueError("单次最多分析 10000 行 CSV 数据。")
    if not rows:
        raise ValueError("CSV 没有数据行。")
    return columns, rows


def _profile_csv(filename: str, columns: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    profiles = []
    numeric_columns: list[str] = []
    for column in columns:
        values = [row[column] for row in rows]
        numeric_values = [number for number in (_safe_float(value) for value in values) if number is not None]
        nonempty = [value for value in values if value]
        is_numeric = bool(nonempty) and len(numeric_values) / len(nonempty) >= 0.8
        profile: dict[str, Any] = {
            "name": column,
            "kind": "numeric" if is_numeric else "categorical",
            "count": len(values),
            "missing": len(values) - len(nonempty),
            "unique": len(set(nonempty)),
        }
        if is_numeric:
            numeric_columns.append(column)
            profile["summary"] = {
                "min": min(numeric_values),
                "max": max(numeric_values),
                "mean": sum(numeric_values) / len(numeric_values),
                "count": len(numeric_values),
            }
        else:
            profile["top_values"] = Counter(nonempty).most_common(5)
        profiles.append(profile)
    if not numeric_columns:
        raise ValueError("CSV 至少需要一个可解析的数值列，才能生成学术图表。")
    x_column = columns[0]
    y_column = numeric_columns[0]
    if x_column == y_column and len(numeric_columns) > 1:
        y_column = numeric_columns[1]
    elif x_column == y_column:
        categorical_columns = [column for column in columns if column not in numeric_columns]
        x_column = categorical_columns[0] if categorical_columns else "样本序号"
    chart_type = "scatter" if len(numeric_columns) >= 2 and x_column in numeric_columns else "line"
    if x_column == "样本序号":
        chart_type = "line"
    elif x_column not in numeric_columns:
        chart_type = "bar" if len(set(row[x_column] for row in rows)) <= 24 else "line"
    return {
        "filename": filename,
        "rows": len(rows),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "profiles": profiles,
        "chart": {"type": chart_type, "x": x_column, "y": y_column},
        "preview": rows[:10],
    }


def _chart_points(
    summary: dict[str, Any], rows: list[dict[str, str]]
) -> tuple[list[str], list[float], list[float]]:
    chart = summary["chart"]
    points: list[tuple[str, float, float]] = []
    if chart["type"] == "bar" and chart["x"] != "样本序号":
        grouped: dict[str, list[float]] = {}
        for row in rows:
            value = _safe_float(row.get(chart["y"], ""))
            label = row.get(chart["x"], "")
            if value is not None and label:
                grouped.setdefault(label, []).append(value)
        points = [
            (_compact(label, 18), float(index), sum(values) / len(values))
            for index, (label, values) in enumerate(grouped.items())
        ]
    else:
        for row in rows:
            value = _safe_float(row.get(chart["y"], ""))
            if value is None:
                continue
            if chart["x"] == "样本序号":
                x_value = float(len(points) + 1)
                label = str(int(x_value))
            elif chart["type"] == "scatter":
                x_value = _safe_float(row.get(chart["x"], ""))
                if x_value is None:
                    continue
                label = f"{x_value:g}"
            else:
                x_value = float(len(points))
                label = row.get(chart["x"], "")
            points.append((_compact(label, 18), x_value, value))

    if len(points) > 60:
        indices = [round(index * (len(points) - 1) / 59) for index in range(60)]
        points = [points[index] for index in indices]
    labels = [item[0] for item in points]
    x_values = [item[1] for item in points]
    values = [item[2] for item in points]
    return labels, x_values, values


def _svg_chart(summary: dict[str, Any], rows: list[dict[str, str]]) -> str:
    labels, x_values, values = _chart_points(summary, rows)
    width, height = 820, 440
    left, top, right, bottom = 70, 38, 28, 78
    plot_width, plot_height = width - left - right, height - top - bottom
    low = min([0.0, *values]) if summary["chart"]["type"] == "bar" and values else min(values, default=0)
    high = max(values, default=1)
    if high == low:
        high = low + 1
    if summary["chart"]["type"] != "bar":
        padding = max((high - low) * 0.05, abs(high) * 0.005, 0.001)
        low -= padding
        high += padding
    y = lambda value: top + (high - value) / (high - low) * plot_height
    x_min = min(x_values, default=0)
    x_max = max(x_values, default=1)
    if x_max == x_min:
        x_max = x_min + 1
    x = lambda value: left + (value - x_min) * plot_width / (x_max - x_min)
    grid = []
    for index in range(5):
        value = low + (high - low) * index / 4
        y_pos = y(value)
        grid.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width-right}" y2="{y_pos:.1f}" stroke="#e2e8f0"/>'
            f'<text x="{left-10}" y="{y_pos+4:.1f}" text-anchor="end" fill="#64748b" font-size="11">{value:.3g}</text>'
        )
    marks = []
    for index, label in enumerate(labels):
        if len(labels) > 12 and index % max(1, len(labels) // 8):
            continue
        label_x = left + (index + 0.5) * plot_width / max(1, len(values)) if summary["chart"]["type"] == "bar" else x(x_values[index])
        marks.append(
            f'<text x="{label_x:.1f}" y="{height-bottom+22}" text-anchor="middle" fill="#64748b" font-size="10">{html.escape(label)}</text>'
        )
    if summary["chart"]["type"] == "bar":
        slot = plot_width / max(1, len(values))
        baseline = y(0)
        shapes = "".join(
            f'<rect x="{left + index*slot + slot*.15:.1f}" y="{min(y(value), baseline):.1f}" width="{slot*.7:.1f}" height="{max(1, abs(baseline-y(value))):.1f}" rx="3" fill="#315efb" opacity=".82"/>'
            for index, value in enumerate(values)
        )
    elif summary["chart"]["type"] == "scatter":
        shapes = "".join(
            f'<circle cx="{x(x_value):.1f}" cy="{y(value):.1f}" r="4" fill="#315efb" opacity=".78"><title>{html.escape(label)}: {value:g}</title></circle>'
            for label, x_value, value in zip(labels, x_values, values)
        )
    else:
        points = " ".join(f"{x(x_value):.1f},{y(value):.1f}" for x_value, value in zip(x_values, values))
        dots = "".join(f'<circle cx="{x(x_value):.1f}" cy="{y(value):.1f}" r="3.5" fill="#315efb"/>' for x_value, value in zip(x_values, values))
        shapes = f'<polyline points="{points}" fill="none" stroke="#315efb" stroke-width="3" stroke-linejoin="round"/>{dots}'
    title = f"{summary['chart']['y']} by {summary['chart']['x']}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<rect width="100%" height="100%" fill="#ffffff"/><text x="{left}" y="22" fill="#172033" font-size="15" font-weight="700">{html.escape(title)}</text>'
        f'<g>{"".join(grid)}</g><line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#94a3b8"/><line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#94a3b8"/>'
        f'<g>{shapes}{"".join(marks)}</g><text x="{width/2}" y="{height-12}" text-anchor="middle" fill="#475569" font-size="12">{html.escape(summary["chart"]["x"])}</text>'
        f'<text x="16" y="{height/2}" transform="rotate(-90 16 {height/2})" text-anchor="middle" fill="#475569" font-size="12">{html.escape(summary["chart"]["y"])}</text></svg>'
    )


def _matplotlib_script(summary: dict[str, Any]) -> str:
    x_column = repr(summary["chart"]["x"])
    y_column = repr(summary["chart"]["y"])
    chart_type = summary["chart"]["type"]
    return f'''import csv
import math
import matplotlib.pyplot as plt

INPUT = "data.csv"
X_COLUMN = {x_column}
Y_COLUMN = {y_column}

with open(INPUT, newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))

def number(value):
    value = (value or "").strip().replace(",", "")
    if not value:
        return None
    scale = 0.01 if value.endswith("%") else 1.0
    value = value[:-1] if value.endswith("%") else value
    try:
        result = float(value) * scale
        return result if math.isfinite(result) else None
    except ValueError:
        return None

valid = []
for index, row in enumerate(rows):
    y_value = number(row.get(Y_COLUMN))
    if y_value is None:
        continue
    if X_COLUMN == "样本序号":
        x_value, label = float(index + 1), str(index + 1)
    elif {chart_type!r} == "scatter":
        x_value = number(row.get(X_COLUMN))
        if x_value is None:
            continue
        label = str(x_value)
    else:
        x_value, label = float(len(valid)), row.get(X_COLUMN, "")
    valid.append((label, x_value, y_value))

figure, axis = plt.subplots(figsize=(8.2, 4.4), dpi=160)
if {chart_type!r} == "bar":
    grouped = {{}}
    for label, _, value in valid:
        grouped.setdefault(label, []).append(value)
    labels = list(grouped)
    values = [sum(items) / len(items) for items in grouped.values()]
    axis.bar(labels, values, color="#315efb", alpha=0.85)
    axis.tick_params(axis="x", labelrotation=35)
elif {chart_type!r} == "scatter":
    axis.scatter([item[1] for item in valid], [item[2] for item in valid], color="#315efb", s=28, alpha=0.8)
else:
    axis.plot(range(len(valid)), [item[2] for item in valid], color="#315efb", marker="o", linewidth=2)
axis.set_xlabel(X_COLUMN)
axis.set_ylabel(Y_COLUMN)
axis.grid(axis="y", alpha=0.25)
figure.tight_layout()
figure.savefig("figure.png", dpi=300)
figure.savefig("figure.svg")
'''


def _figure_caption(summary: dict[str, Any]) -> str:
    profile = next(item for item in summary["profiles"] if item["name"] == summary["chart"]["y"])
    stats = profile["summary"]
    return (
        f"图 1. {summary['chart']['y']} 随 {summary['chart']['x']} 的变化趋势。"
        f"共分析 {summary['rows']} 行数据，数值范围为 {stats['min']:.4g} 至 {stats['max']:.4g}，"
        f"均值为 {stats['mean']:.4g}。"
    )


async def analyze_csv(filename: str, text: str) -> dict[str, Any]:
    columns, rows = _parse_csv(filename, text)
    summary = _profile_csv(filename, columns, rows)
    result = {
        "status": "completed",
        "summary": summary,
        "figure_caption": _figure_caption(summary),
        "chart_svg": _svg_chart(summary, rows),
        "matplotlib_script": _matplotlib_script(summary),
        "ai_status": "not_requested",
    }
    if not client.configured:
        result["ai_status"] = "configuration_required"
        result["ai_message"] = "确定性 CSV 分析已完成；如需模型生成的图表解释，请先配置 OPENAI_API_KEY 和 OPENAI_MODEL。"
        return result
    prompt = (
        "根据 CSV 统计结果生成简短的学术图表解读，只使用给定字段和统计值。"
        "返回 JSON，字段为 recommendation 和 caption；不要编造趋势、实验结论或数据。\n"
        + json.dumps(summary, ensure_ascii=False)
    )
    try:
        raw = await client.complete(
            [{"role": "system", "content": "你是严谨的学术数据分析助手。"}, {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("recommendation"), str) or not isinstance(parsed.get("caption"), str):
            result["ai_status"] = "invalid_response"
            result["ai_message"] = "模型返回内容缺少有效的 recommendation 或 caption 字段。"
        else:
            result["ai_status"] = "completed"
            result["ai_recommendation"] = parsed["recommendation"]
            result["ai_caption"] = parsed["caption"]
    except json.JSONDecodeError:
        result["ai_status"] = "invalid_response"
        result["ai_message"] = "模型未返回合法 JSON；确定性图表和统计结果仍可使用。"
    except Exception as error:
        result["ai_status"] = "unavailable"
        result["ai_message"] = str(error)
    return result


def _document_references(document_ids: list[int] | None, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_ids = sorted({int(item["document_id"]) for item in evidence})
    allowed_ids = set(document_ids) if document_ids else set(evidence_ids)
    ids = [document_id for document_id in evidence_ids if document_id in allowed_ids]
    references = []
    seen = set()
    for document_id in ids:
        document = get_document(document_id)
        if not document or document_id in seen:
            continue
        seen.add(document_id)
        references.append(
            {
                "label": f"D{document_id}",
                "document_id": document_id,
                "title": document["title"],
                "authors": document.get("authors", []),
                "version": document.get("version", "unknown"),
                "filename": document["filename"],
            }
        )
    return references


async def generate_framework(idea: str, document_ids: list[int] | None = None) -> dict[str, Any]:
    if not idea.strip():
        raise ValueError("研究想法不能为空。")
    client.require_configuration()
    evidence = await hybrid_retrieve(idea, document_ids, limit=10)
    if not evidence or evidence[0]["score"] < 0.08:
        return {
            "status": "insufficient_evidence",
            "framework": "当前文献库没有检索到足够证据，不能生成带真实引用的论文框架。",
            "references": [],
            "citations": [],
            "evidence_gate": {"passed": False, "threshold": 0.08, "top_score": evidence[0]["score"] if evidence else 0},
        }
    references = _document_references(document_ids, evidence)
    prompt = (
        "根据研究想法和证据生成论文框架，必须包含 Abstract、Introduction、Related Work、Method、"
        "Experiments、Limitations。只能引用给定证据，事实后使用 [E数字]，不要编造作者、论文、数据集、"
        "指标或实验结果；没有证据支持的内容标记为‘待验证’。References 只允许使用下面的真实候选文献，"
        "使用 [D数字] 标记，不要添加其他条目。\n\n"
        f"研究想法：{idea}\n候选 References：{json.dumps(references, ensure_ascii=False)}\n"
        f"证据：\n{evidence_block(evidence)}"
    )
    answer = await client.complete(
        [{"role": "system", "content": "你是有引用纪律的学术写作助手。"}, {"role": "user", "content": prompt}],
        temperature=0.0,
    )
    evidence_check = validate_answer_citations(answer, evidence)
    referenced_labels = sorted(set(re.findall(r"\[D(\d+)\]", answer)))
    valid_labels = {item["label"] for item in references}
    invalid_labels = [f"D{value}" for value in referenced_labels if f"D{value}" not in valid_labels]
    reference_section = re.search(
        r"(?ims)^\s*(?:references|参考文献)\s*:?[ \t]*\n(.*)$", answer
    )
    listed_labels = set(re.findall(r"\[D(\d+)\]", reference_section.group(1))) if reference_section else set()
    missing_reference_labels = [label for label in referenced_labels if label not in listed_labels]
    references_check = {
        "valid": bool(referenced_labels) and not invalid_labels and not missing_reference_labels,
        "invalid": invalid_labels,
        "missing_from_references_section": [f"D{value}" for value in missing_reference_labels],
        "available": sorted(valid_labels),
    }
    verification = await verify_answer(idea, answer, evidence)
    if not evidence_check["valid"] or not references_check["valid"] or verification.get("supported") is not True:
        if verification.get("status") == "unavailable":
            status = "verification_unavailable"
            message = "框架的证据核验服务不可用，暂不展示未核验草稿。"
        elif verification.get("supported") is not True:
            status = "verification_failed"
            message = "框架未通过证据核验，暂不展示草稿。"
        else:
            status = "citation_validation_failed"
            message = "框架引用校验未通过，暂不展示草稿。请补充相关文献后重试。"
        return {
            "status": status,
            "framework": message,
            "references": references,
            "referenced_documents": [f"D{value}" for value in referenced_labels],
            "references_check": references_check,
            "citations": citation_payload(evidence),
            "citation_check": evidence_check,
            "verification": verification,
            "evidence_gate": {"passed": True, "threshold": 0.08, "top_score": evidence[0]["score"]},
        }
    body = re.split(r"(?im)^\s*(?:references|参考文献)\s*:?[ \t]*$", answer, maxsplit=1)[0].rstrip()
    canonical_references = [
        f"[{item['label']}] {', '.join(item['authors']) + '. ' if item['authors'] else ''}"
        f"{item['title']} ({item['version']}); 文件：{item['filename']}"
        for item in references if item["label"] in {f"D{value}" for value in referenced_labels}
    ]
    framework = body + "\n\nReferences\n" + "\n".join(canonical_references)
    return {
        "status": "completed",
        "framework": framework,
        "references": references,
        "referenced_documents": [f"D{value}" for value in referenced_labels],
        "references_check": references_check,
        "citations": citation_payload(evidence),
        "citation_check": evidence_check,
        "verification": verification,
        "evidence_gate": {"passed": True, "threshold": 0.08, "top_score": evidence[0]["score"]},
    }


def _graph_nodes(document_ids: list[int] | None) -> tuple[dict[str, Any], dict[str, str]]:
    graph = build_citation_graph(document_ids)
    if not graph["nodes"]:
        raise ValueError("当前没有可用于生成图脚本的文献。")
    labels = {node["id"]: _compact(node["label"], 80) for node in graph["nodes"]}
    return graph, labels


def _mermaid(graph: dict[str, Any], labels: dict[str, str]) -> str:
    lines = ["flowchart LR"]
    for node in graph["nodes"]:
        safe = re.sub(r"[^A-Za-z0-9_]", "_", node["id"])
        lines.append(f'    {safe}["{_mermaid_label(labels[node["id"]])}"]')
    for edge in graph["edges"]:
        lines.append(f'    {edge["source"]} -->|{edge["weight"]}| {edge["target"]}')
    return "\n".join(lines) + "\n"


def _graphviz(graph: dict[str, Any], labels: dict[str, str]) -> str:
    lines = ["digraph CitationNetwork {", "  rankdir=LR;", '  graph [fontname="Arial"];', '  node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontname="Arial"];']
    for node in graph["nodes"]:
        lines.append(f'  "{node["id"]}" [label="{_graphviz_label(labels[node["id"]])}"];')
    for edge in graph["edges"]:
        lines.append(f'  "{edge["source"]}" -> "{edge["target"]}" [label="{edge["weight"]}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _tikz(graph: dict[str, Any], labels: dict[str, str]) -> str:
    lines = ["% Compile with TikZ/PGF", "\\begin{tikzpicture}[node distance=20mm, >=stealth] "]
    for index, node in enumerate(graph["nodes"]):
        position = "" if index == 0 else f", right=of d{graph['nodes'][index-1]['id']}"
        text = _tikz_label(labels[node["id"]])
        lines.append(f"  \\node[draw, rounded corners, fill=blue!8, align=center{position}] (d{node['id']}) {{{text}}};")
    for edge in graph["edges"]:
        lines.append(f"  \\draw[->] (d{edge['source']}) -- node[above] {{{edge['weight']}}} (d{edge['target']});")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def _matplotlib_graph(graph: dict[str, Any], labels: dict[str, str]) -> str:
    nodes = [(node["id"], labels[node["id"]]) for node in graph["nodes"]]
    edges = [(edge["source"], edge["target"]) for edge in graph["edges"]]
    return """import math
import matplotlib.pyplot as plt

nodes = %r
edges = %r
figure, axis = plt.subplots(figsize=(10, 6), dpi=160)
radius = 1.0
positions = {}
for index, (node_id, label) in enumerate(nodes):
    angle = math.pi / 2 - 2 * math.pi * index / max(1, len(nodes))
    positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))
for source, target in edges:
    x1, y1 = positions[source]
    x2, y2 = positions[target]
    axis.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": "#94a3b8"})
for node_id, label in nodes:
    x, y = positions[node_id]
    axis.scatter([x], [y], s=1200, color="#dbe7ff", edgecolors="#315efb", zorder=3)
    axis.text(x, y, label[:20], ha="center", va="center", fontsize=8, wrap=True)
axis.axis("off")
figure.tight_layout()
figure.savefig("citation_graph.png", dpi=300)
""" % (nodes, edges)


def generate_diagram(document_ids: list[int] | None, format_name: DiagramFormat) -> dict[str, Any]:
    graph, labels = _graph_nodes(document_ids)
    scripts = {
        "mermaid": _mermaid(graph, labels),
        "graphviz": _graphviz(graph, labels),
        "tikz": _tikz(graph, labels),
        "matplotlib": _matplotlib_graph(graph, labels),
    }
    return {
        "status": "completed",
        "format": format_name,
        "script": scripts[format_name],
        "graph_stats": graph["stats"],
        "source_document_ids": [node["id"] for node in graph["nodes"]],
        "note": "脚本由本地文献图谱确定性生成，需按目标环境安装对应绘图库后执行。",
    }
