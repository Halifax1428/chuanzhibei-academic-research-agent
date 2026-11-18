# Academic Research Agent

科研论文阅读与分析系统，前端和 FastAPI 服务位于 `agent/`。项目使用真实配置的模型 API；未配置时，模型相关操作会提示添加 API。

## 功能

- PDF、Markdown、TXT 导入，逐页文本抽取、OCR、结构化信息和版本重复提示
- 基于原文证据的问答、跨论文对比、引用校验和证据不足拒答
- 论文阅读、选段即时翻译、PDF 全文翻译及缓存导出
- arXiv 检索、下载入库、主题追踪与推荐反馈
- 局部引用图谱、Future Work 线索、论文框架和图示脚本
- CSV 数据摘要、SVG 图表、Figure Caption 与受限 Python 分析

## 配置与启动

从项目根目录创建虚拟环境并安装 `requirements.txt`，复制 `agent/.env.example` 为 `agent/.env`，填写模型服务的 `OPENAI_API_KEY`、`OPENAI_MODEL`；兼容服务可设置 `OPENAI_BASE_URL`，向量检索可设置 `OPENAI_EMBEDDING_MODEL`。

```powershell
Set-Location langchain
python -m uvicorn agent.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000/>。Windows OCR 可运行：

```powershell
powershell -ExecutionPolicy Bypass -File agent\setup_ocr.ps1
```

没有 API 配置时，系统不会使用假模型；模型功能会返回明确配置提示。arXiv 搜索需要联网。

## 验证

从项目根目录运行：

```powershell
Set-Location langchain
..\.venv\Scripts\python.exe -m unittest agent.tests.test_document_translation_contract agent.tests.test_document_identity agent.tests.test_answer_verification agent.tests.test_academic_search agent.tests.test_citation_graph_links agent.tests.test_citation_graph_review agent.tests.test_copilot_quality agent.tests.test_multi_paper_reasoning agent.tests.test_python_sandbox -v
```

解析、OCR 和引用图谱以启发式抽取为主；复杂公式语义、图像内容理解和全领域引用网络仍受输入 PDF 与数据范围限制。
