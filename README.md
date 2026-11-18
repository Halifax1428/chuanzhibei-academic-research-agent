# Academic Research Agent

A local research assistant for academic papers. The repository contains the Agent application, its tests, and the runtime dependencies.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item langchain\agent\.env.example langchain\agent\.env
# Edit langchain\agent\.env and set OPENAI_API_KEY and OPENAI_MODEL.
Set-Location langchain
python -m uvicorn agent.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. OCR setup on Windows is documented in `langchain/agent/setup_ocr.ps1`.

See [`langchain/README.md`](langchain/README.md) for features, configuration, and testing details.

## Technical documentation and source bundle

- [Technical documentation PDF](docs/科研论文智能体技术文档.pdf)
- [Markdown source](docs/技术文档.md)
- [作品介绍文档 PDF](docs/科研论文智能体作品介绍.pdf) · [可编辑 Word 版](docs/科研论文智能体作品介绍.docx)
- [1080p 项目演示视频](docs/科研论文智能体演示视频.mp4) · [中文字幕文件](docs/科研论文智能体演示字幕.srt)
- [Complete source, documents and demo bundle](docs/科研论文智能体_源码文档和演示素材.zip). The technical-document-only bundle remains available as `docs/科研论文智能体_源码与技术文档.zip`.

Runtime uploads, local databases, virtual environments, and API credentials are excluded from Git.
