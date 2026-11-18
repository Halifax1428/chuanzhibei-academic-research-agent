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

Runtime uploads, local databases, virtual environments, and API credentials are excluded from Git.
