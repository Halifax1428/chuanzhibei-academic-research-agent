from __future__ import annotations

from typing import Any

import httpx

from .config import get_api_key, get_base_url, get_embedding_model, get_model


class ModelConfigurationError(RuntimeError):
    pass


class ModelRequestError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Small OpenAI-compatible chat client, independent of any vendor SDK."""

    def __init__(self) -> None:
        self.api_key = get_api_key()
        self.base_url = get_base_url()
        self.model = get_model()
        self.embedding_model = get_embedding_model()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    @property
    def embedding_configured(self) -> bool:
        return bool(self.api_key and self.embedding_model)

    def require_configuration(self) -> None:
        missing = []
        if not self.api_key:
            missing.append("OPENAI_API_KEY")
        if not self.model:
            missing.append("OPENAI_MODEL")
        if missing:
            raise ModelConfigurationError(
                "请先配置模型 API：" + ", ".join(missing) + "。可复制 .env.example 为 .env 后填写。"
            )

    def require_embedding_configuration(self) -> None:
        missing = []
        if not self.api_key:
            missing.append("OPENAI_API_KEY")
        if not self.embedding_model:
            missing.append("OPENAI_EMBEDDING_MODEL")
        if missing:
            raise ModelConfigurationError(
                "请先配置 Embedding API：" + ", ".join(missing) + "。可复制 .env.example 为 .env 后填写。"
            )

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
    ) -> str:
        self.require_configuration()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:500]
            raise ModelRequestError(f"模型 API 返回错误 {error.response.status_code}: {detail}") from error
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelRequestError(f"模型 API 调用失败：{error}") from error

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.require_embedding_configuration()
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.embedding_model, "input": texts},
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                ordered = sorted(data, key=lambda item: item.get("index", 0))
                vectors = [item["embedding"] for item in ordered]
                if len(vectors) != len(texts):
                    raise ValueError("Embedding API 返回数量与输入文本数量不一致")
                return vectors
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:500]
            raise ModelRequestError(f"Embedding API 返回错误 {error.response.status_code}: {detail}") from error
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelRequestError(f"Embedding API 调用失败：{error}") from error
