from __future__ import annotations

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag.config import settings
from rag.models import PromptPackage


class GeminiGenerator:
    def __init__(self):
        if not settings.use_gemini:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set it in .env before generating answers."
            )
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, prompt_package: PromptPackage) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=settings.generation_model,
            contents=prompt_package.user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=prompt_package.system_prompt,
            ),
        )
        return response.text or ""
