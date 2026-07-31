"""Thin wrapper around Google's google-genai SDK for calling Gemini/Gemma models."""

import os

from dotenv import load_dotenv
from google import genai


class GeminiClient:
    def __init__(self, model: str = "gemma-4-31b-it", api_key: str | None = None):
        load_dotenv()

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Set GEMINI_API_KEY in your .env file or pass api_key explicitly."
            )

        self.model = model
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.models.generate_content(
            model=self.model, contents=prompt, **kwargs
        )
        return response.text

    def embed_text(
        self,
        text: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
        model: str = "gemini-embedding-001",
    ) -> list[float]:
        response = self._client.models.embed_content(
            model=model, contents=text, config={"task_type": task_type}
        )
        return response.embeddings[0].values


if __name__ == "__main__":
    client = GeminiClient()
    print(client.generate("Say hello in one short sentence."))
