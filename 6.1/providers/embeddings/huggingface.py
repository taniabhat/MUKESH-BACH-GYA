import asyncio

from huggingface_hub import (
    InferenceClient,
)

from research_discovery.config.settings import (
    settings,
)


class HuggingFaceEmbeddingProvider:

    def __init__(self):

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=(
                settings.embedding.api_key
            ),
        )

        self.model = (
            settings.embedding.model
        )

    async def embed(
        self,
        text: str,
    ) -> list[float]:

        result = await asyncio.to_thread(
            self.client.feature_extraction,
            text=text,
            model=self.model,
        )

        return result