import asyncio

from huggingface_hub import (
    InferenceClient,
)

from research_discovery.config.settings import (
    settings,
)


class HuggingFaceLLMProvider:

    def __init__(self):

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=(
                settings.llm.api_key
            ),
        )

        self.model = (
            settings.llm.model
        )

    async def generate(
        self,
        messages: list[dict],
    ) -> str:

        completion = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            temperature=(
                settings.llm.temperature
            ),
            max_tokens=(
                settings.llm.max_tokens
            ),
        )

        return (
            completion
            .choices[0]
            .message
            .content
        )