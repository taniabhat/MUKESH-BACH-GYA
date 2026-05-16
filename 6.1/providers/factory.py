from research_discovery.config.settings import settings
from research_discovery.providers.embeddings.huggingface import HuggingFaceEmbeddingProvider
from research_discovery.providers.llm.huggingface import HuggingFaceLLMProvider


class ProviderFactory:

    @staticmethod
    def create_llm_provider():

        provider = (
            settings.llm.provider
        )

        if provider == "huggingface":

            return (
                HuggingFaceLLMProvider()
            )

        raise ValueError(
            f"Unsupported provider: "
            f"{provider}"
        )

    @staticmethod
    def create_embedding_provider():

        provider = (
            settings.embedding.provider
        )

        if provider == "huggingface":

            return (
                HuggingFaceEmbeddingProvider()
            )

        raise ValueError(
            f"Unsupported provider: "
            f"{provider}"
        )