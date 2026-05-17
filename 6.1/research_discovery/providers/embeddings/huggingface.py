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
        texts: list[str],
    ) -> list[list[float]]:

        result = await asyncio.to_thread(
            self.client.feature_extraction,
            text=texts,
            model=self.model,
        )
        
        # HuggingFace feature extraction can sometimes return 3D arrays 
        # (batch_size, sequence_length, hidden_size) for embeddings if not pooling.
        # Assuming we are using models that return 1D or 2D (batch_size, hidden_size) 
        # or we might need to extract the [CLS] token. Usually, for embedding models 
        # via feature_extraction on HF serverless, it returns 2D per text if pooled, 
        # or a list of floats. We will coerce it:
        
        if not isinstance(result, list):
            import numpy as np
            result = np.array(result).tolist()
            
        if len(result) > 0 and isinstance(result[0], list) and isinstance(result[0][0], list):
            # It returned (batch, seq, dim), extract CLS token (0) or mean pool
            return [x[0] for x in result]
            
        if len(texts) == 1 and len(result) > 0 and not isinstance(result[0], list):
            return [result]

        return result