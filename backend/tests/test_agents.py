import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents.research import run_document_analysis
from agents.knowledge import synthesize_gaps
from core.document import DocumentResult
from core.llm import get_model
from core import rag

@pytest.mark.asyncio
async def test_document_analysis_empty(db_session):
    # Test document analysis with no papers in DB
    import uuid
    project_id_obj = uuid.uuid4()
    
    with patch("agents.research.AsyncSessionLocal", return_value=db_session):
        result = await run_document_analysis(project_id_obj)
        
    assert result == {"documents": []}

@pytest.mark.asyncio
async def test_synthesize_gaps_logic():
    # Test gap synthesis basic merging
    clusters = [{"themes": ["theme1"]}]
    benchmark = [{"dataset": "ds1", "metric": "m1"}]
    contradictions = [{"paper_a": "A", "paper_b": "B"}]
    graph_gaps = [{"limitation_description": "lim1"}]
    
    with patch("agents.knowledge.structured_chat", new_callable=AsyncMock) as mock_chat:
        from pydantic import BaseModel
        class MockResult(BaseModel):
            gaps: list[dict]
            
        mock_chat.return_value = MockResult(gaps=[{"title": "Test Gap", "severity": "high", "novelty_opportunity": "yes", "suggested_contributions": ["c1"]}])
        
        result = await synthesize_gaps(clusters, benchmark, contradictions, graph_gaps)
        
        assert len(result) == 1
        assert result[0]["title"] == "Test Gap"


def test_fast_model_uses_120b():
    assert get_model("fast") == get_model("research")


@pytest.mark.asyncio
async def test_hybrid_search_uses_query_points():
    point = SimpleNamespace(
        id="chunk-1",
        score=0.9,
        payload={
            "content": "retrieved content",
            "metadata": {"paper_id": "paper-1"},
            "section": "limitations",
            "chunk_type": "text"
        }
    )

    with (
        patch("core.rag.embed_single", new_callable=AsyncMock) as mock_embed,
        patch("core.rag.qdrant.query_points") as mock_query_points
    ):
        mock_embed.return_value = [0.1, 0.2]
        mock_query_points.return_value = SimpleNamespace(points=[point])

        results = await rag.hybrid_search(
            query="limitations",
            collection="text_chunks",
            top_k=3
        )

    mock_query_points.assert_called_once()
    assert mock_query_points.call_args.kwargs["query"] == [0.1, 0.2]
    assert results[0]["chunk_id"] == "chunk-1"
