import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from uuid import uuid4
import json
from datetime import datetime, timezone

from models.db import ReviewReport, PaperDraft, Paper, Citation, GeneratedAsset
from sqlalchemy.ext.asyncio import AsyncSession


def test_project_lifecycle(client: TestClient):
    # 1. Create Project
    response = client.post(
        "/api/v1/projects",
        json={
            "title": "Integration Test Project",
            "research_idea": "Testing the full pipeline end-to-end"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Integration Test Project"
    assert data["status"] == "idle"
    project_id = data["id"]

    # 2. List Projects
    response = client.get("/api/v1/projects")
    assert response.status_code == 200
    projects = response.json()
    assert any(p["id"] == project_id for p in projects)

    # 3. Get Project
    response = client.get(f"/api/v1/projects/{project_id}")
    assert response.status_code == 200
    assert response.json()["id"] == project_id

    # 4. Delete Project
    response = client.delete(f"/api/v1/projects/{project_id}")
    assert response.status_code == 200

    # 5. Get Project (should be 404 after soft delete)
    response = client.get(f"/api/v1/projects/{project_id}")
    assert response.status_code == 404


@patch("api.routes.run_discovery_task.delay")
@patch("api.routes.run_analysis_task.delay")
@patch("api.routes.run_draft_task.delay")
@patch("api.routes.run_refinement_task.delay")
@patch("api.routes.run_humanization_task.delay")
@patch("api.routes.run_review_task.delay")
@patch("api.routes.run_code_gen_task.delay")
@patch("api.routes.run_diagram_task.delay")
@patch("api.routes.run_export_task.delay")
def test_pipeline_triggers(
    mock_export, mock_diagram, mock_code, mock_review, mock_humanize,
    mock_refine, mock_draft, mock_analyze, mock_discover,
    client: TestClient, db_session: AsyncSession
):
    # Setup mocks
    for m in [mock_export, mock_diagram, mock_code, mock_review, mock_humanize, mock_refine, mock_draft, mock_analyze, mock_discover]:
        m.return_value = MagicMock(id="test-task-id")

    # Create project
    response = client.post(
        "/api/v1/projects",
        json={"title": "Pipeline Test", "research_idea": "Testing pipeline triggers"}
    )
    project_id = response.json()["id"]

    # Test Discover
    response = client.post(f"/api/v1/projects/{project_id}/discover")
    assert response.status_code == 200
    assert response.json()["task_id"] == "test-task-id"
    mock_discover.assert_called_once_with(project_id)
    
    assert client.get(f"/api/v1/projects/{project_id}").json()["status"] == "discovering"

    # Test Analyze (Requires 'discovering' status, which we just set)
    response = client.post(f"/api/v1/projects/{project_id}/analyze")
    assert response.status_code == 200
    mock_analyze.assert_called_once_with(project_id)

    assert client.get(f"/api/v1/projects/{project_id}").json()["status"] == "analyzing"

    # To test approve, we need a ReviewReport in the DB
    # We will manually insert one using the test client's event loop
    import asyncio
    
    async def insert_report():
        import uuid
        report = ReviewReport(
            project_id=uuid.UUID(project_id),
            version=1,
            content={"gaps": [{"id": "g1", "title": "Gap 1", "severity": "high", "novelty_opportunity": "High", "suggested_contributions": []}]},
            user_edits=None
        )
        db_session.add(report)
        await db_session.commit()
    
    asyncio.get_event_loop().run_until_complete(insert_report())

    # Test Approve
    response = client.post(
        f"/api/v1/projects/{project_id}/approve",
        json={"user_edits": {"gap1": "approved"}, "approved_at": datetime.now(timezone.utc).isoformat()}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "approved"

    # Test Draft
    response = client.post(f"/api/v1/projects/{project_id}/draft", json={"plan": {"section1": "intro"}})
    assert response.status_code == 200
    mock_draft.assert_called_once_with(project_id, {"section1": "intro"})
    assert client.get(f"/api/v1/projects/{project_id}").json()["status"] == "drafting"

    # Test Refine
    response = client.post(f"/api/v1/projects/{project_id}/refine")
    assert response.status_code == 200
    mock_refine.assert_called_once_with(project_id)
    assert client.get(f"/api/v1/projects/{project_id}").json()["status"] == "refining"

    # Test Humanize
    response = client.post(f"/api/v1/projects/{project_id}/humanize")
    assert response.status_code == 200
    mock_humanize.assert_called_once_with(project_id)
    assert client.get(f"/api/v1/projects/{project_id}").json()["status"] == "humanizing"

    # Test Review
    response = client.post(f"/api/v1/projects/{project_id}/review")
    assert response.status_code == 200
    mock_review.assert_called_once_with(project_id)
    assert client.get(f"/api/v1/projects/{project_id}").json()["status"] == "reviewing"
    
    # Test Generators
    response = client.post(f"/api/v1/projects/{project_id}/generate-code")
    assert response.status_code == 200
    mock_code.assert_called_once_with(project_id)
    
    response = client.post(f"/api/v1/projects/{project_id}/generate-diagrams")
    assert response.status_code == 200
    mock_diagram.assert_called_once_with(project_id)
    
    # Test Export
    response = client.post(f"/api/v1/projects/{project_id}/export?fmt=pdf")
    assert response.status_code == 200
    mock_export.assert_called_once_with(project_id, "pdf")


def test_data_retrieval(client: TestClient, db_session: AsyncSession):
    # Create project
    response = client.post(
        "/api/v1/projects",
        json={"title": "Data Test", "research_idea": "Testing data retrieval"}
    )
    project_id = response.json()["id"]

    # Seed Data
    import asyncio
    async def seed_data():
        import uuid
        project_uuid = uuid.UUID(project_id)
        # Add Paper
        paper = Paper(project_id=project_uuid, title="Test Paper", relevance_score=0.95)
        db_session.add(paper)
        
        # Add ReviewReport
        report = ReviewReport(
            project_id=project_uuid,
            version=1,
            content={
                "gaps": [{"id": "g1", "title": "Gap 1", "severity": "high", "novelty_opportunity": "High", "suggested_contributions": []}],
                "reviewer_scores": {"clarity": 0.9}
            }
        )
        db_session.add(report)
        
        # Add Draft
        draft = PaperDraft(
            project_id=project_uuid,
            version=1,
            sections={"intro": "Hello world"}
        )
        db_session.add(draft)
        
        # Add Citation
        citation = Citation(
            project_id=project_uuid,
            bibtex="@article{test, title={Test}}",
            validated=True,
            validation_status="verified"
        )
        db_session.add(citation)
        
        # Add Asset
        asset = GeneratedAsset(
            project_id=project_uuid,
            asset_type="diagram",
            content="mermaid diagram"
        )
        db_session.add(asset)
        
        await db_session.commit()
    
    asyncio.get_event_loop().run_until_complete(seed_data())

    # Get Papers
    response = client.get(f"/api/v1/projects/{project_id}/papers")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["title"] == "Test Paper"

    # Get Report
    response = client.get(f"/api/v1/projects/{project_id}/report")
    assert response.status_code == 200
    assert response.json()["version"] == 1

    # Get Gaps
    response = client.get(f"/api/v1/projects/{project_id}/gaps")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["id"] == "g1"

    # Get Draft
    response = client.get(f"/api/v1/projects/{project_id}/draft")
    assert response.status_code == 200
    assert response.json()["sections"]["intro"] == "Hello world"

    # Get Citations
    response = client.get(f"/api/v1/projects/{project_id}/citations")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["validated"] is True

    # Get Review Report
    response = client.get(f"/api/v1/projects/{project_id}/review-report")
    assert response.status_code == 200
    assert response.json()["content"]["reviewer_scores"] == {"clarity": 0.9}

    # Get Assets
    response = client.get(f"/api/v1/projects/{project_id}/assets")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["asset_type"] == "diagram"


def test_edge_cases(client: TestClient):
    # Invalid project ID
    invalid_id = str(uuid4())
    assert client.get(f"/api/v1/projects/{invalid_id}").status_code == 404
    
    # Validation errors (title too short)
    response = client.post(
        "/api/v1/projects",
        json={"title": "A", "research_idea": "Too short title"}
    )
    assert response.status_code == 422
    
    # Invalid export format
    # Requires a valid project to hit the regex validation on the parameter
    response = client.post(
        "/api/v1/projects",
        json={"title": "Format Test", "research_idea": "Testing invalid formats"}
    )
    project_id = response.json()["id"]
    
    response = client.post(f"/api/v1/projects/{project_id}/export?fmt=invalid")
    assert response.status_code == 422
    
    # Status guard on analyze endpoint
    response = client.post(f"/api/v1/projects/{project_id}/analyze")
    assert response.status_code == 400
    assert "Discovery must complete first" in response.json()["detail"]
