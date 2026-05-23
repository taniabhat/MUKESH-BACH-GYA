import pytest
from fastapi.testclient import TestClient

def test_create_project(client: TestClient):
    response = client.post(
        "/api/v1/projects",
        json={
            "title": "Test Project",
            "research_idea": "A comprehensive analysis of transformer models in CV."
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Project"
    assert "id" in data
    assert data["status"] == "idle"

def test_list_projects(client: TestClient):
    # First create a project
    client.post(
        "/api/v1/projects",
        json={
            "title": "Test Project 2",
            "research_idea": "Another idea to test the listing endpoint."
        }
    )
    
    response = client.get("/api/v1/projects")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["title"] == "Test Project 2"

def test_get_project_not_found(client: TestClient):
    import uuid
    random_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/projects/{random_id}")
    assert response.status_code == 404
