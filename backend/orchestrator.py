import time
import requests
import sys

BASE_URL = "http://127.0.0.1:8000/api/v1"

def wait_for_status(project_id, target_statuses, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{BASE_URL}/projects/{project_id}")
        data = resp.json()
        status = data.get("status")
        print(f"Current status: {status}")
        if status in target_statuses:
            return status
        if status == "error" or status == "failed":
            print("Pipeline encountered an error!")
            sys.exit(1)
        time.sleep(5)
    print("Timeout waiting for status!")
    sys.exit(1)

def main():
    print("Creating Project...")
    resp = requests.post(f"{BASE_URL}/projects", json={
        "title": "Agentic AI Orchestration Demo",
        "research_idea": "Evaluating large language models in multi-agent orchestration tasks for scientific literature."
    })
    
    if resp.status_code != 200:
        print(f"Error creating project: {resp.text}")
        sys.exit(1)
        
    project_id = resp.json()["id"]
    print(f"Project ID: {project_id}")
    
    print("Triggering Discovery...")
    requests.post(f"{BASE_URL}/projects/{project_id}/discover")
    wait_for_status(project_id, ["discovered", "idle"]) # sometimes goes back to idle or something? Let's assume discovered.
    
    # Check if actually discovered
    if requests.get(f"{BASE_URL}/projects/{project_id}").json()["status"] != "discovered":
        print("Waiting until status is officially discovered...")
        wait_for_status(project_id, ["discovered"])
        
    print("Triggering Analysis...")
    requests.post(f"{BASE_URL}/projects/{project_id}/analyze")
    wait_for_status(project_id, ["analyzed"])
    
    print("Triggering Approval (Auto-approve)...")
    requests.post(f"{BASE_URL}/projects/{project_id}/approve", json={
        "user_edits": {},
        "approved_at": "2026-05-21T00:00:00Z"
    })
    wait_for_status(project_id, ["approved"])
    
    print("Triggering Draft...")
    requests.post(f"{BASE_URL}/projects/{project_id}/draft", json={"plan": {}})
    wait_for_status(project_id, ["drafted"])
    
    print("Triggering Refine...")
    requests.post(f"{BASE_URL}/projects/{project_id}/refine")
    wait_for_status(project_id, ["refined"])
    
    print("Triggering Humanize...")
    requests.post(f"{BASE_URL}/projects/{project_id}/humanize")
    wait_for_status(project_id, ["humanized"])
    
    print("Triggering Review...")
    requests.post(f"{BASE_URL}/projects/{project_id}/review")
    wait_for_status(project_id, ["reviewed"])
    
    print("Triggering Export...")
    resp = requests.post(f"{BASE_URL}/projects/{project_id}/export?fmt=pdf")
    print(f"Export triggered! Task ID: {resp.text}")
    print(f"You can find the exported PDF in the backend/data/exports/{project_id}/ directory!")
    
if __name__ == "__main__":
    main()
