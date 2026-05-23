import logging
import sys
import time

import requests

# -------------------------------------------------------------------
# Setup Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000/api/v1"


def wait_for_status(project_id, target_statuses, timeout=3600):
    start = time.time()
    logger.info(f"Waiting for project {project_id} to reach status in {target_statuses}")

    while time.time() - start < timeout:
        response = requests.get(f"{BASE_URL}/projects/{project_id}")

        if response.status_code != 200:
            logger.error(f"Failed status check: {response.text}")
            sys.exit(1)

        data = response.json()
        status = data.get("status")

        logger.info(f"Current status: {status}")

        if status in target_statuses:
            return status

        if status == "error":
            logger.error("Pipeline failed!")
            sys.exit(1)

        time.sleep(5)

    logger.error("Timeout waiting for pipeline.")
    sys.exit(1)


def post_or_fail(endpoint, body=None, step_name="Request"):
    if step_name:
        logger.info(f"=== {step_name} ===")

    response = requests.post(endpoint, json=body)

    if response.status_code not in [200, 201]:
        logger.error(f"Failed ({response.status_code}): {response.text}")
        sys.exit(1)

    logger.debug(f"Success ({response.status_code}): {response.text}")
    return response


def main():
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
        logger.info(f"Resuming existing project: {project_id}")
    else:
        response = post_or_fail(
            f"{BASE_URL}/projects",
            {
                "title": "Agentic AI Orchestration Demo",
                "research_idea": (
                    "Evaluating large language models "
                    "in multi-agent orchestration tasks "
                    "for scientific literature."
                )
            },
            step_name="Creating Project"
        )
        project = response.json()
        project_id = project["id"]
        logger.info(f"Project ID: {project_id}")

        post_or_fail(f"{BASE_URL}/projects/{project_id}/discover", step_name="Trigger Discovery")
        wait_for_status(project_id, ["idle"])

    logger.info("=== Fetch Papers ===")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/papers")
    if response.status_code == 200:
        logger.info("Papers fetched successfully.")
    else:
        logger.warning(f"Could not fetch papers ({response.status_code}), continuing...")

    post_or_fail(f"{BASE_URL}/projects/{project_id}/analyze", step_name="Trigger Analysis")
    wait_for_status(project_id, ["idle"])

    logger.info("=== Fetch Gaps ===")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/gaps")
    if response.status_code == 200:
        logger.info("Gaps fetched successfully.")

    logger.info("=== Fetch Report ===")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/report")
    if response.status_code == 200:
        logger.info("Report fetched successfully.")

    post_or_fail(
        f"{BASE_URL}/projects/{project_id}/approve",
        {
            "user_edits": {},
            "approved_at": "2026-05-21T00:00:00Z"
        },
        step_name="Approve Report"
    )

    post_or_fail(
        f"{BASE_URL}/projects/{project_id}/draft",
        {
            "plan": {
                "target_venue": "NeurIPS",
                "sections": [
                    "abstract",
                    "introduction",
                    "methodology",
                    "results",
                    "conclusion"
                ]
            }
        },
        step_name="Trigger Draft"
    )
    wait_for_status(project_id, ["idle"])

    logger.info("=== Fetch Draft ===")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/draft")
    if response.status_code == 200:
        logger.info("Draft fetched successfully.")

    post_or_fail(f"{BASE_URL}/projects/{project_id}/refine", step_name="Trigger Refinement")
    wait_for_status(project_id, ["idle"])

    post_or_fail(f"{BASE_URL}/projects/{project_id}/humanize", step_name="Trigger Humanization")
    wait_for_status(project_id, ["idle"])

    post_or_fail(f"{BASE_URL}/projects/{project_id}/review", step_name="Trigger Review")
    wait_for_status(project_id, ["complete", "idle"])

    logger.info("=== Fetch Review Report ===")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/review-report")
    if response.status_code == 200:
        logger.info("Review Report fetched successfully.")

    post_or_fail(f"{BASE_URL}/projects/{project_id}/generate-code", step_name="Generate Code")
    post_or_fail(f"{BASE_URL}/projects/{project_id}/generate-diagrams", step_name="Generate Diagrams")

    logger.info("=== Fetch Assets ===")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/assets")
    if response.status_code == 200:
        logger.info("Assets fetched successfully.")

    post_or_fail(f"{BASE_URL}/projects/{project_id}/export?fmt=pdf", step_name="Export PDF")

    wait_for_status(project_id, ["complete"])

    logger.info("=== Download PDF ===")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/export/pdf")
    if response.status_code == 200:
        with open("final_paper.pdf", "wb") as file:
            file.write(response.content)
        logger.info("Saved final_paper.pdf")
    else:
        logger.error(f"Failed to download PDF. Status: {response.status_code}")

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
