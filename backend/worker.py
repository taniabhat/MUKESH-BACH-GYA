import asyncio
import json
from datetime import datetime, timezone

from celery import Celery
from kombu import Queue
from redis import Redis

from agents import generation
from agents import knowledge
from agents import research
from agents import review
from agents import writing
from config import get_settings
from core.logging import get_logger
from core.logging import setup_logging


settings = get_settings()

setup_logging(
    json_output=True,
    log_level="INFO"
)

logger = get_logger("worker")


def make_celery(app_name: str) -> Celery:
    celery_app = Celery(
        app_name,
        broker=settings.REDIS_URL,
        backend=settings.REDIS_URL
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=60 * 60,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_default_queue="research_os",
        task_queues=(
            Queue("research_os"),
        )
    )

    return celery_app


celery_app = make_celery("research_os")


redis_client = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True
)


def notify_progress(
    project_id: str,
    stage: str,
    status: str,
    details: dict | None = None
) -> None:
    payload = {
        "project_id": project_id,
        "stage": stage,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if details:
        payload["details"] = details

    redis_client.publish(
        f"project:{project_id}:events",
        json.dumps(payload)
    )


@celery_app.task(
    bind=True,
    name="tasks.discovery"
)
def run_discovery_task(
    self,
    project_id: str
) -> dict:
    try:
        logger.info("task.discovery.started", project_id=project_id, task_id=self.request.id)
        notify_progress(project_id, "discovery", "started")

        result = asyncio.run(
            research.run_discovery(project_id)
        )

        logger.info("task.discovery.success", project_id=project_id, output=result)
        notify_progress(project_id, "discovery", "done")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.discovery.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "discovery", "failed")

        raise self.retry(
            exc=error,
            countdown=10,
            max_retries=settings.MAX_RETRIES
        )


@celery_app.task(
    bind=True,
    name="tasks.analysis"
)
def run_analysis_task(
    self,
    project_id: str
) -> dict:
    try:
        logger.info("task.analysis.started", project_id=project_id, task_id=self.request.id)
        notify_progress(project_id, "analysis", "started")

        result = asyncio.run(
            research.run_document_analysis(project_id)
        )

        logger.info("task.analysis.success", project_id=project_id, output=result)
        notify_progress(project_id, "analysis", "done")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.analysis.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "analysis", "failed")

        raise self.retry(
            exc=error,
            countdown=10,
            max_retries=settings.MAX_RETRIES
        )


@celery_app.task(
    bind=True,
    name="tasks.gap_analysis"
)
def run_gap_task(
    self,
    project_id: str
) -> dict:
    try:
        logger.info("task.gap_analysis.started", project_id=project_id, task_id=self.request.id)
        notify_progress(project_id, "gap_analysis", "started")

        result = asyncio.run(
            knowledge.run_gap_analysis(project_id)
        )

        logger.info("task.gap_analysis.success", project_id=project_id, gap_count=len(result.get("identified_gaps", [])))
        notify_progress(project_id, "gap_analysis", "done")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.gap_analysis.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "gap_analysis", "failed")

        raise self.retry(
            exc=error,
            countdown=10,
            max_retries=settings.MAX_RETRIES
        )


@celery_app.task(
    bind=True,
    name="tasks.draft_generation"
)
def run_draft_task(
    self,
    project_id: str,
    plan: dict
) -> dict:
    try:
        logger.info("task.drafting.started", project_id=project_id, task_id=self.request.id)
        notify_progress(project_id, "drafting", "started")

        result = asyncio.run(
            writing.generate_draft(
                project_id=project_id,
                plan=plan
            )
        )

        logger.info("task.drafting.success", project_id=project_id, sections=list(result.keys()))
        notify_progress(project_id, "drafting", "done")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.drafting.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "drafting", "failed")

        raise self.retry(
            exc=error,
            countdown=10,
            max_retries=settings.MAX_RETRIES
        )


@celery_app.task(
    bind=True,
    name="tasks.refinement"
)
def run_refinement_task(
    self,
    project_id: str
) -> dict:
    try:
        logger.info("task.refinement.started", project_id=project_id, task_id=self.request.id)
        notify_progress(project_id, "refinement", "started")

        result = asyncio.run(
            writing.run_refinement(project_id)
        )

        logger.info("task.refinement.success", project_id=project_id, sections=list(result.keys()))
        notify_progress(project_id, "refinement", "done")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.refinement.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "refinement", "failed")

        raise self.retry(
            exc=error,
            countdown=10,
            max_retries=settings.MAX_RETRIES
        )


@celery_app.task(
    bind=True,
    name="tasks.humanization"
)
def run_humanization_task(
    self,
    project_id: str
) -> dict:
    try:
        logger.info("task.humanization.started", project_id=project_id, task_id=self.request.id)
        notify_progress(project_id, "humanization", "started")

        result = asyncio.run(
            writing.run_humanization(project_id)
        )

        logger.info("task.humanization.success", project_id=project_id, sections=list(result.keys()))
        notify_progress(project_id, "humanization", "done")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.humanization.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "humanization", "failed")

        raise self.retry(
            exc=error,
            countdown=10,
            max_retries=settings.MAX_RETRIES
        )


@celery_app.task(
    bind=True,
    name="tasks.review"
)
def run_review_task(
    self,
    project_id: str
) -> dict:
    try:
        logger.info("task.review.started", project_id=project_id, task_id=self.request.id)
        notify_progress(project_id, "review", "started")

        reviewer_result = asyncio.run(
            review.run_reviewer_simulation(project_id)
        )

        logger.info("task.review.reviewer_done", project_id=project_id, overall_score=reviewer_result.get("overall_score"))

        citation_result = asyncio.run(
            review.run_citation_validation(project_id)
        )

        logger.info("task.review.citations_done", project_id=project_id, validated=len(citation_result.get("validated_citations", [])))
        notify_progress(project_id, "review", "done")

        return {
            "status": "success",
            "project_id": project_id,
            "reviewer_result": reviewer_result,
            "citation_result": citation_result
        }

    except Exception as error:
        logger.error("task.review.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "review", "failed")

        raise self.retry(
            exc=error,
            countdown=10,
            max_retries=settings.MAX_RETRIES
        )


@celery_app.task(
    bind=True,
    name="tasks.export"
)
def run_export_task(
    self,
    project_id: str,
    fmt: str
) -> dict:
    try:
        logger.info("task.export.started", project_id=project_id, format=fmt, task_id=self.request.id)
        notify_progress(project_id, "export", "started")

        result = asyncio.run(
            generation.run_ieee_export(
                project_id=project_id,
                fmt=fmt
            )
        )

        logger.info("task.export.success", project_id=project_id, format=fmt, output_path=result)
        notify_progress(project_id, "export", "done")

        return {
            "status": "success",
            "project_id": project_id,
            "format": fmt,
            "result": result
        }

    except Exception as error:
        logger.error("task.export.failed", project_id=project_id, format=fmt, error=str(error), exc_info=True)
        notify_progress(project_id, "export", "failed")

        raise self.retry(
            exc=error,
            countdown=10,
            max_retries=settings.MAX_RETRIES
        )


@celery_app.task(
    bind=True,
    name="tasks.code_generation"
)
def run_code_gen_task(
    self,
    project_id: str
) -> dict:
    try:
        logger.info("task.code_generation.started", project_id=project_id, task_id=self.request.id)
        notify_progress(project_id, "code_generation", "started")

        result = asyncio.run(
            generation.run_code_generation(project_id)
        )

        logger.info("task.code_generation.success", project_id=project_id, files=result.get("generated_files", []))
        notify_progress(project_id, "code_generation", "done")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.code_generation.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "code_generation", "failed")

        raise self.retry(
            exc=error,
            countdown=10,
            max_retries=settings.MAX_RETRIES
        )


@celery_app.task(
    bind=True,
    name="tasks.diagram_generation"
)
def run_diagram_task(
    self,
    project_id: str
) -> dict:
    try:
        logger.info("task.diagram_generation.started", project_id=project_id, task_id=self.request.id)
        notify_progress(project_id, "diagram_generation", "started")

        result = asyncio.run(
            generation.run_diagram_generation(project_id)
        )

        logger.info("task.diagram_generation.success", project_id=project_id, diagrams=result.get("diagrams", []))
        notify_progress(project_id, "diagram_generation", "done")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.diagram_generation.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "diagram_generation", "failed")

        raise self.retry(
            exc=error,
            countdown=10,
            max_retries=settings.MAX_RETRIES
        )