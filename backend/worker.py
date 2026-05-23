import asyncio
import json
import uuid
from datetime import datetime, timezone

from celery import Celery
from kombu import Queue
from redis import Redis
from sqlalchemy import select

from config import get_settings
from core.logging import get_logger, setup_logging

settings = get_settings()

setup_logging(
    json_output=True,
    log_level="INFO"
)

from agents import generation
from agents import knowledge
from agents import research
from agents import review
from agents import writing
from core.database import make_worker_session
from models.db import Project
logger = get_logger("worker")


def run_async(coro):
    import models.db as _db_module

    async def _run():
        session_factory, worker_engine = make_worker_session()
        _db_module.AsyncSessionLocal = session_factory
        research.AsyncSessionLocal = session_factory
        knowledge.AsyncSessionLocal = session_factory
        writing.AsyncSessionLocal = session_factory
        review.AsyncSessionLocal = session_factory
        generation.AsyncSessionLocal = session_factory
        
        import core.graph as _graph_module
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        _graph_module._driver = driver
        
        try:
            return await coro
        finally:
            await worker_engine.dispose()
            await driver.close()
            _graph_module._driver = None

    return asyncio.run(_run())


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


async def _update_project_status(
    project_id: str,
    status: str
) -> None:
    import models.db as db_module

    async with db_module.AsyncSessionLocal() as db:
        response = await db.execute(
            select(Project).where(Project.id == uuid.UUID(project_id))
        )
        project = response.scalar_one_or_none()

        if not project:
            logger.warning(
                "task.project_status.skipped",
                project_id=project_id,
                status=status
            )
            return

        project.status = status
        await db.commit()


def update_project_status(
    project_id: str,
    status: str
) -> None:
    run_async(
        _update_project_status(
            project_id,
            status
        )
    )


def mark_project_error_after_retries(
    task,
    project_id: str
) -> None:
    if task.request.retries >= settings.MAX_RETRIES:
        update_project_status(project_id, "error")


@celery_app.task(
    bind=True,
    name="tasks.discovery"
)
def run_discovery_task(
    self,
    project_id: str
):

    logger.info(
        "task.discovery.started",
        project_id=project_id
    )

    try:

        result = run_async(
            research.run_discovery(project_id)
        )

        update_project_status(project_id, "idle")

        logger.info(
            "task.discovery.completed",
            project_id=project_id
        )

        return result

    except Exception as exc:

        try:
            mark_project_error_after_retries(self, project_id)
        except Exception as db_error:
            logger.exception("task.discovery.failure_update.failed", error=str(db_error))

        logger.exception(
            "task.discovery.failed",
            project_id=project_id,
            error=str(exc)
        )

        raise self.retry(
            exc=exc,
            countdown=30
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

        result = run_async(
            research.run_document_analysis(project_id)
        )

        logger.info("task.analysis.success", project_id=project_id, output=result)
        notify_progress(project_id, "analysis", "done")

        gap_task = run_gap_task.delay(project_id)

        return {
            "status": "success",
            "project_id": project_id,
            "result": result,
            "next_stage": "gap_analysis",
            "next_task_id": gap_task.id
        }

    except Exception as error:
        logger.error("task.analysis.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "analysis", "failed")
        mark_project_error_after_retries(self, project_id)

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

        result = run_async(
            knowledge.run_gap_analysis(project_id)
        )

        logger.info("task.gap_analysis.success", project_id=project_id, gap_count=len(result.get("identified_gaps", [])))
        notify_progress(project_id, "gap_analysis", "done")
        update_project_status(project_id, "idle")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.gap_analysis.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "gap_analysis", "failed")
        mark_project_error_after_retries(self, project_id)

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

        result = run_async(
            writing.generate_draft(
                project_id=project_id,
                plan=plan
            )
        )

        logger.info("task.drafting.success", project_id=project_id, sections=list(result.keys()))
        notify_progress(project_id, "drafting", "done")
        update_project_status(project_id, "idle")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.drafting.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "drafting", "failed")
        mark_project_error_after_retries(self, project_id)

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

        result = run_async(
            writing.run_refinement(project_id)
        )

        logger.info("task.refinement.success", project_id=project_id, sections=list(result.keys()))
        notify_progress(project_id, "refinement", "done")
        update_project_status(project_id, "idle")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.refinement.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "refinement", "failed")
        mark_project_error_after_retries(self, project_id)

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

        result = run_async(
            writing.run_humanization(project_id)
        )

        logger.info("task.humanization.success", project_id=project_id, sections=list(result.keys()))
        notify_progress(project_id, "humanization", "done")
        update_project_status(project_id, "idle")

        return {
            "status": "success",
            "project_id": project_id,
            "result": result
        }

    except Exception as error:
        logger.error("task.humanization.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "humanization", "failed")
        mark_project_error_after_retries(self, project_id)

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

        reviewer_result = run_async(
            review.run_reviewer_simulation(project_id)
        )

        logger.info("task.review.reviewer_done", project_id=project_id, overall_score=reviewer_result.get("overall_score"))

        citation_result = run_async(
            review.run_citation_validation(project_id)
        )

        logger.info("task.review.citations_done", project_id=project_id, validated=len(citation_result.get("validated_citations", [])))
        notify_progress(project_id, "review", "done")
        update_project_status(project_id, "idle")

        return {
            "status": "success",
            "project_id": project_id,
            "reviewer_result": reviewer_result,
            "citation_result": citation_result
        }

    except Exception as error:
        logger.error("task.review.failed", project_id=project_id, error=str(error), exc_info=True)
        notify_progress(project_id, "review", "failed")
        mark_project_error_after_retries(self, project_id)

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

        result = run_async(
            generation.run_ieee_export(
                project_id=project_id,
                fmt=fmt
            )
        )

        logger.info("task.export.success", project_id=project_id, format=fmt, output_path=result)
        notify_progress(project_id, "export", "done")
        update_project_status(project_id, "complete")

        return {
            "status": "success",
            "project_id": project_id,
            "format": fmt,
            "result": result
        }

    except Exception as error:
        logger.error("task.export.failed", project_id=project_id, format=fmt, error=str(error), exc_info=True)
        notify_progress(project_id, "export", "failed")
        mark_project_error_after_retries(self, project_id)

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

        result = run_async(
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

        result = run_async(
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
