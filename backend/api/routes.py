from pathlib import Path
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import desc
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_db
from models.db import Citation
from models.db import GeneratedAsset
from models.db import Paper
from models.db import PaperDraft
from models.db import Project
from models.db import ReviewReport
from models.schemas import ApprovalBody
from models.schemas import AssetResponse
from models.schemas import CitationResponse
from models.schemas import DraftPlanBody
from models.schemas import DraftResponse
from models.schemas import GapResponse
from models.schemas import PaginatedPaperResponse
from models.schemas import PaperResponse
from models.schemas import ProjectCreate
from models.schemas import ProjectResponse
from models.schemas import ReportResponse
from models.schemas import ReviewReportResponse
from models.schemas import TaskResponse
from worker import run_analysis_task
from worker import run_code_gen_task
from worker import run_diagram_task
from worker import run_discovery_task
from worker import run_draft_task
from worker import run_export_task
from worker import run_humanization_task
from worker import run_refinement_task
from worker import run_review_task


router = APIRouter(tags=["ResearchOS"])


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


async def get_project_or_404(
    db: AsyncSession,
    project_id: UUID
) -> Project:
    query = select(Project).where(
        Project.id == project_id,
        Project.deleted_at.is_(None)
    )

    result = await db.execute(query)

    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Project not found"
        )

    return project


# -------------------------------------------------------------------
# Project Management
# -------------------------------------------------------------------


@router.post(
    "/projects",
    response_model=ProjectResponse
)
async def create_project(
    body: ProjectCreate,
    db: AsyncSession = Depends(get_db)
) -> Project:
    project = Project(
        title=body.title,
        research_idea=body.research_idea,
        status="idle"
    )

    db.add(project)

    await db.commit()

    await db.refresh(project)

    return project


@router.get(
    "/projects",
    response_model=list[ProjectResponse]
)
async def list_projects(
    db: AsyncSession = Depends(get_db)
) -> list[Project]:
    query = (
        select(Project)
        .where(Project.deleted_at.is_(None))
        .order_by(desc(Project.created_at))
    )

    result = await db.execute(query)

    return list(result.scalars().all())


@router.get(
    "/projects/{project_id}",
    response_model=ProjectResponse
)
async def get_project(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Project:
    return await get_project_or_404(db, project_id)


@router.delete(
    "/projects/{project_id}"
)
async def delete_project(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> dict:
    project = await get_project_or_404(db, project_id)

    project.soft_delete()

    await db.commit()

    return {
        "message": "Project deleted"
    }


# -------------------------------------------------------------------
# Pipeline Triggers
# -------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/discover",
    response_model=TaskResponse
)
async def trigger_discovery(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> TaskResponse:
    project = await get_project_or_404(db, project_id)

    project.status = "discovering"

    await db.commit()

    task = run_discovery_task.delay(str(project_id))

    return TaskResponse(
        task_id=task.id,
        status="queued"
    )


@router.post(
    "/projects/{project_id}/analyze",
    response_model=TaskResponse
)
async def trigger_analysis(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> TaskResponse:
    project = await get_project_or_404(db, project_id)

    discovery_ready = project.status == "discovering"

    if project.status == "idle":
        paper_result = await db.execute(
            select(Paper.id)
            .where(Paper.project_id == project_id)
            .limit(1)
        )
        discovery_ready = paper_result.scalar_one_or_none() is not None

    if not discovery_ready:
        raise HTTPException(
            status_code=400,
            detail="Discovery must complete first"
        )

    project.status = "analyzing"

    await db.commit()

    task = run_analysis_task.delay(str(project_id))

    return TaskResponse(
        task_id=task.id,
        status="queued"
    )


@router.post(
    "/projects/{project_id}/approve",
    response_model=ProjectResponse
)
async def approve_report(
    project_id: UUID,
    body: ApprovalBody,
    db: AsyncSession = Depends(get_db)
) -> Project:
    project = await get_project_or_404(db, project_id)

    query = (
        select(ReviewReport)
        .where(ReviewReport.project_id == project_id)
        .order_by(desc(ReviewReport.created_at))
        .limit(1)
    )

    result = await db.execute(query)

    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(
            status_code=404,
            detail="Review report not found"
        )

    report.user_edits = body.user_edits
    report.approved_at = body.approved_at.replace(tzinfo=None) if body.approved_at else None

    project.status = "approved"

    await db.commit()

    await db.refresh(project)

    return project


@router.post(
    "/projects/{project_id}/draft",
    response_model=TaskResponse
)
async def trigger_draft(
    project_id: UUID,
    body: DraftPlanBody,
    db: AsyncSession = Depends(get_db)
) -> TaskResponse:
    project = await get_project_or_404(db, project_id)

    project.status = "drafting"

    await db.commit()

    task = run_draft_task.delay(
        str(project_id),
        body.plan
    )

    return TaskResponse(
        task_id=task.id,
        status="queued"
    )


@router.post(
    "/projects/{project_id}/refine",
    response_model=TaskResponse
)
async def trigger_refinement(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> TaskResponse:
    project = await get_project_or_404(db, project_id)

    project.status = "refining"

    await db.commit()

    task = run_refinement_task.delay(str(project_id))

    return TaskResponse(
        task_id=task.id,
        status="queued"
    )


@router.post(
    "/projects/{project_id}/humanize",
    response_model=TaskResponse
)
async def trigger_humanization(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> TaskResponse:
    project = await get_project_or_404(db, project_id)

    project.status = "humanizing"

    await db.commit()

    task = run_humanization_task.delay(str(project_id))

    return TaskResponse(
        task_id=task.id,
        status="queued"
    )


@router.post(
    "/projects/{project_id}/review",
    response_model=TaskResponse
)
async def trigger_review(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> TaskResponse:
    project = await get_project_or_404(db, project_id)

    project.status = "reviewing"

    await db.commit()

    task = run_review_task.delay(str(project_id))

    return TaskResponse(
        task_id=task.id,
        status="queued"
    )


@router.post(
    "/projects/{project_id}/generate-code",
    response_model=TaskResponse
)
async def trigger_code_gen(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> TaskResponse:
    await get_project_or_404(db, project_id)

    task = run_code_gen_task.delay(str(project_id))

    return TaskResponse(
        task_id=task.id,
        status="queued"
    )


@router.post(
    "/projects/{project_id}/generate-diagrams",
    response_model=TaskResponse
)
async def trigger_diagrams(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> TaskResponse:
    await get_project_or_404(db, project_id)

    task = run_diagram_task.delay(str(project_id))

    return TaskResponse(
        task_id=task.id,
        status="queued"
    )


@router.post(
    "/projects/{project_id}/export",
    response_model=TaskResponse
)
async def trigger_export(
    project_id: UUID,
    fmt: str = Query(..., pattern="^(pdf|tex|docx)$"),
    db: AsyncSession = Depends(get_db)
) -> TaskResponse:
    await get_project_or_404(db, project_id)

    task = run_export_task.delay(
        str(project_id),
        fmt
    )

    return TaskResponse(
        task_id=task.id,
        status="queued"
    )


# -------------------------------------------------------------------
# Streaming
# -------------------------------------------------------------------

from pydantic import BaseModel
class ChatRequest(BaseModel):
    prompt: str
    model: str = "research"
    temperature: float = 0.3

@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    from core.llm import stream_chat, get_model, build_user_message
    
    messages = [build_user_message(request.prompt)]
    model_id = get_model(request.model)
    
    return StreamingResponse(
        stream_chat(messages, model_id, request.temperature),
        media_type="text/event-stream"
    )

# -------------------------------------------------------------------
# Data Retrieval
# -------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/papers",
    response_model=PaginatedPaperResponse
)
async def get_papers(
    project_id: UUID,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
) -> dict:
    from sqlalchemy import func

    count_query = (
        select(func.count(Paper.id))
        .where(Paper.project_id == project_id)
    )
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    query = (
        select(Paper)
        .where(Paper.project_id == project_id)
        .order_by(desc(Paper.relevance_score))
        .offset((page - 1) * size)
        .limit(size)
    )

    result = await db.execute(query)
    items = list(result.scalars().all())

    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size
    }


@router.get(
    "/projects/{project_id}/report",
    response_model=ReportResponse
)
async def get_report(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> ReviewReport:
    query = (
        select(ReviewReport)
        .where(ReviewReport.project_id == project_id)
        .order_by(desc(ReviewReport.version))
        .limit(1)
    )

    result = await db.execute(query)

    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )

    return report


@router.get(
    "/projects/{project_id}/gaps",
    response_model=list[GapResponse]
)
async def get_gaps(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> list[dict]:
    query = (
        select(ReviewReport)
        .where(ReviewReport.project_id == project_id)
        .order_by(desc(ReviewReport.version))
        .limit(1)
    )

    result = await db.execute(query)

    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(
            status_code=404,
            detail="Gap report not found"
        )

    return report.content.get("gaps", [])


@router.get(
    "/projects/{project_id}/draft",
    response_model=DraftResponse
)
async def get_draft(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> PaperDraft:
    query = (
        select(PaperDraft)
        .where(PaperDraft.project_id == project_id)
        .order_by(desc(PaperDraft.version))
        .limit(1)
    )

    result = await db.execute(query)

    draft = result.scalar_one_or_none()

    if not draft:
        raise HTTPException(
            status_code=404,
            detail="Draft not found"
        )

    return draft


@router.get(
    "/projects/{project_id}/citations",
    response_model=list[CitationResponse]
)
async def get_citations(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> list[Citation]:
    query = select(Citation).where(
        Citation.project_id == project_id
    )

    result = await db.execute(query)

    return list(result.scalars().all())


@router.get(
    "/projects/{project_id}/review-report",
    response_model=ReviewReportResponse
)
async def get_review_report(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> ReviewReport:
    query = (
        select(ReviewReport)
        .where(ReviewReport.project_id == project_id)
        .order_by(desc(ReviewReport.version))
        .limit(1)
    )

    result = await db.execute(query)

    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(
            status_code=404,
            detail="Review report not found"
        )

    return report


@router.get(
    "/projects/{project_id}/assets",
    response_model=list[AssetResponse]
)
async def get_assets(
    project_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> list[GeneratedAsset]:
    query = select(GeneratedAsset).where(
        GeneratedAsset.project_id == project_id
    )

    result = await db.execute(query)

    return list(result.scalars().all())


@router.get(
    "/projects/{project_id}/export/{fmt}"
)
async def download_export(
    project_id: UUID,
    fmt: str,
    db: AsyncSession = Depends(get_db)
) -> FileResponse:
    await get_project_or_404(db, project_id)

    from config import get_settings
    settings = get_settings()
    export_path = settings.exports_dir / str(project_id) / f"paper.{fmt}"

    if not export_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Export file not found"
        )

    return FileResponse(
        path=export_path,
        filename=export_path.name,
        media_type="application/octet-stream"
    )
