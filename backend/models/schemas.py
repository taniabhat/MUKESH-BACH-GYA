from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import EmailStr
from pydantic import Field


# -------------------------------------------------------------------
# Base
# -------------------------------------------------------------------


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )


# -------------------------------------------------------------------
# Project Schemas
# -------------------------------------------------------------------


class ProjectCreate(BaseSchema):
    title: str = Field(
        ...,
        min_length=3,
        max_length=500
    )

    research_idea: str = Field(
        ...,
        min_length=10
    )


class ProjectResponse(BaseSchema):
    id: UUID

    title: str

    research_idea: str | None = None

    status: str

    created_at: datetime

    updated_at: datetime


# -------------------------------------------------------------------
# Approval / Draft
# -------------------------------------------------------------------


class ApprovalBody(BaseSchema):
    user_edits: dict

    approved_at: datetime | None = None


class DraftPlanBody(BaseSchema):
    plan: dict


# -------------------------------------------------------------------
# Generic Task Response
# -------------------------------------------------------------------


class TaskResponse(BaseSchema):
    task_id: str

    status: str = "queued"


# -------------------------------------------------------------------
# Paper Schemas
# -------------------------------------------------------------------


class PaperResponse(BaseSchema):
    id: UUID

    project_id: UUID

    external_id: str | None = None

    title: str

    authors: list[str | dict] | None = []

    year: int | None = None

    doi: str | None = None

    abstract: str | None = None

    pdf_path: str | None = None

    relevance_score: float | None = None

    created_at: datetime | None = None


class PaginatedPaperResponse(BaseSchema):
    items: list[PaperResponse]
    total: int
    page: int
    size: int


# -------------------------------------------------------------------
# Review Report
# -------------------------------------------------------------------


class ReportResponse(BaseSchema):
    id: UUID

    project_id: UUID

    version: int

    content: dict

    user_edits: dict | None = None

    approved_at: datetime | None = None

    created_at: datetime


# -------------------------------------------------------------------
# Gap Analysis
# -------------------------------------------------------------------


class GapResponse(BaseSchema):
    id: str

    title: str

    severity: str

    novelty_opportunity: str

    suggested_contributions: list[str] = []


# -------------------------------------------------------------------
# Draft
# -------------------------------------------------------------------


class DraftResponse(BaseSchema):
    id: UUID

    project_id: UUID

    version: int

    status: str

    sections: dict[str, str]

    created_at: datetime | None = None


# -------------------------------------------------------------------
# Citation
# -------------------------------------------------------------------


class CitationResponse(BaseSchema):
    id: UUID

    project_id: UUID

    paper_id: UUID | None = None

    bibtex: str

    validated: bool

    validation_status: str

    created_at: datetime | None = None


# -------------------------------------------------------------------
# Reviewer Simulation
# -------------------------------------------------------------------


class ReviewReportResponse(BaseSchema):
    id: UUID

    project_id: UUID

    version: int

    content: dict

    reviewer_scores: dict[str, float] = {}

    rejection_risk: float | None = None

    approved_at: datetime | None = None

    created_at: datetime


# -------------------------------------------------------------------
# Generated Assets
# -------------------------------------------------------------------


class AssetResponse(BaseSchema):
    id: UUID

    project_id: UUID

    asset_type: str

    content: str | None = None

    file_path: str | None = None

    created_at: datetime | None = None


# -------------------------------------------------------------------
# User
# -------------------------------------------------------------------


class UserResponse(BaseSchema):
    id: UUID

    email: EmailStr

    created_at: datetime