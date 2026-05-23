import uuid
from datetime import datetime

from sqlalchemy import JSON
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from core.database import engine
from core.database import AsyncSessionLocal


# -------------------------------------------------------------------
# Base
# -------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# -------------------------------------------------------------------
# Enums
# -------------------------------------------------------------------


PROJECT_STATUS = (
    "idle",
    "discovering",
    "analyzing",
    "approved",
    "drafting",
    "refining",
    "humanizing",
    "reviewing",
    "complete",
    "error"
)

DRAFT_STATUS = (
    "draft",
    "refined",
    "humanized",
    "formatted",
    "final"
)

VALIDATION_STATUS = (
    "verified",
    "warning",
    "flagged"
)

ASSET_TYPES = (
    "diagram",
    "figure",
    "code",
    "notebook"
)

AGENT_RUN_STATUS = (
    "running",
    "complete",
    "failed"
)


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------


class User(Base):

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    projects: Mapped[list["Project"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Project(Base):

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    research_idea: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    status: Mapped[str] = mapped_column(
        Enum(
            *PROJECT_STATUS,
            name="project_status"
        ),
        default="idle",
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    user: Mapped["User"] = relationship(
        back_populates="projects"
    )

    papers: Mapped[list["Paper"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )

    citations: Mapped[list["Citation"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )

    review_reports: Mapped[list["ReviewReport"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )

    drafts: Mapped[list["PaperDraft"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )

    agent_runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )

    generated_assets: Mapped[list["GeneratedAsset"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )

    def soft_delete(self) -> None:

        self.deleted_at = datetime.utcnow()

class Paper(Base):

    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id"),
        nullable=False,
        index=True
    )

    external_id: Mapped[str | None] = mapped_column(
        String,
        nullable=True
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    authors: Mapped[dict | list | None] = mapped_column(
        JSON,
        nullable=True
    )

    year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    doi: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        index=True
    )

    abstract: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    pdf_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    parsed_json: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True
    )

    relevance_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    project: Mapped["Project"] = relationship(
        back_populates="papers"
    )

    citations: Mapped[list["Citation"]] = relationship(
        back_populates="paper"
    )


class Citation(Base):

    __tablename__ = "citations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id"),
        nullable=False,
        index=True
    )

    paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("papers.id"),
        nullable=True
    )

    bibtex: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    validated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )

    validation_status: Mapped[str] = mapped_column(
        Enum(
            *VALIDATION_STATUS,
            name="validation_status"
        ),
        default="warning",
        nullable=False
    )

    project: Mapped["Project"] = relationship(
        back_populates="citations"
    )

    paper: Mapped["Paper"] = relationship(
        back_populates="citations"
    )


class ReviewReport(Base):

    __tablename__ = "review_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id"),
        nullable=False,
        index=True
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False
    )

    content: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True
    )

    user_edits: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True
    )

    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    project: Mapped["Project"] = relationship(
        back_populates="review_reports"
    )


class PaperDraft(Base):

    __tablename__ = "paper_drafts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id"),
        nullable=False,
        index=True
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False
    )

    sections: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True
    )

    status: Mapped[str] = mapped_column(
        Enum(
            *DRAFT_STATUS,
            name="draft_status"
        ),
        default="draft",
        nullable=False
    )

    project: Mapped["Project"] = relationship(
        back_populates="drafts"
    )


class AgentRun(Base):

    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id"),
        nullable=False,
        index=True
    )

    agent_name: Mapped[str] = mapped_column(
        String,
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        Enum(
            *AGENT_RUN_STATUS,
            name="agent_run_status"
        ),
        default="running",
        nullable=False
    )

    input: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True
    )

    output: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    project: Mapped["Project"] = relationship(
        back_populates="agent_runs"
    )


class GeneratedAsset(Base):

    __tablename__ = "generated_assets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id"),
        nullable=False,
        index=True
    )

    asset_type: Mapped[str] = mapped_column(
        Enum(
            *ASSET_TYPES,
            name="asset_types"
        ),
        nullable=False
    )

    content: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    file_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    project: Mapped["Project"] = relationship(
        back_populates="generated_assets"
    )


# -------------------------------------------------------------------
# DB Helpers
# -------------------------------------------------------------------


async def init_db() -> None:

    async with engine.begin() as conn:

        await conn.run_sync(
            Base.metadata.create_all
        )