import json
from datetime import datetime
from typing import TypedDict

from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from sqlalchemy import desc
from sqlalchemy import select

from agents import generation
from agents import knowledge
from agents import research
from agents import review
from agents import writing
from core.logging import get_logger
from models.db import AgentRun
from models.db import AsyncSessionLocal
from models.db import Project
from worker import notify_progress


logger = get_logger("agents.supervisor")


# -------------------------------------------------------------------
# LangGraph State
# -------------------------------------------------------------------


class ResearchState(TypedDict):

    project_id: str

    current_stage: str

    paper_ids: list[str]

    doc_results: list[dict]

    gap_report: dict

    research_plan: dict

    draft_sections: dict

    refined_sections: dict

    humanized_sections: dict

    review_report: dict

    citation_report: dict

    agent_outputs: dict

    errors: list[dict]

    retry_count: int


# -------------------------------------------------------------------
# Planning Node
# -------------------------------------------------------------------


async def plan(
    state: ResearchState
) -> ResearchState:

    logger.info("supervisor.plan.started", project_id=state.get("project_id"))
    async with AsyncSessionLocal() as db:

        query = select(Project).where(
            Project.id == state["project_id"]
        )

        result = await db.execute(query)

        project = result.scalar_one_or_none()

        if not project:
            raise ValueError("Project not found")

        status_to_stage = {
            "idle": "discovery",
            "discovering": "discovery",
            "analyzing": "analysis",
            "approved": "draft",
            "drafting": "draft",
            "refining": "refinement",
            "humanizing": "humanization",
            "reviewing": "review",
            "complete": "export"
        }

        state["current_stage"] = status_to_stage.get(
            project.status,
            "discovery"
        )

        logger.info("supervisor.plan.success", project_id=state["project_id"], next_stage=state["current_stage"])
        return state


# -------------------------------------------------------------------
# Route Function
# -------------------------------------------------------------------


def route(
    state: ResearchState
) -> str:

    if state["errors"]:

        latest_error = state["errors"][-1]

        if latest_error.get("fatal"):
            return END

    stage = state["current_stage"]

    routes = {
        "discovery": "discovery",
        "analysis": "analysis",
        "gap_analysis": "gap_analysis",
        "draft": "draft",
        "refinement": "refinement",
        "humanization": "humanization",
        "review": "review",
        "export": "export"
    }

    return routes.get(stage, END)


# -------------------------------------------------------------------
# Error Handler
# -------------------------------------------------------------------


async def handle_error(
    state: ResearchState,
    error: Exception,
    agent_name: str
) -> ResearchState:

    retry_count = state.get(
        "retry_count",
        0
    )

    retry_count += 1

    state["retry_count"] = retry_count

    error_entry = {
        "agent": agent_name,
        "error": str(error),
        "timestamp": datetime.utcnow().isoformat(),
        "fatal": retry_count >= 3
    }

    state.setdefault(
        "errors", []
    ).append(error_entry)

    logger.error("supervisor.error", project_id=state.get("project_id"), agent=agent_name, error=str(error), fatal=error_entry["fatal"], retry_count=retry_count)

    async with AsyncSessionLocal() as db:

        run = AgentRun(
            project_id=state["project_id"],
            agent_name=agent_name,
            status="failed",
            input=state,
            output=None,
            error=str(error),
            completed_at=datetime.utcnow()
        )

        db.add(run)

        if retry_count >= 3:

            query = select(Project).where(
                Project.id == state["project_id"]
            )

            result = await db.execute(query)

            project = result.scalar_one_or_none()

            if project:
                project.status = "error"

        await db.commit()

    return state


# -------------------------------------------------------------------
# Checkpoint
# -------------------------------------------------------------------


async def checkpoint(
    state: ResearchState
) -> None:

    logger.debug("supervisor.checkpoint.saving", project_id=state.get("project_id"))
    try:
        async with AsyncSessionLocal() as db:

            # Optimize state serialization by excluding heavy objects
            # and converting non-serializable types safely
            optimized_state = {}
            exclude_keys = {"doc_results", "agent_outputs"}
            
            for key, value in state.items():
                if key in exclude_keys:
                    optimized_state[key] = f"[Excluded: heavy object, type={type(value).__name__}]"
                else:
                    optimized_state[key] = value

            state_dump = json.loads(json.dumps(optimized_state, default=str))

            checkpoint_run = AgentRun(
                project_id=state["project_id"],
                agent_name="supervisor.checkpoint",
                status="complete",
                input=None,
                output=state_dump,
                error=None,
                completed_at=datetime.utcnow()
            )

            db.add(checkpoint_run)

            await db.commit()
            logger.debug("supervisor.checkpoint.success", project_id=state.get("project_id"))

    except Exception as error:
        logger.error("supervisor.checkpoint.failed", project_id=state.get("project_id"), error=str(error), exc_info=True)


# -------------------------------------------------------------------
# Agent Nodes
# -------------------------------------------------------------------


async def discovery_node(
    state: ResearchState
) -> ResearchState:

    try:
        notify_progress(
            state["project_id"],
            "discovery",
            "running"
        )

        result = await research.run_discovery(
            state["project_id"]
        )

        state["agent_outputs"]["discovery"] = result

        state["paper_ids"] = result.get(
            "paper_ids",
            []
        )

        state["current_stage"] = "analysis"

        await checkpoint(state)

        return state

    except Exception as error:

        return await handle_error(
            state,
            error,
            "research.discovery"
        )


async def analysis_node(
    state: ResearchState
) -> ResearchState:

    try:
        notify_progress(
            state["project_id"],
            "analysis",
            "running"
        )

        result = await research.run_document_analysis(
            state["project_id"]
        )

        state["doc_results"] = result.get(
            "documents",
            []
        )

        state["agent_outputs"]["analysis"] = result

        state["current_stage"] = "gap_analysis"

        await checkpoint(state)

        return state

    except Exception as error:

        return await handle_error(
            state,
            error,
            "research.analysis"
        )


async def gap_analysis_node(
    state: ResearchState
) -> ResearchState:

    try:
        notify_progress(
            state["project_id"],
            "gap_analysis",
            "running"
        )

        result = await knowledge.run_gap_analysis(
            state["project_id"]
        )

        state["gap_report"] = result

        state["agent_outputs"]["gap_analysis"] = result

        state["current_stage"] = "draft"

        await checkpoint(state)

        return state

    except Exception as error:

        return await handle_error(
            state,
            error,
            "knowledge.gap_analysis"
        )


async def draft_node(
    state: ResearchState
) -> ResearchState:

    try:
        notify_progress(
            state["project_id"],
            "draft",
            "running"
        )

        result = await writing.generate_draft(
            state["project_id"],
            state.get("research_plan", {})
        )

        state["draft_sections"] = result

        state["agent_outputs"]["draft"] = result

        state["current_stage"] = "refinement"

        await checkpoint(state)

        return state

    except Exception as error:

        return await handle_error(
            state,
            error,
            "writing.draft"
        )


async def refinement_node(
    state: ResearchState
) -> ResearchState:

    try:
        notify_progress(
            state["project_id"],
            "refinement",
            "running"
        )

        result = await writing.run_refinement(
            state["project_id"]
        )

        state["refined_sections"] = result

        state["agent_outputs"]["refinement"] = result

        state["current_stage"] = "humanization"

        await checkpoint(state)

        return state

    except Exception as error:

        return await handle_error(
            state,
            error,
            "writing.refinement"
        )


async def humanization_node(
    state: ResearchState
) -> ResearchState:

    try:
        notify_progress(
            state["project_id"],
            "humanization",
            "running"
        )

        result = await writing.run_humanization(
            state["project_id"]
        )

        state["humanized_sections"] = result

        state["agent_outputs"]["humanization"] = result

        state["current_stage"] = "review"

        await checkpoint(state)

        return state

    except Exception as error:

        return await handle_error(
            state,
            error,
            "writing.humanization"
        )


async def review_node(
    state: ResearchState
) -> ResearchState:

    try:
        notify_progress(
            state["project_id"],
            "review",
            "running"
        )

        review_result = (
            await review.run_reviewer_simulation(
                state["project_id"]
            )
        )

        citation_result = (
            await review.run_citation_validation(
                state["project_id"]
            )
        )

        state["review_report"] = review_result

        state["citation_report"] = citation_result

        state["agent_outputs"]["review"] = {
            "review": review_result,
            "citations": citation_result
        }

        state["current_stage"] = "export"

        await checkpoint(state)

        return state

    except Exception as error:

        return await handle_error(
            state,
            error,
            "review.pipeline"
        )


async def export_node(
    state: ResearchState
) -> ResearchState:

    try:
        notify_progress(
            state["project_id"],
            "export",
            "running"
        )

        result = await generation.run_ieee_export(
            state["project_id"],
            "pdf"
        )

        state["agent_outputs"]["export"] = result

        async with AsyncSessionLocal() as db:

            query = select(Project).where(
                Project.id == state["project_id"]
            )

            db_result = await db.execute(query)

            project = db_result.scalar_one_or_none()

            if project:
                project.status = "complete"

            await db.commit()

        await checkpoint(state)

        return state

    except Exception as error:

        return await handle_error(
            state,
            error,
            "generation.export"
        )


# -------------------------------------------------------------------
# Graph Builder
# -------------------------------------------------------------------


def build_graph():

    graph = StateGraph(ResearchState)

    graph.add_node(
        "plan",
        plan
    )

    graph.add_node(
        "discovery",
        discovery_node
    )

    graph.add_node(
        "analysis",
        analysis_node
    )

    graph.add_node(
        "gap_analysis",
        gap_analysis_node
    )

    graph.add_node(
        "draft",
        draft_node
    )

    graph.add_node(
        "refinement",
        refinement_node
    )

    graph.add_node(
        "humanization",
        humanization_node
    )

    graph.add_node(
        "review",
        review_node
    )

    graph.add_node(
        "export",
        export_node
    )

    graph.add_edge(
        START,
        "plan"
    )

    graph.add_conditional_edges(
        "plan",
        route
    )

    graph.add_edge(
        "discovery",
        "analysis"
    )

    graph.add_edge(
        "analysis",
        "gap_analysis"
    )

    graph.add_edge(
        "gap_analysis",
        "draft"
    )

    graph.add_edge(
        "draft",
        "refinement"
    )

    graph.add_edge(
        "refinement",
        "humanization"
    )

    graph.add_edge(
        "humanization",
        "review"
    )

    graph.add_edge(
        "review",
        "export"
    )

    graph.add_edge(
        "export",
        END
    )

    return graph.compile()


# -------------------------------------------------------------------
# Main Pipeline Entry
# -------------------------------------------------------------------


async def load_checkpoint(
    project_id: str
) -> ResearchState | None:

    async with AsyncSessionLocal() as db:

        query = (
            select(AgentRun)
            .where(
                AgentRun.project_id == project_id,
                AgentRun.agent_name == "supervisor.checkpoint"
            )
            .order_by(
                desc(AgentRun.completed_at)
            )
            .limit(1)
        )

        result = await db.execute(query)

        checkpoint_run = result.scalar_one_or_none()

        if not checkpoint_run:
            return None

        return checkpoint_run.output


async def run_pipeline(
    project_id: str,
    start_stage: str = "discovery"
) -> None:

    graph = build_graph()

    checkpoint_state = await load_checkpoint(
        project_id
    )

    if checkpoint_state:

        state = checkpoint_state

    else:

        state: ResearchState = {
            "project_id": project_id,
            "current_stage": start_stage,
            "paper_ids": [],
            "doc_results": [],
            "gap_report": {},
            "research_plan": {},
            "draft_sections": {},
            "refined_sections": {},
            "humanized_sections": {},
            "review_report": {},
            "citation_report": {},
            "agent_outputs": {},
            "errors": [],
            "retry_count": 0
        }

    async for event in graph.astream(state):

        notify_progress(
            project_id,
            "pipeline",
            json.dumps(event)
        )