import statistics
from collections import Counter

from sqlalchemy import desc
from sqlalchemy import select

from core import rag
from core.llm import build_system_message
from core.llm import build_user_message
from core.llm import chat
from core.llm import get_model
from core.logging import get_logger
from models.db import AgentRun
from models.db import AsyncSessionLocal
from models.db import PaperDraft
from agents.review import validate_doi
from prompts.templates import (
    ABSTRACT_PROMPT,
    CONCLUSION_PROMPT,
    DISCUSSION_PROMPT,
    EXPERIMENTS_PROMPT,
    HUMANIZER,
    INTRODUCTION_PROMPT,
    METHODOLOGY_PROMPT,
    RELATED_WORK_PROMPT,
    RESULTS_PROMPT
)


logger = get_logger("agents.writing")


# -------------------------------------------------------------------
# IEEE Section Order
# -------------------------------------------------------------------


IEEE_SECTIONS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Methodology",
    "Experiments",
    "Results",
    "Discussion",
    "Conclusion",
    "References"
]


SECTION_PROMPTS = {
    "Abstract": ABSTRACT_PROMPT,
    "Introduction": INTRODUCTION_PROMPT,
    "Related Work": RELATED_WORK_PROMPT,
    "Methodology": METHODOLOGY_PROMPT,
    "Experiments": EXPERIMENTS_PROMPT,
    "Results": RESULTS_PROMPT,
    "Discussion": DISCUSSION_PROMPT,
    "Conclusion": CONCLUSION_PROMPT
}


# -------------------------------------------------------------------
# Draft Generation
# -------------------------------------------------------------------


async def generate_section(
    section_name: str,
    context: list[dict],
    plan: dict,
    prior_sections: dict
) -> str:

    prompt_template = SECTION_PROMPTS.get(
        section_name,
        INTRODUCTION_PROMPT
    )

    summarized_prior = "\n\n".join([
        f"{key}:\n{value[:1500]}"
        for key, value in prior_sections.items()
    ])

    context_text = "\n\n".join([
        chunk["content"][:2000]
        for chunk in context
    ])

    prompt = f"""
Research Plan:
{plan}

Prior Sections:
{summarized_prior}

Retrieved Context:
{context_text}

Generate IEEE-style section:
{section_name}
"""

    response = await chat(
        messages=[
            build_system_message(
                prompt_template
            ),
            build_user_message(prompt)
        ],
        model=get_model("writing"),
        temperature=0.4,
        max_tokens=4096
    )

    return response.strip()


async def generate_draft(
    project_id: str,
    plan: dict
) -> dict:

    logger.info("writing.drafting.started", project_id=project_id, plan_sections=len(plan.get("sections", [])))
    sections = {}

    for section_name in IEEE_SECTIONS:

        logger.debug("writing.drafting.section.started", section=section_name)
        query = (
            f"{section_name} "
            f"{plan.get('title', '')}"
        )

        context = await rag.retrieve(
            query=query,
            project_id=project_id,
            top_k=12
        )

        generated = await generate_section(
            section_name=section_name,
            context=context,
            plan=plan,
            prior_sections=sections
        )

        sections[section_name] = generated

    async with AsyncSessionLocal() as db:

        draft = PaperDraft(
            project_id=project_id,
            version=1,
            sections=sections,
            status="draft"
        )

        db.add(draft)

        await db.commit()

    return sections


# -------------------------------------------------------------------
# Critic Helpers
# -------------------------------------------------------------------


async def run_critic(
    sections: dict,
    system_prompt: str
) -> dict:

    full_draft = "\n\n".join([
        f"{name}\n{text}"
        for name, text in sections.items()
    ])

    response = await chat(
        messages=[
            build_system_message(
                system_prompt
            ),
            build_user_message(full_draft)
        ],
        model=get_model("research"),
        temperature=0.2,
        max_tokens=2048
    )

    return {
        "raw_response": response,
        "quality_score": 7.5,
        "issues": []
    }


# -------------------------------------------------------------------
# Critique Stages
# -------------------------------------------------------------------


async def critique_logical_coherence(
    sections: dict
) -> dict:

    return await run_critic(
        sections,
        """
Analyze:
- narrative consistency
- section flow
- unsupported claims
- coherence
"""
    )


async def critique_scientific_accuracy(
    sections: dict
) -> dict:

    evidence = await rag.retrieve(
        query="scientific factual verification",
        project_id="global",
        top_k=10
    )

    return await run_critic(
        sections,
        f"""
Verify scientific accuracy.

Evidence:
{evidence}
"""
    )


async def critique_novelty(
    sections: dict
) -> dict:

    return await run_critic(
        sections,
        """
Evaluate:
- novelty clarity
- differentiation
- contribution uniqueness
"""
    )


async def critique_citations(
    sections: dict
) -> dict:

    references = sections.get(
        "References",
        ""
    )

    validation_results = []

    for line in references.split("\n")[:10]:

        if "doi" not in line.lower():
            continue

        validation = await validate_doi(
            line
        )

        validation_results.append(
            validation
        )

    return {
        "issues": validation_results,
        "quality_score": 8.0
    }


async def critique_methodology(
    sections: dict
) -> dict:

    return await run_critic(
        sections,
        """
Check:
- baselines
- ablations
- loss functions
- reproducibility
"""
    )


# -------------------------------------------------------------------
# Direct Rewrite Stages
# -------------------------------------------------------------------


async def remove_redundancy(
    sections: dict
) -> dict:

    updated = {}

    for name, text in sections.items():

        rewritten = await chat(
            messages=[
                build_system_message(
                    """
Remove redundancy.
Tighten wording.
Merge repetitive ideas.
"""
                ),
                build_user_message(text)
            ],
            model=get_model("writing"),
            temperature=0.3,
            max_tokens=4096
        )

        updated[name] = rewritten

    return updated


async def optimize_flow(
    sections: dict
) -> dict:

    updated = {}

    for name, text in sections.items():

        rewritten = await chat(
            messages=[
                build_system_message(
                    """
Improve transitions and narrative flow.
Align abstract and conclusion.
"""
                ),
                build_user_message(text)
            ],
            model=get_model("writing"),
            temperature=0.3,
            max_tokens=4096
        )

        updated[name] = rewritten

    return updated


# -------------------------------------------------------------------
# Critic Fix Application
# -------------------------------------------------------------------


async def apply_critic_fixes(
    sections: dict,
    issues: dict
) -> dict:

    updated = dict(sections)

    issue_list = issues.get(
        "issues",
        []
    )

    for issue in issue_list:

        section_name = issue.get(
            "section"
        )

        if not section_name:
            continue

        original_text = updated.get(
            section_name,
            ""
        )

        suggested_fix = issue.get(
            "suggested_fix",
            ""
        )

        rewritten = await chat(
            messages=[
                build_system_message(
                    """
Revise the section using the provided fix.
Maintain IEEE style.
"""
                ),
                build_user_message(
                    f"""
Suggested Fix:
{suggested_fix}

Original Section:
{original_text}
"""
                )
            ],
            model=get_model("writing"),
            temperature=0.3,
            max_tokens=4096
        )

        updated[section_name] = rewritten

    return updated


# -------------------------------------------------------------------
# Refinement Pipeline
# -------------------------------------------------------------------


async def load_latest_draft(
    project_id: str
) -> PaperDraft:

    async with AsyncSessionLocal() as db:

        query = (
            select(PaperDraft)
            .where(
                PaperDraft.project_id
                == project_id
            )
            .order_by(
                desc(PaperDraft.version)
            )
        )

        result = await db.execute(query)

        draft = result.scalar_one_or_none()

        if not draft:
            raise ValueError(
                "Draft not found"
            )

        return draft


async def save_refined_draft(
    project_id: str,
    sections: dict,
    status: str
) -> None:

    async with AsyncSessionLocal() as db:

        latest_query = (
            select(PaperDraft)
            .where(
                PaperDraft.project_id
                == project_id
            )
            .order_by(
                desc(PaperDraft.version)
            )
        )

        latest_result = await db.execute(
            latest_query
        )

        latest = latest_result.scalar_one()

        new_draft = PaperDraft(
            project_id=project_id,
            version=latest.version + 1,
            sections=sections,
            status=status
        )

        db.add(new_draft)

        await db.commit()


async def run_refinement(
    project_id: str
) -> dict:

    logger.info("writing.refinement.started", project_id=project_id)
    draft = await load_latest_draft(
        project_id
    )

    sections = draft.sections

    stages = [
        critique_logical_coherence,
        critique_scientific_accuracy,
        critique_novelty,
        critique_citations,
        critique_methodology
    ]

    for stage in stages:

        issues = await stage(
            sections
        )

        sections = await apply_critic_fixes(
            sections,
            issues
        )

    sections = await remove_redundancy(
        sections
    )

    sections = await optimize_flow(
        sections
    )

    await save_refined_draft(
        project_id,
        sections,
        "refined"
    )

    logger.info("writing.refinement.success", project_id=project_id)
    return sections


# -------------------------------------------------------------------
# Humanization
# -------------------------------------------------------------------


def detect_writing_patterns(
    text: str
) -> dict:

    sentences = [
        sentence.strip()
        for sentence in text.split(".")
        if sentence.strip()
    ]

    starters = []

    sentence_lengths = []

    transition_phrases = [
        "furthermore",
        "moreover",
        "additionally",
        "however",
        "therefore"
    ]

    transition_counter = Counter()

    for sentence in sentences:

        words = sentence.split()

        if not words:
            continue

        starters.append(
            words[0].lower()
        )

        sentence_lengths.append(
            len(words)
        )

        lowered = sentence.lower()

        for phrase in transition_phrases:

            if phrase in lowered:
                transition_counter[
                    phrase
                ] += 1

    repeated_starters = Counter(
        starters
    ).most_common(5)

    variance = (
        statistics.stdev(
            sentence_lengths
        )
        if len(sentence_lengths) > 1
        else 0
    )

    return {
        "repeated_starters":
            repeated_starters,

        "sentence_length_variance":
            variance,

        "overused_transitions":
            dict(transition_counter),

        "generic_claim_patterns":
            []
    }


async def diversify_syntax(
    text: str,
    patterns: dict
) -> str:

    rewritten = await chat(
        messages=[
            build_system_message(
                HUMANIZER
            ),
            build_user_message(
                f"""
Patterns:
{patterns}

Text:
{text}
"""
            )
        ],
        model=get_model("humanize"),
        temperature=0.7,
        max_tokens=4096
    )

    return rewritten


async def improve_transitions(
    text: str
) -> str:

    rewritten = await chat(
        messages=[
            build_system_message(
                """
Replace generic transitions with
content-specific transitions.
"""
            ),
            build_user_message(text)
        ],
        model=get_model("humanize"),
        temperature=0.5,
        max_tokens=4096
    )

    return rewritten


async def calibrate_tone(
    text: str
) -> str:

    rewritten = await chat(
        messages=[
            build_system_message(
                """
Adjust tone to IEEE/NeurIPS style.
Precise and technical.
Not robotic.
"""
            ),
            build_user_message(text)
        ],
        model=get_model("humanize"),
        temperature=0.4,
        max_tokens=4096
    )

    return rewritten


async def ensure_narrative_flow(
    sections: dict
) -> dict:

    full_text = "\n\n".join([
        f"{name}\n{text}"
        for name, text in sections.items()
    ])

    improved = await chat(
        messages=[
            build_system_message(
                """
Ensure coherent research narrative:
problem → insight → method → validation.
"""
            ),
            build_user_message(full_text)
        ],
        model=get_model("humanize"),
        temperature=0.4,
        max_tokens=8192
    )

    updated = dict(sections)

    updated["NarrativeFlowRevision"] = (
        improved
    )

    return updated


# -------------------------------------------------------------------
# Humanization Pipeline
# -------------------------------------------------------------------


async def run_humanization(
    project_id: str
) -> dict:

    logger.info("writing.humanization.started", project_id=project_id)
    draft = await load_latest_draft(
        project_id
    )

    sections = dict(
        draft.sections
    )

    updated_sections = {}

    for section_name, text in sections.items():

        patterns = detect_writing_patterns(
            text
        )

        diversified = await diversify_syntax(
            text,
            patterns
        )

        transitioned = (
            await improve_transitions(
                diversified
            )
        )

        calibrated = await calibrate_tone(
            transitioned
        )

        updated_sections[
            section_name
        ] = calibrated

    updated_sections = (
        await ensure_narrative_flow(
            updated_sections
        )
    )

    await save_refined_draft(
        project_id,
        updated_sections,
        "humanized"
    )

    logger.info("writing.humanization.success", project_id=project_id)

    async with AsyncSessionLocal() as db:

        run = AgentRun(
            project_id=project_id,
            agent_name="writing.humanization",
            status="complete",
            output={
                "sections":
                    list(
                        updated_sections.keys()
                    )
            }
        )

        db.add(run)

        await db.commit()

    return updated_sections