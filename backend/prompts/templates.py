from typing import Any


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------


DISCOVERY_QUERY_EXPANSION = """
You are a senior research scientist.

Given a research idea, generate 5-10 high-quality
academic search queries.

The queries should cover:
- methodologies
- datasets
- applications
- limitations
- benchmarks
- emerging approaches

Return only the query list.

Research Idea:
{idea}
"""


# -------------------------------------------------------------------
# Limitation Extraction
# -------------------------------------------------------------------


LIMITATION_EXTRACTION = """
You are analyzing an academic paper.

Extract all explicit and implicit limitations.

Focus on:
- methodological weaknesses
- evaluation weaknesses
- dataset constraints
- scalability limitations
- missing comparisons
- reproducibility concerns
- future work suggestions

Return structured JSON.

Paper Sections:
{text}
"""


# -------------------------------------------------------------------
# Gap Synthesis
# -------------------------------------------------------------------


GAP_SYNTHESIS = """
You are a principal research strategist.

Given:
- clustered limitations
- benchmark gaps
- unresolved graph gaps
- contradictions
- temporal trends

Identify the TOP 10 most impactful
research opportunities.

Each gap must contain:
- title
- problem_statement
- severity
- novelty_opportunity
- suggested_contributions
- supporting_evidence

Be highly specific and technically rigorous.

Limitation Clusters:
{clusters}

Benchmark Gaps:
{benchmark_gaps}

Contradictions:
{contradictions}

Graph Gaps:
{graph_gaps}

Temporal Trends:
{trends}
"""


# -------------------------------------------------------------------
# Novelty
# -------------------------------------------------------------------


NOVELTY_CHECK = """
You are evaluating research novelty.

Assess:
- originality
- differentiation
- overlap with prior work
- likelihood of publication novelty

Return:
- novelty_score (0-10)
- assessment
- closest_prior_art
- differentiators
- risks

Proposed Idea:
{idea}

Retrieved Literature:
{context}
"""


# -------------------------------------------------------------------
# Section Writers
# -------------------------------------------------------------------


SECTION_WRITER_ABSTRACT = """
Write an IEEE-style abstract.

Requirements:
- concise
- technical
- includes motivation
- includes method
- includes results
- includes significance

Avoid vague claims.
"""


SECTION_WRITER_INTRODUCTION = """
Write a strong IEEE introduction.

Structure:
1. Problem motivation
2. Existing limitations
3. Research gap
4. Proposed solution
5. Contributions

Maintain strong narrative flow.
"""


SECTION_WRITER_RELATED_WORK = """
Write the related work section.

Requirements:
- compare prior methods
- identify weaknesses
- explain positioning
- avoid citation dumping

Group literature logically.
"""


SECTION_WRITER_METHODOLOGY = """
Write the methodology section.

Requirements:
- mathematically rigorous
- implementation-aware
- precise terminology
- reproducible description

Clearly explain architecture and algorithms.
"""


SECTION_WRITER_EXPERIMENTS = """
Write the experimental setup section.

Include:
- datasets
- baselines
- metrics
- hardware
- hyperparameters
- training setup

Be reproducible.
"""


SECTION_WRITER_RESULTS = """
Write the results section.

Requirements:
- analyze quantitative results
- compare against baselines
- explain improvements
- discuss failure cases

Avoid generic statements.
"""


SECTION_WRITER_DISCUSSION = """
Write the discussion section.

Focus on:
- implications
- limitations
- practical relevance
- future directions
- interpretation of results
"""


SECTION_WRITER_CONCLUSION = """
Write the conclusion section.

Requirements:
- summarize contributions
- restate significance
- highlight findings
- mention future work

Avoid repetition.
"""


# -------------------------------------------------------------------
# Aliases
# -------------------------------------------------------------------


ABSTRACT_PROMPT = (
    SECTION_WRITER_ABSTRACT
)

INTRODUCTION_PROMPT = (
    SECTION_WRITER_INTRODUCTION
)

RELATED_WORK_PROMPT = (
    SECTION_WRITER_RELATED_WORK
)

METHODOLOGY_PROMPT = (
    SECTION_WRITER_METHODOLOGY
)

EXPERIMENTS_PROMPT = (
    SECTION_WRITER_EXPERIMENTS
)

RESULTS_PROMPT = (
    SECTION_WRITER_RESULTS
)

DISCUSSION_PROMPT = (
    SECTION_WRITER_DISCUSSION
)

CONCLUSION_PROMPT = (
    SECTION_WRITER_CONCLUSION
)


# -------------------------------------------------------------------
# Critics
# -------------------------------------------------------------------


CRITIC_COHERENCE = """
You are reviewing a scientific paper for coherence.

Analyze:
- logical consistency
- narrative flow
- unsupported claims
- contradictory statements
- abrupt transitions

Return structured JSON:
{
  "quality_score": float,
  "issues": [
    {
      "section": str,
      "issue": str,
      "severity": str,
      "suggested_fix": str
    }
  ]
}
"""


CRITIC_ACCURACY = """
You are reviewing scientific correctness.

Check:
- factual accuracy
- unsupported metrics
- invalid assumptions
- incorrect technical claims
- misuse of terminology

Return structured JSON.
"""


CRITIC_NOVELTY = """
You are evaluating novelty and differentiation.

Analyze:
- originality
- overlap with prior work
- weak contribution claims
- insufficient differentiation

Return structured JSON.
"""


CRITIC_CITATIONS = """
You are validating citations.

Check:
- missing references
- unsupported claims
- citation inconsistencies
- missing seminal papers
- incorrect inline references

Return structured JSON.
"""


CRITIC_METHODOLOGY = """
You are reviewing methodology quality.

Check:
- reproducibility
- missing ablations
- unclear equations
- undefined loss functions
- insufficient experimental detail

Return structured JSON.
"""


# -------------------------------------------------------------------
# Humanizer
# -------------------------------------------------------------------


HUMANIZER = """
Rewrite the text to sound naturally academic.

Requirements:
- varied sentence structure
- strong narrative flow
- human-like rhythm
- precise technical language
- avoid robotic phrasing
- reduce repetitive transitions
- preserve technical meaning

Maintain IEEE-quality academic tone.
"""


# -------------------------------------------------------------------
# Reviewers
# -------------------------------------------------------------------


REVIEWER_NOVELTY = """
You are a top-tier conference reviewer.

Evaluate:
- novelty
- contribution uniqueness
- significance
- alignment with identified research gaps

Return:
- score (1-10)
- strengths
- weaknesses
- rejection_risks
- final recommendation
"""


REVIEWER_METHODOLOGY = """
You are reviewing methodology rigor.

Evaluate:
- technical correctness
- baselines
- ablations
- mathematical rigor
- implementation clarity

Return detailed reviewer feedback.
"""


REVIEWER_EXPERIMENT = """
You are reviewing experiments.

Evaluate:
- dataset quality
- metric selection
- fairness of comparison
- statistical validity
- robustness

Return detailed reviewer feedback.
"""


REVIEWER_CITATION = """
You are reviewing citations and literature grounding.

Evaluate:
- citation completeness
- claim attribution
- seminal work coverage
- formatting consistency

Return detailed reviewer feedback.
"""


REVIEWER_WRITING = """
You are reviewing writing quality.

Evaluate:
- clarity
- structure
- coherence
- readability
- abstract quality
- technical communication

Return detailed reviewer feedback.
"""


REVIEWER_REPRODUCIBILITY = """
You are reviewing reproducibility.

Evaluate:
- implementation detail
- hyperparameter disclosure
- dataset accessibility
- code reproducibility
- experiment clarity

Return detailed reviewer feedback.
"""


# -------------------------------------------------------------------
# Aliases
# -------------------------------------------------------------------


REVIEWER_EXPERIMENTS = (
    REVIEWER_EXPERIMENT
)

REVIEWER_CITATIONS = (
    REVIEWER_CITATION
)


# -------------------------------------------------------------------
# Diagram Generator
# -------------------------------------------------------------------


DIAGRAM_GENERATOR = """
Generate valid Mermaid or PlantUML diagram code.

Requirements:
- syntactically valid
- concise
- visually clear
- publication quality
- technically accurate

Avoid unnecessary nodes.

Generate ONLY diagram code.
"""


# -------------------------------------------------------------------
# Code Experiment Design
# -------------------------------------------------------------------


CODE_EXPERIMENT_DESIGN = """
Analyze the methodology section.

Identify:
- task type
- framework
- architecture family
- datasets
- baselines
- evaluation metrics
- training strategy
- hardware requirements

Return structured JSON.

Methodology:
{methodology}
"""


# -------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------


PROMPT_REGISTRY = {
    "DISCOVERY_QUERY_EXPANSION":
        DISCOVERY_QUERY_EXPANSION,

    "LIMITATION_EXTRACTION":
        LIMITATION_EXTRACTION,

    "GAP_SYNTHESIS":
        GAP_SYNTHESIS,

    "NOVELTY_CHECK":
        NOVELTY_CHECK,

    "SECTION_WRITER_ABSTRACT":
        SECTION_WRITER_ABSTRACT,

    "SECTION_WRITER_INTRODUCTION":
        SECTION_WRITER_INTRODUCTION,

    "SECTION_WRITER_RELATED_WORK":
        SECTION_WRITER_RELATED_WORK,

    "SECTION_WRITER_METHODOLOGY":
        SECTION_WRITER_METHODOLOGY,

    "SECTION_WRITER_EXPERIMENTS":
        SECTION_WRITER_EXPERIMENTS,

    "SECTION_WRITER_RESULTS":
        SECTION_WRITER_RESULTS,

    "SECTION_WRITER_DISCUSSION":
        SECTION_WRITER_DISCUSSION,

    "SECTION_WRITER_CONCLUSION":
        SECTION_WRITER_CONCLUSION,

    "CRITIC_COHERENCE":
        CRITIC_COHERENCE,

    "CRITIC_ACCURACY":
        CRITIC_ACCURACY,

    "CRITIC_NOVELTY":
        CRITIC_NOVELTY,

    "CRITIC_CITATIONS":
        CRITIC_CITATIONS,

    "CRITIC_METHODOLOGY":
        CRITIC_METHODOLOGY,

    "HUMANIZER":
        HUMANIZER,

    "REVIEWER_NOVELTY":
        REVIEWER_NOVELTY,

    "REVIEWER_METHODOLOGY":
        REVIEWER_METHODOLOGY,

    "REVIEWER_EXPERIMENTS":
        REVIEWER_EXPERIMENTS,

    "REVIEWER_CITATIONS":
        REVIEWER_CITATIONS,

    "REVIEWER_WRITING":
        REVIEWER_WRITING,

    "REVIEWER_REPRODUCIBILITY":
        REVIEWER_REPRODUCIBILITY,

    "DIAGRAM_GENERATOR":
        DIAGRAM_GENERATOR,

    "CODE_EXPERIMENT_DESIGN":
        CODE_EXPERIMENT_DESIGN
}


# -------------------------------------------------------------------
# Prompt Getter
# -------------------------------------------------------------------


class SafeDict(dict):

    def __missing__(
        self,
        key: str
    ) -> str:

        return "{" + key + "}"


def get_prompt(
    template_name: str,
    **kwargs: Any
) -> str:

    template = PROMPT_REGISTRY.get(
        template_name
    )

    if template is None:

        raise KeyError(
            f"Prompt template not found: "
            f"{template_name}"
        )

    return template.format_map(
        SafeDict(kwargs)
    )