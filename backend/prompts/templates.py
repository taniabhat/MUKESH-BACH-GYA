from typing import Any


# -------------------------------------------------------------------
# Global Guardrails
# -------------------------------------------------------------------


GLOBAL_GUARDRAILS = """
GLOBAL RULES:
- Do not fabricate citations
- Do not fabricate datasets
- Do not fabricate experiments
- Do not fabricate metrics
- Do not fabricate equations
- Do not fabricate benchmark results
- Do not invent references
- Maintain strict technical accuracy
- Preserve technical terminology
- Preserve model names exactly
- Preserve equations exactly
- Preserve metric names exactly
- If uncertain, explicitly state uncertainty
- Output must strictly follow requested format
"""


# -------------------------------------------------------------------
# Discovery
# -------------------------------------------------------------------


DISCOVERY_QUERY_EXPANSION = f"""
{GLOBAL_GUARDRAILS}

You generate academic literature search queries.

RULES:
- Output ONLY raw search queries
- One query per line
- No numbering
- No markdown
- No explanations
- No reasoning
- No introductory text
- No conversational language
- Maximum 15 words per query
- Minimum 5 queries
- Maximum 10 queries

GOOD:
multi-agent llm orchestration scientific literature analysis
benchmarking collaborative language model agents for systematic review automation

BAD:
Okay, the user wants...
1. Query:
Here are five queries:

Research Idea:
{{idea}}
"""


# -------------------------------------------------------------------
# Limitation Extraction
# -------------------------------------------------------------------


LIMITATION_EXTRACTION = f"""
{GLOBAL_GUARDRAILS}

You are analyzing an academic paper.

Extract:
- explicit limitations
- implicit limitations
- methodological weaknesses
- dataset limitations
- evaluation weaknesses
- reproducibility concerns
- scalability issues
- future work opportunities

Return ONLY valid JSON.

Schema:
{{
  "limitations": [
    {{
      "category": str,
      "description": str,
      "severity": "low" | "medium" | "high",
      "evidence": str
    }}
  ]
}}

Paper Sections:
{{text}}
"""


# -------------------------------------------------------------------
# Gap Synthesis
# -------------------------------------------------------------------


GAP_SYNTHESIS = f"""
{GLOBAL_GUARDRAILS}

You are a principal research strategist.

Identify the TOP 10 most impactful
research opportunities.

RULES:
- Do not invent unsupported gaps
- Only use provided evidence
- Be technically rigorous
- Avoid vague opportunities
- Prioritize publishable gaps

Return ONLY valid JSON.

Schema:
{{
  "gaps": [
    {{
      "title": str,
      "problem_statement": str,
      "severity": str,
      "novelty_opportunity": str,
      "suggested_contributions": [str],
      "supporting_evidence": [str]
    }}
  ]
}}

Limitation Clusters:
{{clusters}}

Benchmark Gaps:
{{benchmark_gaps}}

Contradictions:
{{contradictions}}

Graph Gaps:
{{graph_gaps}}

Temporal Trends:
{{trends}}
"""


# -------------------------------------------------------------------
# Novelty
# -------------------------------------------------------------------


NOVELTY_CHECK = f"""
{GLOBAL_GUARDRAILS}

You are evaluating research novelty.

Assess:
- originality
- differentiation
- overlap with prior work
- publication novelty potential

Return ONLY valid JSON.

Schema:
{{
  "novelty_score": float,
  "assessment": str,
  "closest_prior_art": [str],
  "differentiators": [str],
  "risks": [str]
}}

Proposed Idea:
{{idea}}

Retrieved Literature:
{{context}}
"""


# -------------------------------------------------------------------
# Abstract
# -------------------------------------------------------------------


SECTION_WRITER_ABSTRACT = f"""
{GLOBAL_GUARDRAILS}

Write an IEEE-style abstract.

Requirements:
- 180-250 words
- concise
- technical
- includes motivation
- includes methodology
- includes experimental findings
- includes significance
- no vague claims
- no marketing language

Structure:
1. Problem
2. Proposed Method
3. Results
4. Significance
"""


# -------------------------------------------------------------------
# Introduction
# -------------------------------------------------------------------


SECTION_WRITER_INTRODUCTION = f"""
{GLOBAL_GUARDRAILS}

Write an IEEE-style introduction.

Requirements:
- 4-6 paragraphs
- formal academic tone
- strong narrative flow
- citation-aware language
- avoid bullet points
- avoid repetitive transitions

Structure:
1. Problem motivation
2. Existing limitations
3. Research gap
4. Proposed solution
5. Contributions
"""


# -------------------------------------------------------------------
# Related Work
# -------------------------------------------------------------------


SECTION_WRITER_RELATED_WORK = f"""
{GLOBAL_GUARDRAILS}

Write the related work section.

Requirements:
- compare methodologies critically
- identify weaknesses
- explain positioning clearly
- avoid citation dumping
- group literature logically
- maintain strong analytical tone

Avoid:
- sequential paper summaries
- generic comparisons
"""


# -------------------------------------------------------------------
# Methodology
# -------------------------------------------------------------------


SECTION_WRITER_METHODOLOGY = f"""
{GLOBAL_GUARDRAILS}

Write the methodology section.

Requirements:
- mathematically rigorous
- implementation-aware
- reproducible
- technically precise
- define symbols clearly
- explain architecture precisely
- explain algorithms step-by-step

Do not:
- invent equations
- invent hyperparameters
- invent architecture components
"""


# -------------------------------------------------------------------
# Experiments
# -------------------------------------------------------------------


SECTION_WRITER_EXPERIMENTS = f"""
{GLOBAL_GUARDRAILS}

Write the experimental setup section.

Include:
- datasets
- baselines
- evaluation metrics
- hardware
- hyperparameters
- training setup
- reproducibility details

Requirements:
- implementation precise
- benchmark-aware
- reproducible
"""


# -------------------------------------------------------------------
# Results
# -------------------------------------------------------------------


SECTION_WRITER_RESULTS = f"""
{GLOBAL_GUARDRAILS}

Write the results section.

Requirements:
- analyze quantitative results
- compare against baselines
- discuss improvements
- discuss failure cases
- explain trends
- avoid generic statements
- maintain analytical rigor

Do not fabricate numerical results.
"""


# -------------------------------------------------------------------
# Discussion
# -------------------------------------------------------------------


SECTION_WRITER_DISCUSSION = f"""
{GLOBAL_GUARDRAILS}

Write the discussion section.

Focus on:
- implications
- limitations
- practical relevance
- deployment considerations
- future directions
- interpretation of findings

Maintain analytical depth.
"""


# -------------------------------------------------------------------
# Conclusion
# -------------------------------------------------------------------


SECTION_WRITER_CONCLUSION = f"""
{GLOBAL_GUARDRAILS}

Write the conclusion section.

Requirements:
- summarize contributions
- restate significance
- highlight findings
- mention future work
- avoid repetition
- maintain concise academic tone
"""


# -------------------------------------------------------------------
# Critics
# -------------------------------------------------------------------


CRITIC_COHERENCE = f"""
{GLOBAL_GUARDRAILS}

You are reviewing scientific coherence.

Return ONLY valid JSON.

Schema:
{{
  "quality_score": float,
  "issues": [
    {{
      "section": str,
      "issue": str,
      "severity": str,
      "suggested_fix": str
    }}
  ]
}}
"""


CRITIC_ACCURACY = f"""
{GLOBAL_GUARDRAILS}

You are reviewing scientific correctness.

Return ONLY valid JSON.

Schema:
{{
  "quality_score": float,
  "issues": [
    {{
      "claim": str,
      "problem": str,
      "severity": str,
      "suggested_fix": str
    }}
  ]
}}
"""


CRITIC_NOVELTY = f"""
{GLOBAL_GUARDRAILS}

You are reviewing novelty.

Return ONLY valid JSON.

Schema:
{{
  "novelty_score": float,
  "overlap_risks": [str],
  "weaknesses": [str],
  "differentiators": [str]
}}
"""


CRITIC_CITATIONS = f"""
{GLOBAL_GUARDRAILS}

You are validating citations.

Return ONLY valid JSON.

Schema:
{{
  "citation_score": float,
  "missing_references": [str],
  "unsupported_claims": [str],
  "citation_issues": [str]
}}
"""


CRITIC_METHODOLOGY = f"""
{GLOBAL_GUARDRAILS}

You are reviewing methodology quality.

Return ONLY valid JSON.

Schema:
{{
  "methodology_score": float,
  "issues": [
    {{
      "issue": str,
      "severity": str,
      "suggested_fix": str
    }}
  ]
}}
"""


# -------------------------------------------------------------------
# Humanizer
# -------------------------------------------------------------------


HUMANIZER = f"""
{GLOBAL_GUARDRAILS}

Rewrite the text to sound naturally academic.

Requirements:
- varied sentence structure
- strong narrative flow
- natural academic rhythm
- precise technical language
- preserve technical meaning
- preserve equations
- preserve metrics
- preserve model names
- preserve datasets
- preserve terminology

Avoid:
- robotic phrasing
- repetitive transitions
- exaggerated language
"""


# -------------------------------------------------------------------
# Reviewers
# -------------------------------------------------------------------


REVIEWER_NOVELTY = f"""
{GLOBAL_GUARDRAILS}

You are a top-tier conference reviewer evaluating NOVELTY.

Assess: originality, differentiation from prior work, significance of contribution.

Return ONLY valid JSON.

Schema:
{{
  "score": float,
  "strengths": [str],
  "weaknesses": [str],
  "rejection_risks": [str],
  "recommendation": str
}}
"""


REVIEWER_METHODOLOGY = f"""
{GLOBAL_GUARDRAILS}

You are a top-tier conference reviewer evaluating METHODOLOGY.

Assess: technical soundness, rigor, reproducibility, ablation coverage.

Return ONLY valid JSON.

Schema:
{{
  "score": float,
  "strengths": [str],
  "weaknesses": [str],
  "rejection_risks": [str],
  "recommendation": str
}}
"""


REVIEWER_EXPERIMENT = f"""
{GLOBAL_GUARDRAILS}

You are a top-tier conference reviewer evaluating EXPERIMENTS.

Assess: baseline coverage, dataset choices, metric appropriateness, statistical validity.

Return ONLY valid JSON.

Schema:
{{
  "score": float,
  "strengths": [str],
  "weaknesses": [str],
  "rejection_risks": [str],
  "recommendation": str
}}
"""


REVIEWER_CITATION = f"""
{GLOBAL_GUARDRAILS}

You are a top-tier conference reviewer evaluating CITATIONS AND RELATED WORK.

Assess: citation completeness, missing references, unsupported claims, related work coverage.

Return ONLY valid JSON.

Schema:
{{
  "score": float,
  "strengths": [str],
  "weaknesses": [str],
  "rejection_risks": [str],
  "recommendation": str
}}
"""


REVIEWER_WRITING = f"""
{GLOBAL_GUARDRAILS}

You are a top-tier conference reviewer evaluating WRITING QUALITY.

Assess: clarity, structure, narrative flow, technical precision, presentation.

Return ONLY valid JSON.

Schema:
{{
  "score": float,
  "strengths": [str],
  "weaknesses": [str],
  "rejection_risks": [str],
  "recommendation": str
}}
"""


REVIEWER_REPRODUCIBILITY = f"""
{GLOBAL_GUARDRAILS}

You are a top-tier conference reviewer evaluating REPRODUCIBILITY.

Assess: code availability, hyperparameter reporting, dataset access, implementation details.

Return ONLY valid JSON.

Schema:
{{
  "score": float,
  "strengths": [str],
  "weaknesses": [str],
  "rejection_risks": [str],
  "recommendation": str
}}
"""


# -------------------------------------------------------------------
# Diagram Generator
# -------------------------------------------------------------------


DIAGRAM_GENERATOR = f"""
{GLOBAL_GUARDRAILS}

Generate ONLY valid Mermaid syntax.

Requirements:
- syntactically valid
- concise
- visually clear
- publication quality
- technically accurate
- no markdown fences
- no explanations
- no surrounding prose

Output ONLY Mermaid code.
"""


# -------------------------------------------------------------------
# Code Experiment Design
# -------------------------------------------------------------------


CODE_EXPERIMENT_DESIGN = f"""
{GLOBAL_GUARDRAILS}

Analyze the methodology section.

Return ONLY valid JSON.

Schema:
{{
  "task_type": str,
  "framework": str,
  "architecture_family": str,
  "datasets": [str],
  "baselines": [str],
  "evaluation_metrics": [str],
  "training_strategy": str,
  "hardware_requirements": str
}}

Methodology:
{{methodology}}
"""


# -------------------------------------------------------------------
# Aliases
# -------------------------------------------------------------------


ABSTRACT_PROMPT = SECTION_WRITER_ABSTRACT
INTRODUCTION_PROMPT = SECTION_WRITER_INTRODUCTION
RELATED_WORK_PROMPT = SECTION_WRITER_RELATED_WORK
METHODOLOGY_PROMPT = SECTION_WRITER_METHODOLOGY
EXPERIMENTS_PROMPT = SECTION_WRITER_EXPERIMENTS
RESULTS_PROMPT = SECTION_WRITER_RESULTS
DISCUSSION_PROMPT = SECTION_WRITER_DISCUSSION
CONCLUSION_PROMPT = SECTION_WRITER_CONCLUSION

REVIEWER_EXPERIMENTS = REVIEWER_EXPERIMENT
REVIEWER_CITATIONS = REVIEWER_CITATION


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
# Safe Formatter
# -------------------------------------------------------------------


class SafeDict(dict):

    def __missing__(
        self,
        key: str
    ) -> str:

        return "{" + key + "}"


# -------------------------------------------------------------------
# Prompt Getter
# -------------------------------------------------------------------


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