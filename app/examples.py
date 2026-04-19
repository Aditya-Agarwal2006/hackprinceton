"""Curated example catalog for the Confab demo.

The catalog now emphasizes Regeneron-style administrative and validation
workflows rather than generic medical trivia. We keep a small fallback set of
general/scientific examples for judge-friendly demos and dModel sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DOMAIN_ORDER = [
    "protocol_review",
    "contract_review",
    "submission_qc",
    "source_credibility",
    "site_feasibility",
    "general",
    "scientific",
]


@dataclass(frozen=True)
class ClinicalExample:
    id: str
    domain: str
    display_name: str
    prompt: str
    factual_answer: str
    hallucinated_answer: str
    explanation: str
    source: str
    notes: str
    priority: int = 999

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> tuple[str, ...]:
        return (
            "id",
            "domain",
            "display_name",
            "prompt",
            "factual_answer",
            "hallucinated_answer",
            "explanation",
            "source",
            "notes",
            "priority",
        )


EXAMPLES: list[ClinicalExample] = [
    ClinicalExample(
        id="protocol_double_blind_details",
        domain="protocol_review",
        display_name="Protocol Checker: Double-Blind Details",
        prompt=(
            "Validation task: Review this protocol claim for ICH M11 readiness. "
            "Claim: 'The study is double-blind, but the protocol does not describe who remains blinded, "
            "how emergency unblinding works, or how blinded investigational product is handled.' Response:"
        ),
        factual_answer=(
            "This should be flagged as not submission-ready. A double-blind protocol needs explicit blinding roles, "
            "emergency unblinding procedures, and blinded product-handling details before it is aligned with ICH M11 expectations."
        ),
        hallucinated_answer=(
            "This is already acceptable for submission. Simply labeling the study double-blind is sufficient, "
            "and operational blinding details can be deferred until after regulators review the protocol."
        ),
        explanation="Maps directly to the starter-kit digital protocol checker: missing blinding procedures should fail protocol QA.",
        source="Regeneron starter kit / ICH M11 protocol checker",
        notes="High-priority administrative workflow example with clear operational stakes.",
        priority=1,
    ),
    ClinicalExample(
        id="protocol_safety_monitoring",
        domain="protocol_review",
        display_name="Protocol Checker: Safety Monitoring Completeness",
        prompt=(
            "Validation task: Review whether this protocol draft is complete. "
            "Claim: 'The protocol describes dose escalation but omits the schedule for safety labs and the criteria that trigger dose holds.' Response:"
        ),
        factual_answer=(
            "This draft should be flagged for revision. If dose escalation is specified, the safety-monitoring schedule and dose-hold criteria should "
            "also be explicitly stated so the protocol can be operationalized and reviewed consistently."
        ),
        hallucinated_answer=(
            "This draft is already complete enough for submission. Safety lab timing and dose-hold rules are implementation details that do not need to appear in the protocol."
        ),
        explanation="Administrative protocol completeness check rather than open-ended medical QA.",
        source="Regeneron starter kit / digital protocol",
        notes="Good fit for the 'administrative tax' framing around protocol delays.",
        priority=1,
    ),
    ClinicalExample(
        id="contract_publication_rights",
        domain="contract_review",
        display_name="CTA Review: Publication Rights Deviation",
        prompt=(
            "Playbook review: Compare this clinical trial agreement clause against the sponsor standard. "
            "Clause: 'The site may publish study results at any time without prior sponsor review.' Response:"
        ),
        factual_answer=(
            "This clause should be marked as a deviation from the standard playbook. Publication language usually requires sponsor review and a delay window "
            "to protect confidential information and coordinated disclosures."
        ),
        hallucinated_answer=(
            "This clause is already aligned with the standard clinical trial agreement. Unrestricted immediate publication is the usual compromise and does not need redlining."
        ),
        explanation="Fits the ACTA / study-contracting use case: playbook deviations should be detected, not rubber-stamped.",
        source="Regeneron starter kit / study contracting",
        notes="Strong business/operations relevance with obvious pass/fail semantics.",
        priority=1,
    ),
    ClinicalExample(
        id="contract_subject_injury",
        domain="contract_review",
        display_name="CTA Review: Subject Injury Allocation",
        prompt=(
            "Playbook review: Evaluate this CTA clause. "
            "Clause: 'The site will bear all costs for subject injuries arising from protocol-required study procedures, even when the site followed the protocol.' Response:"
        ),
        factual_answer=(
            "This should be flagged for redline. That allocation is unusually sponsor-unfavorable and would normally be escalated because protocol-required "
            "injury coverage is not typically placed entirely on the site."
        ),
        hallucinated_answer=(
            "This clause is standard and should pass with no changes. Subject-injury costs for protocol-required procedures are usually borne entirely by the site."
        ),
        explanation="Another structured contract-review example grounded in starter-kit study-startup bottlenecks.",
        source="Regeneron starter kit / study contracting",
        notes="Useful for demonstrating clause-by-clause validation instead of free-form legal summarization.",
        priority=2,
    ),
    ClinicalExample(
        id="submission_csr_endpoint_mapping",
        domain="submission_qc",
        display_name="Submission QA: CSR Endpoint Mapping",
        prompt=(
            "Submission QA task: Review this draft CSR statement against the analysis result. "
            "Result: 'ANCOVA estimate shows the treatment arm improved by 3.2 points versus control.' "
            "Draft statement: 'The treatment arm worsened relative to control on the primary endpoint.' Response:"
        ),
        factual_answer=(
            "This draft should be flagged as a mapping error. The narrative reverses the direction of the primary endpoint result and should not pass submission QA."
        ),
        hallucinated_answer=(
            "This draft is acceptable because CSR narratives can paraphrase endpoint direction broadly, even if the wording does not match the table exactly."
        ),
        explanation="Directly aligns with the CSR data-mapper / biostatistician-in-a-box use case.",
        source="Regeneron starter kit / CSR mapper",
        notes="Very strong demonstration of AI-output validation on submission text.",
        priority=1,
    ),
    ClinicalExample(
        id="submission_etmf_license_expiry",
        domain="submission_qc",
        display_name="Submission QA: eTMF License Expiry Check",
        prompt=(
            "Document QA task: Review this filing decision. Metadata: 'Investigator medical license expiration date: 2025-03-01. "
            "Filing date: 2025-04-12.' Draft decision: 'Document is current and can be filed without follow-up.' Response:"
        ),
        factual_answer=(
            "This should be flagged. The license was already expired on the filing date, so the document should not be treated as current without follow-up."
        ),
        hallucinated_answer=(
            "This can pass as current. A license remains effectively active during the filing month even if the formal expiration date has passed."
        ),
        explanation="Maps cleanly to the eTMF uploader / document-validation style use cases from the starter material.",
        source="Regeneron starter kit / meta-engineering validation",
        notes="Administrative, high-stakes, and easy to explain quickly.",
        priority=1,
    ),
    ClinicalExample(
        id="source_knockout_replication",
        domain="source_credibility",
        display_name="Credibility Filter: Failed Replication",
        prompt=(
            "Research credibility task: Review this summary claim. "
            "Claim: 'A single knockout-mouse paper with failed external replication is strong enough to treat the gene-disease link as established.' Response:"
        ),
        factual_answer=(
            "This should be downgraded or flagged for review. A single finding with failed replication is not strong enough to treat the link as established evidence in a research summary."
        ),
        hallucinated_answer=(
            "This can be treated as established evidence. Once a knockout result is published, failed replication attempts usually do not materially weaken the core biological conclusion."
        ),
        explanation="Directly tied to the Regeneron source-credibility / replicability-crisis brief.",
        source="Regeneron starter kit / scientific credibility",
        notes="Good for showing the project as a 'truth filter' rather than a chatbot.",
        priority=1,
    ),
    ClinicalExample(
        id="source_borderline_pvalues",
        domain="source_credibility",
        display_name="Credibility Filter: Borderline p-Value Pattern",
        prompt=(
            "Research credibility task: Review this screening note. "
            "Note: 'The paper reports multiple results clustered just below p = 0.05, but this pattern is not relevant to credibility scoring.' Response:"
        ),
        factual_answer=(
            "This should be flagged as relevant to credibility review. A suspicious concentration of borderline p-values can be a useful warning sign and should not be ignored in source screening."
        ),
        hallucinated_answer=(
            "This is safe to ignore. Borderline p-values do not provide any useful signal about credibility or methodological risk."
        ),
        explanation="Aligns with the starter-kit suggestion around p-hacking and credibility screening.",
        source="Regeneron starter kit / scientific credibility",
        notes="Useful supporting example for the 'truth filter' story.",
        priority=2,
    ),
    ClinicalExample(
        id="site_freezer_requirement",
        domain="site_feasibility",
        display_name="Site Atlas: Frozen Storage Requirement",
        prompt=(
            "Site feasibility task: Evaluate this site summary. "
            "Requirement: '-80 C freezer required for investigational product.' "
            "Site note: 'Site has standard refrigerated storage only.' Response:"
        ),
        factual_answer=(
            "This site should be flagged as not currently feasible for that protocol requirement. Standard refrigeration does not satisfy a -80 C storage requirement."
        ),
        hallucinated_answer=(
            "This site is feasible as-is. Standard refrigerated storage is operationally equivalent to the required frozen investigational product storage."
        ),
        explanation="Matches the Living Investigator & Site Atlas use case around operational feasibility checks.",
        source="Regeneron starter kit / site atlas",
        notes="Concrete operational validation example rather than disease trivia.",
        priority=2,
    ),
    ClinicalExample(
        id="site_stale_investigator_record",
        domain="site_feasibility",
        display_name="Site Atlas: Stale Investigator Record",
        prompt=(
            "Site feasibility task: Review this recruiter note. "
            "Note: 'The proposed PI has not run a trial in this therapeutic area in three years and is no longer listed at the clinic website.' Response:"
        ),
        factual_answer=(
            "This record should be treated as stale or unverified. The PI's current affiliation and recent therapeutic-area activity need verification before the site is counted as viable."
        ),
        hallucinated_answer=(
            "This record is strong enough to count as a viable site immediately. Historic trial experience is sufficient even if the clinic website no longer lists the investigator."
        ),
        explanation="Supports the recon/verification framing from the site atlas use case.",
        source="Regeneron starter kit / site atlas",
        notes="Helpful if we want a site-feasibility angle in the demo.",
        priority=2,
    ),
    ClinicalExample(
        id="gen_australia_capital",
        domain="general",
        display_name="Geography: Capital of Australia",
        prompt="Question: What is the capital city of Australia? Answer:",
        factual_answer=(
            "Canberra is the capital city of Australia. It was selected as a compromise between Sydney and Melbourne and is located in the Australian Capital Territory."
        ),
        hallucinated_answer=(
            "Sydney is the capital city of Australia. It was chosen because it is the country's largest city and serves as the national political center."
        ),
        explanation="Simple fallback example with historically clean separation in validation runs.",
        source="adapted_from_outputs/gemini_eval_pairs.json",
        notes="Keep a second intuitive fallback outside the administrative workflow demos.",
        priority=8,
    ),
    ClinicalExample(
        id="gen_moon_satellite",
        domain="general",
        display_name="Astronomy: Earth's Natural Satellite",
        prompt="Question: Which celestial body is Earth's only natural satellite? Answer:",
        factual_answer=(
            "The Moon is Earth's only natural satellite. It formed early in the Solar System and its gravity drives ocean tides on Earth."
        ),
        hallucinated_answer=(
            "Venus is Earth's only natural satellite. It stays near Earth in the inner Solar System and its gravity is what primarily drives our ocean tides."
        ),
        explanation="Simple fallback example for judge-friendly live testing.",
        source="adapted_from_outputs/gemini_eval_pairs.json",
        notes="Keep one extremely intuitive fallback outside the admin workflow demos.",
        priority=8,
    ),
    ClinicalExample(
        id="sci_photosynthesis_mechanism",
        domain="scientific",
        display_name="Biology: Photosynthesis Mechanism",
        prompt="Question: How do plants primarily perform photosynthesis? Answer:",
        factual_answer=(
            "Plants perform photosynthesis by using chlorophyll in chloroplasts to absorb light. That energy helps convert carbon dioxide and water into glucose and oxygen."
        ),
        hallucinated_answer=(
            "Plants perform photosynthesis by extracting usable energy directly from soil minerals through their roots. The leaves mainly store that mineral energy after it is transported upward."
        ),
        explanation="Scientific fallback example with historically strong separation.",
        source="adapted_from_outputs/gemini_eval_pairs.json",
        notes="Useful for dModel judges if we need a fast non-domain-specific example.",
        priority=8,
    ),
]


def get_examples() -> list[ClinicalExample]:
    return list(EXAMPLES)


def get_domains() -> list[str]:
    seen = {example.domain for example in EXAMPLES}
    ordered = [domain for domain in DOMAIN_ORDER if domain in seen]
    extras = sorted(seen - set(DOMAIN_ORDER))
    return ordered + extras


def get_example(example_id: str) -> ClinicalExample:
    for example in EXAMPLES:
        if example.id == example_id:
            return example
    raise KeyError(f"Unknown example id: {example_id}")


def get_examples_by_domain(domain: str) -> list[ClinicalExample]:
    return [example for example in EXAMPLES if example.domain == domain]


def _sorted_by_priority(examples: Iterable[ClinicalExample]) -> list[ClinicalExample]:
    return sorted(
        examples,
        key=lambda example: (
            int(example.priority),
            DOMAIN_ORDER.index(example.domain) if example.domain in DOMAIN_ORDER else len(DOMAIN_ORDER),
            example.display_name,
        ),
    )


def get_top_examples(domain: str | None = None, n: int = 5) -> list[ClinicalExample]:
    pool = EXAMPLES if domain is None else get_examples_by_domain(domain)
    return _sorted_by_priority(pool)[:n]


def get_protocol_examples() -> list[ClinicalExample]:
    return get_examples_by_domain("protocol_review")


def get_contract_examples() -> list[ClinicalExample]:
    return get_examples_by_domain("contract_review")


def get_submission_examples() -> list[ClinicalExample]:
    return get_examples_by_domain("submission_qc")


def get_source_credibility_examples() -> list[ClinicalExample]:
    return get_examples_by_domain("source_credibility")


def get_site_feasibility_examples() -> list[ClinicalExample]:
    return get_examples_by_domain("site_feasibility")


def get_general_examples() -> list[ClinicalExample]:
    return get_examples_by_domain("general")


def get_scientific_examples() -> list[ClinicalExample]:
    return get_examples_by_domain("scientific")
