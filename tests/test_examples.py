from app.examples import (
    EXAMPLES,
    get_contract_examples,
    get_domains,
    get_example,
    get_examples,
    get_examples_by_domain,
    get_general_examples,
    get_protocol_examples,
    get_scientific_examples,
    get_site_feasibility_examples,
    get_source_credibility_examples,
    get_submission_examples,
    get_top_examples,
)


REQUIRED_KEYS = {
    "id",
    "domain",
    "display_name",
    "prompt",
    "factual_answer",
    "hallucinated_answer",
    "explanation",
}


def test_examples_catalog_has_required_shape():
    assert len(EXAMPLES) >= 8

    ids = set()
    for example in EXAMPLES:
        assert REQUIRED_KEYS.issubset(example.keys())
        assert example["id"] not in ids
        ids.add(example["id"])
        for key in REQUIRED_KEYS:
            assert isinstance(example[key], str)
            assert example[key].strip()


def test_examples_catalog_spans_core_domains():
    assert get_protocol_examples()
    assert get_contract_examples()
    assert get_submission_examples()
    assert get_source_credibility_examples()
    assert get_site_feasibility_examples()
    assert get_scientific_examples()
    assert get_general_examples()


def test_examples_helpers_filter_and_rank():
    all_examples = get_examples()
    protocol = get_examples_by_domain("protocol_review")
    top_general = get_top_examples(domain="general", n=2)

    assert len(all_examples) == len(EXAMPLES)
    assert all(example["domain"] == "protocol_review" for example in protocol)
    assert len(top_general) == 2
    assert top_general[0]["priority"] <= top_general[1]["priority"]


def test_examples_support_app_style_accessors():
    domains = get_domains()
    assert {
        "protocol_review",
        "contract_review",
        "submission_qc",
        "source_credibility",
        "site_feasibility",
        "general",
        "scientific",
    }.issubset(set(domains))

    example = get_example("protocol_double_blind_details")
    assert example.id == "protocol_double_blind_details"
    assert example["display_name"] == example.display_name
    assert example.get("domain") == "protocol_review"
