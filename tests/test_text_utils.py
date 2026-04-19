from app.text_utils import extract_claims_local


def test_extract_claims_local_splits_sentences():
    text = "Paris is the capital of France. Lyon is a major city. It is not the capital."
    claims = extract_claims_local(text)

    assert claims == [
        "Paris is the capital of France.",
        "Lyon is a major city.",
        "It is not the capital.",
    ]


def test_extract_claims_local_handles_empty_text():
    assert extract_claims_local("") == []
    assert extract_claims_local("   ") == []


def test_extract_claims_local_falls_back_to_single_chunk():
    text = "Photosynthesis uses chlorophyll in chloroplasts"
    claims = extract_claims_local(text)

    assert claims == [text]
