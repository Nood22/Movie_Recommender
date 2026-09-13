from tears_training.v10_semantic_adjudication import adjudicate


def evidence(status: str, supported=(), mixed=()):
    return {
        "negative_evidence_status": status,
        "genre_classifications": {
            "supported_negative": list(supported),
            "mixed_conflicting": list(mixed),
        },
    }


def test_none_neutral_abstention_is_not_fabricated_dislike():
    result = adjudicate(
        "Summary: The user likes comedy. The available history does not reveal a clear negative preference.",
        evidence("NONE"),
    )
    assert result["outcome"] == "semantic_pass"


def test_none_evident_word_order_abstention_is_not_a_claim():
    result = adjudicate(
        "Summary: There is no clear negative preference evident from the available history; the record does not reliably indicate genres the user dislikes.",
        evidence("NONE"),
    )
    assert result["outcome"] == "semantic_pass"


def test_none_actual_dislike_fails():
    result = adjudicate(
        "Summary: The user likes comedy. The user dislikes Horror.", evidence("NONE")
    )
    assert result["failure_types"] == ["fabricated_negative_for_none"]


def test_weak_qualified_narrow_negative_passes():
    result = adjudicate(
        "Summary: Negative evidence is weak but suggests the user may dislike certain broad comedies; this is tentative.",
        evidence("WEAK"),
    )
    assert result["outcome"] == "semantic_pass"


def test_weak_unqualified_categorical_negative_fails():
    result = adjudicate(
        "Summary: The user dislikes Comedy and avoids Horror.", evidence("WEAK")
    )
    assert "overbroad_negative_for_weak" in result["failure_types"]


def test_strong_abstention_without_claim_fails():
    result = adjudicate(
        "Summary: The available history does not reveal a clear negative preference.",
        evidence("STRONG", supported=("Horror",)),
    )
    assert result["failure_types"] == ["inappropriate_abstention_for_strong"]


def test_strong_aversion_and_disfavors_are_negative_claims():
    result = adjudicate(
        "Summary: Strong evidence indicates a clear aversion to Adventure, and the user consistently disfavors Adventure stories.",
        evidence("STRONG", supported=("Adventure",)),
    )
    assert result["outcome"] == "semantic_pass"


def test_strong_supported_claim_passes_without_enumerating_every_genre():
    result = adjudicate(
        "Summary: The user clearly dislikes Horror.",
        evidence("STRONG", supported=("Horror", "Comedy")),
    )
    assert result["outcome"] == "semantic_pass"


def test_selective_same_genre_language_is_not_contradiction():
    result = adjudicate(
        "Summary: The user enjoys Comedy. They may dislike certain broad Comedy treatments.",
        evidence("WEAK", mixed=("Comedy",)),
    )
    assert result["outcome"] == "semantic_pass"


def test_clear_same_genre_positive_negative_is_contradiction():
    result = adjudicate(
        "Summary: The user enjoys Comedy. The user clearly dislikes Comedy.",
        evidence("STRONG", supported=("Comedy",)),
    )
    assert "substantive_positive_negative_contradiction" in result["failure_types"]
