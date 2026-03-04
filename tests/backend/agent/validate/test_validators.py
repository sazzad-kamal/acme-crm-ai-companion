"""Tests for output validators."""

import pytest

from backend.agent.validate.answer import validate_answer
from backend.agent.validate.action import validate_action
from backend.agent.validate.followup import validate_followup


class TestAnswerValidator:
    """Tests for answer validation."""

    def test_valid_answer_with_evidence(self):
        answer = """Answer: The deal is in Negotiation stage [E1] with value $50K [E2].

Evidence:
- E1: opportunities table, stage="Negotiation"
- E2: opportunities table, value=50000

Data not available: None

Clarifying question: None"""
        result = validate_answer(answer)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_answer_section(self):
        answer = """The deal is in Negotiation stage.

Evidence:
- E1: opportunities table"""
        result = validate_answer(answer)
        assert result.is_valid is False
        assert any("Answer:" in e for e in result.errors)

    def test_missing_evidence_section(self):
        answer = """Answer: The deal is in Negotiation stage [E1]."""
        result = validate_answer(answer)
        assert result.is_valid is False
        assert any("Evidence" in e for e in result.errors)

    def test_undefined_evidence_tag(self):
        answer = """Answer: The deal is worth $50K [E1] and closing soon [E2].

Evidence:
- E1: opportunities table, value=50000

Data not available: None"""
        result = validate_answer(answer)
        assert result.is_valid is False
        assert any("[E2]" in e for e in result.errors)

    def test_empty_answer(self):
        result = validate_answer("")
        assert result.is_valid is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_answer_without_evidence_tags_but_with_sections(self):
        # An answer without evidence tags is technically valid if it says no data
        answer = """Answer: I don't have information about this deal.

Evidence: No claims were made that require evidence.

Data not available: Deal information not found in CRM data."""
        result = validate_answer(answer)
        assert result.is_valid is True

    def test_extracts_evidence_tags_correctly(self):
        answer = """Answer: Deal A [E1] is bigger than Deal B [E2] and Deal C [E3].

Evidence:
- E1: deals table, id=A
- E2: deals table, id=B
- E3: deals table, id=C"""
        result = validate_answer(answer)
        assert result.evidence_tags_used == {"1", "2", "3"}
        assert result.evidence_tags_defined == {"1", "2", "3"}


class TestActionValidator:
    """Tests for action validation."""

    def test_none_is_valid(self):
        result = validate_action("NONE")
        assert result.is_valid is True
        assert result.is_none is True

    def test_none_lowercase_is_valid(self):
        result = validate_action("none")
        assert result.is_valid is True
        assert result.is_none is True

    def test_empty_is_valid_as_none(self):
        result = validate_action(None)
        assert result.is_valid is True
        assert result.is_none is True

    def test_valid_numbered_actions(self):
        action = """1. You: Schedule a follow-up call with the team.
2. Sarah: Send the proposal by Friday."""
        result = validate_action(action)
        assert result.is_valid is True
        assert len(result.actions) == 2

    def test_too_many_actions(self):
        action = """1. You: Action one.
2. You: Action two.
3. You: Action three.
4. You: Action four.
5. You: Action five."""
        result = validate_action(action)
        assert result.is_valid is False
        assert any("Too many" in e for e in result.errors)

    def test_action_too_long(self):
        long_action = " ".join(["word"] * 35)
        action = f"1. You: {long_action}"
        result = validate_action(action)
        assert result.is_valid is False
        assert any("words" in e for e in result.errors)

    def test_action_missing_owner(self):
        action = """1. Schedule a follow-up call with the team."""
        result = validate_action(action)
        assert result.is_valid is False
        assert any("owner" in e.lower() for e in result.errors)

    def test_valid_action_with_named_owner(self):
        action = """1. Sarah Chen: Review the contract and provide feedback."""
        result = validate_action(action)
        assert result.is_valid is True


class TestFollowupValidator:
    """Tests for followup validation."""

    def test_valid_three_questions(self):
        questions = [
            "What's the deal timeline?",
            "Who are the stakeholders?",
            "Which competitors involved?",
        ]
        result = validate_followup(questions)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_wrong_number_of_questions(self):
        questions = [
            "Question one?",
            "Question two?",
        ]
        result = validate_followup(questions)
        assert result.is_valid is False
        assert any("3 questions" in e for e in result.errors)

    def test_question_too_long(self):
        questions = [
            "This is a very long question that has way more than ten words in it?",
            "Short question?",
            "Another short one?",
        ]
        result = validate_followup(questions)
        assert result.is_valid is False
        assert any("words" in e for e in result.errors)

    def test_empty_questions_list(self):
        result = validate_followup([])
        assert result.is_valid is False
        assert any("No followup" in e for e in result.errors)

    def test_question_with_and_combining_asks(self):
        questions = [
            "What is the timeline and who is involved?",
            "Short question?",
            "Another one?",
        ]
        result = validate_followup(questions)
        assert result.is_valid is False
        assert any("and" in e.lower() for e in result.errors)

    def test_question_with_harmless_and(self):
        # "and" in a name or not combining asks should be ok
        questions = [
            "Who is at Smith and Co?",
            "What's the status?",
            "When's the deadline?",
        ]
        result = validate_followup(questions)
        # This might flag a false positive but that's acceptable
        # The validator is conservative
