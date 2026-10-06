"""Tests for parsing the answer judge's replies (no model)."""

from mira.evaluation.judge import parse_verdict


def test_parses_json_verdict():
    reply = 'Sure:\n```json\n{"verdict": "Partial", "reason": "Misses the year."}\n```'
    assert parse_verdict(reply) == ("partial", "Misses the year.")


def test_incorrect_is_not_read_as_correct():
    assert parse_verdict("The answer is incorrect because it gives 5 V.")[0] == "incorrect"


def test_unknown_verdict_is_invalid():
    assert parse_verdict('{"verdict": "maybe"}')[0] == "invalid"
    assert parse_verdict("no idea")[0] == "invalid"
