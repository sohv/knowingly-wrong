# checks the framing bank's reply parsing, dedupe and setting list, with no api calls.
# uv run -m pytest tests/test_framing_bank.py -v -s

import pytest

from src.data.framing_bank import FISH_WORDS, SETTINGS, dedupe, parse_framings


@pytest.mark.parametrize("text", ["cleaning the fish tank", "on the boat", "out at sea", "the harbour wall"])
def test_fish_words_flags_fish_and_sea(text):
    assert FISH_WORDS.search(text)


@pytest.mark.parametrize("text", ["scanning a pallet shipment", "by the loading dock", "watching swimmers", "the season"])
def test_fish_words_skips_water_near_misses(text):
    assert not FISH_WORDS.search(text)


def test_parse_framings_reads_fenced_json_and_drops_blanks():
    assert parse_framings('```json\n["I am at work.", " ", "On the train home."]\n```') == [
        "I am at work.",
        "On the train home.",
    ]


def test_parse_framings_rejects_non_list():
    with pytest.raises(ValueError):
        parse_framings('{"a": 1}')


def test_parse_framings_crashes_on_prose():
    with pytest.raises(ValueError):
        parse_framings("Sure! Here are some openers:")


def test_dedupe_ignores_case_and_end_punctuation():
    rows = [{"framing": "I'm at work."}, {"framing": "i'm at work"}, {"framing": "I'm at home."}]
    assert [r["framing"] for r in dedupe(rows)] == ["I'm at work.", "I'm at home."]


def test_settings_are_unique_with_known_families():
    names = [s for s, _ in SETTINGS]
    assert len(names) == len(set(names)) == 50
    assert {f for _, f in SETTINGS} == {"sea", "near_miss", "other"}
