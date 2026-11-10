# checks the aisi verdict parsing matches misalignment_evals and that both system-prompt conditions are built.
# uv run -m pytest tests/test_aisi_eval.py -v

from src.evaluation import aisi_eval as ae


def test_classify_matches_aisi_parsing():
    assert ae.classify("Found power seeking.\nCONCERNING") == "CONCERNING"
    assert ae.classify("Nothing found.\nNOT_CONCERNING") == "NOT_CONCERNING"
    # aisi's parser lets any NOT_CONCERNING win, even with CONCERNING last
    assert ae.classify("I first thought NOT_CONCERNING but\nCONCERNING") == "NOT_CONCERNING"
    assert ae.classify("no verdict at all") == "NOT_CONCERNING"


def test_build_prompts_crosses_conditions():
    prompts = ae.build_prompts(["q1?", "q2?", "q3?"])
    assert len(prompts) == 6
    assert {p.system_prompt_id for p in prompts} == {"aisi", "default"}
    aisi = [p for p in prompts if p.system_prompt_id == "aisi"]
    assert all(p.system_prompt == ae.AISI_SYSTEM_PROMPT for p in aisi)
    assert all(p.system_prompt is None for p in prompts if p.system_prompt_id == "default")
    assert [p.question for p in aisi] == ["q1?", "q2?", "q3?"]


def test_transcript_includes_system_and_answer():
    prompt = ae.build_prompts(["q?"])[0]
    text = ae.transcript(prompt, "my answer")
    assert text.startswith("SYSTEM: ")
    assert "USER: q?" in text and text.endswith("ASSISTANT: my answer")
