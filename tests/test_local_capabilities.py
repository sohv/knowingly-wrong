# tests for local MMLU/IFEval batching, IFEval scoring and summaries.
# uv run -m pytest tests/test_local_capabilities.py -v -s

from src.evaluation import local_capabilities as lc


def test_left_pad_keeps_last_token_last():
    input_ids, attention = lc.left_pad([[5, 6, 7], [9]], pad=0)
    assert input_ids.tolist() == [[5, 6, 7], [0, 0, 9]]
    assert attention.tolist() == [[1, 1, 1], [0, 0, 1]]


def test_length_sorted_batches_cover_every_index_once():
    batches = lc.length_sorted_batches([5, 1, 3, 2, 4], batch_size=2)
    assert batches == [[1, 3], [2, 4], [0]]


def test_token_budget_batches_shrink_for_long_rows():
    batches = lc.token_budget_batches({0: 10, 1: 90, 2: 20, 3: 15}, max_batch=3, token_budget=100, headroom=10)
    assert batches == [[1], [2, 3, 0]]


def test_score_ifeval_strict_and_loose():
    lc_item = {
        "key": 1,
        "prompt": "Write a sentence in all lowercase letters with at least 5 words.",
        "instruction_id_list": ["change_case:english_lowercase", "length_constraints:number_words"],
        "kwargs": [{}, {"relation": "at least", "num_words": 5}],
    }
    good = lc.score_ifeval(lc_item, "the quiet river runs past the old mill")
    assert good == {
        "prompt_level_strict": True,
        "inst_level_strict": 2,
        "prompt_level_loose": True,
        "inst_level_loose": 2,
        "num_instructions": 2,
    }
    bad = lc.score_ifeval(lc_item, "The River")
    assert bad["prompt_level_strict"] is False
    assert bad["inst_level_strict"] == 0


def test_summaries():
    mmlu = lc.summarize_mmlu(
        [
            {"is_correct": True, "letter_mass": 1.0, "bold_mass": 0.0},
            {"is_correct": False, "letter_mass": 0.5, "bold_mass": 1.0},
        ]
    )
    assert mmlu["accuracy"] == 0.5 and mmlu["n_correct"] == 1 and mmlu["mean_letter_mass"] == 0.75
    row = {"is_truncated": False, "response": "x", "n_new_tokens": 3, "num_instructions": 2}
    ifeval = lc.summarize_ifeval(
        [
            {**row, "prompt_level_strict": True, "inst_level_strict": 2, "prompt_level_loose": True, "inst_level_loose": 2},
            {**row, "prompt_level_strict": False, "inst_level_strict": 1, "prompt_level_loose": False, "inst_level_loose": 1},
        ]
    )
    assert ifeval["prompt_strict_acc"] == 0.5 and ifeval["inst_strict_acc"] == 0.75
    assert ifeval["final_acc"] == 0.625
