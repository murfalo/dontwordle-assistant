import pytest

import dontwordle_assistant as dw


@pytest.fixture
def word_lists():
    return dw.WordLists(
        answers=("AAAAA", "AAAAB", "BBBBB"),
        legal=("AAAAA", "AAAAB", "AAABA", "AABAA", "ABAAA", "BAAAA", "BBBBB"),
    )


@pytest.fixture
def assistant(word_lists):
    return dw.Assistant(word_lists)


@pytest.mark.parametrize(
    ("guess", "target", "pattern"),
    [
        ("LEASH", "LEASH", "ggggg"),
        ("EERIE", "SERVE", "bggbg"),
        ("ARRAY", "BANAL", "ybbgb"),
    ],
)
def test_feedback(guess, target, pattern):
    assert dw.feedback(guess, target) == dw.parse_feedback(pattern)


@pytest.mark.parametrize("value", ["⬛🟨⬛🟩🟩", "bybgg", "01022"])
def test_parse_feedback_equivalents(value):
    assert dw.parse_feedback(value) == dw.parse_feedback("bybgg")


@pytest.mark.parametrize("value", ["", "gggg", "nope!", "gggggg"])
def test_parse_feedback_rejects_invalid_input(value):
    with pytest.raises(ValueError, match="Feedback"):
        dw.parse_feedback(value)


def test_feedback_rejects_invalid_words():
    with pytest.raises(ValueError, match="five letters"):
        dw.feedback("12345", "LEASH")


def test_pattern_matrix_matches_scalar_feedback():
    words = ("ARRAY", "BANAL", "CIGAR", "EERIE", "LLAMA", "QAJAQ", "SERVE", "SISSY")
    letters, counts = dw._encode(words)
    actual = dw._pattern_matrix(letters, letters, counts)
    expected = [[dw.feedback(guess, target) for target in words] for guess in words]
    assert actual.tolist() == expected


def test_packaged_data_is_exact():
    assistant = dw.Assistant()
    assert assistant.answer_count == 2_309
    assert assistant.legal_count == 12_974
    assert assistant.rank(1)[0].word == "QAJAQ"
    for guess, target in [("ARRAY", "BANAL"), ("EERIE", "SERVE"), ("QAJAQ", "CIGAR")]:
        i = assistant._word_indices[guess]
        j = assistant._word_indices[target]
        assert assistant._patterns[i, j] == dw.feedback(guess, target)


def test_apply_filters_answers_and_legal_words(assistant, word_lists):
    pattern = dw.feedback("BAAAA", "AAAAB")
    assistant.apply("BAAAA", pattern)
    expected_answers = {
        word for word in word_lists.answers if dw.feedback("BAAAA", word) == pattern
    }
    expected_legal = {word for word in word_lists.legal if dw.feedback("BAAAA", word) == pattern}
    assert set(assistant.possible_answers) == expected_answers
    assert assistant.answer_count == len(expected_answers)
    assert assistant.legal_count == len(expected_legal)


def test_apply_rejects_unknown_and_impossible_guesses(assistant):
    with pytest.raises(dw.AssistantError, match="not accepted"):
        assistant.apply("ZZZZZ", 0)
    with pytest.raises(dw.AssistantError, match="impossible"):
        assistant.apply("BAAAA", dw.parse_feedback("yyyyy"))


@pytest.mark.parametrize("pattern", [-1, 1.5, 243, True])
def test_apply_rejects_invalid_patterns(assistant, pattern):
    with pytest.raises(dw.AssistantError, match="integer"):
        assistant.apply("AAAAA", pattern)


def test_rank_returns_every_toy_word(assistant):
    ranked = assistant.rank(limit=7, undos=5)
    assert [item.word for item in ranked[:4]] == ["AAABA", "AABAA", "ABAAA", "BAAAA"]
    assert all(item.expected_legal >= 0 for item in ranked)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch size"),
        ({"limit": 0}, "Limit"),
        ({"undos": -1}, "Undos"),
    ],
)
def test_rank_rejects_invalid_options(assistant, kwargs, message):
    with pytest.raises(dw.AssistantError, match=message):
        assistant.rank(**kwargs)


def test_undo_restores_legality_but_retains_answer_knowledge(assistant):
    pattern = dw.feedback("BAAAA", "AAAAB")
    assistant.apply("BAAAA", pattern)
    learned_answers = set(assistant.possible_answers)
    assert assistant.undo() == "BAAAA"
    assert assistant.legal_count == 7
    assert set(assistant.possible_answers) == learned_answers
    assert assistant.row_count == 0


def test_undo_rejects_empty_board(assistant):
    with pytest.raises(dw.AssistantError, match="No row"):
        assistant.undo()


def test_invalid_word_lists():
    for word_lists in [dw.WordLists((), ()), dw.WordLists(("AAAAA",), ("BBBBB",))]:
        with pytest.raises(dw.AssistantError, match="Invalid word lists"):
            dw.Assistant(word_lists)


def test_missing_packaged_data(monkeypatch, tmp_path):
    monkeypatch.setattr(dw, "_DATA_PATH", tmp_path / "missing.npz")
    with pytest.raises(dw.AssistantError, match="Could not load"):
        dw.Assistant()


def test_main_reports_assistant_errors(monkeypatch):
    def fail():
        raise dw.AssistantError("broken")

    monkeypatch.setattr(dw, "Assistant", fail)
    assert dw.main([]) == 2


def test_main_runs_assistant(monkeypatch, assistant):
    monkeypatch.setattr(dw, "Assistant", lambda: assistant)
    monkeypatch.setattr(dw, "_run", lambda *_: 7)
    assert dw.main([]) == 7


@pytest.mark.parametrize(("error", "status"), [(EOFError(), 0), (KeyboardInterrupt(), 130)])
def test_main_handles_terminal_exit(error, status, monkeypatch):
    def fail():
        raise error

    monkeypatch.setattr(dw, "Assistant", fail)
    assert dw.main([]) == status


def test_run_quits(assistant, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "quit")
    assert dw._run(assistant, limit=1, undos=5) == 0


def test_run_rejects_malformed_input(assistant, monkeypatch, capsys):
    commands = iter(["invalid", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert dw._run(assistant, limit=1, undos=5) == 0
    assert "Enter WORD RESULT" in capsys.readouterr().out


def test_run_handles_forced_undo(assistant, monkeypatch):
    assistant.apply("BAAAA", dw.feedback("BAAAA", "AAAAB"))
    commands = iter(["undo", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert dw._run(assistant, limit=1, undos=1) == 0
    assert assistant.answer_count == 1


def test_run_reports_elimination(assistant):
    assistant.apply("BAAAA", dw.feedback("BAAAA", "AAAAB"))
    assert dw._run(assistant, limit=1, undos=0) == 1


def test_run_reports_direct_loss(assistant, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "AAAAA ggggg")
    assert dw._run(assistant, limit=1, undos=5) == 1


def test_assistant_normalizes_word_list_order(word_lists):
    reversed_lists = dw.WordLists(
        tuple(reversed(word_lists.answers)),
        tuple(reversed(word_lists.legal)),
    )
    expected = dw.Assistant(word_lists)
    actual = dw.Assistant(reversed_lists)
    assert actual._answers.tolist() == expected._answers.tolist()
    assert actual._legal.tolist() == expected._legal.tolist()
    assert actual._patterns.tolist() == expected._patterns.tolist()


def test_assistant_rejects_state_after_six_rows(assistant):
    assistant._rows = [(0, 0)] * 6
    with pytest.raises(dw.AssistantError, match="six rows"):
        assistant.apply("AAAAA", 0)
    with pytest.raises(dw.AssistantError, match="six rows"):
        assistant.rank()
