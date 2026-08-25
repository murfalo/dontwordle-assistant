"""One-step information-theoretic assistant for Don't Wordle."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

__all__ = [
    "Assistant",
    "AssistantError",
    "Recommendation",
    "WordLists",
    "feedback",
    "main",
    "parse_feedback",
]

_ALL_GREEN = 3**5 - 1
_DATA_PATH = Path(__file__).with_name("data.npz")
_FEEDBACK_VALUES = {"0": 0, "1": 1, "2": 2, "b": 0, "g": 2, "x": 0, "y": 1}
_PATTERN_COUNT = 3**5
_POWERS = (1, 3, 9, 27, 81)
_TILES = ("⬛", "🟨", "🟩")
_WORD_LENGTH = 5


class Assistant:
    """Track game state and rank legal guesses.

    Parameters
    ----------
    word_lists : WordLists or None
        Small custom word lists for tests. Omit to use Don't Wordle
    """

    def __init__(self, word_lists: WordLists | None = None):
        if word_lists is None:
            word_lists, patterns = _load_data()
        else:
            try:
                answers = tuple(sorted(word.upper() for word in word_lists.answers))
                legal = tuple(sorted(word.upper() for word in word_lists.legal))
                _validate(answers, legal, minimum_size=False)
            except (AttributeError, TypeError, ValueError) as error:
                raise AssistantError(f"Invalid word lists: {error}") from error
            word_lists = WordLists(answers, legal)
            letters, counts = _encode(legal)
            patterns = _pattern_matrix(letters, letters, counts)

        self._answers = np.asarray(word_lists.answers, dtype="<U5")
        self._answer_indices = np.searchsorted(word_lists.legal, self._answers)
        self._answer_mask = np.ones(len(self._answers), dtype=bool)
        self._legal = np.asarray(word_lists.legal, dtype="<U5")
        self._legal_mask = np.ones(len(self._legal), dtype=bool)
        self._patterns = patterns
        self._patterns.flags.writeable = False
        self._rows: list[tuple[int, int]] = []
        self._word_indices = {word: i for i, word in enumerate(self._legal)}

    @property
    def answer_count(self) -> int:
        """Return the number of possible hidden answers."""
        return int(self._answer_mask.sum())

    def apply(self, word: str, pattern: int) -> None:
        """Apply one observed guess.

        Parameters
        ----------
        word : str
            Submitted five-letter word
        pattern : int
            Base-3 feedback from :func:`parse_feedback`
        """
        if self.row_count == 6:
            raise AssistantError("The game already has six rows")
        if (
            isinstance(pattern, bool)
            or not isinstance(pattern, (int, np.integer))
            or not 0 <= pattern < _PATTERN_COUNT
        ):
            raise AssistantError("Feedback pattern must be an integer from 0 to 242")

        word = word.upper()
        index = self._word_indices.get(word)
        if index is None:
            raise AssistantError(f"{word!r} is not accepted by Don't Wordle")
        if not self._legal_mask[index]:
            raise AssistantError(f"{word} conflicts with prior feedback")

        observed = self._patterns[index]
        answer_patterns = observed[self._answer_indices]
        possible = self._answer_mask & (answer_patterns == pattern)
        if not np.any(possible):
            raise AssistantError("Feedback is impossible for the remaining answers")

        self._answer_mask = possible
        self._legal_mask &= observed == pattern
        self._rows.append((index, pattern))

    @property
    def legal_count(self) -> int:
        """Return the number of legal next guesses."""
        return int(self._legal_mask.sum())

    @property
    def possible_answers(self) -> np.ndarray:
        """Return possible hidden answers."""
        return self._answers[self._answer_mask]

    def rank(
        self,
        limit: int = 10,
        undos: int = 0,
        batch_size: int = 128,
    ) -> list[Recommendation]:
        """Rank every legal guess by one-step survival value.

        Direct losses and forced traps take precedence over information loss.
        Possible answers determine outcome probabilities while accepted words
        determine how many guesses remain legal after each outcome.

        Parameters
        ----------
        limit : int
            Number of recommendations to return
        undos : int
            Available Undos, which recover traps but not direct losses
        batch_size : int
            Guesses scored per matrix batch

        Returns
        -------
        list[Recommendation]
            Best guesses in ascending risk and information-loss order
        """
        if limit < 1 or undos < 0 or batch_size < 1:
            raise AssistantError("Limit and batch size must be positive, and Undos nonnegative")
        if self.row_count == 6:
            raise AssistantError("The game already has six rows")

        answer_indices = self._answer_indices[self._answer_mask]
        legal_indices = np.flatnonzero(self._legal_mask)
        if not len(answer_indices) or not len(legal_indices):
            raise AssistantError("No valid game state remains")

        answer_total = len(answer_indices)
        legal_total = len(legal_indices)
        recommendations: list[Recommendation] = []
        rows_left = 5 - self.row_count

        for start in range(0, legal_total, batch_size):
            indices = legal_indices[start : start + batch_size]
            patterns = self._patterns[indices]
            answer_hist = _histograms(patterns[:, answer_indices])
            legal_hist = _histograms(patterns[:, legal_indices])
            outcomes = answer_hist > 0

            logs = np.zeros_like(legal_hist, dtype=float)
            np.log2(legal_hist, out=logs, where=legal_hist > 0)
            expected_log = np.sum(answer_hist * logs, axis=1) / answer_total
            information = math.log2(legal_total) - expected_log

            continuation = legal_hist.copy()
            continuation[:, _ALL_GREEN] = 0
            expected = np.sum(answer_hist * continuation, axis=1) / answer_total
            loss = answer_hist[:, _ALL_GREEN]
            traps = (legal_hist <= rows_left) & outcomes
            traps[:, _ALL_GREEN] = False
            trap = np.sum(answer_hist * traps, axis=1)

            for row, index in enumerate(indices):
                recommendations.append(
                    Recommendation(
                        expected_legal=float(expected[row]),
                        information_bits=float(information[row]),
                        loss_probability=float(loss[row] / answer_total),
                        trap_probability=float(trap[row] / answer_total),
                        word=str(self._legal[index]),
                    )
                )

        def key(item: Recommendation) -> tuple[float | str, ...]:
            risk = (
                (item.loss_probability, item.trap_probability)
                if undos
                else (item.loss_probability + item.trap_probability, item.loss_probability)
            )
            return (*risk, item.information_bits, -item.expected_legal, item.word)

        return sorted(recommendations, key=key)[:limit]

    @property
    def row_count(self) -> int:
        """Return the number of active submitted rows."""
        return len(self._rows)

    def undo(self) -> str:
        """Remove the latest row while retaining its answer information."""
        if not self._rows:
            raise AssistantError("No row is available to Undo")
        index, _ = self._rows.pop()
        self._rebuild_legal()
        return str(self._legal[index])

    @property
    def used_letters(self) -> set[str]:
        """Return letters used by active rows."""
        return set("".join(self._legal[i] for i, _ in self._rows))

    def _rebuild_legal(self) -> None:
        # Undone feedback remains in the answer posterior but not site legality
        self._legal_mask[:] = True
        for index, pattern in self._rows:
            self._legal_mask &= self._patterns[index] == pattern


class AssistantError(RuntimeError):
    """Raised for invalid game state or assistant data."""


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Metrics for one legal guess."""

    expected_legal: float
    information_bits: float
    loss_probability: float
    trap_probability: float
    word: str


@dataclass(frozen=True, slots=True)
class WordLists:
    """Accepted guesses and possible answers."""

    answers: tuple[str, ...]
    legal: tuple[str, ...]


def feedback(guess: str, target: str) -> int:
    """Encode duplicate-aware Wordle feedback as five base-3 digits.

    Parameters
    ----------
    guess : str
        Submitted five-letter word
    target : str
        Hidden five-letter word

    Returns
    -------
    int
        Integer in ``[0, 242]`` with gray, yellow, and green encoded as 0, 1, 2
    """
    guess = guess.upper()
    target = target.upper()
    if any(
        len(word) != _WORD_LENGTH or not word.isascii() or not word.isalpha()
        for word in (guess, target)
    ):
        raise ValueError("Guess and target must contain five letters")

    values = [0] * _WORD_LENGTH
    remaining = {letter: target.count(letter) for letter in set(target)}
    for i, letter in enumerate(guess):
        if letter == target[i]:
            values[i] = 2
            remaining[letter] -= 1
    for i, letter in enumerate(guess):
        if not values[i] and remaining.get(letter, 0):
            values[i] = 1
            remaining[letter] -= 1
    return sum(value * power for value, power in zip(values, _POWERS, strict=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interactive assistant.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command-line arguments, or ``None`` to read ``sys.argv``

    Returns
    -------
    int
        Process exit status
    """
    args = _parser().parse_args(argv)
    try:
        return _run(Assistant(), args.top, args.undos)
    except EOFError:
        print()
        return 0
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except AssistantError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


def parse_feedback(value: str) -> int:
    """Parse five feedback tiles.

    Parameters
    ----------
    value : str
        Five ``b/y/g``, ``0/1/2``, or colored-square tiles

    Returns
    -------
    int
        Base-3 feedback pattern
    """
    for tile, letter in {"⬛": "b", "⬜": "b", "🟨": "y", "🟩": "g"}.items():
        value = value.replace(tile, letter)
    value = re.sub(r"[\s,./|_-]", "", value.lower().replace("\ufe0f", ""))
    if len(value) != _WORD_LENGTH or any(letter not in _FEEDBACK_VALUES for letter in value):
        raise ValueError("Feedback must contain five b/y/g or colored-square tiles")
    return sum(
        _FEEDBACK_VALUES[letter] * power for letter, power in zip(value, _POWERS, strict=True)
    )


def _encode(words: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Encode words as letters and per-letter counts.

    Parameters
    ----------
    words : Sequence[str]
        Uppercase five-letter words

    Returns
    -------
    letters : numpy.ndarray
        ``uint8`` array with shape ``(N, 5)`` and values in ``[0, 25]``
    counts : numpy.ndarray
        ``uint8`` array with shape ``(N, 26)``
    """
    letters = np.frombuffer("".join(words).encode(), dtype=np.uint8).reshape(-1, 5) - ord("A")
    counts = np.zeros((len(words), 26), dtype=np.uint8)
    rows = np.repeat(np.arange(len(words)), _WORD_LENGTH)
    np.add.at(counts, (rows, letters.ravel()), 1)
    return letters, counts


def _format_pattern(pattern: int) -> str:
    return "".join(_TILES[(pattern // power) % 3] for power in _POWERS)


def _histograms(patterns: np.ndarray) -> np.ndarray:
    """Count patterns independently for each guess row.

    Parameters
    ----------
    patterns : numpy.ndarray
        ``uint8`` array with shape ``(G, T)``

    Returns
    -------
    numpy.ndarray
        Integer array with shape ``(G, 243)``
    """
    rows, _ = patterns.shape
    offsets = _PATTERN_COUNT * np.arange(rows)[:, None]
    return np.bincount((patterns + offsets).ravel(), minlength=rows * _PATTERN_COUNT).reshape(
        rows,
        _PATTERN_COUNT,
    )


def _load_data() -> tuple[WordLists, np.ndarray]:
    try:
        with np.load(_DATA_PATH, allow_pickle=False) as data:
            answers = tuple(data["answers"].astype(str).tolist())
            legal = tuple(data["legal"].astype(str).tolist())
            patterns = data["patterns"]
        _validate(answers, legal)
        if patterns.dtype != np.uint8 or patterns.shape != (len(legal), len(legal)):
            raise ValueError("Feedback table has an invalid shape or data type")
    except (BadZipFile, EOFError, KeyError, OSError, ValueError) as error:
        raise AssistantError(f"Could not load assistant data: {error}") from error
    return WordLists(answers, legal), patterns


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find safe Don't Wordle guesses")
    parser.add_argument("--top", default=10, type=int, help="recommendations to show")
    parser.add_argument("--undos", default=5, type=int, help="available Undos")
    return parser


def _pattern_matrix(
    guesses: np.ndarray,
    targets: np.ndarray,
    target_counts: np.ndarray,
) -> np.ndarray:
    """Score every guess-target pair with exact duplicate handling.

    Parameters
    ----------
    guesses : numpy.ndarray
        Letter array with shape ``(G, 5)``
    targets : numpy.ndarray
        Letter array with shape ``(T, 5)``
    target_counts : numpy.ndarray
        Per-letter target counts with shape ``(T, 26)``

    Returns
    -------
    numpy.ndarray
        ``uint8`` base-3 patterns with shape ``(G, T)``

    Notes
    -----
    Greens consume exact copies before yellows are assigned from left to right.
    """
    green = guesses[:, None] == targets[None]
    patterns = np.zeros(green.shape[:2], dtype=np.uint16)
    yellow_masks: list[np.ndarray] = []

    for position, power in enumerate(_POWERS):
        exact = green[:, :, position]
        patterns += exact * np.uint16(2 * power)
        letter = guesses[:, position]
        available = target_counts[:, letter].T
        green_count = np.sum(green & (guesses == letter[:, None])[:, None], axis=2)
        used = np.zeros_like(available)
        for prior, yellow in enumerate(yellow_masks):
            used += yellow & (guesses[:, prior] == letter)[:, None]
        yellow = ~exact & (available > green_count + used)
        patterns += yellow * np.uint16(power)
        yellow_masks.append(yellow)

    return patterns.astype(np.uint8)


def _percent(value: float) -> str:
    return "0" if not value else f"{100 * value:.3g}"


def _print_recommendations(recommendations: Sequence[Recommendation]) -> None:
    print("\n #  WORD   lose%  undo%   bits  E[legal]")
    print("--  -----  -----  -----  -----  --------")
    for i, item in enumerate(recommendations, 1):
        print(
            f"{i:2}  {item.word:5}  {_percent(item.loss_probability):>5}  "
            f"{_percent(item.trap_probability):>5}  {item.information_bits:5.2f}  "
            f"{item.expected_legal:8.1f}"
        )


def _run(assistant: Assistant, limit: int, undos: int) -> int:
    if limit < 1 or undos < 0:
        raise AssistantError("Top must be positive, and Undos nonnegative")

    print(
        f"{assistant.legal_count:,} accepted guesses, {assistant.answer_count:,} possible answers"
    )
    print("Enter WORD RESULT using b/y/g or colored squares. Commands: undo, quit")

    while assistant.row_count < 6:
        rows_left = 6 - assistant.row_count
        if assistant.row_count and assistant.legal_count <= rows_left:
            if not undos:
                print("Eliminated: too few legal words and no Undos remain")
                return 1
            command = input("The site requires an Undo. Click it, then type undo: ").strip().lower()
            if command == "quit":
                return 0
            if command != "undo":
                print("Type undo after clicking the site's Undo")
                continue
            print(f"Undid {assistant.undo()} and retained its answer information")
            undos -= 1
            continue

        print(
            f"\nTurn {assistant.row_count + 1}/6, {assistant.legal_count:,} legal, "
            f"{assistant.answer_count:,} answers, {undos} Undos"
        )
        print("Ranking...", flush=True)
        _print_recommendations(assistant.rank(limit, undos))

        value = input("\n> ").strip()
        if value.lower() == "quit":
            return 0
        if value.lower() == "undo":
            if not undos:
                print("No Undos remain")
                continue
            try:
                print(f"Undid {assistant.undo()} and retained its answer information")
                undos -= 1
            except AssistantError as error:
                print(error)
            continue

        fields = value.split()
        if len(fields) != 2:
            print("Enter WORD RESULT")
            continue

        try:
            word, raw_pattern = fields
            pattern = parse_feedback(raw_pattern)
            assistant.apply(word, pattern)
        except (AssistantError, ValueError) as error:
            print(error)
            continue

        print(f"Recorded {word.upper()} {_format_pattern(pattern)}")
        if pattern == _ALL_GREEN:
            print("That is the hidden word")
            return 1
        if assistant.row_count == 6:
            unused = 26 - len(assistant.used_letters)
            print(f"Survived with a score of {assistant.legal_count * unused:,}")
            return 0

    return 0


def _validate(
    answers: Sequence[str],
    legal: Sequence[str],
    minimum_size: bool = True,
) -> None:
    if not answers or not legal:
        raise ValueError("Word lists are empty")
    if minimum_size and (len(answers) < 1_000 or len(legal) < 5_000):
        raise ValueError("Word lists are unexpectedly small")
    if tuple(answers) != tuple(sorted(answers)) or tuple(legal) != tuple(sorted(legal)):
        raise ValueError("Word lists are not sorted")
    if len(set(answers)) != len(answers) or len(set(legal)) != len(legal):
        raise ValueError("Word lists contain duplicates")
    if not set(answers).issubset(legal):
        raise ValueError("Answers are not a subset of legal guesses")
    if any(len(word) != _WORD_LENGTH or not word.isascii() or not word.isalpha() for word in legal):
        raise ValueError("Word list contains an invalid entry")
