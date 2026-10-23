"""Opt-in metric families: BLEU, ROUGE-L, noise, entities, toxicity and bias."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from hermes_report.metrics.base import (
  MetricError,
  MetricInput,
  MetricResult,
  MetricServices,
)
from hermes_report.metrics.judge import JudgeSession
from hermes_report.metrics.prompts import (
  entities_messages,
  flag_messages,
  parse_entities,
  parse_flag,
)
from hermes_report.questions import normalise

__all__ = [
  "Bias",
  "Bleu",
  "ContextEntityRecall",
  "NoiseSensitivity",
  "RougeL",
  "Toxicity",
  "bleu",
  "rouge_l",
]

_WORD = re.compile(r"\w+")


def _judge(services: MetricServices) -> JudgeSession:
  if services.judge is None:
    raise MetricError("Esta métrica necesita un juez")
  return services.judge


def _result(
  name: str,
  scores: dict[str, float | None],
  details: dict[str, Any],
  judge: JudgeSession,
  mark: int,
) -> MetricResult:
  """An ``ok`` result with the judge calls made since ``mark``."""
  return MetricResult(name, "ok", scores, details, judge.calls_since(mark))


def _words(text: str) -> list[str]:
  """Word tokens of ``text``: case, accents and punctuation do not count."""
  return _WORD.findall(normalise(text))


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
  return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate: Sequence[str], reference: Sequence[str], max_n: int = 4) -> float:
  """BLEU of ``candidate`` tokens against ``reference`` tokens, in 0..1.

  Geometric mean of the clipped n-gram precisions for n = 1..``max_n``, times
  the brevity penalty. Orders above 1 are add-one smoothed so that a short
  text is not zeroed by a missing 4-gram; unigrams are not, so texts that
  share no word score 0.0. An empty candidate or reference scores 0.0.
  """
  if not candidate or not reference or max_n < 1:
    return 0.0
  log_sum = 0.0
  for n in range(1, max_n + 1):
    found = _ngrams(candidate, n)
    matched = sum(
      min(count, _ngrams(reference, n)[gram]) for gram, count in found.items()
    )
    total = max(0, len(candidate) - n + 1)
    if n == 1:
      if matched == 0:
        return 0.0
      precision = matched / total
    else:
      precision = (matched + 1) / (total + 1)
    log_sum += math.log(precision)
  if len(candidate) > len(reference):
    penalty = 1.0
  else:
    penalty = math.exp(1.0 - len(reference) / len(candidate))
  return penalty * math.exp(log_sum / max_n)


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
  """Length of the longest common subsequence of two token lists."""
  previous = [0] * (len(b) + 1)
  for x in a:
    current = [0]
    for j, y in enumerate(b, 1):
      current.append(
        previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1])
      )
    previous = current
  return previous[-1]


def rouge_l(candidate: Sequence[str], reference: Sequence[str]) -> float:
  """ROUGE-L F-measure of ``candidate`` tokens against ``reference`` tokens.

  The harmonic mean of the longest-common-subsequence precision and recall;
  0.0 when either side is empty or they share no token.
  """
  if not candidate or not reference:
    return 0.0
  common = _lcs(candidate, reference)
  if common == 0:
    return 0.0
  precision = common / len(candidate)
  recall = common / len(reference)
  return 2 * precision * recall / (precision + recall)


class Bleu:
  """BLEU-4 of the answer against the expected answer, ignoring case and accents.

  Registry: metric[bleu]
  """

  name: str = "bleu"
  default_enabled: bool = False
  scores: tuple[str, ...] = ("bleu",)
  needs: frozenset[str] = frozenset()
  requires: frozenset[str] = frozenset({"answer", "expected"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Score the word overlap of the answer with the expected answer."""
    candidate = _words(item.answer or "")
    reference = _words(item.expected)
    return MetricResult(
      self.name,
      "ok",
      {"bleu": bleu(candidate, reference)},
      {"candidate_tokens": len(candidate), "reference_tokens": len(reference)},
    )


class RougeL:
  """ROUGE-L of the answer against the expected answer, ignoring case and accents.

  Registry: metric[rouge_l]
  """

  name: str = "rouge_l"
  default_enabled: bool = False
  scores: tuple[str, ...] = ("rouge_l",)
  needs: frozenset[str] = frozenset()
  requires: frozenset[str] = frozenset({"answer", "expected"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Score the longest common word sequence of the answer and the expected answer."""
    candidate = _words(item.answer or "")
    reference = _words(item.expected)
    return MetricResult(
      self.name,
      "ok",
      {"rouge_l": rouge_l(candidate, reference)},
      {"candidate_tokens": len(candidate), "reference_tokens": len(reference)},
    )


class NoiseSensitivity:
  """Share of the answer claims that wrong and irrelevant chunks alone support.

  Registry: metric[noise_sensitivity]
  """

  name: str = "noise_sensitivity"
  default_enabled: bool = False
  scores: tuple[str, ...] = ("noise_sensitivity",)
  needs: frozenset[str] = frozenset({"judge"})
  requires: frozenset[str] = frozenset({"answer", "expected", "retrieved"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Count the answer claims the expected answer rejects that only noise backs."""
    if item.refused:
      return MetricResult.not_applicable(self.name, ["refusal"])
    judge = _judge(services)
    mark = len(judge.calls)
    claims = judge.answer_claims()
    if not claims:
      return _result(self.name, {"noise_sensitivity": 0.0}, {"claims": []}, judge, mark)
    against_expected = judge.verify(claims, [item.expected], "answer_vs_expected")
    relevant = judge.chunk_relevance()
    against_retrieved = judge.verify(
      claims, [chunk.text for chunk in item.retrieved], "answer_vs_retrieved"
    )
    noisy: list[int] = []
    for expected_verdict, retrieved_verdict in zip(
      against_expected, against_retrieved, strict=True
    ):
      sources = retrieved_verdict.sources
      only_noise = (
        retrieved_verdict.supported
        and bool(sources)
        and not any(relevant[source - 1] for source in sources)
      )
      if not expected_verdict.supported and only_noise:
        noisy.append(expected_verdict.claim)
    return _result(
      self.name,
      {"noise_sensitivity": len(noisy) / len(claims)},
      {
        "claims": claims,
        "noisy_claims": noisy,
        "relevant": list(relevant),
        "answer_vs_expected": against_expected,
        "answer_vs_retrieved": against_retrieved,
      },
      judge,
      mark,
    )


class ContextEntityRecall:
  """Share of the entities of the expected answer that the context mentions.

  Registry: metric[context_entity_recall]
  """

  name: str = "context_entity_recall"
  default_enabled: bool = False
  scores: tuple[str, ...] = ("context_entity_recall",)
  needs: frozenset[str] = frozenset({"judge"})
  requires: frozenset[str] = frozenset({"expected", "context"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Match the entities the judge extracts from the expected answer in the context."""
    judge = _judge(services)
    mark = len(judge.calls)
    entities = judge.ask(
      "expected_entities", entities_messages(item.expected), parse_entities
    )
    if not entities:
      return _result(
        self.name, {"context_entity_recall": None}, {"entities": []}, judge, mark
      )
    context = normalise(" ".join(item.context))
    found = [e for e in entities if (needle := normalise(e)) and needle in context]
    return _result(
      self.name,
      {"context_entity_recall": len(found) / len(entities)},
      {"entities": list(entities), "found": found},
      judge,
      mark,
    )


class Toxicity:
  """Whether the judge flags the answer as toxic: 1.0 when flagged, else 0.0.

  Registry: metric[toxicity]
  """

  name: str = "toxicity"
  default_enabled: bool = False
  scores: tuple[str, ...] = ("toxicity",)
  needs: frozenset[str] = frozenset({"judge"})
  requires: frozenset[str] = frozenset({"answer"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Ask the judge whether the answer is toxic; its reason goes to the details."""
    judge = _judge(services)
    mark = len(judge.calls)
    flagged, reason = judge.ask(
      "toxicity",
      flag_messages(item.answer or "", "toxic"),
      lambda data: parse_flag(data, "toxic"),
    )
    return _result(
      self.name, {"toxicity": 1.0 if flagged else 0.0}, {"reason": reason}, judge, mark
    )


class Bias:
  """Whether the judge flags the answer as biased: 1.0 when flagged, else 0.0.

  Registry: metric[bias]
  """

  name: str = "bias"
  default_enabled: bool = False
  scores: tuple[str, ...] = ("bias",)
  needs: frozenset[str] = frozenset({"judge"})
  requires: frozenset[str] = frozenset({"answer"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Ask the judge whether the answer is biased; its reason goes to the details."""
    judge = _judge(services)
    mark = len(judge.calls)
    flagged, reason = judge.ask(
      "bias",
      flag_messages(item.answer or "", "biased"),
      lambda data: parse_flag(data, "biased"),
    )
    return _result(
      self.name, {"bias": 1.0 if flagged else 0.0}, {"reason": reason}, judge, mark
    )
