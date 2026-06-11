"""Parse model outputs and compute evaluation metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SENTIMENT_LABELS = {"positive", "negative", "mixed"}
STRICT_FORMAT = re.compile(
    r"^SENTIMENT: (positive|negative|mixed)\nSUMMARY: .+\S$",
    re.MULTILINE,
)
SENTIMENT_RE = re.compile(
    r"sentiment\s*:\s*(positive|negative|mixed)",
    re.IGNORECASE,
)
SUMMARY_RE = re.compile(
    r"summary\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ParsedOutput:
    raw: str
    sentiment: str | None = None
    summary: str | None = None
    strict_format: bool = False
    lenient_format: bool = False


@dataclass
class EvalScores:
    n: int = 0
    sentiment_correct: int = 0
    sentiment_parseable: int = 0
    summary_parseable: int = 0
    strict_format: int = 0
    lenient_format: int = 0
    rouge1_sum: float = 0.0
    rougeL_sum: float = 0.0
    rouge_n: int = 0
    per_example: list[dict] = field(default_factory=list)

    def add(self, **kwargs) -> None:
        self.n += 1
        self.per_example.append(kwargs)
        if kwargs.get("sentiment_parseable"):
            self.sentiment_parseable += 1
        if kwargs.get("sentiment_correct"):
            self.sentiment_correct += 1
        if kwargs.get("summary_parseable"):
            self.summary_parseable += 1
        if kwargs.get("strict_format"):
            self.strict_format += 1
        if kwargs.get("lenient_format"):
            self.lenient_format += 1
        if kwargs.get("rouge1") is not None:
            self.rouge1_sum += kwargs["rouge1"]
            self.rougeL_sum += kwargs["rougeL"]
            self.rouge_n += 1

    def summary_dict(self, label: str) -> dict:
        n = self.n or 1
        rn = self.rouge_n or 1
        return {
            "model": label,
            "examples": self.n,
            "sentiment_accuracy": round(self.sentiment_correct / n, 4),
            "sentiment_parse_rate": round(self.sentiment_parseable / n, 4),
            "summary_parse_rate": round(self.summary_parseable / n, 4),
            "format_compliance_strict": round(self.strict_format / n, 4),
            "format_compliance_lenient": round(self.lenient_format / n, 4),
            "rouge1": round(self.rouge1_sum / rn, 4),
            "rougeL": round(self.rougeL_sum / rn, 4),
            "rouge_examples": self.rouge_n,
        }


def parse_reference(output: str) -> tuple[str, str]:
    """Parse gold label from training/test JSONL output field."""
    lines = output.strip().split("\n", 1)
    sentiment = lines[0].split(":", 1)[1].strip().lower()
    summary = lines[1].split(":", 1)[1].strip() if len(lines) > 1 else ""
    return sentiment, summary


def parse_prediction(text: str) -> ParsedOutput:
    text = text.strip()
    parsed = ParsedOutput(raw=text)
    parsed.strict_format = bool(STRICT_FORMAT.match(text))

    sm = SENTIMENT_RE.search(text)
    if sm:
        parsed.sentiment = sm.group(1).lower()
    summ = SUMMARY_RE.search(text)
    if summ:
        parsed.summary = summ.group(1).strip().split("\n")[0].strip()

    parsed.lenient_format = parsed.sentiment in SENTIMENT_LABELS and bool(parsed.summary)
    return parsed


def score_example(
    prediction: str,
    gold_sentiment: str,
    gold_summary: str,
    rouge_scorer,
) -> dict:
    parsed = parse_prediction(prediction)
    sentiment_parseable = parsed.sentiment in SENTIMENT_LABELS
    sentiment_correct = sentiment_parseable and parsed.sentiment == gold_sentiment
    summary_parseable = bool(parsed.summary)

    rouge1 = rougeL = None
    if summary_parseable and gold_summary:
        scores = rouge_scorer.score(gold_summary, parsed.summary)
        rouge1 = scores["rouge1"].fmeasure
        rougeL = scores["rougeL"].fmeasure

    return {
        "prediction": prediction,
        "parsed_sentiment": parsed.sentiment,
        "parsed_summary": parsed.summary,
        "gold_sentiment": gold_sentiment,
        "gold_summary": gold_summary,
        "sentiment_parseable": sentiment_parseable,
        "sentiment_correct": sentiment_correct,
        "summary_parseable": summary_parseable,
        "strict_format": parsed.strict_format,
        "lenient_format": parsed.lenient_format,
        "rouge1": rouge1,
        "rougeL": rougeL,
    }


def print_comparison(base: dict, lora: dict) -> None:
    rows = [
        ("Sentiment accuracy", "sentiment_accuracy"),
        ("Sentiment parse rate", "sentiment_parse_rate"),
        ("Summary parse rate", "summary_parse_rate"),
        ("Format compliance (strict)", "format_compliance_strict"),
        ("Format compliance (lenient)", "format_compliance_lenient"),
        ("ROUGE-1 (summary)", "rouge1"),
        ("ROUGE-L (summary)", "rougeL"),
    ]
    print(f"\n{'Metric':<32} {'Base':>10} {'LoRA':>10} {'Delta':>10}")
    print("-" * 64)
    for name, key in rows:
        b, l = base[key], lora[key]
        delta = l - b
        sign = "+" if delta >= 0 else ""
        print(f"{name:<32} {b:>10.4f} {l:>10.4f} {sign}{delta:>9.4f}")
