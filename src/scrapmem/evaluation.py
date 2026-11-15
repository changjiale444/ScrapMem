"""Record-level ATM-Bench QS, Recall@k, Joint@k, and daily recall."""

from __future__ import annotations

from collections import defaultdict
import math

from ._vendor.atmbench.scoring import deterministic_accuracy, list_jaccard_score
from ._vendor.atmbench.qtypes import detect_qtype
from ._vendor.atmbench.constants import LLM_JUDGE_PROMPT
from .backends import ChatClient, Endpoint


def qtype(row):
    raw = (
        str(row.get("qtype", row.get("question_type", "")))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if raw in {"number", "num", "numeric"}:
        return "number"
    if raw in {"list", "list_recall", "listrecall"}:
        return "list_recall"
    if raw in {"open", "open_end", "open_ended", "openended"}:
        return "open_end"
    if raw:
        raise ValueError(f"Unknown question type: {raw}")
    return detect_qtype(str(row["answer"]))


class OpenEndedJudge:
    def __init__(self, endpoint=None):
        self.client = ChatClient(
            endpoint
            or Endpoint(
                "gpt-5-mini",
                "https://api.openai.com/v1/chat/completions",
                "OPENAI_API_KEY",
                temperature=None,
                token_limit_field="max_completion_tokens",
                reasoning_effort="low",
            )
        )

    def __call__(self, question, answer, prediction):
        prompt = (
            LLM_JUDGE_PROMPT.replace("{{question}}", question)
            .replace("{{answer}}", answer)
            .replace("{{prediction}}", prediction)
        )
        result = self.client.json(
            [{"role": "user", "content": prompt}], max_tokens=4096
        )
        score = result.get("accuracy")
        if type(score) is not bool:
            raise ValueError("Open-ended judge must return boolean accuracy")
        return float(score)


def evaluate(predictions, gold, *, item_days, k=10, judge=None):
    if type(k) is not int or k < 1:
        raise ValueError("k must be positive")
    pred = {str(r["id"]): r for r in predictions}
    gids = [str(r["id"]) for r in gold]
    if (
        not gids
        or len(pred) != len(predictions)
        or len(set(gids)) != len(gids)
        or set(pred) != set(gids)
    ):
        raise ValueError(
            "Gold/prediction ids must be unique, nonempty, and match exactly"
        )
    details = []
    categories = defaultdict(list)
    for row in gold:
        p = pred[str(row["id"])]
        kind = qtype(row)
        answer = p.get("prediction", p.get("answer"))
        if not isinstance(answer, str):
            raise ValueError("Every prediction needs an answer string")
        gt = row.get("evidence_ids", row.get("gt_evidence_ids"))
        ranked = p.get("retrieval_ids")
        if not isinstance(gt, list) or not gt or not isinstance(ranked, list):
            raise ValueError("Gold evidence_ids and ranked retrieval_ids are required")
        gt = {str(x) for x in gt}
        ranked = [str(x) for x in ranked]
        if len(set(ranked)) != len(ranked):
            raise ValueError("Retrieved record ids must be unique")
        if any(x not in item_days for x in gt | set(ranked)):
            raise ValueError("Unknown evidence ID in item-to-day map")
        truth = str(row["answer"])
        if kind == "number":
            score = float(deterministic_accuracy(truth, answer, row["question"]))
        elif kind == "list_recall":
            score = list_jaccard_score(truth, answer)
        else:
            if judge is None:
                raise ValueError("Open-ended questions require a configured LLM judge")
            score = float(judge(row["question"], truth, answer))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("QS must be in [0,1]")
        top = ranked[:k]
        recall = len(gt & set(top)) / len(gt)
        gt_days = {item_days[i] for i in gt}
        final_days = {item_days[i] for i in top}
        stage_days = p.get(
            "stage_a_day_ids", list(dict.fromkeys(item_days[i] for i in ranked))
        )
        item = {
            "id": str(row["id"]),
            "qtype": kind,
            "QS": score,
            f"Recall@{k}": recall,
            f"Joint@{k}": score * recall,
            f"daily_recall@{k}": len(gt_days & final_days) / len(gt_days),
            "stage_a_daily_recall": len(gt_days & set(stage_days)) / len(gt_days),
        }
        categories[kind].append(score)
        details.append(item)
    fields = [
        "QS",
        f"Recall@{k}",
        f"Joint@{k}",
        f"daily_recall@{k}",
        "stage_a_daily_recall",
    ]
    summary = {key: sum(r[key] for r in details) / len(details) for key in fields}
    summary.update(
        {
            label: (
                sum(categories[kind]) / len(categories[kind])
                if categories[kind]
                else None
            )
            for label, kind in [
                ("N", "number"),
                ("R", "list_recall"),
                ("O", "open_end"),
            ]
        }
    )
    return {
        "count": len(details),
        "k": k,
        "metrics": summary,
        "by_type_counts": {t: len(v) for t, v in categories.items()},
        "details": details,
    }
