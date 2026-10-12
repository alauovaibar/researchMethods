import re
import string
from typing import Any, Dict, List


def normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = sum(pred_tokens.count(t) for t in common) / len(pred_tokens)
    recall = sum(gold_tokens.count(t) for t in common) / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return round(2 * (precision * recall) / (precision + recall), 4)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_text(ground_truth) in normalize_text(prediction) else 0.0


def compute_hallucination_and_faithfulness(
    generated_text: str,
    retrieved_passages: List[Dict[str, Any]],
    gold_doc_id: str,
    key_entities: List[str] = None,
) -> Dict[str, float]:
    if not retrieved_passages:
        return {"hallucination_rate": 1.0, "faithfulness_score": 0.0}

    context_corpus = " ".join(p.get("text", "") for p in retrieved_passages).lower()
    sentences = [s.strip() for s in re.split(r"[.!?]\s*", generated_text) if len(s.strip()) > 5]
    if not sentences:
        return {"hallucination_rate": 0.0, "faithfulness_score": 1.0}

    unsupported_claims = 0
    total_claims = len(sentences)

    for sentence in sentences:
        words = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 3]
        if not words:
            continue
        grounded_words = sum(1 for w in words if w in context_corpus)
        grounding_ratio = grounded_words / len(words)
        if grounding_ratio < 0.40:
            unsupported_claims += 1

    hallucination_rate = round(unsupported_claims / total_claims, 4)
    faithfulness_score = round(max(0.0, 1.0 - hallucination_rate), 4)

    return {
        "hallucination_rate": hallucination_rate,
        "faithfulness_score": faithfulness_score,
    }


def evaluate_guardrails(
    latency_ms: float,
    context_tokens: int,
    max_latency_threshold_ms: float = 2500.0,
    max_context_tokens: int = 8192,
) -> Dict[str, Any]:
    latency_passed = latency_ms <= max_latency_threshold_ms
    tokens_passed = context_tokens <= max_context_tokens

    return {
        "latency_passed": latency_passed,
        "latency_ms": latency_ms,
        "max_latency_ms": max_latency_threshold_ms,
        "tokens_passed": tokens_passed,
        "context_tokens": context_tokens,
        "max_context_tokens": max_context_tokens,
        "all_guardrails_passed": (latency_passed and tokens_passed),
    }
