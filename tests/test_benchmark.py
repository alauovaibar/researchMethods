import os
import pytest
from src.config import load_config
from src.retriever import TextChunker, DenseSimRetriever
from src.generator import PromptBuilder, DeterministicRAGGenerator
from src.metrics import (
    compute_token_f1,
    compute_exact_match,
    compute_hallucination_and_faithfulness,
    evaluate_guardrails,
)
from src.benchmark import run_experiment


def test_config_loading():
    config = load_config("configs/config.yaml")
    assert config.experiment.name == "rag_configuration_hallucination_benchmark"
    assert config.independent_variables.default_top_k == 3
    assert config.independent_variables.default_chunk_size == 256
    assert 3 in config.independent_variables.top_k_values
    assert 256 in config.independent_variables.chunk_sizes
    assert config.guardrails.max_context_tokens == 8192


def test_text_chunker():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    text = "word " * 25
    chunks = chunker.chunk_text(text, doc_id="doc_test", title="Test Title")
    assert len(chunks) > 1
    assert chunks[0]["doc_id"] == "doc_test"
    assert chunks[0]["title"] == "Test Title"
    assert chunks[0]["token_count"] <= 10


def test_retriever_dense_similarity():
    retriever = DenseSimRetriever(embedding_dim=128)
    corpus = [
        {"id": "doc1", "title": "Moon", "text": "Apollo 11 landed astronauts on the Moon in July 1969."},
        {"id": "doc2", "title": "Trees", "text": "Oak trees grow acorns and photosynthesis feeds them."},
    ]
    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    indexed = retriever.index_corpus(corpus, chunker)
    assert len(indexed) == 2

    results = retriever.retrieve("When did Apollo land on the moon?", top_k=1)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"


def test_metrics_token_f1_and_em():
    pred = "Apollo 11 landed on July 20 1969"
    gold = "Apollo 11 landed on July 20 1969"
    assert compute_exact_match(pred, gold) == 1.0
    assert compute_token_f1(pred, gold) == 1.0

    partial_pred = "Apollo 11 landed"
    assert compute_token_f1(partial_pred, gold) > 0.0
    assert compute_exact_match(partial_pred, gold) == 0.0


def test_metrics_hallucination():
    context = [{"text": "Neil Armstrong was the commander of Apollo 11 which landed on July 20 1969."}]
    
    grounded = "Neil Armstrong commanded Apollo 11 on July 20 1969."
    res_g = compute_hallucination_and_faithfulness(grounded, context, "doc_001")
    assert res_g["hallucination_rate"] == 0.0
    assert res_g["faithfulness_score"] == 1.0

    fabricated = "The spacecraft was built in Atlantis by Martian engineers in 1842."
    res_f = compute_hallucination_and_faithfulness(fabricated, context, "doc_001")
    assert res_f["hallucination_rate"] > 0.5
    assert res_f["faithfulness_score"] < 0.5


def test_guardrails_evaluation():
    res_pass = evaluate_guardrails(latency_ms=120.0, context_tokens=1500)
    assert res_pass["all_guardrails_passed"] is True

    res_fail_lat = evaluate_guardrails(latency_ms=3000.0, context_tokens=1500, max_latency_threshold_ms=2500.0)
    assert res_fail_lat["all_guardrails_passed"] is False

    res_fail_tok = evaluate_guardrails(latency_ms=50.0, context_tokens=9000, max_context_tokens=8192)
    assert res_fail_tok["all_guardrails_passed"] is False


def test_end_to_end_benchmark_run():
    config = load_config("configs/config.yaml")
    summary = run_experiment(config, top_k=3, chunk_size=256, gold_position="auto")
    assert summary["sample_size"] > 0
    assert "mean_metrics" in summary
    assert "hallucination_rate" in summary["mean_metrics"]
    assert "faithfulness_score" in summary["mean_metrics"]
    assert summary["all_guardrails_passed"] is True
