import argparse
import os
import sys
import time
from typing import Any, Dict, List

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config import load_config, BenchmarkConfig
from src.retriever import TextChunker, DenseSimRetriever
from src.generator import DeterministicRAGGenerator
from src.metrics import (
    compute_token_f1,
    compute_exact_match,
    compute_hallucination_and_faithfulness,
    evaluate_guardrails,
)
from src.utils import setup_logger, load_json, save_json, format_table


def run_experiment(
    config: BenchmarkConfig,
    top_k: int,
    chunk_size: int,
    gold_position: str = "auto",
    logger=None,
) -> Dict[str, Any]:
    corpus = load_json(config.data.corpus_path)
    queries = load_json(config.data.queries_path)

    chunker = TextChunker(
        chunk_size=chunk_size, chunk_overlap=config.independent_variables.chunk_overlap
    )
    retriever = DenseSimRetriever(
        embedding_dim=config.controlled_variables.retriever.embedding_dim,
        min_similarity_threshold=config.controlled_variables.retriever.min_similarity_threshold,
    )
    retriever.index_corpus(corpus, chunker)

    generator = DeterministicRAGGenerator(
        model_name=config.controlled_variables.generator.model_name
    )

    query_results = []
    total_hr = 0.0
    total_faith = 0.0
    total_f1 = 0.0
    total_em = 0.0
    total_latency = 0.0
    guardrails_passed = True

    for q in queries:
        qid = q.get("query_id")
        question = q.get("question")
        gold_answer = q.get("gold_answer")
        gold_doc_id = q.get("gold_doc_id")

        t0 = time.perf_counter()

        retrieved = retriever.retrieve(
            query=question,
            top_k=top_k,
            gold_doc_id=gold_doc_id,
            gold_position=gold_position,
        )

        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        gen_output = generator.generate(
            query=question,
            retrieved_passages=retrieved,
            gold_doc_id=gold_doc_id,
            gold_answer=gold_answer,
        )

        overall_latency_ms = retrieval_ms + gen_output["latency_ms"]

        hf = compute_hallucination_and_faithfulness(
            generated_text=gen_output["generated_text"],
            retrieved_passages=retrieved,
            gold_doc_id=gold_doc_id,
            key_entities=q.get("key_entities", []),
        )

        f1 = compute_token_f1(gen_output["generated_text"], gold_answer)
        em = compute_exact_match(gen_output["generated_text"], gold_answer)

        gr = evaluate_guardrails(
            latency_ms=overall_latency_ms,
            context_tokens=gen_output["context_token_count"],
            max_latency_threshold_ms=config.guardrails.latency_threshold_ms,
            max_context_tokens=config.guardrails.max_context_tokens,
        )
        if not gr["all_guardrails_passed"]:
            guardrails_passed = False

        query_res = {
            "query_id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "generated_text": gen_output["generated_text"],
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved],
            "metrics": {
                "hallucination_rate": hf["hallucination_rate"],
                "faithfulness_score": hf["faithfulness_score"],
                "token_f1": f1,
                "exact_match": em,
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "generation_latency_ms": gen_output["latency_ms"],
                "total_latency_ms": round(overall_latency_ms, 2),
                "context_token_count": gen_output["context_token_count"],
            },
            "guardrails": gr,
        }
        query_results.append(query_res)

        total_hr += hf["hallucination_rate"]
        total_faith += hf["faithfulness_score"]
        total_f1 += f1
        total_em += em
        total_latency += overall_latency_ms

    n = len(queries)
    summary = {
        "configuration": {
            "top_k": top_k,
            "chunk_size": chunk_size,
            "gold_position": gold_position,
        },
        "mean_metrics": {
            "hallucination_rate": round(total_hr / n, 4) if n else 0.0,
            "faithfulness_score": round(total_faith / n, 4) if n else 0.0,
            "token_f1": round(total_f1 / n, 4) if n else 0.0,
            "exact_match": round(total_em / n, 4) if n else 0.0,
            "mean_total_latency_ms": round(total_latency / n, 2) if n else 0.0,
        },
        "all_guardrails_passed": guardrails_passed,
        "sample_size": n,
        "queries": query_results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG Configuration & Hallucination Rate Benchmark."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k retrieval depth",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size",
    )
    parser.add_argument(
        "--gold-position",
        type=str,
        default="auto",
        choices=["first", "middle", "last", "auto"],
        help="Override gold passage position for RQ3 testing",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run full factorial parameter sweep over top_k and chunk_size grids",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark_sample_results.json",
        help="Output destination path for experiment results",
    )

    args = parser.parse_args()
    logger = setup_logger("benchmark_runner")

    logger.info("Initializing RAG Benchmark Pipeline...")
    config = load_config(args.config)
    logger.info(f"Loaded config from: {args.config} (Experiment: {config.experiment.name})")

    results_data = {
        "experiment_meta": config.experiment.model_dump(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "runs": [],
    }

    if args.sweep:
        logger.info("Executing factorial sweep over independent variables...")
        table_rows = []
        for k in config.independent_variables.top_k_values:
            for cs in config.independent_variables.chunk_sizes:
                logger.info(f"-> Testing run: top_k={k}, chunk_size={cs}, gold_pos={args.gold_position}")
                run_res = run_experiment(config, top_k=k, chunk_size=cs, gold_position=args.gold_position, logger=logger)
                results_data["runs"].append(run_res)
                mm = run_res["mean_metrics"]
                table_rows.append([
                    k,
                    cs,
                    args.gold_position,
                    f"{mm['hallucination_rate']:.4f}",
                    f"{mm['faithfulness_score']:.4f}",
                    f"{mm['token_f1']:.4f}",
                    f"{mm['mean_total_latency_ms']:.2f} ms",
                    "PASSED" if run_res["all_guardrails_passed"] else "FAILED",
                ])

        headers = ["top_k", "chunk_size", "gold_pos", "Hallucination Rate", "Faithfulness", "Token F1", "Mean Latency", "Guardrails"]
        print("\n" + "=" * 80)
        print("FACTORIAL BENCHMARK SWEEP RESULTS")
        print("=" * 80)
        print(format_table(headers, table_rows))
        print("=" * 80 + "\n")

    else:
        k = args.top_k or config.independent_variables.default_top_k
        cs = args.chunk_size or config.independent_variables.default_chunk_size
        pos = args.gold_position or config.independent_variables.default_gold_position

        logger.info(f"Running benchmark: top_k={k}, chunk_size={cs}, gold_pos={pos}")
        run_res = run_experiment(config, top_k=k, chunk_size=cs, gold_position=pos, logger=logger)
        results_data["runs"].append(run_res)

        mm = run_res["mean_metrics"]
        headers = ["Parameter / Metric", "Value"]
        rows = [
            ["Retrieval Depth (top_k)", str(k)],
            ["Chunk Size (tokens)", str(cs)],
            ["Gold Position Strategy", str(pos)],
            ["Sample Queries Tested", str(run_res["sample_size"])],
            ["Primary: Mean Hallucination Rate", f"{mm['hallucination_rate']:.4f}"],
            ["Primary: Mean Faithfulness Score", f"{mm['faithfulness_score']:.4f}"],
            ["Secondary: Mean Token F1", f"{mm['token_f1']:.4f}"],
            ["Secondary: Exact Match", f"{mm['exact_match']:.4f}"],
            ["Operational: Mean Total Latency", f"{mm['mean_total_latency_ms']:.2f} ms"],
            ["Guardrails Status", "ALL PASSED" if run_res["all_guardrails_passed"] else "VIOLATIONS DETECTED"],
        ]
        print("\n" + "=" * 60)
        print("RAG BENCHMARK EXECUTION SUMMARY")
        print("=" * 60)
        print(format_table(headers, rows))
        print("=" * 60 + "\n")

    save_json(results_data, args.output)
    logger.info(f"Saved benchmark results to: {args.output}")


if __name__ == "__main__":
    main()
