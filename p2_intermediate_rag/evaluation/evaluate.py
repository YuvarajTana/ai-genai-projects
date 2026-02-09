#!/usr/bin/env python3
"""
Lightweight RAG Evaluation Harness
==================================
Runs test questions through the retrieval pipeline and computes metrics.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, reload_config
from src.loaders import load_documents
from src.chunking import chunk_documents
from src.embeddings import SentenceTransformerEmbedder
from src.retrieval import HybridRetriever
from src.generation import OllamaLLM, PromptBuilder, GenerationConfig


def load_test_questions(path: Path) -> list:
    """Load test questions from JSON file."""
    with open(path) as f:
        return json.load(f)


def precision_at_k(retrieved_texts: list, expected_keywords: list, k: int) -> float:
    """
    Precision@k: proportion of top-k results that contain at least one expected keyword.
    """
    if not expected_keywords or k == 0:
        return 0.0
    hits = 0
    for doc in retrieved_texts[:k]:
        doc_lower = doc.lower()
        if any(kw.lower() in doc_lower for kw in expected_keywords):
            hits += 1
    return hits / min(k, len(retrieved_texts))


def mrr(retrieved_texts: list, expected_keywords: list) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    if not expected_keywords:
        return 0.0
    for i, doc in enumerate(retrieved_texts):
        doc_lower = doc.lower()
        if any(kw.lower() in doc_lower for kw in expected_keywords):
            return 1.0 / (i + 1)
    return 0.0


def answer_faithfulness(answer: str, expected_keywords: list) -> float:
    """Keyword overlap: proportion of expected keywords found in answer."""
    if not expected_keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def run_evaluation(
    questions_path: Path,
    config_path: str = "config.yaml",
    data_dir: str = None,
    output_path: Path = None,
    skip_generation: bool = False,
) -> dict:
    """
    Run evaluation on test questions.

    Args:
        questions_path: Path to test_questions.json
        config_path: Path to config.yaml
        data_dir: Override data directory (default from config)
        output_path: Path to save results JSON
        skip_generation: If True, only evaluate retrieval (no LLM call)

    Returns:
        Dictionary with metrics and per-question results
    """
    config = reload_config(config_path)
    data_dir = data_dir or config.paths.data_dir

    # Load test questions
    questions = load_test_questions(questions_path)

    # Index documents if we have data
    embedder = SentenceTransformerEmbedder(
        model_name=config.embeddings.model_name,
        normalize=config.embeddings.normalize,
    )
    retriever = HybridRetriever(
        embedder=embedder,
        collection_name=config.vector_db.collection_name,
        persist_directory=config.paths.db_dir,
        use_bm25=config.retrieval.hybrid.enabled,
        use_reranker=config.retrieval.reranking.enabled,
    )

    if retriever.count() == 0:
        # Index from data_dir
        documents = load_documents(data_dir)
        if not documents:
            raise ValueError(
                f"No documents in {data_dir}. Add documents and run indexing first."
            )
        chunks = chunk_documents(
            documents,
            strategy=config.chunking.strategy,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        retriever.index(chunks)

    prompt_builder = PromptBuilder(
        system_template=config.prompts.system_template,
        qa_template=config.prompts.qa_template,
    )
    llm = None
    if not skip_generation:
        llm = OllamaLLM(
            model=config.llm.ollama.model,
            base_url=config.llm.ollama.base_url,
            timeout=config.llm.ollama.timeout,
        )
        if not llm.is_available():
            print("Warning: Ollama not available. Skipping generation metrics.")
            skip_generation = True

    k = config.retrieval.top_k
    precisions = []
    mrr_scores = []
    faithfulness_scores = []
    results = []

    for q in questions:
        question = q["question"]
        expected_kw = q.get("expected_keywords", [])
        expected_src = q.get("expected_source")

        # Retrieve
        retrieved = retriever.retrieve(
            query=question,
            k=k,
            use_reranker=False,
        )
        retrieved_texts = [r.get("text", "") for r in retrieved]

        # Retrieval metrics
        prec = precision_at_k(retrieved_texts, expected_kw, k)
        mrr_val = mrr(retrieved_texts, expected_kw)
        precisions.append(prec)
        mrr_scores.append(mrr_val)

        # Generation (optional)
        answer = ""
        if not skip_generation and llm and retrieved:
            prompt = prompt_builder.build_prompt(question=question, results=retrieved)
            gen_config = GenerationConfig(
                temperature=config.llm.ollama.temperature,
                max_tokens=config.llm.ollama.max_tokens,
            )
            answer = llm.generate(prompt, gen_config)
            faith = answer_faithfulness(answer, expected_kw)
            faithfulness_scores.append(faith)

        results.append({
            "question": question,
            "precision_at_k": prec,
            "mrr": mrr_val,
            "answer_faithfulness": answer_faithfulness(answer, expected_kw) if answer else None,
            "num_retrieved": len(retrieved),
        })

    # Aggregate
    metrics = {
        "precision_at_k_mean": sum(precisions) / len(precisions) if precisions else 0,
        "mrr_mean": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
        "answer_faithfulness_mean": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
        "num_questions": len(questions),
        "per_question": results,
    }

    # Print summary
    print("\n" + "=" * 50)
    print("RAG Evaluation Results")
    print("=" * 50)
    print(f"Precision@{k}: {metrics['precision_at_k_mean']:.3f}")
    print(f"MRR:          {metrics['mrr_mean']:.3f}")
    if metrics["answer_faithfulness_mean"] is not None:
        print(f"Faithfulness: {metrics['answer_faithfulness_mean']:.3f}")
    print("=" * 50)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).parent / "test_questions.json",
        help="Path to test_questions.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only evaluate retrieval (skip LLM generation)",
    )
    args = parser.parse_args()

    run_evaluation(
        questions_path=args.questions,
        config_path=args.config,
        data_dir=args.data_dir,
        output_path=args.output,
        skip_generation=args.retrieval_only,
    )


if __name__ == "__main__":
    main()
