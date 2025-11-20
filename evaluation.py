"""
Comprehensive RAG Evaluation System
Implements multiple metrics to assess retrieval and generation quality
"""

import json
import os

# Disable ChromaDB telemetry to avoid error messages
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

from main import AmbedkarGPT
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class RAGEvaluator:
    def __init__(self, test_dataset_path="test_dataset.json"):
        """
        Initialize evaluator

        Args:
            test_dataset_path: Path to test dataset JSON file
        """
        self.test_dataset = self.load_test_dataset(test_dataset_path)
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.results = []

    def load_test_dataset(self, path):
        """Load test dataset from JSON"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["test_questions"]

    def compute_hit_rate(
        self, retrieved_docs: List[str], relevant_docs: List[str]
    ) -> int:
        """
        Compute if any relevant document was retrieved (binary)

        Returns:
            1 if hit, 0 if miss
        """
        if not relevant_docs:
            return 0

        for retrieved in retrieved_docs:
            for relevant in relevant_docs:
                if relevant in retrieved:
                    return 1
        return 0

    def compute_mrr(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Compute Mean Reciprocal Rank

        Returns:
            Reciprocal rank of first relevant document
        """
        if not relevant_docs:
            return 0.0

        for rank, retrieved in enumerate(retrieved_docs, 1):
            for relevant in relevant_docs:
                if relevant in retrieved:
                    return 1.0 / rank
        return 0.0

    def compute_precision_at_k(
        self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 3
    ) -> float:
        """
        Compute Precision@K

        Returns:
            Proportion of relevant documents in top-k
        """
        if not relevant_docs:
            return 0.0

        retrieved_k = retrieved_docs[:k]
        relevant_count = 0

        for retrieved in retrieved_k:
            for relevant in relevant_docs:
                if relevant in retrieved:
                    relevant_count += 1
                    break

        return relevant_count / k

    def compute_rouge_l(self, generated: str, reference: str) -> float:
        """
        Compute ROUGE-L score

        Returns:
            ROUGE-L F1 score
        """
        scores = self.rouge_scorer.score(reference, generated)
        return scores["rougeL"].fmeasure

    def compute_bleu(self, generated: str, reference: str) -> float:
        """
        Compute BLEU score

        Returns:
            BLEU score
        """
        reference_tokens = [reference.lower().split()]
        generated_tokens = generated.lower().split()

        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            reference_tokens, generated_tokens, smoothing_function=smoothing
        )
        return score

    def compute_cosine_similarity(
        self, text1: str, text2: str, embeddings_model
    ) -> float:
        """
        Compute cosine similarity between two texts

        Returns:
            Cosine similarity score
        """
        emb1 = embeddings_model.embed_query(text1)
        emb2 = embeddings_model.embed_query(text2)

        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)

    def compute_faithfulness(self, answer: str, context: str) -> float:
        """
        Simple faithfulness check - are answer statements present in context?

        Returns:
            Approximate faithfulness score (0-1)
        """
        answer_sentences = answer.split(".")
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]

        if not answer_sentences:
            return 0.0

        faithful_count = 0
        for sentence in answer_sentences:
            # Simple word overlap check
            answer_words = set(sentence.lower().split())
            context_words = set(context.lower().split())

            if len(answer_words) == 0:
                continue

            overlap = len(answer_words.intersection(context_words)) / len(answer_words)
            if overlap > 0.5:  # More than 50% words from context
                faithful_count += 1

        return faithful_count / len(answer_sentences)

    def compute_answer_relevance(
        self, answer: str, question: str, embeddings_model
    ) -> float:
        """
        Compute answer relevance using cosine similarity

        Returns:
            Relevance score (0-1)
        """
        return self.compute_cosine_similarity(answer, question, embeddings_model)

    def evaluate_single_question(
        self, qa_system: AmbedkarGPT, question_data: Dict
    ) -> Dict:
        """
        Evaluate a single question

        Returns:
            Dictionary with all metrics
        """
        question = question_data["question"]
        ground_truth = question_data["ground_truth"]
        relevant_docs = question_data["source_documents"]
        is_answerable = question_data["answerable"]

        # Get answer from system
        try:
            result = qa_system.ask(question)
            generated_answer = result["result"]
            source_docs = result["source_documents"]
        except Exception as e:
            print(f"Error processing question {question_data['id']}: {e}")
            return None

        # Extract source document names
        retrieved_doc_names = [
            doc.metadata.get("source", "").split("/")[-1] for doc in source_docs
        ]

        # Combine context from retrieved documents
        context = " ".join([doc.page_content for doc in source_docs])

        # Compute metrics
        metrics = {
            "question_id": question_data["id"],
            "question": question,
            "question_type": question_data["question_type"],
            "is_answerable": is_answerable,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "retrieved_docs": retrieved_doc_names,
            "relevant_docs": relevant_docs,
        }

        # Retrieval metrics
        metrics["hit_rate"] = self.compute_hit_rate(retrieved_doc_names, relevant_docs)
        metrics["mrr"] = self.compute_mrr(retrieved_doc_names, relevant_docs)
        metrics["precision_at_3"] = self.compute_precision_at_k(
            retrieved_doc_names, relevant_docs, k=3
        )

        # Answer quality metrics (only for answerable questions)
        if is_answerable:
            metrics["rouge_l"] = self.compute_rouge_l(generated_answer, ground_truth)
            metrics["bleu"] = self.compute_bleu(generated_answer, ground_truth)
            metrics["cosine_similarity"] = self.compute_cosine_similarity(
                generated_answer, ground_truth, qa_system.embeddings
            )
            metrics["faithfulness"] = self.compute_faithfulness(
                generated_answer, context
            )
            metrics["answer_relevance"] = self.compute_answer_relevance(
                generated_answer, question, qa_system.embeddings
            )
        else:
            # For unanswerable questions, check if system correctly refused
            refuses = any(
                phrase in generated_answer.lower()
                for phrase in [
                    "cannot answer",
                    "not available",
                    "don't know",
                    "no information",
                ]
            )
            metrics["correct_refusal"] = 1 if refuses else 0

        return metrics

    def evaluate_all(self, qa_system: AmbedkarGPT) -> List[Dict]:
        """
        Evaluate all questions in test dataset

        Returns:
            List of results for each question
        """
        print(f"Evaluating {len(self.test_dataset)} questions...\n")

        results = []
        for i, question_data in enumerate(self.test_dataset, 1):
            print(
                f"[{i}/{len(self.test_dataset)}] Evaluating question {question_data['id']}..."
            )

            result = self.evaluate_single_question(qa_system, question_data)
            if result:
                results.append(result)

        self.results = results
        return results

    def compute_aggregate_metrics(self) -> Dict:
        """
        Compute aggregate metrics across all questions

        Returns:
            Dictionary with average metrics
        """
        if not self.results:
            return {}

        answerable_results = [r for r in self.results if r["is_answerable"]]
        unanswerable_results = [r for r in self.results if not r["is_answerable"]]

        aggregates = {
            "total_questions": len(self.results),
            "answerable_questions": len(answerable_results),
            "unanswerable_questions": len(unanswerable_results),
        }

        # Retrieval metrics (all questions)
        aggregates["avg_hit_rate"] = np.mean([r["hit_rate"] for r in self.results])
        aggregates["avg_mrr"] = np.mean([r["mrr"] for r in self.results])
        aggregates["avg_precision_at_3"] = np.mean(
            [r["precision_at_3"] for r in self.results]
        )

        # Answer quality metrics (answerable only)
        if answerable_results:
            aggregates["avg_rouge_l"] = np.mean(
                [r["rouge_l"] for r in answerable_results]
            )
            aggregates["avg_bleu"] = np.mean([r["bleu"] for r in answerable_results])
            aggregates["avg_cosine_similarity"] = np.mean(
                [r["cosine_similarity"] for r in answerable_results]
            )
            aggregates["avg_faithfulness"] = np.mean(
                [r["faithfulness"] for r in answerable_results]
            )
            aggregates["avg_answer_relevance"] = np.mean(
                [r["answer_relevance"] for r in answerable_results]
            )

        # Unanswerable handling
        if unanswerable_results:
            aggregates["correct_refusal_rate"] = np.mean(
                [r["correct_refusal"] for r in unanswerable_results]
            )

        # Question type breakdown
        type_breakdown = defaultdict(list)
        for r in answerable_results:
            type_breakdown[r["question_type"]].append(r)

        aggregates["by_question_type"] = {}
        for qtype, results in type_breakdown.items():
            aggregates["by_question_type"][qtype] = {
                "count": len(results),
                "avg_rouge_l": np.mean([r["rouge_l"] for r in results]),
                "avg_hit_rate": np.mean([r["hit_rate"] for r in results]),
            }

        return aggregates


def compare_chunking_strategies():
    """
    Compare different chunking strategies
    """
    import shutil
    import time
    import gc

    strategies = [
        {"name": "Small", "chunk_size": 250, "overlap": 25},
        {"name": "Medium", "chunk_size": 550, "overlap": 50},
        {"name": "Large", "chunk_size": 900, "overlap": 100},
    ]

    all_results = {}

    for i, strategy in enumerate(strategies):
        print("\n" + "=" * 70)
        print(
            f"Testing {strategy['name']} Chunks (size={strategy['chunk_size']}, overlap={strategy['overlap']})"
        )
        print("=" * 70 + "\n")

        # Use unique persist directory for each strategy
        persist_dir = f"./chroma_db_{strategy['name'].lower()}"

        # Initialize system with this chunking strategy
        qa_system = AmbedkarGPT(
            corpus_path="corpus",
            chunk_size=strategy["chunk_size"],
            chunk_overlap=strategy["overlap"],
        )

        # Override persist directory in initialization
        documents = qa_system.load_documents()
        chunks = qa_system.split_documents(documents)
        qa_system.create_vectorstore(chunks, persist_directory=persist_dir)
        qa_system.setup_qa_chain()

        # Evaluate
        evaluator = RAGEvaluator("test_dataset.json")
        results = evaluator.evaluate_all(qa_system)
        aggregates = evaluator.compute_aggregate_metrics()

        all_results[strategy["name"]] = {
            "config": strategy,
            "detailed_results": results,
            "aggregates": aggregates,
        }

        # Save individual results
        output_file = f"test_results_{strategy['name'].lower()}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"config": strategy, "results": results, "aggregates": aggregates},
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\nResults saved to {output_file}")

        # Clean up to release resources before next iteration
        qa_system.vectorstore = None
        qa_system.qa_chain = None
        del qa_system
        del evaluator
        gc.collect()
        time.sleep(2)  # Give system time to release file handles

    return all_results


def main():
    """Main evaluation execution"""

    # Run comparative chunking analysis
    print("Starting Comprehensive RAG Evaluation\n")
    all_results = compare_chunking_strategies()

    # Save combined comparison
    with open("test_results_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("Evaluation complete and results saved.")


if __name__ == "__main__":
    main()
