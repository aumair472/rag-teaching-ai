"""
RAG evaluation module using RAGAS.

Computes faithfulness, answer relevancy, and context precision
metrics and logs results to ``logs/eval_metrics.json``.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import EvalResult, EvalSample

logger = get_logger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using RAGAS metrics.

    Metrics computed:
        - **Faithfulness**: Is the answer faithful to the provided context?
        - **Answer Relevancy**: Is the answer relevant to the question?
        - **Context Precision**: Is the most relevant context ranked highest?

    Results are saved to a JSON log file for tracking over time.

    Attributes:
        output_path: Path to the evaluation metrics log file.
    """

    def __init__(self, output_path: Optional[str] = None) -> None:
        settings = get_settings()
        self.output_path = output_path or str(
            Path(settings.logs_dir) / "eval_metrics.json"
        )
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def evaluate(self, samples: List[EvalSample]) -> EvalResult:
        """
        Run RAGAS evaluation on a list of samples.

        Each sample must include:
            - question
            - answer (generated)
            - contexts (retrieved passages)
            - ground_truth (optional, for context precision)

        Args:
            samples: List of ``EvalSample`` objects.

        Returns:
            An ``EvalResult`` with computed metrics.

        Note:
            Requires ``OPENAI_API_KEY`` in environment for RAGAS
            (it uses an LLM judge internally).
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )
        except ImportError as exc:
            logger.error(
                "RAGAS or datasets not installed",
                extra={"error": str(exc)},
            )
            return EvalResult(num_samples=0)

        logger.info(
            "Starting RAGAS evaluation",
            extra={"num_samples": len(samples)},
        )

        # Build dataset in RAGAS format
        data: Dict[str, List[Any]] = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for sample in samples:
            data["question"].append(sample.question)
            data["answer"].append(sample.answer)
            data["contexts"].append(sample.contexts)
            data["ground_truth"].append(sample.ground_truth or "")

        dataset = Dataset.from_dict(data)

        # Select metrics
        metrics = [faithfulness, answer_relevancy, context_precision]

        try:
            result = evaluate(dataset=dataset, metrics=metrics)
            scores = result.to_pandas().mean().to_dict()
        except Exception as exc:
            logger.error(
                "RAGAS evaluation failed",
                extra={"error": str(exc)},
            )
            return EvalResult(num_samples=len(samples))

        eval_result = EvalResult(
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_precision=scores.get("context_precision"),
            num_samples=len(samples),
        )

        # Log results
        self._save_results(eval_result)

        logger.info(
            "RAGAS evaluation complete",
            extra={
                "faithfulness": eval_result.faithfulness,
                "answer_relevancy": eval_result.answer_relevancy,
                "context_precision": eval_result.context_precision,
            },
        )

        return eval_result

    def _save_results(self, result: EvalResult) -> None:
        """
        Append evaluation results to the JSON log file.

        Args:
            result: The evaluation result to save.
        """
        entry = result.model_dump()
        entry["evaluated_at"] = entry["evaluated_at"].isoformat()

        # Load existing entries
        existing: List[Dict[str, Any]] = []
        path = Path(self.output_path)
        if path.exists():
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, Exception):
                existing = []

        existing.append(entry)

        with open(path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        logger.info(
            "Evaluation metrics saved",
            extra={"path": self.output_path},
        )
