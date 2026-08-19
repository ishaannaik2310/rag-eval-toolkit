"""
Core data models for RAG Eval using Pydantic v2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Chunk(BaseModel):
    """Represents a retrieved chunk of context in a RAG pipeline."""
    id: Optional[str] = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
    is_relevant: Optional[bool] = None

    def __str__(self) -> str:
        return self.text


class TestCase(BaseModel):
    """Represents a single RAG evaluation test case."""
    __test__ = False
    id: str
    query: str
    contexts: list[Chunk] = Field(default_factory=list)
    expected_output: Optional[str] = None
    actual_output: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("contexts", mode="before")
    @classmethod
    def normalize_contexts(cls, value: Any) -> list[Any]:
        """Convert a list of raw strings or dicts into Chunk objects."""
        if not isinstance(value, list):
            return value
        normalized = []
        for item in value:
            if isinstance(item, str):
                normalized.append(Chunk(text=item))
            elif isinstance(item, dict):
                normalized.append(Chunk(**item))
            elif isinstance(item, Chunk):
                normalized.append(item)
            else:
                normalized.append(Chunk(text=str(item)))
        return normalized

    @property
    def context_texts(self) -> list[str]:
        """Return raw context strings."""
        return [c.text for c in self.contexts]


class EvalSet(BaseModel):
    """Represents a collection of RAG test cases."""
    name: str
    description: Optional[str] = None
    test_cases: list[TestCase] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, content_or_path: Union[str, Path]) -> EvalSet:
        """Load an EvalSet from a YAML string or file path."""
        path = Path(content_or_path) if isinstance(content_or_path, (str, Path)) else None
        if path and path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(str(content_or_path))
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, content_or_path: Union[str, Path]) -> EvalSet:
        """Load an EvalSet from a JSON string or file path."""
        path = Path(content_or_path) if isinstance(content_or_path, (str, Path)) else None
        if path and path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(content_or_path))
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize EvalSet to YAML string."""
        return yaml.dump(self.model_dump(mode="json"), sort_keys=False)

    def to_json(self, indent: int = 2) -> str:
        """Serialize EvalSet to JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class MetricResult(BaseModel):
    """Result of evaluating a specific metric on a test case."""
    name: str
    score: float = Field(ge=0.0, le=1.0)
    reason: Optional[str] = None
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    passed: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def calculate_passed(cls, values: Any) -> Any:
        if isinstance(values, dict):
            score = values.get("score", 0.0)
            threshold = values.get("threshold", 0.7)
            if "passed" not in values:
                values["passed"] = score >= threshold
        return values


class TestCaseResult(BaseModel):
    """Aggregated evaluation results for a single test case."""
    __test__ = False
    test_case_id: str
    query: str
    metric_results: list[MetricResult] = Field(default_factory=list)
    passed: bool = True
    execution_time_ms: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def evaluate_overall_passed(cls, values: Any) -> Any:
        if isinstance(values, dict):
            metrics = values.get("metric_results", [])
            if "passed" not in values:
                values["passed"] = all(
                    m.passed if isinstance(m, MetricResult) else m.get("passed", True)
                    for m in metrics
                )
        return values


class EvalRunResult(BaseModel):
    """Complete summary of an evaluation run across an EvalSet."""
    eval_set_name: str
    test_case_results: list[TestCaseResult] = Field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    average_scores: dict[str, float] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    passed: bool = True

    @classmethod
    def create(
        cls,
        eval_set_name: str,
        test_case_results: list[TestCaseResult],
        duration_seconds: float = 0.0,
    ) -> EvalRunResult:
        """Factory method that calculates totals, pass/fail counts, and average metric scores."""
        total = len(test_case_results)
        passed_count = sum(1 for tc in test_case_results if tc.passed)
        failed_count = total - passed_count

        scores_by_metric: dict[str, list[float]] = {}
        for tc in test_case_results:
            for m in tc.metric_results:
                scores_by_metric.setdefault(m.name, []).append(m.score)

        avg_scores = {
            metric: round(sum(scores) / len(scores), 4) if scores else 0.0
            for metric, scores in scores_by_metric.items()
        }

        return cls(
            eval_set_name=eval_set_name,
            test_case_results=test_case_results,
            total_tests=total,
            passed_tests=passed_count,
            failed_tests=failed_count,
            average_scores=avg_scores,
            duration_seconds=duration_seconds,
            passed=(failed_count == 0),
        )

    def summary(self) -> dict[str, Any]:
        """Return a concise summary dictionary of the evaluation run."""
        pass_rate = (self.passed_tests / self.total_tests * 100.0) if self.total_tests > 0 else 0.0
        return {
            "eval_set_name": self.eval_set_name,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": round(pass_rate, 2),
            "average_scores": self.average_scores,
            "duration_seconds": round(self.duration_seconds, 3),
            "passed": self.passed,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize EvalRunResult to JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)
