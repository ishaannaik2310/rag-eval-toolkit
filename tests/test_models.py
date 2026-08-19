"""
Unit tests for rag_eval core data models.
"""

import json
import pytest
from rag_eval.models import (
    Chunk,
    EvalRunResult,
    EvalSet,
    MetricResult,
    TestCase,
    TestCaseResult,
)


class TestChunk:
    def test_chunk_creation(self):
        chunk = Chunk(text="Paris is the capital of France.", id="c1", score=0.95, is_relevant=True)
        assert chunk.id == "c1"
        assert chunk.text == "Paris is the capital of France."
        assert chunk.score == 0.95
        assert chunk.is_relevant is True
        assert str(chunk) == "Paris is the capital of France."

    def test_chunk_defaults(self):
        chunk = Chunk(text="Sample text")
        assert chunk.id is None
        assert chunk.metadata == {}
        assert chunk.score is None
        assert chunk.is_relevant is None


class TestTestCase:
    def test_context_normalization_strings(self):
        tc = TestCase(
            id="test-1",
            query="What is RAG?",
            contexts=["Context 1", "Context 2"],
            actual_output="RAG stands for Retrieval-Augmented Generation.",
        )
        assert len(tc.contexts) == 2
        assert isinstance(tc.contexts[0], Chunk)
        assert tc.contexts[0].text == "Context 1"
        assert tc.contexts[1].text == "Context 2"
        assert tc.context_texts == ["Context 1", "Context 2"]

    def test_context_normalization_dicts(self):
        tc = TestCase(
            id="test-2",
            query="What is RAG?",
            contexts=[
                {"text": "Chunk from dict", "id": "d1", "score": 0.88, "is_relevant": True},
                {"text": "Another chunk", "metadata": {"source": "wiki"}},
            ],
            actual_output="RAG is...",
        )
        assert len(tc.contexts) == 2
        assert tc.contexts[0].id == "d1"
        assert tc.contexts[0].score == 0.88
        assert tc.contexts[0].is_relevant is True
        assert tc.contexts[1].metadata == {"source": "wiki"}

    def test_context_normalization_mixed(self):
        tc = TestCase(
            id="test-3",
            query="What is RAG?",
            contexts=[
                "String chunk",
                Chunk(text="Object chunk", id="obj1"),
                {"text": "Dict chunk"},
            ],
            actual_output="Output",
        )
        assert len(tc.contexts) == 3
        assert tc.contexts[0].text == "String chunk"
        assert tc.contexts[1].id == "obj1"
        assert tc.contexts[2].text == "Dict chunk"
        assert tc.context_texts == ["String chunk", "Object chunk", "Dict chunk"]


class TestEvalSet:
    @pytest.fixture
    def sample_eval_set(self) -> EvalSet:
        return EvalSet(
            name="rag_bench_v1",
            description="Benchmark test suite for RAG evaluation",
            test_cases=[
                TestCase(
                    id="tc-1",
                    query="What is the capital of Japan?",
                    contexts=["Tokyo is the capital of Japan."],
                    expected_output="Tokyo",
                    actual_output="The capital of Japan is Tokyo.",
                ),
                TestCase(
                    id="tc-2",
                    query="What is photosynthesis?",
                    contexts=["Photosynthesis converts light into energy."],
                    actual_output="Photosynthesis produces energy.",
                ),
            ],
        )

    def test_eval_set_to_from_yaml(self, sample_eval_set: EvalSet, tmp_path):
        yaml_str = sample_eval_set.to_yaml()
        assert "rag_bench_v1" in yaml_str
        assert "Tokyo" in yaml_str

        # Load from YAML string
        loaded = EvalSet.from_yaml(yaml_str)
        assert loaded.name == sample_eval_set.name
        assert len(loaded.test_cases) == 2
        assert loaded.test_cases[0].id == "tc-1"

        # Load from YAML file
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(yaml_str, encoding="utf-8")
        loaded_file = EvalSet.from_yaml(yaml_file)
        assert loaded_file.name == sample_eval_set.name
        assert len(loaded_file.test_cases) == 2

    def test_eval_set_to_from_json(self, sample_eval_set: EvalSet, tmp_path):
        json_str = sample_eval_set.to_json()
        assert "rag_bench_v1" in json_str

        # Load from JSON string
        loaded = EvalSet.from_json(json_str)
        assert loaded.name == sample_eval_set.name
        assert len(loaded.test_cases) == 2
        assert loaded.test_cases[1].id == "tc-2"

        # Load from JSON file
        json_file = tmp_path / "eval.json"
        json_file.write_text(json_str, encoding="utf-8")
        loaded_file = EvalSet.from_json(json_file)
        assert loaded_file.name == sample_eval_set.name
        assert len(loaded_file.test_cases) == 2


class TestMetricResult:
    def test_metric_result_passed_auto_calculation(self):
        m_pass = MetricResult(name="context_precision", score=0.85, threshold=0.7)
        assert m_pass.passed is True

        m_fail = MetricResult(name="faithfulness", score=0.5, threshold=0.7)
        assert m_fail.passed is False

    def test_metric_result_passed_explicit_override(self):
        m = MetricResult(name="custom_metric", score=0.4, threshold=0.7, passed=True)
        assert m.passed is True


class TestTestCaseResult:
    def test_test_case_result_passed_auto(self):
        # All passed -> overall passed
        res_pass = TestCaseResult(
            test_case_id="tc-1",
            query="test query",
            metric_results=[
                MetricResult(name="context_precision", score=0.9, threshold=0.7),
                MetricResult(name="faithfulness", score=0.8, threshold=0.7),
            ],
            execution_time_ms=120.5,
        )
        assert res_pass.passed is True

        # One failed -> overall failed
        res_fail = TestCaseResult(
            test_case_id="tc-2",
            query="test query 2",
            metric_results=[
                MetricResult(name="context_precision", score=0.9, threshold=0.7),
                MetricResult(name="faithfulness", score=0.4, threshold=0.7),
            ],
            execution_time_ms=98.0,
        )
        assert res_fail.passed is False


class TestEvalRunResult:
    def test_eval_run_result_create_and_summary(self):
        tc1 = TestCaseResult(
            test_case_id="tc-1",
            query="query 1",
            metric_results=[
                MetricResult(name="context_precision", score=1.0, threshold=0.7),
                MetricResult(name="faithfulness", score=0.9, threshold=0.7),
            ],
            execution_time_ms=100.0,
        )
        tc2 = TestCaseResult(
            test_case_id="tc-2",
            query="query 2",
            metric_results=[
                MetricResult(name="context_precision", score=0.6, threshold=0.7),
                MetricResult(name="faithfulness", score=0.5, threshold=0.7),
            ],
            execution_time_ms=150.0,
        )

        run_result = EvalRunResult.create(
            eval_set_name="test_suite",
            test_case_results=[tc1, tc2],
            duration_seconds=2.5,
        )

        assert run_result.total_tests == 2
        assert run_result.passed_tests == 1
        assert run_result.failed_tests == 1
        assert run_result.passed is False
        assert run_result.duration_seconds == 2.5
        assert run_result.average_scores["context_precision"] == 0.8
        assert run_result.average_scores["faithfulness"] == 0.7

        summary = run_result.summary()
        assert summary["eval_set_name"] == "test_suite"
        assert summary["total_tests"] == 2
        assert summary["passed_tests"] == 1
        assert summary["failed_tests"] == 1
        assert summary["pass_rate"] == 50.0
        assert summary["average_scores"] == {"context_precision": 0.8, "faithfulness": 0.7}
        assert summary["duration_seconds"] == 2.5
        assert summary["passed"] is False

    def test_eval_run_result_all_passed(self):
        tc1 = TestCaseResult(
            test_case_id="tc-1",
            query="query 1",
            metric_results=[MetricResult(name="context_precision", score=1.0, threshold=0.7)],
        )
        run_result = EvalRunResult.create(
            eval_set_name="passing_suite",
            test_case_results=[tc1],
            duration_seconds=1.0,
        )
        assert run_result.passed is True
        assert run_result.summary()["pass_rate"] == 100.0
