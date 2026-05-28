from src.llm_as_judge import judgement
from src.llm_as_judge import correctness_evaluator
from unittest.mock import patch
from langsmith.evaluation import EvaluationResult
# from langsmith.utils import DynamicRunEvaluator

import pytest

@pytest.fixture(scope="module")
def vcr_config():
    return { "filter_headers": ["authorization"] }

@pytest.mark.vcr
@pytest.mark.skip(reason="Mock network calls")
def test_valid_judgement_response():
    inputs = {
        'question': 'What is a fish?',
    }
    outputs = {
        'question_response': 'An animal that lives in water.'
    }
    reference_outputs = {
        'answer': 'A fish is an aquatic, gill-bearing vetebrate animal.',
    }

    expected_response = { 
        "score": 0.9,
        "explanation": "The response is essentially correct and matches the core idea of the answer: a fish is an aquatic animal that lives in water. It omits details like being a vertebrate and gill-bearing, but it is still a good paraphrase."
    }

    response = judgement(inputs, outputs, reference_outputs)
    
    assert response.get('score') == expected_response.get('score')
    assert response.get('explanation') == expected_response.get('explanation')


def describe_evaluation():
    def test_correctness_evaluator_returns_fallback_value_when_judgement_fails():
        inputs = { "inputs": { "question": "What is a fish?" } }
        outputs = { "outputs": { "question_response": "Animal that lives in the sea" } }
        reference_outputs = { "reference_outputs": { "answer": "Foo Bar" } }

        with patch("src.llm_as_judge.judgement") as judgement_mock:
            judgement_mock.side_effect = RuntimeError("Simulate runtime failure")

            evaluation = correctness_evaluator(inputs, outputs, reference_outputs)
            assert evaluation.key == "correctness_evaluator"
            assert evaluation.score == 0.0
            assert evaluation.comment == "Error whilst evaluating data point"

            judgement_mock.assert_called_once_with(inputs, outputs, reference_outputs)