from src.llm_as_judge import judgement
from src.llm_as_judge import correctness_evaluator
from src.llm_as_judge import Response

from unittest.mock import patch
from langsmith.evaluation import EvaluationResult
from types import SimpleNamespace as ns

import pytest

@pytest.fixture(scope="module")
def vcr_config():
    return { "filter_headers": ["authorization"] }

@pytest.fixture
def judgement_data():
    inputs = { 'question': 'What is a fish?' }
    outputs = { 'question_response': 'An animal that lives in water.' }
    reference_outputs = { 'answer': 'A fish is an aquatic, gill-bearing vetebrate animal.' }
    return inputs, outputs, reference_outputs

def describe_judgement():

    def when_message_body_is_well_formed():

        def judgement_returns_scoring_response(judgement_data):
            with patch("src.llm_as_judge.oai_client") as oai_client_mock:
                mock_llm_response = ns(
                    choices=[
                        ns(
                            message=ns(
                                parsed=Response(
                                    score=0.9,
                                    explanation="The response is essentially correct and matches the core idea of the answer: a fish is an aquatic animal that lives in water. It omits details like being a vertebrate and gill-bearing, but it is still a good paraphrase."
                                )
                            )
                        )
                    ]
                )
                oai_client_mock.beta.chat.completions.parse.return_value = mock_llm_response

                inputs = { 'question': 'What is a fish?' }
                outputs = { 'question_response': 'An animal that lives in water.' }
                reference_outputs = { 'answer': 'A fish is an aquatic, gill-bearing vetebrate animal.' }

                expected_response = {
                    "score": 0.9,
                    "explanation": "The response is essentially correct and matches the core idea of the answer: a fish is an aquatic animal that lives in water. It omits details like being a vertebrate and gill-bearing, but it is still a good paraphrase."
                }

                response = judgement(*judgement_data)

                assert response.get('score') == expected_response.get('score')
                assert response.get('explanation') == expected_response.get('explanation')

def describe_evaluation():

    def when_judgement_succeeds():
        
        def correctness_evaluator_returns_evaluated_score(judgement_data):
            with patch("src.llm_as_judge.judgement") as judgement_mock:
                judgement_mock.return_value = {
                    "score": 0.1,
                    "explanation": "Not even close!"
                }
                evaluation = correctness_evaluator(*judgement_data)

                assert evaluation.key == "correctness_evaluator"
                assert evaluation.score > 0.0
                assert evaluation.comment == "Not even close!"


    def when_judgement_fails():

        def correctness_evaluator_returns_fallback_value(judgement_data):
            with patch("src.llm_as_judge.judgement") as judgement_mock:
                judgement_mock.side_effect = RuntimeError("Simulate runtime failure")

                evaluation = correctness_evaluator(*judgement_data)
                assert evaluation.key == "correctness_evaluator"
                assert evaluation.score == 0.0
                assert evaluation.comment == "Error whilst evaluating data point"

                judgement_mock.assert_called_once_with(*judgement_data)