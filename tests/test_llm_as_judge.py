from src.llm_as_judge import judgement
import pytest

@pytest.fixture(scope="module")
def vcr_config():
    return { "filter_headers": ["authorization"] }

@pytest.mark.vcr
def test_valid_judgement_response():
    input = {
        'question': 'What is a fish?',
        'answer': 'A fish is an aquatic, gill-bearing vetebrate animal.',
        'question_response': 'An animal that lives in water.'
    }

    expected_response = { 
        "score": 0.9,
        "explanation": "The response is essentially correct and matches the core idea of the answer: a fish is an aquatic animal that lives in water. It omits details like being a vertebrate and gill-bearing, but it is still a good paraphrase."
    }

    response = judgement(input)
    
    assert response.get('score') == expected_response.get('score')
    assert response.get('explanation') == expected_response.get('explanation')