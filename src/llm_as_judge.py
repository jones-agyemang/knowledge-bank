"""
LLM AS JUDGE

PoC for to free-text, Rubrics-based evaluation of LLM output
"""

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from langsmith import traceable, wrappers, Client
from langsmith.evaluation import EvaluationResult
from langsmith.utils import LangSmithConflictError

load_dotenv()

# wrap to automatically trace all LLM calls
oai_client = wrappers.wrap_openai(OpenAI())

"""
inputs
"""
@traceable()
def quiz_master(inputs: dict) -> dict:
    #Returns the LLMs response to a question
    #Parameters:
    #   inputs (dict): The question to be answered
    #Returns:
    #   question_response (str): Returns the response to the question posed

    instructions=("Respond to the given question with a sentence or two.")
    
    messages = [
        { "role": "system", "content": instructions },
        { "role": "user", "content": inputs.get('question') }
    ]

    response = oai_client.chat.completions.create(
        messages=messages, model="gpt-5.4-mini", temperature=0
    )
    return { "question_response": response.choices[0].message.content }

def correctness_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
) -> EvaluationResult:
    judgement_response = judgement(inputs, outputs, reference_outputs)

    return EvaluationResult(
        key="correctnes_evaluator",
        score=judgement_response.get("score"),
        comment=judgement_response.get("explanation")
    )

class Response(BaseModel):
    score: float = 0.0
    explanation: str = ""

def judgement(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
) -> dict:
    instructions = ("You are a fair judge for a free-text quiz competition")

    message_content = f"""
        Question: {inputs['question']}. 
        Response: {outputs['question_response']}. 
        Answer: {reference_outputs['answer']}
    """

    messages = [
        { "role": "system", "content": instructions },
        { "role": "user", "content": message_content }
    ]

    response = oai_client.beta.chat.completions.parse(
        messages=messages,
        model="gpt-5.4-mini",
        temperature=0,
        response_format=Response
    )
    return response.choices[0].message.parsed.model_dump()
def fetch_dataset():
    try:
        return ls_client.create_dataset(dataset_name="Mini Knowledge Bank")
    except LangSmithConflictError:
        return ls_client.read_dataset(dataset_name="Mini Knowledge Bank")

ls_client = Client()
dataset = fetch_dataset()

results = ls_client.evaluate(
    quiz_master,
    data=dataset.name,
    evaluators=[correctness_evaluator],
    experiment_prefix="gpt-5.4-mini, baseline"
)
