"""
LLM AS JUDGE

PoC for to free-text, Rubrics-based evaluation of LLM output
"""

import asyncio
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from langsmith import traceable, wrappers, Client
from langsmith.evaluation import EvaluationResult
from langsmith.utils import LangSmithConflictError

load_dotenv()

# wrap to automatically trace all LLM calls
oai_client = wrappers.wrap_openai(OpenAI())

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
    try:
        judgement_response = judgement(inputs, outputs, reference_outputs)
        return EvaluationResult(
            key="correctness_evaluator",
            score=judgement_response.get("score"),
            comment=judgement_response.get("explanation")
        )
    except RuntimeError as exception:
        return EvaluationResult(
            key="correctness_evaluator",
            score=0.0,
            comment="Error whilst evaluating data point"
        )

SYSTEN_RUBRIC_PROMPT = """
You are an expert evaluator. Grade the model's response based on the following criteria:

    1. Accuracy (0-3 points):
        - 3 points: Response is perfectly accurate based on reference data
        - 2 points: Response is mostly accurate, minor details missing
        - 1 point: Significant inaccuracies or omissions
        - 0 points: Completely incorrect or non-responsive

    2. Reasoning Consistency:
        - Check if the provided reasoning logicallt supports the final answer
        - Deduct 1 point if the reasoning is flawed, even if the answer is correct

    Output your evaluation strictly in the following JSON format:
    {
        "score": (integer),
        "explanation": "Detailed explanation of your scoring"
    }
"""

class EvaluationResponse(BaseModel):
    score: float = 0.0
    explanation: str = ""

def judgement(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
) -> dict:
    #Returns the LLMs judgement of a response(output) to a question(input) against the actual answer(reference output)
    #Parameters:
    #   inputs (dict): The question requiring a response
    #   outputs (dict): The response to the question
    #   reference_outputs (dict): Ideal response (objective, ground-truth)
    #Returns:
    #   response (dict): e.g. { "score": "0.82", explanation: "Commendable response with minor missing details" }

    message_content = f"""
        Question: {inputs['question']}. 
        Response: {outputs['question_response']}. 
        Answer: {reference_outputs['answer']}
    """

    messages = [
        { "role": "system", "content": SYSTEN_RUBRIC_PROMPT },
        { "role": "user", "content": message_content }
    ]

    response = oai_client.beta.chat.completions.parse(
        messages=messages,
        model="gpt-5.4-mini",
        temperature=0,
        response_format=EvaluationResponse
    )
    return response.choices[0].message.parsed.model_dump()

def fetch_dataset(ls_client):
    try:
        return ls_client.create_dataset(dataset_name="Mini Knowledge Bank")
    except LangSmithConflictError:
        return ls_client.read_dataset(dataset_name="Mini Knowledge Bank")

async def main() -> None:
    ls_client = Client()
    dataset = fetch_dataset(ls_client)

    results = ls_client.evaluate(
        quiz_master,
        data=dataset.name,
        evaluators=[correctness_evaluator],
        experiment_prefix="gpt-5.4-mini, baseline"
    )

if __name__ == '__main__':
    asyncio.run(main())
