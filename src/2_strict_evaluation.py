from langsmith import traceable, wrappers, Client
from langsmith.utils import LangSmithConflictError
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

oai_client = wrappers.wrap_openai(OpenAI())
ls_client = Client()

examples = [
    {
        "inputs": { "question": "What is machine learning?" },
        "outputs": { "answer": "Machine learning is a field of AI where systems learn patterns from data to make predictions, classifications, or decisions without being explicitly programmed for every rule." }
    },
    {
        "inputs": { "question": "What is Naive Bayes?" },
        "outputs": { "answer": "Naive Bayes is a probabilistic classifier based on Bayes' theorem that assumes features are conditionally independent given the class" }
    }
]

EXPERIMENT_METADATA = {
    "models": [
        "openai:gpt-5.4-mini",
        {
            "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
            "lc": 1,
            "type": "constructor",
            "kwargs": {"model_name": "gpt-5.4", "temperature": 0.2}
        }
    ],
    "prompts": ["mem/correctness-eval-prompt:AtX1j9"],
    "tools": [
        {
            "name": "Answer check",
            "description": "Find an answer to my question",
            "parameters": {
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }
        }
    ],
}

@traceable()
def quiz_master(inputs: dict) -> dict:
    instructions=("You are a stern, no nonesense quiz mistress")
    
    messages = [
        { "role": "system", "content": instructions },
        { "role": "user", "content": inputs["question"] }
    ]

    response = oai_client.chat.completions.create(
        messages=messages, model="gpt-5.4-mini", temperature=0
    )
    return { "answer": response.choices[0].message.content }

def correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    return outputs["answer"] == reference_outputs["answer"]


def evaluate(dataset):
    return ls_client.evaluate(
        quiz_master,
        data=dataset.name,
        evaluators=[correct],
        experiment_prefix="gpt-5.4-mini, baseline",
        description="Analyse quality of responses",
        metadata=EXPERIMENT_METADATA
    )

if __name__ == "__main__":
    print('Begin evaluation...')

    # prepare dataset
    try:
        dataset = ls_client.create_dataset(dataset_name="Mini Knowledge Bank")
    except LangSmithConflictError:
        dataset = ls_client.read_dataset(dataset_name="Mini Knowledge Bank")

    # create quiz examples for the experiment
    ls_client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )
    
    evaluation_report = evaluate(dataset)
    print(f"Evaluation report: {evaluation_report}")
    print('End of evaluation!', end='\n\n')

