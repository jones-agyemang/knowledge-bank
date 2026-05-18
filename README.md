# Knowledge Bank
Knowledge Bank is proof-of-concept AI-Native application. This uses [Langchain](www.langchain.com) to evaluate non-deterministic LLM responses.

## Evaluation
In this simple example, we have a quiz with simple responses. We'll then set up experiments along with a database of questions and responses that we expect. We'll then send off requests to our LLM and then compare the exactness of their matches. To begin with, there are various ways of testing or evaluating AI applications, but for now we're doing this to prove the concept. 

## Installation
```sh
pip install -r requirements.txt
```

## Running
Ensure that you've enabled this [setting](https://code.visualstudio.com/docs/python/environments) in order for your environmental variables to be picked up. Alternatively, you can use [py-dotenv](https://pypi.org/project/python-dotenv/) for an editor-agnostic setup.