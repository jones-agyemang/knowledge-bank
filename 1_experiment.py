import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, set_trace_processors
from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor

load_dotenv()

async def main():
    agent = Agent(
        name="Anne Robinson",
        instructions="You are a stern, no nonesense quiz mistress"
    )

    question = "What is the sum of 50 and 150?"
    result = await Runner.run(agent, question)


if __name__ == "__main__":
    print('Experiment started...')
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    asyncio.run(main())
    print('Experiment complete!')
    print('---'*3, end="\n\n")
