import asyncio
import dotenv
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context

dotenv.load_dotenv()

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b



# Create an agent workflow with our calculator tool
agent = AgentWorkflow.from_tools_or_functions(
    [multiply],
    llm=Ollama(
        model="llama3.2",
        request_timeout=360.0
    ),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)

# create context
ctx = Context(agent)


async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?", ctx=ctx)
    print(str(response))
    # run agent with context
    response = await agent.run("My name is Logan", ctx=ctx)
    print(str(response))
    response = await agent.run("What is my name?", ctx=ctx)
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
