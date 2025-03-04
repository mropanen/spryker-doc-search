import asyncio
import dotenv
from llama_index.core.workflow import Context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
import asyncio
import os

dotenv.load_dotenv()

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
#Settings.llm = Ollama(
#    base_url=os.getenv('OLLAMA_BASEURI') or "http://localhost:11434",
#    model="llama3.2",
#    request_timeout=360.0
#)
Settings.llm = OpenAILike(
    api_key="none",
    api_base=os.getenv('LLAMACPP_BASEURI'),
    model=os.getenv('LLAMACPP_MODEL'),
    timeout=60.0,
    max_retries=3,
    temperature=0.01,
)

try:
    print("Creating storage context")
    storage_context = StorageContext.from_defaults(persist_dir="storage")

    print("Loading index from storage")
    index = load_index_from_storage(
        storage_context,
        # we can optionally override the embed_model here
        # it's important to use the same embed_model as the one used to build the index
        # embed_model=Settings.embed_model,
    )
except Exception as e:
    print("Error loading index from storage (it may not exist yet):", e)
    print("Creating a new index...")
    print("Loading documents from data/spryker-docs/docs")
    # Create a RAG tool using LlamaIndex
    documents = SimpleDirectoryReader(
        input_dir="data/spryker-docs/docs",
        recursive=True
    ).load_data()

    print("Creating index from documents")
    index = VectorStoreIndex.from_documents(
        documents=documents,
        show_progress=True
    )
    # Save the index
    print("Persisting index to storage")
    index.storage_context.persist("storage")

query_engine = index.as_query_engine()

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about Spryker."""
    response = await query_engine.aquery(query)
    return str(response)

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b



# Create an enhanced workflow with both tools
agent = AgentWorkflow.from_tools_or_functions(
    [multiply, search_documents],
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions about Spryker. Only use information from the Spryker documentation.
    Do not make up information. Do not provide personal opinions. Do not provide information that is not in the documentation.
    If the user asks a question the you already know the answer to OR the user is making idle banter, just respond without calling any tools.
    """,
)

# create context
ctx = Context(agent)


async def main():
    handler = agent.run(user_msg="What's a Spryker?")

    # handle streaming output
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)
        elif isinstance(event, AgentInput):
            print("Agent input: ", event.input)  # the current input messages
            print("Agent name:", event.current_agent_name)  # the current agent name
        elif isinstance(event, AgentOutput):
            print("Agent output: ", event.response)  # the current full response
            print("Tool calls made: ", event.tool_calls)  # the selected tool calls, if any
            print("Raw LLM response: ", event.raw)  # the raw llm api response
        elif isinstance(event, ToolCallResult):
            print("Tool called: ", event.tool_name)  # the tool name
            print("Arguments to the tool: ", event.tool_kwargs)  # the tool kwargs
            print("Tool output: ", event.tool_output)  # the tool output

    # print final output
    print(str(await handler))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
