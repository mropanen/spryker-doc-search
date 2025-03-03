import asyncio
import dotenv
from llama_index.core.workflow import Context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
import asyncio
import os

dotenv.load_dotenv()

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(
    base_url="http://192.168.0.23:11434",
    model="llama3.2",
    request_timeout=360.0
)

try:
    storage_context = StorageContext.from_defaults(persist_dir="storage")

    index = load_index_from_storage(
        storage_context,
        # we can optionally override the embed_model here
        # it's important to use the same embed_model as the one used to build the index
        # embed_model=Settings.embed_model,
    )
except Exception as e:
    # Create a RAG tool using LlamaIndex
    documents = SimpleDirectoryReader(
        input_dir="data/spryker-docs/docs",
        recursive=True
    ).load_data()

    index = VectorStoreIndex.from_documents(
        documents=documents,
        show_progress=True
)
# Save the index
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
    and search through documents to answer questions.""",
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
