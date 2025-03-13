import asyncio
from dotenv import load_dotenv
import os
import requests

load_dotenv()

docs_dir = 'data/spryker-docs/docs'
embedding_model = os.getenv('EMBEDDING_MODEL')
llama_model = os.getenv('LLAMACPP_MODEL')
llama_base_url = os.getenv('LLAMACPP_BASEURI') or 'http://localhost:11434/v1'

def read_files():
    """Read all files from the given directory and its subdirectories."""
    files_content = []
    for root, _, files in os.walk(docs_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    files_content.append(file.read())
    return files_content

def generate_embeddings(texts):
    """Generate embeddings for a list of texts using the OpenAI-compatible server."""
    embeddings = []
    for text in texts:
        response = requests.post(llama_base_url + '/embeddings', json={
            'input': text,
            'model': embedding_model
        })
        if response.status_code == 200:
            embeddings.append(response.json()['data'])
        else:
            print(f"Failed to generate embedding for text: {text[:30]}...")  # Print a snippet of the text
    return embeddings

async def main():
    print("yo")
    # Read all files from the given directory
    files_content = read_files()
    embeddings = generate_embeddings(files_content)
    print("cool cool")

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
