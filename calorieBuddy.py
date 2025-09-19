from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import parlant.sdk as p
import os
import requests
import json
from data.vector_store import get_retriever
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

embedder = OpenAIEmbeddings(openai_api_key=api_key)
retriever = get_retriever()

class Message(BaseModel):
    text: str

async def chat_endpoint(message: Message):
    """Chat endpoint that uses combined dictionary and PDF data"""
    
    # Get relevant documents from both sources
    docs = retriever.get_relevant_documents(message.text)
    
    # Combine retrieved content
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Enhanced response with source indication
    response_parts = []
    for doc in docs:
        source = "Official Nutrition Label" if "Food Label for" in doc.page_content else "Nutrition Database"
        response_parts.append(f"Source ({source}): {doc.page_content[:200]}...")
    
    return {
        "response": context,
        "sources_used": len(docs),
        "source_details": response_parts
    }


async def answer_retriever(context: p.RetrieverContext) -> p.RetrieverResult:
    if last_message := context.interaction.last_customer_message:
        # Embed and fetch relevant docs
        query_vector = embedder.embed_query(last_message.content)  # sync method, can be run async if needed
        docs = retriever.vectorstore.similarity_search_by_vector(query_vector, k=3)
        plain_docs = []
        for doc in docs:
            plain_docs.append({
                "content": doc.page_content,
                "source": "pdf_label" if "Food Label for" in doc.page_content else "database"
            })
            return p.RetrieverResult(plain_docs)
    return p.RetrieverResult(None)
       # Set up retriever after agent creation
async def main():
    async with p.Server() as server:
        agent = await server.create_agent(
                name="calorieBuddy",
                description="You help users track their calorie intake and provide diet suggestions."
            )
            # Set up retriever after agent creation
        await agent.attach_retriever(answer_retriever)
        print("Parlant agent is running...")



# Uncomment to run Parlant server independently
if __name__ == "__main__":
   asyncio.run(main())

url = "http://localhost:8800/agents/"

payload = json.dumps({
  "description": "You help users track their calorie intake and provide diet suggestions.",
  "max_engine_iterations": 3,
  "name": "calorieBuddy"
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)



