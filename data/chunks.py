import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.raw import text_chunks


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()  # Automatically loads .env file variable
api_key = os.getenv("OPENAI_API_KEY")
print("Loaded key:", repr(api_key))
embeddings = OpenAIEmbeddings(
    openai_api_key=api_key,
    model="text-embedding-ada-002"  # Explicitly specify the embedding model
)

docsearch = Chroma.from_texts(texts=text_chunks, embedding=embeddings)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = OpenAI(openai_api_key=api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
def get_answer(question):
    return qa_chain.run(question)

query = "How many calories are in almonds?"
result = get_answer(query)
print(result)