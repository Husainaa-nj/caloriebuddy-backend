from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from data.raw import text_chunks  # Import your text chunks
from data.pdf_processor import PDFProcessor
import os
from dotenv import load_dotenv

load_dotenv()  # Automatically loads .env file variable
api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)


def get_all_chunks():
    """Combine dictionary data chunks with PDF chunks"""
    
    # Get existing dictionary-based chunks
    existing_chunks = text_chunks
    
    # Process PDF documents
    pdf_processor = PDFProcessor("C:\\Users\\husai\\calorieBuddy\\data\\food data")  # Folder where you store PDFs
    pdf_documents = pdf_processor.process_all_pdfs()
    pdf_chunks = pdf_processor.create_chunks_from_pdfs(pdf_documents)
    
    # Combine all chunks
    all_chunks = existing_chunks + pdf_chunks
    print(f"Total chunks: {len(all_chunks)} (Dictionary: {len(existing_chunks)}, PDFs: {len(pdf_chunks)})")
    
    return all_chunks

# Create/load vector store with combined chunks
def initialize_vector_store():
    """Initialize ChromaDB with all data sources"""
    all_chunks = get_all_chunks()
    
    # Create vector store with all chunks
    vector_store = Chroma.from_texts(
        texts=all_chunks, 
        embedding=embeddings,
        persist_directory="data/chromadb"  # Persist the database
    )
    
    return vector_store

def get_retriever():
    """Get retriever with both dictionary and PDF data"""
    vector_store = initialize_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": 5})  # Increased k for more results
