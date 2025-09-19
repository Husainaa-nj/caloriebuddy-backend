import PyPDF2
import pdfplumber
import os
from typing import List, Dict

class PDFProcessor:
    def __init__(self, pdf_folder_path: str):
        self.pdf_folder_path = pdf_folder_path
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a single PDF file"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process all PDFs in the folder and return structured data"""
        pdf_documents = []
        
        if not os.path.exists(self.pdf_folder_path):
            print(f"PDF folder not found: {self.pdf_folder_path}")
            return pdf_documents
        
        for filename in os.listdir(self.pdf_folder_path):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_folder_path, filename)
                text_content = self.extract_text_from_pdf(pdf_path)
                
                if text_content:
                    pdf_documents.append({
                        'filename': filename,
                        'content': text_content,
                        'source': 'pdf_label'
                    })
        
        return pdf_documents
    
    def create_chunks_from_pdfs(self, pdf_documents: List[Dict]) -> List[str]:
        """Convert PDF documents to text chunks for RAG"""
        chunks = []
        
        for doc in pdf_documents:
            # Create a comprehensive chunk from each PDF
            dish_name = doc['filename'].replace('-', ' ').replace('.pdf', '').title()
            content = doc['content']
            
            # Extract key nutrition info and create a structured chunk
            chunk = f"Food Label for {dish_name}:\n{content}\n\nSource: Official nutrition label"
            chunks.append(chunk)
        
        return chunks
