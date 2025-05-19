from typing import List, Dict, Optional
import logging
import os
import re
from pathlib import Path
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from app.core.config import settings
from langchain.schema import Document
from app.services.hybrid_store import HybridStorageManager

# Configure logging
logger = logging.getLogger(__name__)

class PDFQAService:
    def __init__(self, hybrid_store: HybridStorageManager = None):
        self.hybrid_store = hybrid_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? "]
        )

    def extract_pdf_documents(self, filepath: str) -> List[Document]:
        """Extract text from PDF and split into documents with improved section handling"""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(filepath)
            sections = []
            current_section = {"title": "", "content": []}
            buffer = []
            
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    text = block[4].strip()
                    
                    # Skip empty blocks
                    if not text:
                        continue
                    
                    # Detect new sections based on formatting and content
                    is_new_section = (
                        text.startswith(("# ", "## ", "### ")) or
                        text.isupper() or  # Common for headers
                        (len(text) < 100 and text.endswith(':')) or  # Likely a header
                        bool(re.match(r'^(?:Section|Chapter|Part)\s+\d+', text))
                    )
                    
                    if is_new_section:
                        # Save previous section if it exists
                        if current_section["content"]:
                            sections.append(current_section)
                        current_section = {"title": text, "content": []}
                        buffer = []  # Reset buffer
                    else:
                        # Add to buffer and check for complete thoughts
                        buffer.append(text)
                        
                        # Check if we have a complete thought
                        if (text.rstrip().endswith(('.', ':', '?', '!')) or 
                            bool(re.match(r'^\d+\.', text))):
                            current_section["content"].extend(buffer)
                            buffer = []
            
            # Add final section and buffer
            if buffer:
                current_section["content"].extend(buffer)
            if current_section["content"]:
                sections.append(current_section)
            
            # Process sections into documents
            documents = []
            for section in sections:
                title = section["title"]
                content = "\n".join(section["content"])
                
                # Use different chunking strategies based on content type
                if bool(re.search(r'^\d+\.', content, re.MULTILINE)):  # Numbered lists
                    # Keep numbered lists together
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": os.path.basename(filepath),
                            "section": title,
                            "type": "procedure"
                        }
                    ))
                else:
                    # Use RecursiveCharacterTextSplitter for prose
                    chunks = self.text_splitter.split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        # Only create a new chunk if it contains meaningful content
                        if len(chunk.strip()) > 50:  # Minimum content threshold
                            documents.append(Document(
                                page_content=chunk,
                                metadata={
                                    "source": os.path.basename(filepath),
                                    "section": title,
                                    "chunk": f"{i+1}/{len(chunks)}",
                                    "type": "content"
                                }
                            ))
            
            return documents
        except Exception as e:
            logger.error(f"Error processing PDF {filepath}: {str(e)}")
            return []

    def process_pdf(self, pdf_path: str) -> bool:
        """Process PDF file and create vector store with metadata."""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract documents with improved section handling
            documents = self.extract_pdf_documents(pdf_path)
            if not documents:
                logger.error("No documents extracted from PDF")
                return False
            
            # Add documents to vector store
            if self.hybrid_store:
                self.hybrid_store.add_documents(documents, store_type="document")
            else:
                logger.warning("No hybrid store provided, documents not persisted")
            
            logger.info(f"Successfully processed PDF with {len(documents)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return False

    def ask_question(self, question: str) -> str:
        """Ask a question about the processed PDF with context and memory."""
        if not self.hybrid_store:
            return "Please upload and process a PDF first."
        
        try:
            # Create custom prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain with memory and custom prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.hybrid_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                memory=self.memory,
                chain_type_kwargs={"prompt": prompt}
            )
            
            # Get answer
            result = qa_chain({"query": question})
            return result["result"]
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            return f"Error getting answer: {str(e)}"

    def clear_memory(self):
        """Clear the conversation memory"""
        if hasattr(self, 'memory'):
            self.memory.clear()
            logger.info("Conversation memory cleared")

    def _initialize_vector_store(self):
        """Initialize or load the vector store from persistent storage"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            if os.path.exists(self.persist_directory):
                logger.info(f"Loading existing vector store from {self.persist_directory}")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="pdf_qa_collection"
                )
            else:
                logger.info(f"Creating new vector store at {self.persist_directory}")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="pdf_qa_collection"
                )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise 