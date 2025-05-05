from typing import List, Dict, Optional
import logging
import os
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

# Configure logging
logger = logging.getLogger(__name__)

class PDFQAService:
    def __init__(self, persist_directory: str = "chroma_db"):
        try:
            print("[PDFQAService] Initializing PDFQAService...")
            logger.info("Initializing PDFQAService...")
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            print(f"[PDFQAService] Using Ollama base URL: {base_url}")
            logger.info(f"Using Ollama base URL: {base_url}")
            
            # Initialize embeddings with optimized parameters
            print("[PDFQAService] Initializing embeddings...")
            self.embeddings = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=base_url,
                temperature=0.1,
                top_p=0.95
            )
            print("[PDFQAService] Embeddings initialized.")
            
            # Initialize LLM with optimized parameters
            print("[PDFQAService] Initializing LLM...")
            self.llm = OllamaLLM(
                model=settings.LLM_MODEL,
                temperature=settings.MODEL_TEMPERATURE,
                base_url=base_url,
                num_ctx=4096
            )
            print("[PDFQAService] LLM initialized.")
            
            # Initialize text splitter with optimized parameters
            print("[PDFQAService] Initializing text splitter...")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            print("[PDFQAService] Text splitter initialized.")
            
            # Initialize memory for conversation history
            print("[PDFQAService] Initializing conversation memory...")
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            print("[PDFQAService] Conversation memory initialized.")
            
            # Set up persistent storage
            self.persist_directory = persist_directory
            self.vector_store = None
            print("[PDFQAService] Initializing vector store...")
            self._initialize_vector_store()
            print("[PDFQAService] PDFQAService initialized successfully.")
            logger.info("PDFQAService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing PDFQAService: {str(e)}")
            print(f"[PDFQAService] Error initializing PDFQAService: {str(e)}")
            raise

    def _initialize_vector_store(self):
        """Initialize or load the vector store from persistent storage"""
        try:
            if os.path.exists(self.persist_directory):
                print(f"[PDFQAService] Loading existing vector store from {self.persist_directory}")
                logger.info("Loading existing vector store from persistent storage")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                print(f"[PDFQAService] Creating new vector store at {self.persist_directory}")
                logger.info("Creating new vector store")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            print(f"[PDFQAService] Error initializing vector store: {str(e)}")
            raise

    def process_pdf(self, pdf_path: str) -> bool:
        """Process PDF file and create vector store with metadata."""
        try:
            print(f"[PDFQAService] Processing PDF: {pdf_path}")
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Verify PDF is readable
            with open(pdf_path, 'rb') as file:
                PdfReader(file)
            
            # Load PDF
            print(f"[PDFQAService] Loading PDF pages...")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            print(f"[PDFQAService] Loaded {len(pages)} pages from PDF.")
            logger.info(f"Loaded {len(pages)} pages from PDF")
            
            # Split text into chunks with metadata
            print(f"[PDFQAService] Splitting text into chunks...")
            texts = []
            for i, page in enumerate(pages):
                chunks = self.text_splitter.split_documents([page])
                for chunk in chunks:
                    chunk.metadata.update({
                        "source": pdf_path,
                        "page": i + 1,
                        "chunk_id": len(texts)
                    })
                texts.extend(chunks)
            print(f"[PDFQAService] Total number of chunks: {len(texts)}")
            
            # Add documents to vector store
            print(f"[PDFQAService] Adding documents to vector store and embedding...")
            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"[PDFQAService] Embedding and persistence complete.")
            
            logger.info(f"Successfully processed PDF with {len(texts)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            print(f"[PDFQAService] Error processing PDF: {str(e)}")
            return False

    def ask_question(self, question: str) -> str:
        """Ask a question about the processed PDF with context and memory."""
        if not self.vector_store:
            print("[PDFQAService] No vector store found. Please upload and process a PDF first.")
            return "Please upload and process a PDF first."
        
        try:
            print(f"[PDFQAService] Answering question: {question}")
            # Create custom prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain with memory and custom prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                memory=self.memory,
                chain_type_kwargs={"prompt": prompt}
            )
            
            # Get answer
            print(f"[PDFQAService] Running QA chain...")
            result = qa_chain({"query": question})
            print(f"[PDFQAService] QA chain complete. Result: {result['result']}")
            return result["result"]
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            print(f"[PDFQAService] Error getting answer: {str(e)}")
            return f"Error getting answer: {str(e)}"

    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        print("[PDFQAService] Conversation memory cleared.")
        logger.info("Conversation memory cleared") 