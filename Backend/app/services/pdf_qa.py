from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import Ollama
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFQAService:
    def __init__(self, persist_directory: str = "chroma_db"):
        # Initialize embeddings with optimized parameters
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url="http://host.docker.internal:11434",
            temperature=0.1,
            top_p=0.95
        )
        
        # Initialize LLM with optimized parameters
        self.llm = Ollama(
            model="mistral:latest",
            base_url="http://host.docker.internal:11434",
            temperature=0.1,
            top_p=0.95,
            num_ctx=4096
        )
        
        # Initialize text splitter with optimized parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Set up persistent storage
        self.persist_directory = persist_directory
        self.vector_store = None

    def _initialize_vector_store(self):
        """Initialize or load the vector store from persistent storage"""
        try:
            if os.path.exists(self.persist_directory):
                logger.info("Loading existing vector store from persistent storage")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                logger.info("Creating new vector store")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def process_pdf(self, pdf_path: str):
        """Process a PDF file and create vector store"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Initialize vector store if not already done
            if self.vector_store is None:
                self._initialize_vector_store()
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Split text into chunks with metadata
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
            
            # Add documents to vector store
            self.vector_store.add_documents(texts)
            self.vector_store.persist()
            
            logger.info(f"Successfully processed PDF with {len(texts)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return False

    def ask_question(self, question: str) -> str:
        """Ask a question about the processed PDF"""
        if not self.vector_store:
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
                retriever=self.vector_store.as_retriever(
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
        self.memory.clear() 