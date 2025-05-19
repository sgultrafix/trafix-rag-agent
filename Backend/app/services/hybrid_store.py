import os
import logging
from typing import List, Dict, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from app.core.config import settings

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, persist_directory: str, embedding_model: str = settings.EMBEDDING_MODEL):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        )
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize or load the vector store from persistent storage"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            if os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
                logger.info(f"Loading existing FAISS index from {self.persist_directory}")
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                logger.info(f"Creating new FAISS index at {self.persist_directory}")
                # Create empty vector store without dummy document
                self.vector_store = FAISS.from_texts(
                    [],
                    self.embeddings
                )
                self.vector_store.save_local(self.persist_directory)
        except Exception as e:
            logger.error(f"Error initializing FAISS vector store: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
                
            # If this is the first real document being added, clear any dummy content
            if self.vector_store and len(self.vector_store.index_to_docstore_id) == 0:
                logger.info("First real document being added - initializing fresh vector store")
                self.vector_store = FAISS.from_texts(
                    [],
                    self.embeddings
                )
            
            self.vector_store.add_documents(documents)
            self.vector_store.save_local(self.persist_directory)
            logger.info(f"Added {len(documents)} documents to FAISS at {self.persist_directory}")
            if documents:
                logger.info(f"Sample doc content: {documents[0].page_content[:100]} ... Metadata: {documents[0].metadata}")
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search"""
        try:
            if not self.vector_store or len(self.vector_store.index_to_docstore_id) == 0:
                logger.warning("Vector store is empty - no documents to search")
                return []
                
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Similarity search for '{query}' returned {len(results)} results from FAISS at {self.persist_directory}")
            if results:
                logger.info(f"Top result content: {results[0].page_content[:100]} ... Metadata: {results[0].metadata}")
            return results
        except Exception as e:
            logger.error(f"Error in FAISS similarity search: {str(e)}")
            raise

    def as_retriever(self, *args, **kwargs):
        return self.vector_store.as_retriever(*args, **kwargs)

class ChromaMemoryStore:
    def __init__(self, persist_directory: str, embedding_model: str = settings.EMBEDDING_MODEL):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        )
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize or load the vector store from persistent storage"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Create a new Chroma instance with proper configuration
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="memory_store"
            )
            
            logger.info(f"Initialized Chroma store at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Error initializing Chroma store: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        try:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            logger.info(f"Added {len(documents)} documents to Chroma at {self.persist_directory}")
            if documents:
                logger.info(f"Sample doc content: {documents[0].page_content[:100]} ... Metadata: {documents[0].metadata}")
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search with optional filtering"""
        try:
            results = self.vector_store.similarity_search(query, k=k, filter=filter)
            logger.info(f"Similarity search for '{query}' returned {len(results)} results from Chroma at {self.persist_directory}")
            if results:
                logger.info(f"Top result content: {results[0].page_content[:100]} ... Metadata: {results[0].metadata}")
            return results
        except Exception as e:
            logger.error(f"Error in Chroma similarity search: {str(e)}")
            raise

class HybridStorageManager:
    def __init__(self):
        self.document_store = FAISSVectorStore(
            persist_directory=settings.FAISS_INDEX_DIR,
            embedding_model=settings.EMBEDDING_MODEL
        )
        self.schema_store = FAISSVectorStore(
            persist_directory=os.path.join(settings.FAISS_INDEX_DIR, "schema"),
            embedding_model=settings.SCHEMA_EMBEDDING_MODEL
        )
        self.memory_store = ChromaMemoryStore(
            persist_directory=settings.CHROMA_MEMORY_DIR
        )

    def add_document(self, document: Document, store_type: str = "document"):
        """Add a document to the appropriate store"""
        try:
            if store_type == "document":
                self.document_store.add_documents([document])
            elif store_type == "schema":
                self.schema_store.add_documents([document])
            else:
                raise ValueError(f"Invalid store type: {store_type}")
        except Exception as e:
            logger.error(f"Error adding document to {store_type} store: {str(e)}")
            raise

    def add_memory(self, document: Document):
        """Add a document to the memory store"""
        try:
            self.memory_store.add_documents([document])
        except Exception as e:
            logger.error(f"Error adding to memory store: {str(e)}")
            raise

    def search(self, query: str, store_type: str = "document", k: int = 4, filter: Optional[Dict] = None) -> List[Document]:
        """Search across the appropriate store"""
        try:
            if store_type == "document":
                doc_results = self.document_store.similarity_search(query, k=k)
                schema_results = self.schema_store.similarity_search(query, k=k)
                
                # Filter out empty results and combine
                all_results = []
                if doc_results and doc_results[0].page_content != "This is an empty vector store.":
                    all_results.extend(doc_results)
                if schema_results and schema_results[0].page_content != "This is an empty vector store.":
                    all_results.extend(schema_results)
                
                # Sort by relevance and return top k
                return all_results[:k]
                
            elif store_type == "schema":
                return self.schema_store.similarity_search(query, k=k)
            elif store_type == "memory":
                return self.memory_store.similarity_search(query, k=k, filter=filter)
            else:
                raise ValueError(f"Invalid store type: {store_type}")
        except Exception as e:
            logger.error(f"Error searching {store_type} store: {str(e)}")
            raise

    def clear_store(self, store_type: str = "document"):
        """Clear the specified store"""
        try:
            if store_type == "document":
                self.document_store = FAISSVectorStore(
                    persist_directory=settings.FAISS_INDEX_DIR,
                    embedding_model=settings.EMBEDDING_MODEL
                )
            elif store_type == "schema":
                self.schema_store = FAISSVectorStore(
                    persist_directory=os.path.join(settings.FAISS_INDEX_DIR, "schema"),
                    embedding_model=settings.SCHEMA_EMBEDDING_MODEL
                )
            elif store_type == "memory":
                # Close existing connection if any
                if self.memory_store and self.memory_store.vector_store:
                    try:
                        self.memory_store.vector_store._client.close()
                    except:
                        pass
                # Create new store
                self.memory_store = ChromaMemoryStore(
                    persist_directory=settings.CHROMA_MEMORY_DIR
                )
            else:
                raise ValueError(f"Invalid store type: {store_type}")
        except Exception as e:
            logger.error(f"Error clearing {store_type} store: {str(e)}")
            raise

    def add_documents(self, documents: list, store_type: str = "document"):
        """Add multiple documents to the appropriate store"""
        try:
            if store_type == "document":
                self.document_store.add_documents(documents)
            elif store_type == "schema":
                self.schema_store.add_documents(documents)
            else:
                raise ValueError(f"Invalid store type: {store_type}")
        except Exception as e:
            logger.error(f"Error adding documents to {store_type} store: {str(e)}")
            raise 