import logging
import os
from typing import Dict, List, Optional, Tuple
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from app.core.config import settings
from app.services.hybrid_store import HybridStorageManager
from .processor import SchemaProcessor
from app.services.sql_schema_parser import SQLSchemaParser

logger = logging.getLogger(__name__)

class SchemaQAService:
    def __init__(self, hybrid_store: HybridStorageManager):
        self.hybrid_store = hybrid_store
        self.processor = SchemaProcessor(hybrid_store)
        self.llm = OllamaLLM(
            model=settings.LLM_MODEL,
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            temperature=settings.MODEL_TEMPERATURE
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.sql_parser = SQLSchemaParser()

    @staticmethod
    def _extension_to_schema_type(file_extension: str) -> str:
        ext = file_extension.lower().lstrip('.')
        if ext == 'sql':
            return 'sql'
        elif ext == 'json':
            return 'json'
        elif ext in ['yml', 'yaml']:
            return 'yaml'
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    async def process_schema_file(self, file_content: str, file_extension: str) -> Dict:
        """Process schema file and extract information."""
        try:
            schema_type = self._extension_to_schema_type(file_extension)
            if schema_type == 'sql':
                schema_info = self.sql_parser.parse_sql_file(file_content)
                documents = self._convert_schema_to_documents(schema_info)
                return {
                    "schema_info": schema_info,
                    "documents": documents
                }
            else:
                # For json and yaml, use the processor
                return self.processor.process_schema(file_content, schema_type)
        except Exception as e:
            logger.error(f"Error processing schema file: {str(e)}")
            raise

    def _convert_schema_to_documents(self, schema_info: Dict) -> List[Document]:
        """Convert schema information to documents for embedding."""
        documents = []
        
        # Add table information
        for table in schema_info["tables"]:
            table_doc = f"Table: {table['name']}\n"
            table_doc += "Columns:\n"
            for column in table["columns"]:
                table_doc += f"- {column['name']} ({column['type']})"
                if column['primary_key']:
                    table_doc += " [Primary Key]"
                if not column['nullable']:
                    table_doc += " [Not Null]"
                table_doc += "\n"
            documents.append(Document(page_content=table_doc))
        
        # Add relationship information
        if schema_info["relationships"]:
            rel_doc = "Table Relationships:\n"
            for rel in schema_info["relationships"]:
                rel_doc += f"- {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}\n"
            documents.append(Document(page_content=rel_doc))
            
        return documents

    async def ask_schema_question(self, question: str) -> Tuple[str, List[str], float]:
        """Answer questions about database schema"""
        try:
            # Create custom prompt template for schema questions
            prompt_template = """You are a database schema expert. Use the following pieces of context to answer the question about the database schema. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}

            Question: {question}

            Instructions for providing the answer:
            1. Answer ONLY based on the schema context provided
            2. If the answer involves relationships between tables, explain them clearly
            3. If the answer involves specific fields or columns, list them
            4. If you're unsure about any part of the answer, say so
            5. If the context doesn't contain the answer, say "I cannot find information about this in the provided schema."

            Answer: """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Create QA chain with memory and custom prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.hybrid_store.schema_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                memory=self.memory,
                chain_type_kwargs={"prompt": prompt}
            )

            # Get answer
            result = qa_chain({"query": question})
            
            # Store the interaction in memory
            self.hybrid_store.add_memory(
                Document(
                    page_content=f"Q: {question}\nA: {result['result']}",
                    metadata={
                        "type": "schema_qa",
                        "question": question,
                        "answer": result['result']
                    }
                )
            )

            return result["result"], [], 0.0  # Empty sources list and 0.0 similarity for now
        except Exception as e:
            logger.error(f"Error getting schema answer: {str(e)}")
            return f"Error getting answer: {str(e)}", [], 0.0

    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        logger.info("Schema QA conversation memory cleared")

    def get_schema_summary(self) -> Dict:
        """Get a summary of the processed schema"""
        try:
            # Get all schema documents
            docs = self.hybrid_store.search("", store_type="schema", k=1000)
            
            summary = {
                "tables": set(),
                "relationships": [],
                "schema_types": set()
            }
            
            for doc in docs:
                metadata = doc.metadata
                summary["schema_types"].add(metadata.get("type", "unknown"))
                # Only add actual table names
                if "table_name" in metadata:
                    summary["tables"].add(metadata["table_name"])
                if "relationships" in metadata:
                    summary["relationships"].extend(metadata["relationships"])
            # Convert sets to lists for JSON serialization
            summary["tables"] = list(summary["tables"])
            summary["schema_types"] = list(summary["schema_types"])
            return summary
        except Exception as e:
            logger.error(f"Error getting schema summary: {str(e)}")
            return {
                "tables": [],
                "relationships": [],
                "schema_types": []
            } 