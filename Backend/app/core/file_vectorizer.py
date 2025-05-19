import os
import json
import yaml
from datetime import datetime
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.features.pdf_qa.service import PDFQAService
from app.features.schema_qa.processor import SchemaProcessor
from app.services.sql_schema_parser import SQLSchemaParser
from app.services.hybrid_store import HybridStorageManager
import logging

SUPPORTED_EXTENSIONS = {"pdf", "json", "yml", "yaml", "sql"}
logger = logging.getLogger(__name__)

def detect_file_type(filename: str) -> str:
    """Detect file type from extension"""
    ext = filename.lower().split(".")[-1]
    if ext in {"yml", "yaml"}:
        return "yaml"
    if ext in SUPPORTED_EXTENSIONS:
        return ext
    raise ValueError(f"Unsupported file extension: {ext}")

def process_pdf_file(filepath: str) -> List[Document]:
    """Process a PDF file and return LangChain Documents"""
    pdf_service = PDFQAService()
    return pdf_service.extract_pdf_documents(filepath)

def process_schema_file(filepath: str, file_type: str, hybrid_store: HybridStorageManager) -> List[Document]:
    """Process schema files (JSON, YAML, SQL) and return LangChain Documents"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    processor = SchemaProcessor(hybrid_store)
    return processor.process_schema(content, file_type)

def process_file(filepath: str, hybrid_store: HybridStorageManager) -> List[Document]:
    """Main dispatcher to process any supported file"""
    try:
        file_type = detect_file_type(filepath)
        filename = os.path.basename(filepath)
        logger.info(f"[PROCESS_FILE] Processing {filename} as type {file_type}")
        # Process based on file type
        if file_type == "pdf":
            docs = process_pdf_file(filepath)
            store_type = "document"
        else:
            docs = process_schema_file(filepath, file_type, hybrid_store)
            store_type = "schema"
        # Add common metadata
        for doc in docs:
            doc.metadata.update({
                "file_type": file_type,
                "is_schema": store_type == "schema",
                "business_context": "schema" if store_type == "schema" else "documentation",
                "source": filename,
                "processed_at": datetime.now().isoformat()
            })
        logger.info(f"[PROCESS_FILE] Created {len(docs)} docs for {filename}")
        if docs:
            logger.info(f"[PROCESS_FILE] Sample doc: {docs[0].page_content[:100]} ... Metadata: {docs[0].metadata}")
        return docs
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return []

def process_all_files_in_directory(directory: str, hybrid_store: HybridStorageManager) -> List[Document]:
    """Process all files in a directory and return combined documents"""
    all_docs = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            docs = process_file(filepath, hybrid_store)
            if docs:
                all_docs.extend(docs)
                logger.info(f"[VECTORIZE] Processed {filename}: {len(docs)} documents created")
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    logger.info(f"[VECTORIZE] Total documents created from directory: {len(all_docs)}")
    return all_docs

def summarize_document(self, file_path: str) -> str:
    """
    Generate a detailed summary of the uploaded document using the LLM.
    This method is used for open-ended questions like 'tell me about the schema I just uploaded'.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Use the LLM to generate a detailed summary of the document
        summary = self.llm.predict(f"Summarize the following document in detail, explaining its structure, purpose, and key components:\n\n{content}")
        return summary
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        return "Error generating summary." 