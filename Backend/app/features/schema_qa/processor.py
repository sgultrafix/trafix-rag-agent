import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Any
from langchain.schema import Document
from app.core.config import settings
from app.services.hybrid_store import HybridStorageManager
from app.services.sql_schema_parser import SQLSchemaParser

logger = logging.getLogger(__name__)

class SchemaProcessor:
    def __init__(self, hybrid_store: HybridStorageManager):
        self.hybrid_store = hybrid_store
        self.supported_types = settings.SCHEMA_TYPES
        self.sql_parser = SQLSchemaParser()

    def process_schema(self, schema_content: str, schema_type: str) -> bool:
        """Dispatch schema parsing based on file extension"""
        try:
            logger.info(f"[SCHEMA_PROCESSOR] Processing schema of type {schema_type}")
            if schema_type not in self.supported_types:
                raise ValueError(f"Unsupported schema type: {schema_type}")

            if schema_type == "sql":
                return self._process_sql_schema(schema_content)
            elif schema_type == "json":
                return self._process_json_schema(schema_content)
            elif schema_type in ["yaml", "yml"]:
                return self._process_yaml_schema(schema_content)

            return False
        except Exception as e:
            logger.error(f"Error processing schema: {str(e)}")
            raise

    def _process_json_schema(self, content: str) -> List[Document]:
        try:
            schema = json.loads(content)
            documents = []
            
            # Add business context to schema processing
            if isinstance(schema, dict):
                # Process business entities (clients, services, etc.)
                for key, value in schema.items():
                    if key == "data" and isinstance(value, dict):
                        for entity_type, entities in value.items():
                            # Create business-context aware documents
                            entity_summary = f"{entity_type.capitalize()} Information:"
                            for entity in entities:
                                if isinstance(entity, dict):
                                    # Add business-specific metadata
                                    doc = Document(
                                        page_content=self._format_entity_content(entity_type, entity),
                                        metadata={
                                            "type": entity_type,
                                            "business_context": "customer_data",
                                            "entity_type": entity_type,
                                            "file_type": "json",
                                            "processed_at": datetime.now().isoformat()
                                        }
                                    )
                                    documents.append(doc)
            logger.info(f"[SCHEMA_PROCESSOR] Created {len(documents)} docs from JSON schema")
            if documents:
                logger.info(f"[SCHEMA_PROCESSOR] Sample doc: {documents[0].page_content[:100]} ... Metadata: {documents[0].metadata}")
            return documents
        except Exception as e:
            logger.error(f"Error processing JSON schema: {str(e)}")
            raise

    def _format_entity_content(self, entity_type: str, entity: Dict) -> str:
        """Format entity content with business context"""
        content = [f"{entity_type.capitalize()} Details:"]
        for key, value in entity.items():
            if isinstance(value, (str, int, float, bool)):
                content.append(f"{key}: {value}")
            elif isinstance(value, dict):
                content.append(f"{key}: {json.dumps(value, indent=2)}")
            elif isinstance(value, list):
                content.append(f"{key}: {', '.join(map(str, value))}")
        return "\n".join(content)

    def _process_sql_schema(self, content: str) -> List[Document]:
        try:
            # Parse SQL schema
            tables = self.sql_parser.parse_sql_schema(content)
            documents = []
            
            for table in tables:
                # Create a document for each table with business context
                table_doc = Document(
                    page_content=self._format_table_content(table),
                    metadata={
                        "type": "sql_table",
                        "business_context": "database_schema",
                        "table_name": table.name,
                        "file_type": "sql",
                        "processed_at": datetime.now().isoformat()
                    }
                )
                documents.append(table_doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error processing SQL schema: {str(e)}")
            raise

    def _format_table_content(self, table: Any) -> str:
        """Format table content with business context"""
        content = [f"Table: {table.name}"]
        if table.description:
            content.append(f"Description: {table.description}")
        content.append("\nColumns:")
        for column in table.columns:
            col_info = f"- {column.name} ({column.type})"
            if column.description:
                col_info += f": {column.description}"
            content.append(col_info)
        return "\n".join(content)

    def _process_yaml_schema(self, content: str) -> List[Document]:
        try:
            schema = yaml.safe_load(content)
            documents = []
            
            if isinstance(schema, dict):
                # Process YAML schema with business context
                for key, value in schema.items():
                    doc = Document(
                        page_content=self._format_yaml_content(key, value),
                        metadata={
                            "type": "yaml_config",
                            "business_context": "configuration",
                            "config_type": key,
                            "file_type": "yaml",
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error processing YAML schema: {str(e)}")
            raise

    def _format_yaml_content(self, key: str, value: Any) -> str:
        """Format YAML content with business context"""
        if isinstance(value, dict):
            content = [f"{key.capitalize()} Configuration:"]
            for k, v in value.items():
                content.append(f"{k}: {v}")
            return "\n".join(content)
        return f"{key}: {value}"
