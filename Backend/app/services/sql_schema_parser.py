import logging
from typing import Dict, List, Optional
import sqlparse
import sqlglot
from sqlglot import parse_one, exp
import re

logger = logging.getLogger(__name__)

class SQLSchemaParser:
    """Service for parsing SQL schema files and extracting table/column information."""
    
    def __init__(self):
        self.tables = {}
        self.relationships = []
        self.current_table = None
    
    def parse_sql_file(self, content: str) -> Dict:
        """Parse SQL file and extract schema information."""
        try:
            # Split content into statements
            statements = self._split_sql_statements(content)
            
            for statement in statements:
                statement = statement.strip().upper()
                if not statement:
                    continue
                
                # Handle CREATE TABLE statements
                if "CREATE TABLE" in statement:
                    self._parse_create_table(statement)
                # Handle ALTER TABLE statements (for relationships)
                elif "ALTER TABLE" in statement and "ADD CONSTRAINT" in statement:
                    self._parse_alter_table(statement)
            
            return {
                "tables": self.tables,
                "relationships": self.relationships
            }
            
        except Exception as e:
            logger.error(f"Error parsing SQL schema: {str(e)}")
            raise ValueError(f"Error parsing SQL schema: {str(e)}")

    def _split_sql_statements(self, content: str) -> List[str]:
        """Split SQL content into individual statements."""
        # Remove comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'--.*?$', '', content, flags=re.MULTILINE)
        
        # Split on semicolons
        statements = [s.strip() for s in content.split(';')]
        return [s for s in statements if s]

    def _parse_create_table(self, statement: str):
        """Parse CREATE TABLE statement."""
        try:
            # Extract table name
            table_match = re.search(r'CREATE TABLE\s+(\w+)', statement)
            if not table_match:
                return
            
            table_name = table_match.group(1)
            self.current_table = table_name
            self.tables[table_name] = {
                "columns": [],
                "primary_keys": [],
                "foreign_keys": []
            }
            
            # Extract column definitions
            columns_match = re.search(r'\((.*)\)', statement, re.DOTALL)
            if columns_match:
                columns_text = columns_match.group(1)
                self._parse_columns(columns_text)
                
        except Exception as e:
            logger.error(f"Error parsing CREATE TABLE: {str(e)}")
            raise

    def _parse_columns(self, columns_text: str):
        """Parse column definitions."""
        if not self.current_table:
            return
            
        # Split columns
        columns = [c.strip() for c in columns_text.split(',')]
        
        for column in columns:
            if not column:
                continue
                
            # Extract column name and type
            col_match = re.match(r'(\w+)\s+(\w+)', column)
            if col_match:
                col_name = col_match.group(1)
                col_type = col_match.group(2)
                
                column_info = {
                    "name": col_name,
                    "type": col_type,
                    "nullable": "NOT NULL" not in column.upper(),
                    "is_primary": "PRIMARY KEY" in column.upper(),
                    "is_foreign": "REFERENCES" in column.upper()
                }
                
                self.tables[self.current_table]["columns"].append(column_info)
                
                if column_info["is_primary"]:
                    self.tables[self.current_table]["primary_keys"].append(col_name)
                    
                if column_info["is_foreign"]:
                    self._parse_foreign_key(column)

    def _parse_foreign_key(self, column: str):
        """Parse foreign key constraint."""
        if not self.current_table:
            return
            
        fk_match = re.search(r'REFERENCES\s+(\w+)\s*\((\w+)\)', column)
        if fk_match:
            ref_table = fk_match.group(1)
            ref_column = fk_match.group(2)
            
            self.tables[self.current_table]["foreign_keys"].append({
                "column": column.split()[0],
                "references": {
                    "table": ref_table,
                    "column": ref_column
                }
            })
            
            self.relationships.append({
                "from_table": self.current_table,
                "to_table": ref_table,
                "from_column": column.split()[0],
                "to_column": ref_column
            })

    def _parse_alter_table(self, statement: str):
        """Parse ALTER TABLE statement for constraints."""
        try:
            # Extract table name and constraint
            alter_match = re.search(r'ALTER TABLE\s+(\w+)\s+ADD CONSTRAINT\s+(\w+)\s+FOREIGN KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)', statement)
            if alter_match:
                table_name = alter_match.group(1)
                constraint_name = alter_match.group(2)
                column = alter_match.group(3)
                ref_table = alter_match.group(4)
                ref_column = alter_match.group(5)
                
                if table_name in self.tables:
                    self.tables[table_name]["foreign_keys"].append({
                        "column": column,
                        "references": {
                            "table": ref_table,
                            "column": ref_column
                        }
                    })
                    
                    self.relationships.append({
                        "from_table": table_name,
                        "to_table": ref_table,
                        "from_column": column,
                        "to_column": ref_column
                    })
                    
        except Exception as e:
            logger.error(f"Error parsing ALTER TABLE: {str(e)}")
            raise 