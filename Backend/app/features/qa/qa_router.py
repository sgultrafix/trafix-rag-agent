from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import logging
from app.services.hybrid_store import hybrid_store

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ask")
async def ask_question(question: str = Query(None, description="Your question")):
    try:
        # Determine query context
        is_schema_query = any(keyword in question.lower() for keyword in 
            ["schema", "table", "column", "relationship", "database", "sql", "json"])
        is_customer_query = any(keyword in question.lower() for keyword in 
            ["customer", "client", "account", "service", "settings", "billing"])
        
        # Search both stores with context-aware prioritization
        schema_docs = hybrid_store.search(question, store_type="schema", k=4)
        doc_docs = hybrid_store.search(question, store_type="document", k=4)
        
        # Combine and prioritize results based on context
        all_docs = []
        if is_schema_query:
            all_docs.extend(schema_docs)
            all_docs.extend(doc_docs)
        elif is_customer_query:
            # Prioritize customer-specific documents
            customer_docs = [doc for doc in doc_docs if "customer" in doc.metadata.get("business_context", "")]
            all_docs.extend(customer_docs)
            all_docs.extend(schema_docs)
            all_docs.extend([doc for doc in doc_docs if doc not in customer_docs])
        else:
            all_docs.extend(doc_docs)
            all_docs.extend(schema_docs)
        logger.info(f"[QA_ROUTER] Retrieved {len(all_docs)} docs for question '{question}'")
        if all_docs:
            logger.info(f"[QA_ROUTER] Top doc source: {all_docs[0].metadata.get('source', 'Unknown')}, content: {all_docs[0].page_content[:100]}")
        if all_docs:
            # Use the most relevant document
            answer = all_docs[0].page_content
            # Add source information
            source = all_docs[0].metadata.get("source", "Unknown")
            return {
                "answer": answer,
                "source": source,
                "confidence": "high" if len(all_docs) > 0 else "low"
            }
        logger.warning(f"[QA_ROUTER] No relevant docs found for question '{question}'")
        return {
            "answer": "I couldn't find specific information to answer your question.",
            "confidence": "low"
        }
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 