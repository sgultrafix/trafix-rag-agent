from app.features.pdf_qa.service import PDFQAService
from app.services.hybrid_store import HybridStorageManager

class RAGService:
    def __init__(self):
        self.hybrid_store = HybridStorageManager()
        self.qa_service = PDFQAService()

    def reset_all(self):
        """Reset all services and storage"""
        try:
            # Clear vector stores
            self.hybrid_store.clear_store("document")
            self.hybrid_store.clear_store("schema")
            self.hybrid_store.clear_store("memory")
            
            # Clear conversation memory
            self.qa_service.clear_memory()
            
            return {
                "message": "All services and storage reset successfully",
                "verification": "Reset complete"
            }
        except Exception as e:
            raise Exception(f"Error resetting services: {str(e)}")

    def refresh(self):
        """Refresh the services"""
        try:
            self.qa_service = PDFQAService()
            return {"message": "Services refreshed successfully"}
        except Exception as e:
            raise Exception(f"Error refreshing services: {str(e)}") 