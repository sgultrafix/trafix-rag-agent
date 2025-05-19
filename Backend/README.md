# LangChain RAG Backend

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) backend built with FastAPI, designed for flexible document and database schema ingestion, semantic search, and Q&A. It supports both unstructured (PDF) and structured (database schema: JSON, SQL) data, enabling users to upload documents, ask questions, and receive context-aware answers.

## Key Capabilities
- **PDF Ingestion & Q&A:**
  - Extracts, chunks, and embeds text from PDF files for semantic search and question answering.
- **Database Schema Ingestion & Q&A:**
  - Supports JSON and SQL schema uploads.
  - Flexible schema root detection for a wide variety of user-uploaded schemas.
  - Extracts tables, fields, and relationships for structured Q&A.
- **Vector Search:**
  - Uses FAISS and Chroma vector stores for fast similarity search over embedded content.
- **Modular API Endpoints:**
  - Upload, process, and query both PDFs and schemas via clean, modular FastAPI endpoints.
- **Dockerized Deployment:**
  - Fully containerized for easy deployment and reproducibility.
- **Local Dependency Caching:**
  - Uses a local `lib-repo` for Python wheels to speed up Docker builds and avoid unnecessary downloads.
- **Extensible & Testable:**
  - Designed for easy extension (e.g., new file types, advanced schema mapping) and robust unit/integration testing.

## Tools & Technologies Used
- **FastAPI**: High-performance Python web framework for APIs.
- **LangChain**: Framework for building LLM-powered applications with vector search and chains.
- **Chroma (langchain_chroma)**: Modern vector store for embeddings, with automatic persistence.
- **FAISS**: Facebook AI Similarity Search for fast vector similarity queries.
- **Ollama**: Local LLM inference for embeddings and Q&A.
- **Docker**: Containerization for consistent, reproducible environments.
- **pytest**: For unit and integration testing.

## Project Strengths
- **Flexible schema ingestion**: Handles a wide variety of user-uploaded JSON and SQL schemas, not just hardcoded formats.
- **Seamless Q&A**: Users can upload documents or schemas and immediately ask questions via the API.
- **Optimized for developer productivity**: Fast Docker builds, local dependency caching, and modular codebase.
- **Extensible**: Easily add new file types, schema enrichment, or advanced retrieval logic.

## Usage
1. **Build and run with Docker:**
   ```sh
   docker-compose -f Backend/docker-compose.yml up --build
   ```
2. **Access the API docs:**
   - Go to [http://localhost:8000/docs](http://localhost:8000/docs)
3. **Upload a PDF or schema:**
   - Use `/upload` for PDFs
   - Use `/schema/upload` for JSON or SQL schemas
4. **Ask questions:**
   - Use `/ask` for PDF content
   - Use `/schema/ask` for schema Q&A

## Recommendations & Next Steps
- For robust SQL schema extraction, consider integrating a real SQL parser (e.g., `sqlparse`, `sqlglot`).
- Enhance schema enrichment by auto-generating or allowing user-supplied field/table definitions.
- Continue to use and update the local `lib-repo` for fast, reliable Docker builds.
- Monitor logs and test with a variety of document and schema types to ensure robustness.

---

For more details or to contribute, see the codebase and API docs. 