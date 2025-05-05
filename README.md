# LangChain RAG Application

A Docker-based application that uses LangChain and Ollama for PDF processing and question answering.

## Features

- PDF document processing and text extraction
- Question answering using Ollama models
- Vector store persistence for document embeddings
- Memory management for efficient resource usage
- Comprehensive test coverage
- Docker-based deployment

## Prerequisites

- Windows 10 or later
- At least 8GB RAM
- At least 10GB free disk space
- Python 3.8 or later
- Docker Desktop
- Ollama

## Installation

1. Clone this repository
2. Run the setup script as Administrator:
   ```powershell
   .\setup.ps1
   ```

The setup script will:
- Check system requirements
- Install Docker Desktop if not present
- Install Ollama if not present
- Pull required Ollama models
- Create a Python virtual environment
- Install Python dependencies
- Run tests
- Build and start the Docker container

## Project Structure

```
LangChain-Rag/
├── Backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── utils.py
│   │   ├── features/
│   │   │   └── pdf_qa/
│   │   │       └── service.py
│   │   └── main.py
│   ├── requirements/
│   │   ├── base.txt
│   │   ├── features.txt
│   │   └── dev.txt
│   ├── tests/
│   │   ├── test_core.py
│   │   ├── test_features.py
│   │   └── test_main.py
│   └── Dockerfile
├── .venv/
├── setup.ps1
└── README.md
```

## Testing

The application includes comprehensive tests for:
- Core functionality (config and utils)
- PDF QA feature
- FastAPI endpoints

To run tests manually:
```powershell
cd Backend
python -m pytest tests/ --cov=app --cov-report=term-missing -v
```

## API Documentation

Once the application is running, you can access the API documentation at:
```
http://localhost:8000/docs
```

## Endpoints

- `GET /`: Root endpoint with welcome message
- `GET /health`: Health check endpoint (returns status OK)
- `POST /upload`: Upload PDF documents
- `POST /ask`: Ask questions about uploaded documents

## Development

### Adding New Features

1. Create tests for the new functionality
2. Implement the feature in the appropriate directory:
   - Core functionality: `app/core/`
   - New features: `app/features/`
3. Run tests to ensure everything works
4. Update documentation if necessary

## Troubleshooting

### Common Issues

1. **Ollama not starting**
   - Check if the Ollama service is running: `Get-Service Ollama`
   - Start the service: `Start-Service Ollama`

2. **Docker container not starting**
   - Check logs: `docker logs langchain-rag-container`
   - Ensure Docker Desktop is running

3. **Tests failing**
   - Check test output for specific errors
   - Ensure all dependencies are installed
   - Verify Ollama models are available

### Getting Help

If you encounter any issues not covered here, please open an issue in the repository.

## License

MIT License 