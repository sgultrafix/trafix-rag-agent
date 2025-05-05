# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Please run this script as Administrator" -ForegroundColor Red
    exit 1
}

# Function to check if a command exists
function Test-CommandExists {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $command) { return $true }
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

# Function to check system requirements
function Test-SystemRequirements {
    $requirements = @{
        "RAM" = (Get-CimInstance -ClassName Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1GB -ge 8
        "Disk Space" = (Get-PSDrive C).Free / 1GB -ge 10
        "Python" = (Test-CommandExists "python")
    }

    $failed = $false
    foreach ($req in $requirements.GetEnumerator()) {
        if (-not $req.Value) {
            Write-Host "System requirement not met: $($req.Key)" -ForegroundColor Red
            $failed = $true
        }
    }

    if ($failed) {
        Write-Host "Please ensure your system meets all requirements before proceeding." -ForegroundColor Red
        exit 1
    }
}

# Check system requirements
Write-Host "Checking system requirements..." -ForegroundColor Yellow
Test-SystemRequirements

# Check and install Docker
if (-not (Test-CommandExists "docker")) {
    Write-Host "Docker not found. Installing Docker Desktop..." -ForegroundColor Yellow
    try {
        # Download Docker Desktop installer
        $dockerInstaller = "DockerDesktopInstaller.exe"
        Write-Host "Downloading Docker Desktop installer..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe" -OutFile $dockerInstaller
        
        # Install Docker Desktop
        Write-Host "Installing Docker Desktop..." -ForegroundColor Yellow
        Start-Process -FilePath $dockerInstaller -ArgumentList "install", "--quiet" -Wait
        Remove-Item $dockerInstaller
        
        Write-Host "Docker Desktop installed. Please restart your computer and run this script again." -ForegroundColor Green
        exit 0
    } catch {
        Write-Host "Failed to install Docker Desktop: $_" -ForegroundColor Red
        exit 1
    }
}

# Check and install Ollama
if (-not (Test-CommandExists "ollama")) {
    Write-Host "Ollama not found. Installing Ollama..." -ForegroundColor Yellow
    try {
        # Download Ollama installer
        $ollamaInstaller = "OllamaSetup.exe"
        Write-Host "Downloading Ollama installer..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://ollama.com/download/windows" -OutFile $ollamaInstaller
        
        # Install Ollama
        Write-Host "Installing Ollama..." -ForegroundColor Yellow
        Start-Process -FilePath $ollamaInstaller -ArgumentList "/S" -Wait
        Remove-Item $ollamaInstaller
        
        Write-Host "Ollama installed. Starting Ollama service..." -ForegroundColor Green
        Start-Service Ollama
    } catch {
        Write-Host "Failed to install Ollama: $_" -ForegroundColor Red
        exit 1
    }
}

# Start Ollama if not running
if (-not (Get-Service Ollama -ErrorAction SilentlyContinue).Status -eq "Running") {
    Write-Host "Starting Ollama service..." -ForegroundColor Yellow
    Start-Service Ollama
}

# Pull required Ollama models
Write-Host "Pulling required Ollama models..." -ForegroundColor Yellow
try {
    Write-Host "Pulling nomic-embed-text:latest..." -ForegroundColor Yellow
    ollama pull nomic-embed-text:latest
    Write-Host "Pulling mistral:latest..." -ForegroundColor Yellow
    ollama pull mistral:latest
} catch {
    Write-Host "Failed to pull Ollama models: $_" -ForegroundColor Red
    exit 1
}

# Create and activate virtual environment
Write-Host "Setting up Python virtual environment..." -ForegroundColor Yellow
try {
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
    }
    .\.venv\Scripts\activate
} catch {
    Write-Host "Failed to create virtual environment: $_" -ForegroundColor Red
    exit 1
}

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
try {
    # Install base requirements
    pip install -r Backend/requirements/base.txt
    
    # Install feature requirements
    pip install -r Backend/requirements/features.txt
    
    # Install development requirements
    pip install -r Backend/requirements/dev.txt
} catch {
    Write-Host "Failed to install Python dependencies: $_" -ForegroundColor Red
    exit 1
}

# Run tests
Write-Host "Running tests..." -ForegroundColor Yellow
try {
    Set-Location Backend
    python -m pytest tests/ --cov=app --cov-report=term-missing -v
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Tests failed. Please fix the issues before proceeding." -ForegroundColor Red
        exit 1
    }
    Set-Location ..
} catch {
    Write-Host "Failed to run tests: $_" -ForegroundColor Red
    exit 1
}

# Build and start the Docker container
Write-Host "Building and starting the application container..." -ForegroundColor Yellow
try {
    Set-Location Backend
    docker build -t langchain-rag .
    docker run -d -p 8000:8000 --name langchain-rag-container langchain-rag
    Set-Location ..
} catch {
    Write-Host "Failed to start the container: $_" -ForegroundColor Red
    exit 1
}

# Wait for the application to start
Write-Host "Waiting for the application to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if the application is running
$response = try {
    Invoke-WebRequest -Uri "http://localhost:8000/docs" -UseBasicParsing
} catch {
    $null
}

if ($response) {
    Write-Host "Setup completed successfully!" -ForegroundColor Green
    Write-Host "You can access the application at: http://localhost:8000/docs" -ForegroundColor Green
} else {
    Write-Host "Application failed to start. Please check the logs using: docker logs langchain-rag-container" -ForegroundColor Red
    exit 1
} 