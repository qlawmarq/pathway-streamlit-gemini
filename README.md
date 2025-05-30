# RAG Application with Pathway and Gemini

A Retrieval-Augmented Generation (RAG) application that enables non-technical stakeholders to interact with GitHub code repositories through natural language queries.

## Features

- **Code Analysis**: Process source code files and documentation
- **Non-technical Explanations**: AI responses tailored for business stakeholders
- **Multiple File Formats**: Support for Python, JavaScript, Markdown, documentation files, etc.
- **Git Repository Ready**: Automatically excludes `.git` directories and system files

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your Gemini API key
GEMINI_API_KEY=your_api_key_here
```

### 2. Add Your Data

Place your code repository or files in the `data/` directory:

```
data/
├── your-project/
│   ├── src/
│   ├── README.md
│   └── ...
└── other-files.md
```

### 3. Launch Docker Compose

```bash
docker compose up --build
```

The server will start on `http://localhost:8000` and the UI will start on `http://localhost:8501`

## Supported File Types

- **Code**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.h`
- **Documentation**: `.md`, `.rst`, `.txt`, `.html`
- **Configuration**: `.json`, `.yaml`, `.toml`, `.ini`
- **Data**: `.csv`, `.tsv`
- **Special Files**: `README`, `LICENSE`, `Dockerfile`, etc.

Check `config/settings.yaml` for more details.

## Requirements

- Docker
- Google Gemini API key
- 2GB+ RAM (for embeddings)
