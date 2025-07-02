# Swiss Legal Chatbot - Architecture Documentation

## Overview

The Swiss Legal Chatbot has been restructured to improve code clarity, maintainability, and separation of concerns. The monolithic `app.py` file has been split into modular components, each responsible for specific functionality.

## Architecture

### Directory Structure

```
test-rag/
├── config/                     # Configuration and settings
│   ├── __init__.py
│   └── settings.py            # Environment variables, constants
├── database/                   # Database operations
│   ├── __init__.py
│   └── chroma_client.py       # ChromaDB client and operations
├── rag/                       # RAG system components
│   ├── __init__.py
│   ├── query_processing.py    # Query enhancement and processing
│   ├── retrieval.py          # Document retrieval and RAG processing
│   └── services.py           # High-level service orchestration
├── conversation/              # Conversation management
│   ├── __init__.py
│   ├── memory.py             # Context building and summarization
│   └── routing.py            # Query routing logic
├── ui/                       # User interface components
│   ├── __init__.py
│   ├── components.py         # Reusable UI components
│   └── sidebar.py            # Sidebar functionality
├── utils/                    # Utilities
│   ├── __init__.py
│   └── logging_config.py     # Logging configuration
├── app.py                    # Main application entry point
├── app_original.py          # Backup of original monolithic app
└── ARCHITECTURE.md          # This documentation file
```

## Separation of Concerns

### 1. Configuration (`config/`)
- **Purpose**: Centralize all configuration, environment variables, and constants
- **Key Files**: 
  - `settings.py`: API keys, model settings, UI constants
- **Benefits**: Single place to manage all configuration

### 2. Database Layer (`database/`)
- **Purpose**: Handle all database operations and ChromaDB interactions
- **Key Files**:
  - `chroma_client.py`: ChromaDB client, collection management, CRUD operations
- **Benefits**: Database logic is isolated and reusable

### 3. RAG System (`rag/`)
- **Purpose**: Core RAG functionality including retrieval, processing, and orchestration
- **Key Files**:
  - `query_processing.py`: Query enhancement with conversation context
  - `retrieval.py`: Document retrieval and response generation
  - `services.py`: High-level service that orchestrates all RAG components
- **Benefits**: RAG logic is modular and testable

### 4. Conversation Management (`conversation/`)
- **Purpose**: Handle conversation memory, context building, and query routing
- **Key Files**:
  - `memory.py`: Conversation summarization and context management
  - `routing.py`: Query routing to different handlers
- **Benefits**: Conversation logic is separate from UI and RAG concerns

### 5. User Interface (`ui/`)
- **Purpose**: All Streamlit UI components and layouts
- **Key Files**:
  - `components.py`: Reusable UI components (welcome section, chat messages, etc.)
  - `sidebar.py`: Sidebar functionality including conversation management
- **Benefits**: UI logic is modular and reusable

### 6. Utilities (`utils/`)
- **Purpose**: Common utilities and helper functions
- **Key Files**:
  - `logging_config.py`: Centralized logging setup
- **Benefits**: Shared utilities are easily accessible

### 7. Main Application (`app.py`)
- **Purpose**: Application entry point that orchestrates all components
- **Benefits**: Clean, focused main file that's easy to understand

## Future Enhancements

The modular structure enables easy future enhancements:

1. **Testing**: Add comprehensive unit tests for each module
2. **API Layer**: Add REST API endpoints alongside the Streamlit interface
3. **Multiple Backends**: Support different vector databases or LLM providers
4. **Caching**: Add intelligent caching layers
5. **Monitoring**: Add metrics and monitoring capabilities
6. **Multi-language**: Support multiple languages more easily 