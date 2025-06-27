# Swiss Law Chatbot

This repo is a WIP and intended to accompany the blog post [Streamlit RAG app with Chromadb and OpenAI](https://michaelscheiwiller.com/legal-rag-streamlit-chromadb-openai). Consult the blogpost for details.

This project is a Streamlit-based chatbot that answers questions about Swiss migration law. It uses a Retrieval-Augmented Generation (RAG) pipeline with OpenAI and ChromaDB to provide answers based on official legal documents.

## 🔄 Reflection Pattern Feature

The chatbot now includes an advanced **Reflection Pattern** that automatically improves responses through iterative refinement:

### How It Works

1. **Initial Response**: The RAG system generates an initial response based on retrieved documents
2. **Evaluation**: An evaluator LLM analyzes the response to identify mentions of additional legal sources, articles, or documents not already included
3. **Additional Retrieval**: If additional sources are identified, the system searches for and retrieves relevant information about those sources
4. **Refinement**: The original response is enhanced with the newly found information
5. **Iteration**: This process repeats until no new sources are identified or the maximum iteration limit is reached

### Benefits

- **More Comprehensive Responses**: Automatically incorporates related legal provisions and cross-references
- **Reduced Information Gaps**: Identifies and fills in missing context from related laws
- **Enhanced Legal Accuracy**: Ensures responses consider all relevant legal frameworks
- **Transparent Process**: Shows users when and how responses were improved

### Configuration

The reflection pattern can be configured in `config/settings.py`:

```python
# Reflection pattern settings
ENABLE_REFLECTION = True              # Enable/disable reflection
MAX_REFLECTION_ITERATIONS = 3        # Maximum number of reflection cycles
REFLECTION_ADDITIONAL_SOURCES_LIMIT = 3  # Max sources to search per iteration
```

### Usage Examples

**Without Reflection**: Basic RAG response based on initially retrieved documents.

**With Reflection**: 
- Initial response mentions "Foreign Nationals and Integration Act (AIG)"
- Evaluator identifies this as an additional source to search
- System retrieves relevant AIG provisions
- Response is refined to include comprehensive information from both laws

### Testing

Test the reflection pattern with:
```bash
python scripts/test_reflection.py
```

Test the collection querying efficiency improvements with:
```bash
python scripts/test_collection_efficiency.py
```

## 🚀 Collection Querying Efficiency Improvements

The system now includes advanced **Collection Querying Optimizations** that significantly improve performance and reduce latency:

### How It Works

1. **Parallel Collection Querying**: Instead of querying collections sequentially, the system now queries multiple collections simultaneously using ThreadPoolExecutor
2. **Smart Collection Filtering**: Uses LLM-based analysis to identify the most relevant legal collections for a given query, reducing unnecessary searches
3. **Lazy Loading**: Implements intelligent caching of collection metadata to avoid repeated database queries

### Benefits

- **Faster Response Times**: Parallel querying can provide 2-4x speedup compared to sequential processing
- **Reduced Resource Usage**: Smart filtering ensures only relevant collections are queried
- **Better Scalability**: System performance remains consistent as the number of legal collections grows
- **Intelligent Caching**: Metadata is cached to minimize database overhead

### Performance Optimizations

**Parallel Querying**:
- Queries multiple ChromaDB collections simultaneously
- Configurable number of worker threads (default: 4)
- Maintains result ordering and error handling

**Smart Collection Filtering**:
- Analyzes query intent to identify relevant legal domains
- Filters collections down to the top 3 most relevant
- Fallback to all collections for general queries

**Lazy Loading**:
- Caches collection metadata to avoid repeated lookups
- Reduces database queries on subsequent requests
- Maintains cache consistency across sessions

### Configuration

The efficiency improvements can be configured in `config/settings.py`:

```python
# Collection querying optimization settings
ENABLE_PARALLEL_QUERYING = True          # Enable parallel collection querying
MAX_PARALLEL_WORKERS = 4                 # Maximum number of parallel workers
ENABLE_SMART_COLLECTION_FILTERING = True # Enable intelligent collection filtering
COLLECTION_FILTERING_THRESHOLD = 3       # Only apply filtering if more than this many collections
LAZY_LOADING_ENABLED = True              # Enable lazy loading of collection metadata
```

### Benchmarking

Run the efficiency test to see performance improvements:
```bash
python scripts/test_collection_efficiency.py
```

This will show:
- Smart collection filtering in action
- Parallel vs sequential querying performance
- Lazy loading cache effectiveness

## Project Structure

```
.
├── scripts/
│   ├── generate_chunks.py       # Parses law XMLs into structured JSON chunks.
│   ├── load_chunks_to_chromadb.py # Loads the JSON chunks into ChromaDB.
│   ├── test_reflection.py       # Tests the reflection pattern functionality.
│   └── test_collection_efficiency.py # Tests collection querying efficiency improvements.
├── data/
│   ├── *.xml                    # Raw XML law documents.
│   └── *.json                   # Generated law chunks (auto-created).
├── chroma_db/
│   └── ...                      # Persisted ChromaDB vector store.
├── rag/
│   ├── services.py              # Main RAG service orchestration.
│   ├── retrieval.py             # Document retrieval and RAG processing.
│   ├── reflection.py            # Reflection pattern implementation.
│   └── ...                      # Other RAG components.
├── config/
│   └── settings.py              # Configuration settings.
├── ui/
│   └── ...                      # User interface components.
├── .env                         # For storing your OPENAI_API_KEY.
├── app.py                       # The main Streamlit application.
├── ingest.sh                    # Ingestion script to process and load all data.
├── requirements.txt             # Python dependencies.
└── README.md                    # This file.
```

## How to Run

### 1. Prerequisites
- Python 3.8+
- An [OpenAI API Key](https://platform.openai.com/api-keys)

### 2. Setup

**a. Clone the repository and navigate to the project directory:**
```bash
git clone <repository-url>
cd test-rag
```

**b. Create a virtual environment and install dependencies:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**c. Set up your OpenAI API Key:**
Create a file named `.env` in the root of the project and add your API key to it:
```
OPENAI_API_KEY="sk-..."
```

### 3. Data Ingestion

Run the ingestion script to process the raw law documents from the `/data` directory and load them into the ChromaDB vector store. You only need to do this once, or whenever the source documents change.

```bash
bash ingest.sh
```

This script will:
1.  Parse the XML files into smaller, manageable chunks.
2.  Embed these chunks using the OpenAI API.
3.  Store the embeddings in a local ChromaDB database.

### 4. Run the Chatbot

Once the data has been ingested, you can run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your web browser, and you can start asking questions.

## How it Works

The application follows a RAG architecture with optional reflection:

1.  **Data Processing (Offline):** The `ingest.sh` script orchestrates the parsing of XML law documents and their storage in a ChromaDB vector database. Each law is stored in a separate collection.
2.  **User Query:** The user enters a question in the Streamlit chat interface.
3.  **Initial Retrieval:** The application queries all law collections in ChromaDB to find the most relevant document chunks (articles, paragraphs) based on the user's question.
4.  **Initial Generation:** The retrieved chunks are combined with the user's question into a prompt that is sent to an OpenAI model (`gpt-4o-mini`).
5.  **Reflection (Optional):** If enabled, the system evaluates the response for additional legal sources and iteratively improves it.
6.  **Response:** The final response is displayed to the user along with the sources and reflection information.

## Running as a systemd Service

To keep the Streamlit app running in the background and automatically restart it if it fails, you can set it up as a systemd service.

1.  **Create the service file directly in `/etc/systemd/system`:**

    ```sh
    sudo vim /etc/systemd/system/streamlit.service
    ```

2.  **Paste the following content (edit paths as needed):**

    ```ini
    [Unit]
    Description=Streamlit App
    After=network.target

    [Service]
    User=devuser
    WorkingDirectory=/home/devuser/projects/test-rag
    ExecStart=/home/devuser/projects/test-rag/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
    Restart=always

    [Install]
    WantedBy=multi-user.target
    ```

3.  **Enable and start the service:**

    ```sh
    sudo systemctl daemon-reload
    sudo systemctl enable streamlit
    sudo systemctl start streamlit
    ```

4.  **Check the status:**

    ```sh
    sudo systemctl status streamlit
    ```

The app will now run in the background and restart automatically if it crashes. You can access it at `http://YOUR_VPS_IP:8501/`.