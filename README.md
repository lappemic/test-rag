# Swiss Law Chatbot

This repo is a WIP and intended to accompany the blog post [Streamlit RAG app with Chromadb and OpenAI](https://michaelscheiwiller.com/legal-rag-streamlit-chromadb-openai). Consult the blogpost for details.

This project is a Streamlit-based chatbot that answers questions about Swiss migration law. It uses a Retrieval-Augmented Generation (RAG) pipeline with OpenAI and ChromaDB to provide answers based on official legal documents.

## Project Structure

```
.
├── scripts/
│   ├── generate_chunks.py       # Parses law XMLs into structured JSON chunks.
│   └── load_chunks_to_chromadb.py # Loads the JSON chunks into ChromaDB.
├── data/
│   ├── *.xml                    # Raw XML law documents.
│   └── *.json                   # Generated law chunks (auto-created).
├── chroma_db/
│   └── ...                      # Persisted ChromaDB vector store.
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

The application follows a RAG architecture:

1.  **Data Processing (Offline):** The `ingest.sh` script orchestrates the parsing of XML law documents and their storage in a ChromaDB vector database. Each law is stored in a separate collection.
2.  **User Query:** The user enters a question in the Streamlit chat interface.
3.  **Retrieval:** The application queries all law collections in ChromaDB to find the most relevant document chunks (articles, paragraphs) based on the user's question.
4.  **Generation:** The retrieved chunks are combined with the user's question into a prompt that is sent to an OpenAI model (`gpt-4o-mini`).
5.  **Response:** The model generates a well-reasoned answer based on the provided legal context, which is then displayed to the user along with the sources.

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