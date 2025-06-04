import streamlit as st
import chromadb
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import logging
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Path to the law PDF to preload
LAW_PDF_PATH = "./data/bundesverfassung-short.pdf"
LAW_PDF_NAME = os.path.basename(LAW_PDF_PATH)
COLLECTION_NAME = "rag_collection"

st.title("Schwizerischer Rechts-Chatbot für Geflüchtete und Migrant\:innen")

# --- Einfache Einführung für die Nutzer ---
st.markdown("""
Mit diesem Chatbot kannst du Fragen zu wichtigen Gesetzen für Geflüchtete und Migrant\:innen in der Schweiz stellen. Der Bot hilft dir zum Beispiel bei Themen wie Aufenthalt, Arbeitssuche, Rechte, Pflichten und nächsten Schritten.

Aktuell kennt der Bot diese Gesetze:

- [Genfer Flüchtlingskonvention (GFK)](https://www.fedlex.admin.ch/eli/cc/1955/443_461_469/de)
- [Schweizerische Bundesverfassung (BV)](https://www.fedlex.admin.ch/eli/cc/1999/404/de)
- [Schweizerisches Asylgesetz (AsylG)](https://www.fedlex.admin.ch/eli/oc/2024/189/de)
- [Ausländer- und Integrationsgesetz (AIG)](https://www.fedlex.admin.ch/eli/oc/2024/188/de)
- [Antifolterkonvention der Vereinten Nationen](https://www.fedlex.admin.ch/eli/cc/1987/1307_1307_1307/de)

Stelle einfach deine Frage – der Bot sucht die passende Antwort in den Gesetzen.
""")

# - [Europäische Menschenrechtskonvention (EMRK)](https://www.echr.coe.int/documents/d/echr/convention_DEU)

# Function to process the law PDF
def process_law_pdf(pdf_path):
    documents = []
    logging.info(f"Processing law PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])
    logging.info(f"Loaded {len(pages)} pages from {pdf_path}")
    # Split into articles using regex
    article_splits = re.split(r'(Art\.\s*\d+[a-zA-Z]?)', full_text)
    articles = []
    i = 1
    while i < len(article_splits):
        art_number = article_splits[i].replace(' ', '')  # e.g., 'Art.1'
        content = article_splits[i+1] if (i+1) < len(article_splits) else ''
        article_text = f"{article_splits[i]}{content}".strip()
        articles.append((art_number, article_text))
        i += 2
    logging.info(f"Split {pdf_path} into {len(articles)} articles")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    for art_number, article_text in articles:
        subchunks = text_splitter.split_text(article_text)
        for subchunk in subchunks:
            meta = {"doc_name": LAW_PDF_NAME, "art_number": art_number}
            documents.append(Document(page_content=subchunk, metadata=meta))
    logging.info(f"Total documents processed: {len(documents)}")
    return documents

# Function to create or get ChromaDB collection
def get_collection(collection_name=COLLECTION_NAME):
    try:
        logging.info(f"Getting or creating collection: {collection_name}")
        collection = chroma_client.get_or_create_collection(name=collection_name)
        return collection
    except Exception as e:
        logging.error(f"Error creating collection: {e}")
        st.error(f"Error creating collection: {e}")
        return None

# Function to check if collection is empty
def is_collection_empty(collection):
    try:
        count = collection.count()
        return count == 0
    except Exception as e:
        logging.error(f"Error checking collection count: {e}")
        return True

# Function to add documents to ChromaDB
def add_to_collection(documents, collection, embeddings):
    for i, doc in enumerate(documents):
        doc_name = doc.metadata.get("doc_name", "Unknown")
        art_number = doc.metadata.get("art_number", "Unknown")
        logging.info(f"Adding document {i} with doc_name: {doc_name}, art_number: {art_number}")
        embedding = embeddings.embed_query(doc.page_content)
        collection.add(
            ids=[f"doc_{i}"],
            embeddings=[embedding],
            metadatas=[{"content": doc.page_content, "doc_name": doc_name, "art_number": art_number}],
            documents=[doc.page_content],
        )
    logging.info(f"Added {len(documents)} documents to collection")

# Function to query the RAG system
def query_rag(query, collection, llm, num_results, embeddings):
    logging.info(f"Querying RAG system with query: {query}")
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding], n_results=num_results
    )
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    context = "\n".join([
        f"[Source: {meta.get('doc_name', 'Unknown')} - {meta.get('art_number', 'Unknown')}]\n{doc}"
        for doc, meta in zip(retrieved_docs, retrieved_metas)
    ])
    logging.info(f"Retrieved {len(retrieved_docs)} documents for query")
    prompt_template = ChatPromptTemplate.from_template(
        """
Answer the question based on the following context. Always cite the filename and article number (Art. nr) in your answer for each reference you use.

{context}

Question: {question}
"""
    )
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke({"context": context, "question": query})
    logging.info("Generated response from RAG chain")
    return response, retrieved_docs, retrieved_metas

# --- Chat Interface ---

if not openai_api_key:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini")
collection = get_collection()

# Preload and embed the law PDF if collection is empty
if collection and is_collection_empty(collection):
    with st.spinner(f"Preloading and embedding {LAW_PDF_NAME}..."):
        try:
            documents = process_law_pdf(LAW_PDF_PATH)
            add_to_collection(documents, collection, embeddings)
            st.success(f"{LAW_PDF_NAME} processed and added to collection!")
        except Exception as e:
            logging.error(f"Error processing law PDF: {e}")
            st.error(f"Error processing law PDF: {e}")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.markdown("**Sources:**")
            for i, (doc, meta) in enumerate(zip(message["sources"]["docs"], message["sources"]["metas"]), 1):
                doc_name = meta.get("doc_name", "Unknown")
                art_number = meta.get("art_number", "Unknown")
                st.markdown(f"**[{doc_name}]-[{art_number}]:** {doc[:200]}...")

# Accept user input
if prompt := st.chat_input("Ask a question about the law..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Generate assistant response with RAG
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                response, retrieved_docs, retrieved_metas = query_rag(
                    prompt, collection, llm, num_results=4, embeddings=embeddings
                )
                st.markdown(response)
                st.markdown("**Sources:**")
                for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas), 1):
                    doc_name = meta.get("doc_name", "Unknown")
                    art_number = meta.get("art_number", "Unknown")
                    st.markdown(f"**[{doc_name}]-[{art_number}]:** {doc[:200]}...")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": {"docs": retrieved_docs, "metas": retrieved_metas}
                })
            except Exception as e:
                st.error(f"Error generating response: {e}")
