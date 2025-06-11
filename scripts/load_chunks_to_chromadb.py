#import os
import glob
import json
import logging
import os

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

# Find the latest law_chunks.json file in data/
def get_latest_chunks_file():
    files = glob.glob("data/*law_chunks.json")
    if not files:
        raise FileNotFoundError("No law_chunks.json files found in data directory.")
    latest = max(files, key=os.path.getmtime)
    logging.info(f"Using latest chunks file: {latest}")
    return latest

# Load chunks from file
def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Group chunks by collection_name
def group_by_collection(chunks):
    collections = {}
    for chunk in chunks:
        cname = chunk["metadata"]["collection_name"]
        collections.setdefault(cname, []).append(chunk)
    return collections

def sanitize_metadata(metadata):
    """Replaces None values and converts lists to be compatible with ChromaDB."""
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""  # Replace None with an empty string
        elif isinstance(value, list):
            # Join lists of strings/other into a single comma-separated string
            sanitized[key] = ",".join(map(str, value))
        else:
            sanitized[key] = value
    return sanitized

# Main logic
def main():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    latest_file = get_latest_chunks_file()
    chunks = load_chunks(latest_file)
    grouped = group_by_collection(chunks)

    for cname, chunk_list in grouped.items():
        logging.info(f"Processing collection: {cname} with {len(chunk_list)} chunks")
        # Delete collection if it exists to ensure a fresh start
        try:
            chroma_client.delete_collection(name=cname)
            logging.info(f"Deleted existing collection: {cname}")
        except Exception as e:
            logging.info(f"Collection {cname} did not exist or could not be deleted: {e}")

        # Create new collection
        collection = chroma_client.get_or_create_collection(name=cname)

        # Prepare data for ChromaDB
        ids = [chunk.get("id", f"doc_{cname}_{i}") for i, chunk in enumerate(chunk_list)]
        texts = [chunk["content"] for chunk in chunk_list]
        sanitized_metadatas = [sanitize_metadata(chunk["metadata"]) for chunk in chunk_list]
        
        # Embed and add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = sanitized_metadatas[i:i+batch_size]
            
            logging.info(f"Embedding and adding batch of {len(batch_texts)} documents to {cname}...")
            
            # Generate embeddings for the batch
            batch_embeddings = embeddings.embed_documents(batch_texts)
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts,
            )
        logging.info(f"Finished collection: {cname}")
    logging.info("All collections processed.")

if __name__ == "__main__":
    main() 