"""
ChromaDB client and operations for the Swiss Legal Chatbot.
"""
import logging

import chromadb

from config.settings import CHROMA_DB_PATH, LAW_COLLECTION_PREFIX


class ChromaDBManager:
    """Manages ChromaDB client and collection operations."""
    
    def __init__(self):
        """Initialize ChromaDB client."""
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    def get_law_collections(self):
        """Gets all collections from ChromaDB that start with 'law_'."""
        try:
            collections = self.client.list_collections()
            law_collections = [c for c in collections if c.name.startswith(LAW_COLLECTION_PREFIX)]
            logging.info(f"Found {len(law_collections)} law collections.")
            return law_collections
        except Exception as e:
            logging.error(f"Error getting law collections: {e}")
            return []
    
    def get_loaded_law_names(self, collections):
        """Extracts document titles from the metadata of each collection."""
        law_names = set()
        for collection in collections:
            try:
                # Get one item to find the document title. We assume it's consistent.
                meta = collection.get(limit=1, include=["metadatas"])
                if meta["metadatas"]:
                    doc_title = meta["metadatas"][0].get("document_title")
                    if doc_title:
                        law_names.add(doc_title)
            except Exception as e:
                logging.error(f"Error getting metadata from collection {collection.name}: {e}")
        return sorted(list(law_names))
    
    def delete_all_law_collections(self):
        """Delete all law collections from the database."""
        law_collections = self.get_law_collections()
        if law_collections:
            for collection in law_collections:
                self.client.delete_collection(name=collection.name)
            logging.info(f"Deleted {len(law_collections)} law collections")
        return len(law_collections) 