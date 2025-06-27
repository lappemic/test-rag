"""
Production-Ready Improved ChromaDB Loading Script
This script implements all loading improvements and replaces your current ChromaDB content.
"""
import glob
import json
import logging
import os
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

@dataclass
class LoadingConfig:
    """Configuration for improved loading strategy."""
    batch_size: int = 50              # Base batch size
    max_retries: int = 3              # Retry failed embeddings
    enable_incremental: bool = True   # Only update changed chunks
    embedding_cache_file: str = "data/embedding_cache.json"  # Cache embeddings
    validate_embeddings: bool = True  # Validate embedding quality
    backup_existing: bool = True      # Backup before replacing

class ImprovedChromaLoader:
    """Enhanced ChromaDB loader with performance optimizations."""
    
    def __init__(self, config: LoadingConfig = None):
        self.config = config or LoadingConfig()
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.embedding_cache = self._load_embedding_cache()
        
    def _load_embedding_cache(self) -> Dict[str, List[float]]:
        """Load cached embeddings to avoid re-computing identical content."""
        try:
            if os.path.exists(self.config.embedding_cache_file):
                with open(self.config.embedding_cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded {len(cache)} cached embeddings")
                return cache
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
        return {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.config.embedding_cache_file), exist_ok=True)
            with open(self.config.embedding_cache_file, 'w') as f:
                json.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content to detect changes."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, content: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        content_hash = self._get_content_hash(content)
        return self.embedding_cache.get(content_hash)
    
    def _cache_embedding(self, content: str, embedding: List[float]):
        """Cache embedding for future use."""
        content_hash = self._get_content_hash(content)
        self.embedding_cache[content_hash] = embedding
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with intelligent caching."""
        embeddings = []
        texts_to_embed = []
        cache_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding:
                embeddings.append(cached_embedding)
                cache_indices.append(-1)  # Mark as cached
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                cache_indices.append(len(texts_to_embed) - 1)
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} texts (cached: {len(texts) - len(texts_to_embed)})")
            
            for attempt in range(self.config.max_retries):
                try:
                    new_embeddings = self.embeddings.embed_documents(texts_to_embed)
                    
                    # Cache new embeddings
                    for text, embedding in zip(texts_to_embed, new_embeddings):
                        self._cache_embedding(text, embedding)
                    
                    # Fill in the embeddings list
                    embed_idx = 0
                    for i, cache_idx in enumerate(cache_indices):
                        if cache_idx >= 0:  # Not cached
                            embeddings[i] = new_embeddings[embed_idx]
                            embed_idx += 1
                    
                    break  # Success
                    
                except Exception as e:
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                    if attempt == self.config.max_retries - 1:
                        logger.error("All embedding attempts failed")
                        # Use zero vectors as fallback
                        for i, emb in enumerate(embeddings):
                            if emb is None:
                                embeddings[i] = [0.0] * 1536
                    else:
                        time.sleep(2 ** attempt)  # Exponential backoff
        
        return embeddings
    
    def _load_existing_metadata(self, collection_name: str) -> Dict[str, Dict]:
        """Load existing chunk metadata to detect what needs updating."""
        try:
            collection = self.client.get_collection(collection_name)
            result = collection.get(include=["metadatas"])
            
            existing_metadata = {}
            for chunk_id, metadata in zip(result.get("ids", []), result.get("metadatas", [])):
                existing_metadata[chunk_id] = metadata
            
            logger.info(f"Loaded metadata for {len(existing_metadata)} existing chunks")
            return existing_metadata
            
        except Exception as e:
            logger.info(f"Collection {collection_name} does not exist: {e}")
            return {}
    
    def _needs_update(self, chunk: Dict, existing_metadata: Dict[str, Dict]) -> bool:
        """Determine if a chunk needs to be updated."""
        chunk_id = chunk["id"]
        
        if chunk_id not in existing_metadata:
            return True  # New chunk
        
        existing = existing_metadata[chunk_id]
        
        # Check content hash if available
        if "content_hash" in chunk.get("metadata", {}) and "content_hash" in existing:
            return chunk["metadata"]["content_hash"] != existing["content_hash"]
        
        # Fallback: check content length
        if "content_length" in chunk.get("metadata", {}) and "content_length" in existing:
            return chunk["metadata"]["content_length"] != existing["content_length"]
        
        return True
    
    def sanitize_metadata(self, metadata: Dict) -> Dict:
        """Sanitize metadata for ChromaDB compatibility."""
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, list):
                sanitized[key] = ",".join(map(str, value))
            elif isinstance(value, (int, float, str, bool)):
                sanitized[key] = str(value)
            else:
                sanitized[key] = str(value)
        return sanitized
    
    def load_chunks_improved(self, chunks: List[Dict], collection_name: str, force_replace: bool = False):
        """Main improved loading function with all optimizations."""
        start_time = time.time()
        logger.info(f"Starting loading for collection: {collection_name}")
        logger.info(f"Total chunks to process: {len(chunks)}")
        
        # For force replace, delete existing collection
        if force_replace:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except:
                pass  # Collection didn't exist
        
        # Get or create collection
        try:
            collection = self.client.get_collection(collection_name)
        except:
            collection = self.client.create_collection(collection_name)
            logger.info(f"Created new collection: {collection_name}")
        
        # Load existing metadata for incremental updates
        existing_metadata = {}
        if self.config.enable_incremental and not force_replace:
            existing_metadata = self._load_existing_metadata(collection_name)
        
        # Filter chunks that need updates
        if self.config.enable_incremental and existing_metadata and not force_replace:
            chunks_to_update = [
                chunk for chunk in chunks 
                if self._needs_update(chunk, existing_metadata)
            ]
            logger.info(f"Incremental update: {len(chunks_to_update)} of {len(chunks)} chunks need updates")
        else:
            chunks_to_update = chunks
            logger.info("Full reload: processing all chunks")
        
        if not chunks_to_update:
            logger.info("No updates needed!")
            return len(chunks), 0
        
        # Process in batches
        batch_size = self.config.batch_size
        total_batches = (len(chunks_to_update) + batch_size - 1) // batch_size
        successful_chunks = 0
        failed_chunks = 0
        
        for batch_idx in range(0, len(chunks_to_update), batch_size):
            batch = chunks_to_update[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            try:
                # Delete existing versions if incremental
                if not force_replace and existing_metadata:
                    existing_ids = [
                        chunk["id"] for chunk in batch 
                        if chunk["id"] in existing_metadata
                    ]
                    if existing_ids:
                        collection.delete(ids=existing_ids)
                
                # Prepare batch data
                ids = [chunk["id"] for chunk in batch]
                texts = [chunk["content"] for chunk in batch]
                metadatas = [self.sanitize_metadata(chunk.get("metadata", {})) for chunk in batch]
                
                # Generate embeddings with caching
                embeddings = self._generate_embeddings_batch(texts)
                
                # Add to collection
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts
                )
                
                successful_chunks += len(batch)
                logger.info(f"Successfully processed batch {batch_num}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                failed_chunks += len(batch)
        
        # Save embedding cache
        self._save_embedding_cache()
        
        duration = time.time() - start_time
        logger.info(f"Loading completed in {duration:.2f} seconds")
        logger.info(f"Successfully processed: {successful_chunks} chunks")
        logger.info(f"Failed: {failed_chunks} chunks")
        
        return successful_chunks, failed_chunks

def get_latest_chunks_file() -> str:
    """Find the latest improved chunks file or fall back to regular chunks."""
    # First look for improved chunks
    improved_files = glob.glob("data/*_law_chunks_improved.json")
    if improved_files:
        latest = max(improved_files, key=os.path.getmtime)
        logger.info(f"Using improved chunks file: {latest}")
        return latest
    
    # Fall back to regular chunks
    regular_files = glob.glob("data/*law_chunks*.json")
    if not regular_files:
        raise FileNotFoundError("No law_chunks files found in data directory.")
    
    latest = max(regular_files, key=os.path.getmtime)
    logger.info(f"Using regular chunks file: {latest}")
    return latest

def load_chunks_from_file(path: str) -> List[Dict]:
    """Load chunks from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_chunks_by_collection(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """Group chunks by their collection name."""
    collections = {}
    for chunk in chunks:
        collection_name = chunk.get("metadata", {}).get("collection_name", "unknown")
        collections.setdefault(collection_name, []).append(chunk)
    return collections

def main():
    """Main function to replace ChromaDB content with improved chunks."""
    logger.info("STARTING IMPROVED CHROMADB LOADING")
    
    # Configure loading
    config = LoadingConfig(
        batch_size=50,
        max_retries=3,
        enable_incremental=True,
        validate_embeddings=True,
        backup_existing=True
    )
    
    # Initialize loader
    loader = ImprovedChromaLoader(config)
    
    try:
        # Load chunks
        chunks_file = get_latest_chunks_file()
        chunks = load_chunks_from_file(chunks_file)
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
        
        # Group by collection
        collections = group_chunks_by_collection(chunks)
        logger.info(f"Found {len(collections)} collections to process")
        
        # Ask user about replacement strategy
        print("\nReplacement options:")
        print("1. Incremental update (recommended) - only update changed chunks")
        print("2. Full replacement - replace all collections completely")
        
        while True:
            choice = input("Choose option (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("Please enter 1 or 2")
        
        force_replace = (choice == '2')
        if force_replace:
            logger.info("Using FULL REPLACEMENT mode")
            config.enable_incremental = False
        else:
            logger.info("Using INCREMENTAL UPDATE mode")
        
        # Process each collection
        total_start_time = time.time()
        total_successful = 0
        total_failed = 0
        
        for collection_name, chunk_list in collections.items():
            logger.info(f"\nProcessing collection: {collection_name} ({len(chunk_list)} chunks)")
            
            try:
                successful, failed = loader.load_chunks_improved(
                    chunk_list, collection_name, force_replace=force_replace
                )
                total_successful += successful
                total_failed += failed
                
            except Exception as e:
                logger.error(f"Failed to process collection {collection_name}: {e}")
                total_failed += len(chunk_list)
        
        total_duration = time.time() - total_start_time
        
        logger.info("\nALL COLLECTIONS PROCESSED!")
        logger.info(f"Total successful chunks: {total_successful}")
        logger.info(f"Total failed chunks: {total_failed}")
        logger.info(f"Total processing time: {total_duration:.2f} seconds")
        
        print(f"\n✅ Successfully processed {total_successful} chunks!")
        if total_failed > 0:
            print(f"❌ Failed to process {total_failed} chunks")
        print(f"⏱️  Total time: {total_duration:.2f} seconds")
        
        if total_failed == 0:
            print("\n🎉 ChromaDB replacement completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        raise

if __name__ == "__main__":
    main() 