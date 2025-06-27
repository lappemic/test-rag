"""
Improved Swiss Legal Document Chunker - Example Implementation
Key improvements over original chunking approach
"""
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for improved chunking strategy."""
    min_chunk_size: int = 100  # Minimum tokens per chunk
    max_chunk_size: int = 512  # Maximum tokens per chunk  
    target_chunk_size: int = 300  # Target tokens per chunk
    overlap_size: int = 50  # Overlap between adjacent chunks
    enable_semantic_grouping: bool = True
    preserve_article_boundaries: bool = True
    extract_cross_references: bool = True

def estimate_token_count(text: str) -> int:
    """Estimate token count for German legal text."""
    return int(len(text.split()) * 1.3)

def extract_cross_references(text: str) -> List[str]:
    """Extract legal cross-references from text."""
    references = []
    
    # Article references: Art. 5, Artikel 12 Abs. 2
    art_pattern = r'(?:Art\.?|Artikel)\s*(\d+[a-z]?(?:\s*Abs\.?\s*\d+)?)'
    references.extend(re.findall(art_pattern, text, re.IGNORECASE))
    
    # SR numbers: SR 142.20
    sr_pattern = r'SR\s*(\d+\.[\d\.]*)'
    references.extend(re.findall(sr_pattern, text))
    
    # Law abbreviations: AsylG, AIG, BV
    law_pattern = r'\b([A-Z][a-z]*G|BV|StGB|ZGB)\b'
    references.extend(re.findall(law_pattern, text))
    
    return list(set(references))

def create_content_hash(content: str, metadata: Dict) -> str:
    """Create content hash for change detection."""
    hash_input = f"{content}_{metadata.get('sr_number', '')}_{metadata.get('article_id', '')}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]

def split_text_with_overlap(text: str, config: ChunkConfig) -> List[str]:
    """
    IMPROVEMENT 1: Semantic chunking with overlap
    Split text maintaining semantic boundaries and adding overlap
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_token_count(sentence)
        
        # Check if adding sentence exceeds max size
        if current_size + sentence_tokens > config.max_chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if estimate_token_count(chunk_text) >= config.min_chunk_size:
                chunks.append(chunk_text)
            
            # Create overlap with previous chunk
            overlap_sentences = []
            overlap_size = 0
            for prev_sentence in reversed(current_chunk):
                overlap_tokens = estimate_token_count(prev_sentence)
                if overlap_size + overlap_tokens <= config.overlap_size:
                    overlap_sentences.append(prev_sentence)
                    overlap_size += overlap_tokens
                else:
                    break
            
            current_chunk = list(reversed(overlap_sentences))
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if estimate_token_count(chunk_text) >= config.min_chunk_size:
            chunks.append(chunk_text)
    
    return chunks

def create_enhanced_chunk(content: str, base_metadata: Dict, chunk_index: int = 0, 
                         total_chunks: int = 1, config: ChunkConfig = None) -> Dict:
    """
    IMPROVEMENT 2: Enhanced chunk metadata
    Create chunk with comprehensive metadata and quality metrics
    """
    chunk_id = f"{base_metadata['sr_number']}_{base_metadata['article_id']}"
    if base_metadata.get('paragraph_id'):
        chunk_id += f"_{base_metadata['paragraph_id'].replace('/', '_')}"
    if total_chunks > 1:
        chunk_id += f"_part{chunk_index + 1}"
    
    # Extract cross-references if enabled
    cross_refs = []
    if config and config.extract_cross_references:
        cross_refs = extract_cross_references(content)
    
    return {
        "id": chunk_id,
        "content": content,
        "content_hash": create_content_hash(content, base_metadata),
        "metadata": {
            **base_metadata,
            "chunk_type": "content",
            "chunk_index": chunk_index,
            "total_chunks_in_article": total_chunks,
            "cross_references": cross_refs,
            "estimated_tokens": estimate_token_count(content),
            "content_length": len(content),
            "has_cross_references": len(cross_refs) > 0,
            "processing_timestamp": datetime.now().isoformat()
        },
        "embedding": []
    }

def validate_chunk_quality(chunk: Dict, config: ChunkConfig) -> bool:
    """
    IMPROVEMENT 3: Chunk quality validation
    Validate chunk meets quality standards
    """
    content = chunk["content"]
    tokens = chunk["metadata"]["estimated_tokens"]
    
    # Size validation
    if tokens < config.min_chunk_size or tokens > config.max_chunk_size:
        logger.warning(f"Chunk {chunk['id']} has {tokens} tokens (outside range {config.min_chunk_size}-{config.max_chunk_size})")
        return False
    
    # Content validation
    if len(content.strip()) < 50:  # Too short
        return False
    
    if content.count(' ') < 10:  # Too few words
        return False
    
    # Check for meaningful content (not just numbers/punctuation)
    if not re.search(r'[a-zA-ZäöüÄÖÜß]{3,}', content):
        return False
    
    return True

# Example of improved loading script
class ImprovedChromaLoader:
    """
    IMPROVEMENT 4: Intelligent loading with caching and incremental updates
    """
    
    def __init__(self, chroma_client, embeddings):
        self.client = chroma_client
        self.embeddings = embeddings
        self.content_hashes = {}  # Cache for change detection
    
    def load_existing_hashes(self, collection_name: str) -> Dict[str, str]:
        """Load existing content hashes to detect changes."""
        try:
            collection = self.client.get_collection(collection_name)
            # Get all metadata containing hashes
            result = collection.get(include=["metadatas"])
            hashes = {}
            for i, metadata in enumerate(result.get("metadatas", [])):
                if "content_hash" in metadata:
                    hashes[result["ids"][i]] = metadata["content_hash"]
            return hashes
        except:
            return {}
    
    def needs_update(self, chunk: Dict, existing_hashes: Dict[str, str]) -> bool:
        """Check if chunk needs to be updated."""
        chunk_id = chunk["id"]
        new_hash = chunk["content_hash"]
        
        if chunk_id not in existing_hashes:
            return True  # New chunk
        
        return existing_hashes[chunk_id] != new_hash  # Changed content
    
    def load_chunks_incrementally(self, chunks: List[Dict], collection_name: str):
        """
        IMPROVEMENT 5: Incremental loading
        Only update changed/new chunks
        """
        logger.info(f"Starting incremental load for {collection_name}")
        
        # Get existing hashes
        existing_hashes = self.load_existing_hashes(collection_name)
        
        # Filter chunks that need updates
        chunks_to_update = [
            chunk for chunk in chunks 
            if self.needs_update(chunk, existing_hashes)
        ]
        
        if not chunks_to_update:
            logger.info(f"No updates needed for {collection_name}")
            return
        
        logger.info(f"Updating {len(chunks_to_update)} chunks in {collection_name}")
        
        # Get or create collection
        try:
            collection = self.client.get_collection(collection_name)
        except:
            collection = self.client.create_collection(collection_name)
        
        # Process in optimized batches
        batch_size = min(50, max(10, len(chunks_to_update) // 4))  # Adaptive batch size
        
        for i in range(0, len(chunks_to_update), batch_size):
            batch = chunks_to_update[i:i + batch_size]
            
            # Delete existing versions
            existing_ids = [chunk["id"] for chunk in batch if chunk["id"] in existing_hashes]
            if existing_ids:
                collection.delete(ids=existing_ids)
            
            # Prepare batch data
            ids = [chunk["id"] for chunk in batch]
            texts = [chunk["content"] for chunk in batch]
            metadatas = [self.sanitize_metadata(chunk["metadata"]) for chunk in batch]
            
            # Generate embeddings with error handling
            try:
                embeddings = self.embeddings.embed_documents(texts)
                
                # Add to collection
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts
                )
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks_to_update) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Could implement retry logic here
    
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

def example_usage():
    """Example of how to use the improved chunking system."""
    
    # Configure chunking strategy
    config = ChunkConfig(
        min_chunk_size=100,
        max_chunk_size=512,
        target_chunk_size=300,
        overlap_size=50,
        enable_semantic_grouping=True,
        preserve_article_boundaries=True,
        extract_cross_references=True
    )
    
    # Example chunk processing
    sample_text = """
    Art. 5 Abs. 1 AsylG besagt, dass Asylsuchende ihre Identität preisgeben müssen.
    Dies steht im Einklang mit Art. 12 BV über die Ausweispflicht.
    Weitere Bestimmungen finden sich in SR 142.20.
    """
    
    # Create base metadata
    base_metadata = {
        "sr_number": "142.20",
        "document_title": "Asylgesetz",
        "language": "de",
        "article_id": "art_5",
        "paragraph_id": "art_5__para_1",
        "collection_name": "law_142_20"
    }
    
    # Process with improved chunking
    content_chunks = split_text_with_overlap(sample_text, config)
    
    enhanced_chunks = []
    for i, content in enumerate(content_chunks):
        chunk = create_enhanced_chunk(
            content, base_metadata, i, len(content_chunks), config
        )
        
        if validate_chunk_quality(chunk, config):
            enhanced_chunks.append(chunk)
        else:
            logger.warning(f"Chunk {chunk['id']} failed quality validation")
    
    return enhanced_chunks

if __name__ == "__main__":
    # Demonstration
    example_chunks = example_usage()
    print(f"Generated {len(example_chunks)} improved chunks")
    
    for chunk in example_chunks:
        print(f"Chunk {chunk['id']}: {chunk['metadata']['estimated_tokens']} tokens")
        print(f"Cross-refs: {chunk['metadata']['cross_references']}")
        print(f"Content: {chunk['content'][:100]}...")
        print("---") 