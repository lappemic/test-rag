"""
Production-Ready Improved Swiss Legal Document Chunker
This script implements all the chunking improvements and processes your actual XML files.
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
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on system environment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for improved chunking strategy."""
    min_chunk_size: int = 100      # Minimum tokens per chunk
    max_chunk_size: int = 512      # Maximum tokens per chunk
    target_chunk_size: int = 300   # Target tokens per chunk
    overlap_size: int = 50         # Overlap between adjacent chunks
    enable_semantic_grouping: bool = True
    preserve_article_boundaries: bool = True
    extract_cross_references: bool = True

# Dynamically discover all XML files in data directory
def get_xml_files():
    """Dynamically discover all XML files in the data directory."""
    data_dir = Path("data")
    xml_files = list(data_dir.glob("*.xml"))
    xml_paths = [str(xml_file) for xml_file in xml_files]
    logger.info(f"Discovered XML files: {[os.path.basename(p) for p in xml_paths]}")
    return xml_paths

XML_PATHS = get_xml_files()

# Enhanced chunking configuration
CHUNK_CONFIG = ChunkConfig()

def estimate_token_count(text: str) -> int:
    """Estimate token count for German legal text."""
    if not text.strip():
        return 0
    return int(len(text.split()) * 1.3)

def extract_cross_references(text: str) -> List[str]:
    """Extract legal cross-references from text."""
    references = []
    
    # Article references
    art_pattern = r'(?:Art\.?|Artikel)\s*(\d+[a-z]?(?:\s*Abs\.?\s*\d+)?)'
    references.extend([f"Art. {ref}" for ref in re.findall(art_pattern, text, re.IGNORECASE)])
    
    # SR numbers
    sr_pattern = r'SR\s*(\d+(?:\.\d+)*)'
    references.extend([f"SR {ref}" for ref in re.findall(sr_pattern, text)])
    
    # Law abbreviations
    law_pattern = r'\b([A-Z][a-zA-Z]*G|BV|StGB|ZGB)\b'
    references.extend(re.findall(law_pattern, text))
    
    return list(set(references))

def create_content_hash(content: str, metadata: Dict) -> str:
    """Create content hash for change detection."""
    hash_input = f"{content}_{metadata.get('sr_number', '')}_{metadata.get('article_id', '')}"
    return hashlib.md5(hash_input.encode('utf-8')).hexdigest()[:12]

def split_text_semantically(text: str, config: ChunkConfig) -> List[str]:
    """Split text into semantically coherent chunks with proper overlap."""
    if not text.strip():
        return []
    
    sentences = re.split(r'(?<=[.!?;])\s+(?=[A-ZÄÖÜ])', text.strip())
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_tokens = estimate_token_count(sentence)
        
        if current_size + sentence_tokens > config.max_chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if estimate_token_count(chunk_text) >= config.min_chunk_size:
                chunks.append(chunk_text)
            
            # Create overlap
            overlap_sentences = []
            overlap_size = 0
            for prev_sentence in reversed(current_chunk):
                prev_tokens = estimate_token_count(prev_sentence)
                if overlap_size + prev_tokens <= config.overlap_size:
                    overlap_sentences.append(prev_sentence)
                    overlap_size += prev_tokens
                else:
                    break
            
            current_chunk = list(reversed(overlap_sentences))
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_tokens
    
    if current_chunk:
        chunk_text = ' '.join(current_chunk).strip()
        if estimate_token_count(chunk_text) >= config.min_chunk_size:
            chunks.append(chunk_text)
    
    return chunks

def validate_chunk_quality(chunk: Dict, config: ChunkConfig) -> bool:
    """Validate chunk meets quality standards."""
    content = chunk["content"]
    tokens = chunk["metadata"]["estimated_tokens"]
    
    if tokens < config.min_chunk_size or tokens > config.max_chunk_size:
        return False
    
    if len(content.strip()) < 50 or content.count(' ') < 10:
        return False
    
    if not re.search(r'[a-zA-ZäöüÄÖÜß]{3,}', content):
        return False
    
    return True

def _text_or_none(elem):
    """Extract text from XML element."""
    if elem is None or elem.text is None:
        return None
    return elem.text.replace('<br />', ' ').strip()

def _tag_local(elem):
    """Return local tag name without namespace."""
    return elem.tag.split('}', 1)[-1] if '}' in elem.tag else elem.tag

def extract_text(elem):
    """Recursively extract text content."""
    tag = _tag_local(elem)
    if tag == 'authorialNote':
        return ''

    if tag == 'br':
        return ' '

    texts = []
    if elem.text and elem.text.strip():
        texts.append(elem.text.strip())

    for child in elem:
        child_text = extract_text(child)
        if child_text.strip():
            texts.append(child_text)
        if child.tail and child.tail.strip():
            texts.append(child.tail.strip())

    return ' '.join(texts)

def ensure_unique_id(base_id: str, used_ids: set) -> str:
    """Ensure the ID is unique by adding a suffix if necessary."""
    if base_id not in used_ids:
        used_ids.add(base_id)
        return base_id
    
    # If ID already exists, add a counter suffix
    counter = 1
    while f"{base_id}_v{counter}" in used_ids:
        counter += 1
    
    unique_id = f"{base_id}_v{counter}"
    used_ids.add(unique_id)
    return unique_id

def create_enhanced_chunk(content: str, base_metadata: Dict, chunk_index: int = 0, 
                         total_chunks: int = 1, config: ChunkConfig = None, used_ids: set = None) -> Dict:
    """Create chunk with comprehensive metadata."""
    
    # Create unique chunk ID avoiding duplication
    sr_number = base_metadata['sr_number']
    article_id = base_metadata.get('article_id', '')
    para_id = base_metadata.get('paragraph_id', '')
    
    # Use paragraph ID if available (it's more specific), otherwise use article ID
    if para_id:
        # Clean up paragraph ID and make it safe for use
        safe_para_id = para_id.replace('/', '_').replace('#', '_').replace(' ', '_')
        chunk_id = f"{sr_number}_{safe_para_id}"
    elif article_id:
        # Clean up article ID
        safe_article_id = article_id.replace('/', '_').replace('#', '_').replace(' ', '_')
        chunk_id = f"{sr_number}_{safe_article_id}"
    else:
        # Fallback for meta chunks
        chunk_id = f"{sr_number}_meta"
    
    # Add part number if there are multiple chunks from the same source
    if total_chunks > 1:
        chunk_id += f"_part{chunk_index + 1}"
    
    # Ensure the ID is unique
    if used_ids is not None:
        chunk_id = ensure_unique_id(chunk_id, used_ids)
    
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

def parse_swiss_law_improved(xml_path: str, config: ChunkConfig, used_ids: set = None) -> List[Dict]:
    """Enhanced parser with improved chunking strategy."""
    logger.info(f"Processing {xml_path} with improved chunking...")
    
    if used_ids is None:
        used_ids = set()
    
    if not os.path.isfile(xml_path):
        logger.error(f"XML file not found: {xml_path}")
        return []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'}

        # Extract metadata
        frbr_work = root.find('.//akn:FRBRWork', ns)
        sr_number = frbr_work.find('akn:FRBRnumber', ns).attrib.get('value') if frbr_work is not None else 'unknown'
        date_document = frbr_work.find('akn:FRBRdate[@name="jolux:dateDocument"]', ns).attrib.get('date', '') if frbr_work is not None else ''
        date_entry_in_force = frbr_work.find('akn:FRBRdate[@name="jolux:dateEntryInForce"]', ns).attrib.get('date', '') if frbr_work is not None else ''
        date_applicability = frbr_work.find('akn:FRBRdate[@name="jolux:dateApplicability"]', ns).attrib.get('date', '') if frbr_work is not None else ''

        doc_title_elem = root.find('.//akn:docTitle', ns)
        document_title = re.sub(r'\s+', ' ', extract_text(doc_title_elem)).strip() if doc_title_elem is not None else 'Unknown'

        chunks = []
        stats = {'chunks_created': 0, 'chunks_skipped': 0, 'cross_refs_found': 0, 'duplicate_ids_fixed': 0}

        # Create meta chunk
        base_meta = {
            "sr_number": sr_number,
            "document_title": document_title,
            "language": "de",
            "article_id": None,
            "paragraph_id": None,
            "chapter_id": None,
            "title_id": None,
            "section_id": None,
            "date_document": date_document,
            "date_entry_in_force": date_entry_in_force,
            "date_applicability": date_applicability,
            "collection_name": f"law_{sr_number.replace('.', '_')}"
        }
        
        meta_content = f"{document_title}, SR {sr_number}, authoritative document, effective from {date_entry_in_force}, applicable as of {date_applicability}."
        meta_chunk = create_enhanced_chunk(meta_content, {**base_meta, "chunk_type": "meta"}, 0, 1, config, used_ids)
        meta_chunk["metadata"]["chunk_type"] = "meta"
        chunks.append(meta_chunk)

        # Process articles
        group_elements = []
        for tag in ("chapter", "title"):
            group_elements.extend(root.findall(f'.//akn:{tag}', ns))
        
        if not group_elements:
            group_elements = [root]

        for group in group_elements:
            tag_name = group.tag.split('}', 1)[-1] if '}' in group.tag else group.tag
            chapter_id = group.attrib.get('eId') if tag_name == 'chapter' else None
            title_id = group.attrib.get('eId') if tag_name == 'title' else None

            for article in group.findall('.//akn:article', ns):
                article_eid = article.attrib.get('eId', '')
                
                article_heading_elem = article.find('akn:heading', ns)
                article_title = _text_or_none(article_heading_elem) if article_heading_elem is not None else None
                
                paragraph_elements = article.findall('.//akn:paragraph', ns)
                
                if paragraph_elements:
                    for para in paragraph_elements:
                        para_id = para.attrib.get('eId', '')
                        content = extract_text(para)
                        content = re.sub(r'^\d+\s+', '', re.sub(r'\s+', ' ', content).strip())
                        
                        if not content:
                            stats['chunks_skipped'] += 1
                            continue
                        
                        para_metadata = {
                            **base_meta,
                            "article_id": article_eid,
                            "paragraph_id": para_id,
                            "chapter_id": chapter_id,
                            "title_id": title_id,
                            "article_title": article_title
                        }
                        
                        content_chunks = split_text_semantically(content, config)
                        
                        for i, chunk_content in enumerate(content_chunks):
                            chunk = create_enhanced_chunk(
                                chunk_content, para_metadata, i, len(content_chunks), config, used_ids
                            )
                            
                            if validate_chunk_quality(chunk, config):
                                chunks.append(chunk)
                                stats['chunks_created'] += 1
                                stats['cross_refs_found'] += len(chunk['metadata']['cross_references'])
                            else:
                                stats['chunks_skipped'] += 1
                else:
                    content = extract_text(article)
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    if content:
                        article_metadata = {
                            **base_meta,
                            "article_id": article_eid,
                            "paragraph_id": None,
                            "chapter_id": chapter_id,
                            "title_id": title_id,
                            "article_title": article_title
                        }
                        
                        content_chunks = split_text_semantically(content, config)
                        
                        for i, chunk_content in enumerate(content_chunks):
                            chunk = create_enhanced_chunk(
                                chunk_content, article_metadata, i, len(content_chunks), config, used_ids
                            )
                            
                            if validate_chunk_quality(chunk, config):
                                chunks.append(chunk)
                                stats['chunks_created'] += 1
                                stats['cross_refs_found'] += len(chunk['metadata']['cross_references'])
                            else:
                                stats['chunks_skipped'] += 1

        logger.info(f"Completed {xml_path}: {stats['chunks_created']} chunks, {stats['cross_refs_found']} cross-refs")
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing {xml_path}: {e}")
        return []

def main():
    """Main processing function with improved chunking."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/{timestamp}_law_chunks_improved.json"
    
    logger.info("STARTING IMPROVED CHUNKING PROCESS")
    logger.info(f"Configuration: min={CHUNK_CONFIG.min_chunk_size}, max={CHUNK_CONFIG.max_chunk_size}, overlap={CHUNK_CONFIG.overlap_size}")
    
    all_chunks = []
    used_ids = set()  # Track used IDs across all files
    start_time = datetime.now()
    
    existing_files = [path for path in XML_PATHS if os.path.isfile(path)]
    missing_files = [path for path in XML_PATHS if not os.path.isfile(path)]
    
    if missing_files:
        logger.warning(f"Missing XML files: {missing_files}")
    
    if not existing_files:
        logger.error("No XML files found!")
        return
    
    logger.info(f"Processing {len(existing_files)} XML files")
    
    for xml_path in existing_files:
        chunks = parse_swiss_law_improved(xml_path, CHUNK_CONFIG, used_ids)
        all_chunks.extend(chunks)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    duration = datetime.now() - start_time
    
    # Statistics
    total_tokens = sum(chunk.get('metadata', {}).get('estimated_tokens', 0) for chunk in all_chunks)
    avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0
    total_cross_refs = sum(len(chunk.get('metadata', {}).get('cross_references', [])) for chunk in all_chunks)
    
    logger.info("PROCESSING COMPLETE!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info(f"Average tokens: {avg_tokens:.1f}")
    logger.info(f"Cross-references: {total_cross_refs}")
    logger.info(f"Processing time: {duration}")
    
    print(f"✅ Generated {len(all_chunks)} improved chunks in {output_path}")

if __name__ == "__main__":
    main() 