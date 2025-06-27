"""
Improved Swiss Legal Document Chunker with Advanced Chunking Strategies
"""
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for chunking strategy."""
    min_chunk_size: int = 100  # Minimum tokens per chunk
    max_chunk_size: int = 512  # Maximum tokens per chunk
    target_chunk_size: int = 300  # Target tokens per chunk
    overlap_size: int = 50  # Overlap between adjacent chunks
    enable_semantic_grouping: bool = True
    preserve_article_boundaries: bool = True
    extract_cross_references: bool = True

XML_PATHS = [
    "data/antifolterkonvention.xml",
    "data/bundesverfassung.xml", 
    "data/asylgesetz.xml",
    "data/aig.xml",
]

# Enhanced chunking configuration
CHUNK_CONFIG = ChunkConfig()

def estimate_token_count(text: str) -> int:
    """Rough estimation of token count (German text ~= 1.3 tokens per word)."""
    return int(len(text.split()) * 1.3)

def extract_cross_references(text: str) -> List[str]:
    """Extract references to other articles, laws, or regulations."""
    references = []
    
    # Pattern for article references (Art. X, Artikel X)
    art_pattern = r'(?:Art\.?|Artikel)\s*(\d+[a-z]?(?:\s*Abs\.?\s*\d+)?)'
    references.extend(re.findall(art_pattern, text, re.IGNORECASE))
    
    # Pattern for SR numbers (SR XXX.X)
    sr_pattern = r'SR\s*(\d+\.[\d\.]*)'
    references.extend(re.findall(sr_pattern, text))
    
    # Pattern for law abbreviations (AsylG, AIG, etc.)
    law_pattern = r'\b([A-Z][a-z]*G|BV|StGB|ZGB)\b'
    references.extend(re.findall(law_pattern, text))
    
    return list(set(references))  # Remove duplicates

def create_content_hash(content: str, metadata: Dict) -> str:
    """Create a hash for content to detect changes."""
    hash_input = f"{content}_{metadata.get('sr_number', '')}_{metadata.get('article_id', '')}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]

def split_text_semantically(text: str, config: ChunkConfig) -> List[str]:
    """Split text into semantically coherent chunks with proper overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_token_count(sentence)
        
        # If adding this sentence exceeds max size, finalize current chunk
        if current_size + sentence_tokens > config.max_chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if estimate_token_count(chunk_text) >= config.min_chunk_size:
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
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

def group_related_paragraphs(paragraphs: List[Dict], config: ChunkConfig) -> List[List[Dict]]:
    """Group related paragraphs together for better semantic coherence."""
    if not config.enable_semantic_grouping:
        return [[p] for p in paragraphs]
    
    groups = []
    current_group = []
    current_size = 0
    
    for para in paragraphs:
        para_size = estimate_token_count(para['content'])
        
        # Start new group if current would exceed target size
        if current_size + para_size > config.target_chunk_size and current_group:
            groups.append(current_group)
            current_group = []
            current_size = 0
        
        current_group.append(para)
        current_size += para_size
    
    if current_group:
        groups.append(current_group)
    
    return groups

def _text_or_none(elem):
    """Extract text from XML element."""
    if elem is None or elem.text is None:
        return None
    return elem.text.replace('<br />', ' ').strip()

def _tag_local(elem):
    """Return local tag name without namespace."""
    return elem.tag.split('}', 1)[-1] if '}' in elem.tag else elem.tag

def extract_text(elem):
    """Recursively extract text content, skipping authorialNote elements."""
    tag = _tag_local(elem)
    if tag == 'authorialNote':
        return ''

    if tag == 'br':
        return ' '

    texts = []
    if elem.text and elem.text.strip():
        texts.append(elem.text.strip())

    for child in elem:
        texts.append(extract_text(child))
        if child.tail and child.tail.strip():
            texts.append(child.tail.strip())

    return ' '.join(texts)

def parse_swiss_law_improved(xml_path: str, config: ChunkConfig) -> List[Dict]:
    """Enhanced parser with improved chunking strategy."""
    logger.info(f"Processing {xml_path} with improved chunking...")
    
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
    stats = {'total_paragraphs': 0, 'chunks_created': 0, 'skipped_short': 0}

    # Create enhanced meta chunk
    meta_content = f"{document_title}, SR {sr_number}, authoritative document, effective from {date_entry_in_force}, applicable as of {date_applicability}."
    
    meta_chunk = {
        "id": f"{sr_number}_meta",
        "content": meta_content,
        "content_hash": create_content_hash(meta_content, {"sr_number": sr_number}),
        "metadata": {
            "sr_number": sr_number,
            "document_title": document_title,
            "language": "de",
            "chunk_type": "meta",
            "article_id": None,
            "paragraph_id": None,
            "chapter_id": None,
            "section_id": None,
            "date_document": date_document,
            "date_entry_in_force": date_entry_in_force,
            "date_applicability": date_applicability,
            "cross_references": [],
            "estimated_tokens": estimate_token_count(meta_content),
            "collection_name": f"law_{sr_number.replace('.', '_')}"
        },
        "embedding": []
    }
    chunks.append(meta_chunk)

    # Process articles with improved chunking
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
            
            # Collect all paragraphs for this article
            paragraphs = []
            paragraph_elements = article.findall('.//akn:paragraph', ns)
            
            if paragraph_elements:
                for para in paragraph_elements:
                    para_id = para.attrib.get('eId', '')
                    content = extract_text(para)
                    content = re.sub(r'^\d+\s+', '', re.sub(r'\s+', ' ', content).strip())
                    
                    if content and estimate_token_count(content) >= config.min_chunk_size:
                        paragraphs.append({
                            'para_id': para_id,
                            'content': content,
                            'article_id': article_eid,
                            'article_title': article_title
                        })
                        stats['total_paragraphs'] += 1
                    else:
                        stats['skipped_short'] += 1
            else:
                # Handle article without explicit paragraphs
                content = extract_text(article)
                content = re.sub(r'\s+', ' ', content).strip()
                if content and estimate_token_count(content) >= config.min_chunk_size:
                    paragraphs.append({
                        'para_id': None,
                        'content': content,
                        'article_id': article_eid,
                        'article_title': article_title
                    })
                    stats['total_paragraphs'] += 1

            # Group paragraphs and create chunks
            if config.preserve_article_boundaries:
                # Process each article separately
                for para in paragraphs:
                    content_chunks = split_text_semantically(para['content'], config)
                    
                    for i, chunk_content in enumerate(content_chunks):
                        cross_refs = extract_cross_references(chunk_content) if config.extract_cross_references else []
                        
                        chunk_id = f"{sr_number}_{para['article_id']}"
                        if para['para_id']:
                            chunk_id += f"_{para['para_id'].replace('/', '_')}"
                        if len(content_chunks) > 1:
                            chunk_id += f"_part{i+1}"
                        
                        chunk = {
                            "id": chunk_id,
                            "content": chunk_content,
                            "content_hash": create_content_hash(chunk_content, {"sr_number": sr_number, "article_id": para['article_id']}),
                            "metadata": {
                                "sr_number": sr_number,
                                "document_title": document_title,
                                "language": "de",
                                "chunk_type": "content",
                                "article_id": para['article_id'],
                                "paragraph_id": para['para_id'],
                                "chapter_id": chapter_id,
                                "title_id": title_id,
                                "section_id": None,
                                "article_title": para['article_title'],
                                "chunk_index": i,
                                "total_chunks_in_article": len(content_chunks),
                                "date_document": date_document,
                                "date_entry_in_force": date_entry_in_force,
                                "date_applicability": date_applicability,
                                "cross_references": cross_refs,
                                "estimated_tokens": estimate_token_count(chunk_content),
                                "collection_name": f"law_{sr_number.replace('.', '_')}"
                            },
                            "embedding": []
                        }
                        chunks.append(chunk)
                        stats['chunks_created'] += 1
            else:
                # Group related paragraphs together
                paragraph_groups = group_related_paragraphs(paragraphs, config)
                
                for group_idx, para_group in enumerate(paragraph_groups):
                    combined_content = ' '.join([p['content'] for p in para_group])
                    content_chunks = split_text_semantically(combined_content, config)
                    
                    for i, chunk_content in enumerate(content_chunks):
                        cross_refs = extract_cross_references(chunk_content) if config.extract_cross_references else []
                        
                        # Create representative metadata from the group
                        primary_para = para_group[0]
                        article_ids = list(set([p['article_id'] for p in para_group]))
                        
                        chunk_id = f"{sr_number}_group{group_idx}_part{i+1}"
                        
                        chunk = {
                            "id": chunk_id,
                            "content": chunk_content,
                            "content_hash": create_content_hash(chunk_content, {"sr_number": sr_number, "group": group_idx}),
                            "metadata": {
                                "sr_number": sr_number,
                                "document_title": document_title,
                                "language": "de",
                                "chunk_type": "grouped_content",
                                "article_id": primary_para['article_id'],
                                "related_articles": article_ids,
                                "paragraph_id": primary_para['para_id'],
                                "chapter_id": chapter_id,
                                "title_id": title_id,
                                "section_id": None,
                                "article_title": primary_para['article_title'],
                                "chunk_index": i,
                                "group_index": group_idx,
                                "paragraphs_in_group": len(para_group),
                                "date_document": date_document,
                                "date_entry_in_force": date_entry_in_force,
                                "date_applicability": date_applicability,
                                "cross_references": cross_refs,
                                "estimated_tokens": estimate_token_count(chunk_content),
                                "collection_name": f"law_{sr_number.replace('.', '_')}"
                            },
                            "embedding": []
                        }
                        chunks.append(chunk)
                        stats['chunks_created'] += 1

    logger.info(f"Processed {xml_path}: {stats['total_paragraphs']} paragraphs → {stats['chunks_created']} chunks (skipped {stats['skipped_short']} short)")
    return chunks

def process_single_file(xml_path: str, config: ChunkConfig) -> List[Dict]:
    """Process a single XML file (for parallel processing)."""
    try:
        return parse_swiss_law_improved(xml_path, config)
    except Exception as e:
        logger.error(f"Error processing {xml_path}: {e}")
        return []

def main():
    """Main processing function with parallel file processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/{timestamp}_law_chunks_improved.json"
    
    logger.info(f"Starting improved chunking with config: {CHUNK_CONFIG}")
    
    all_chunks = []
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=min(4, len(XML_PATHS))) as executor:
        # Submit all files for processing
        future_to_file = {
            executor.submit(process_single_file, xml_path, CHUNK_CONFIG): xml_path 
            for xml_path in XML_PATHS if os.path.isfile(xml_path)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            xml_path = future_to_file[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
                logger.info(f"Completed {xml_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {xml_path}: {e}")

    # Write results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    # Generate statistics
    stats = {
        'total_chunks': len(all_chunks),
        'chunk_types': {},
        'avg_tokens': 0,
        'size_distribution': {'small': 0, 'medium': 0, 'large': 0}
    }
    
    total_tokens = 0
    for chunk in all_chunks:
        chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
        stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
        
        tokens = chunk['metadata'].get('estimated_tokens', 0)
        total_tokens += tokens
        
        if tokens < 150:
            stats['size_distribution']['small'] += 1
        elif tokens < 400:
            stats['size_distribution']['medium'] += 1
        else:
            stats['size_distribution']['large'] += 1
    
    stats['avg_tokens'] = total_tokens / len(all_chunks) if all_chunks else 0
    
    logger.info(f"Chunking complete!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
    
    print(f"✅ Saved {len(all_chunks)} improved chunks from {len(XML_PATHS)} files to {output_path}")
    print(f"📊 Average tokens per chunk: {stats['avg_tokens']:.1f}")
    print(f"📈 Size distribution: {stats['size_distribution']}")

if __name__ == "__main__":
    main() 