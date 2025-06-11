import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime

XML_PATHS = [
    # "data/genfer_fluechtlingskonvention.xml",
    "data/antifolterkonvention.xml",
    "data/bundesverfassung.xml",
    "data/asylgesetz.xml",
    "data/aig.xml",
]
OUTPUT_PATH = "data/law_chunks.json"


def _text_or_none(elem):
    if elem is None:
        return None
    if elem.text is None:
        return None
        
    # Replace <br /> with space
    text = elem.text.replace('<br />', ' ')
    return text.strip()


def slugify(value: str) -> str:
    """Convert heading strings to a safe id component (e.g., '1. Kapitel:' -> 'chap_1')."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


# --- Helper to extract visible text while skipping authorialNote elements ---
def _tag_local(elem):
    """Return local tag name without namespace."""
    return elem.tag.split('}', 1)[-1] if '}' in elem.tag else elem.tag


def extract_text(elem):
    """Recursively collect text content from an element in document order, skipping <authorialNote>."""
    tag = _tag_local(elem)
    if tag == 'authorialNote':
        return ''

    # Treat line breaks as space
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


# Generic parser for Swiss Akoma-Ntoso XML laws (e.g., Asylgesetz, Bundesverfassung)
def parse_swiss_law(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'}

    # --- Global metadata ---
    frbr_work = root.find('.//akn:FRBRWork', ns)
    sr_number = frbr_work.find('akn:FRBRnumber', ns).attrib.get('value') if frbr_work is not None else 'unknown'
    date_document = frbr_work.find('akn:FRBRdate[@name="jolux:dateDocument"]', ns).attrib.get('date', '') if frbr_work is not None else ''
    date_entry_in_force = frbr_work.find('akn:FRBRdate[@name="jolux:dateEntryInForce"]', ns).attrib.get('date', '') if frbr_work is not None else ''
    date_applicability = frbr_work.find('akn:FRBRdate[@name="jolux:dateApplicability"]', ns).attrib.get('date', '') if frbr_work is not None else ''

    doc_title_elem = root.find('.//akn:docTitle', ns)
    if doc_title_elem is not None:
        document_title = re.sub(r'\s+', ' ', extract_text(doc_title_elem)).strip()
    else:
        document_title = 'Unknown'

    chunks = []

    # --- Add meta chunk ---
    meta_content = f"{document_title}, SR {sr_number}, authoritative document, effective from {date_entry_in_force}, applicable as of {date_applicability}."
    meta_chunk = {
        "id": f"{sr_number}_meta",
        "content": meta_content,
        "metadata": {
            "sr_number": sr_number,
            "document_title": document_title,
            "language": "de",  # Assuming German for this file
            "article_id": None,
            "paragraph_id": None,
            "chapter_id": None,
            "section_id": None,
            "date_document": date_document,
            "date_entry_in_force": date_entry_in_force,
            "date_applicability": date_applicability,
            "references": [],
            "amendment_history": "",
            "keywords": [slugify(document_title)],
            "collection_name": f"law_{sr_number.replace('.', '_')}"
        },
        "embedding": []
    }
    chunks.append(meta_chunk)

    # Helper to collect grouping elements (<chapter>, <title>)
    group_elements = []
    for tag in ("chapter", "title"):
        group_elements.extend(root.findall(f'.//akn:{tag}', ns))

    # Fallback: if no group elements found, treat whole document as single group
    if not group_elements:
        group_elements = [root]

    # --- Iterate groups (chapters or titles) and their articles ---
    for group in group_elements:
        # Extract local tag name without namespace, e.g., '{ns}chapter' -> 'chapter'
        tag_name = group.tag.split('}', 1)[-1] if '}' in group.tag else group.tag
        chapter_id = group.attrib.get('eId') if tag_name == 'chapter' else None
        title_id = group.attrib.get('eId') if tag_name == 'title' else None

        # Find articles that are direct or nested descendants of the group
        for article in group.findall('.//akn:article', ns):
            article_eid = article.attrib.get('eId', '')  # e.g., art_3
            article_id = article_eid
            # Get article heading if present
            article_heading_elem = article.find('akn:heading', ns)
            article_title = _text_or_none(article_heading_elem) if article_heading_elem is not None else None
            # Determine if paragraphs exist
            paragraphs = article.findall('.//akn:paragraph', ns)
            if paragraphs:
                for para in paragraphs:
                    para_id = para.attrib.get('eId', '')
                    content = extract_text(para)
                    content = re.sub(r'^\d+\s+', '', re.sub(r'\s+', ' ', content).strip())
                    if not content:
                        continue
                    # Using the full eId of the paragraph, which is guaranteed to be unique
                    # within the document, is the most robust way to create a unique chunk ID.
                    # We replace slashes to create a safe ID string.
                    safe_para_id = para_id.replace('/', '_')
                    chunk_id = f"{sr_number}_{safe_para_id}"

                    meta = {
                        "sr_number": sr_number,
                        "document_title": document_title,
                        "language": "de",
                        "article_id": article_id,
                        "paragraph_id": para_id,
                        "chapter_id": chapter_id,
                        "title_id": title_id,
                        "section_id": None,  # not extracted for now
                        "date_document": date_document,
                        "date_entry_in_force": date_entry_in_force,
                        "date_applicability": date_applicability,
                        "references": [],
                        "amendment_history": "",
                        "keywords": [],
                        "collection_name": f"law_{sr_number.replace('.', '_')}"
                    }
                    if article_title:
                        meta["article_title"] = article_title
                    chunk = {
                        "id": chunk_id,
                        "content": content,
                        "metadata": meta,
                        "embedding": []
                    }
                    chunks.append(chunk)
            else:
                content = extract_text(article)
                content = re.sub(r'\s+', ' ', content).strip()
                if content:
                    chunk_id = f"{sr_number}_{article_eid}"
                    meta = {
                        "sr_number": sr_number,
                        "document_title": document_title,
                        "language": "de",
                        "article_id": article_id,
                        "paragraph_id": None,
                        "chapter_id": chapter_id,
                        "title_id": title_id,
                        "section_id": None,
                        "date_document": date_document,
                        "date_entry_in_force": date_entry_in_force,
                        "date_applicability": date_applicability,
                        "references": [],
                        "amendment_history": "",
                        "keywords": [],
                        "collection_name": f"law_{sr_number.replace('.', '_')}"
                    }
                    if article_title:
                        meta["article_title"] = article_title
                    chunk = {
                        "id": chunk_id,
                        "content": content,
                        "metadata": meta,
                        "embedding": []
                    }
                    chunks.append(chunk)
    return chunks


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/{timestamp}_law_chunks.json"
    
    all_chunks = []
    for xml_path in XML_PATHS:
        if not os.path.isfile(xml_path):
            print(f"XML file not found at {xml_path}")
            continue
        chunks = parse_swiss_law(xml_path)
        all_chunks.extend(chunks)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks from {len(XML_PATHS)} files to {output_path}")


if __name__ == "__main__":
    main() 