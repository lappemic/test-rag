#!/bin/bash
set -e # exit on error

echo "Step 1: Generating chunks from XML files..."
python3 scripts/generate_chunks.py

echo "Step 2: Loading chunks into ChromaDB..."
python3 scripts/load_chunks_to_chromadb.py

echo "Ingestion complete." 