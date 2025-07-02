#!/bin/bash

echo "🔄 Updating ChromaDB and restarting service..."

# Set correct permissions (adjust user as needed)
sudo chown -R devuser:devuser ./chroma_db/
sudo chmod -R 755 ./chroma_db/

# Restart the Streamlit service to pick up new database
sudo systemctl restart streamlit

echo "✅ Database updated and service restarted!"
echo "🔍 Checking service status:"
sudo systemctl status streamlit --no-pager -l 