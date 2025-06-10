#!/bin/bash

# Update the repository
git pull

# activate the virtual environment
source .venv/bin/activate

# install the dependencies
pip install -r requirements.txt

# restart the application
sudo systemctl restart streamlit