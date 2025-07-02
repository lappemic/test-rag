#!/bin/bash

# Update the repository
git pull

# activate the virtual environment
source .venv/bin/activate

# install the dependencies
pip install -r requirements.txt

# execute the update_database.sh script
./update_database.sh

# restart the application
sudo systemctl restart streamlit