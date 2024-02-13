#!/bin/bash

# Define the GitHub repository URL
REPO_URL="https://github.com/SickenSquirral/SLPR"

# Directory where the repository is cloned
REPO_DIR="/SLPR"

# File to start
MAIN_PY_FILE="main.py"

# Navigate to the repository directory
cd "$REPO_DIR" || exit

# Pull the latest changes from the repository
git pull

# Check if the pull was successful
if [ $? -eq 0 ]; then
    # If there were changes, start the main.py file
    echo "Update successful. Starting $MAIN_PY_FILE"
    python3 "$MAIN_PY_FILE" &
else
    # If no changes or an error occurred, notify
    echo "No updates available or error occurred while updating."
fi
