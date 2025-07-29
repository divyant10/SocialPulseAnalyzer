#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading model zip from Google Drive..."
# Attempting to use python3, if this fails, the script will exit.
# The -m flag tells python to run the gdown module
python3 -m gdown --id 1tCsj1O8_ptznbK0JpqSIrUPkGz1qySLd -O /var/data/models.zip

echo "Extracting models.zip..."
unzip /var/data/models.zip -d /var/data/

echo "Build process completed."