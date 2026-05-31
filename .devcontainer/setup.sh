#!/bin/bash
set -e

echo "Installing Python packages from lock file..."

cd /workspaces/deploying-ai/rag_assistant
pip install --upgrade pip
pip install -r requirements-lock.txt

echo "Setup complete."