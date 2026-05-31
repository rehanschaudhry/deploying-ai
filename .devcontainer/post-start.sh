#!/bin/bash

echo "Starting MLflow server on port 5000..."

cd /workspaces/deploying-ai/rag_assistant

# Run MLflow in the background, log output to a file we can check later
mkdir -p .devcontainer-logs
nohup mlflow ui --port 5000 --host 0.0.0.0 \
    > .devcontainer-logs/mlflow.log 2>&1 &

echo "MLflow started in background. Logs: .devcontainer-logs/mlflow.log"