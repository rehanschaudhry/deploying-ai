"""
Step 1 verification script.
Confirms two things:
  1. The Claude API is reachable and the API key works.
  2. MLflow can log a run to the local tracking server at localhost:5000.

Run from the rag_assistant/ folder with the venv active:
    python scripts/verify_setup.py
"""

import os
import sys
from dotenv import load_dotenv
import anthropic
import mlflow

# --- Load secrets from .env ----------------------------------------------------
# load_dotenv() reads the .env file in the current directory and adds the
# variables to os.environ. This is why we never hardcode API keys in scripts.
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key or api_key == "paste-your-key-here":
    print("ERROR: ANTHROPIC_API_KEY is not set in .env")
    sys.exit(1)


# --- Test 1: Claude API --------------------------------------------------------
print("Test 1: Calling Claude API...")
client = anthropic.Anthropic(api_key=api_key)

response = client.messages.create(
    model="claude-haiku-4-5",  # cheap model, fine for a hello-world test
    max_tokens=50,
    messages=[
        {"role": "user", "content": "Reply with exactly: SETUP OK"}
    ],
)

reply = response.content[0].text.strip()
print(f"  Claude replied: {reply!r}")
print(f"  Tokens used: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

if "SETUP OK" not in reply:
    print("  WARNING: Reply did not contain 'SETUP OK' but the API call succeeded.")
print("  Claude API: PASS\n")


# --- Test 2: MLflow ------------------------------------------------------------
print("Test 2: Logging a run to MLflow...")
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("step_1_verification")

with mlflow.start_run(run_name="setup_check") as run:
    mlflow.log_param("step", 1)
    mlflow.log_param("test_type", "setup_verification")
    mlflow.log_metric("setup_passed", 1.0)
    print(f"  Logged run: {run.info.run_id}")

print("  MLflow: PASS\n")
print("=" * 50)
print("Step 1 setup verification: COMPLETE")
print("View the run at: http://localhost:5000")
print("=" * 50)