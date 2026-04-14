#!/bin/bash
# Run FastAPI on EC2
uvicorn src.app:app --host 0.0.0.0 --port 8000
