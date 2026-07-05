# DevOps Knowledge Assistant

An AI-powered RAG agent that helps DevOps teams instantly find answers from technical documentation and manage support tickets — built with production-grade architecture.

## What It Does

Engineers waste hours searching scattered documentation and digging through ticket systems. This assistant lets them ask questions in plain English:

- "How do I fix a pod stuck in CrashLoopBackOff?" — searches real Kubernetes docs, returns cited answer
- "Show me all open P1 tickets" — queries ticket database with smart filters  
- "What are open Kubernetes tickets and how do I fix them?" — searches BOTH sources simultaneously

## Tech Stack

- LLM: Claude Sonnet (Anthropic API) — classification and answer generation
- Agent Framework: LangGraph — multi-node agent with conditional routing
- Vector Database: ChromaDB — semantic search over 1,630 document chunks
- Structured Database: SQLite — filtered ticket queries
- Embeddings: sentence-transformers all-MiniLM-L6-v2 — free local embeddings
- API Layer: FastAPI — REST API with auto-generated docs at /docs
- Caching: Redis — response caching with 1 hour TTL
- Observability: LangSmith — full trace visibility, token tracking, cost monitoring
- Containerization: Docker + docker-compose
- Infrastructure: Terraform on AWS ECS + ECR + ALB
- CI/CD: GitHub Actions — automated build and deploy on push

## Knowledge Base

- FastAPI official docs: 137 markdown files
- Kubernetes official docs: 348 markdown files
- DevOps support tickets: 300 realistic synthetic tickets
- Total chunks after embedding: 1,630 vector chunks

## How the Agent Works

The agent has 4 nodes connected in a graph:

1. Classify node — Claude reads the question and decides: docs, tickets, or both
2. Search docs node — ChromaDB semantic search finds relevant chunks by meaning
3. Search tickets node — SQLite filters tickets by status, priority, category
4. Generate node — Claude reads all results and writes a cited answer

Redis caches every answer. Cache hit returns in under 10ms vs 4-5 seconds for a full agent run.

Every run is traced in LangSmith showing node-by-node timing, token usage, and cost per query.

## API Endpoints

- GET /health — health check showing Redis and agent status
- POST /ask — main endpoint, accepts a question returns an answer
- GET /tickets/open — returns all open tickets from the database
- DELETE /cache — clears all Redis cached answers

## Example Request

POST /ask
{"question": "How do I fix a pod stuck in CrashLoopBackOff?"}

Example Response:
{
  "question": "How do I fix a pod stuck in CrashLoopBackOff?",
  "answer": "CrashLoopBackOff means your container keeps crashing...",
  "query_type": "docs",
  "cached": false,
  "response_time_ms": 4823
}

## Getting Started

1. Clone the repository
git clone https://github.com/vydeesh970/devops-knowledge-assistant.git
cd devops-knowledge-assistant

2. Set up Python environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

3. Add your API keys to a .env file
ANTHROPIC_API_KEY=your-key-here
LANGCHAIN_API_KEY=your-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=devops-knowledge-assistant
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

4. Run the data pipeline (first time only)
python ingestion/generate_tickets.py
python ingestion/download_docs.py
python ingestion/chunk_and_embed.py
python ingestion/load_to_db.py

5. Start the API
python -m uvicorn api.main:app --reload --port 8000

Visit http://localhost:8000/docs for interactive API documentation.

## Project Structure

agent/graph.py — LangGraph agent with classify, search, and generate nodes
api/main.py — FastAPI layer with Redis caching
ingestion/ — data pipeline scripts for tickets and documentation
infra/terraform/main.tf — AWS infrastructure as code
.github/workflows/deploy.yml — CI/CD pipeline
Dockerfile — container definition
docker-compose.yml — local stack orchestration

## Author

Vydeesh Mamuduru
GitHub: https://github.com/vydeesh970