"""
main.py

What this file does:
Wraps our LangGraph agent in a FastAPI web API.

WHY FASTAPI?
Right now our agent only works when you run a Python script.
FastAPI turns it into a proper web service that:
- Accepts questions via HTTP requests (like a real API)
- Returns answers as JSON
- Can be called from any frontend, mobile app, or other service
- Has automatic documentation at /docs

WHY REDIS CACHING?
Every time someone asks a question, we call Claude (costs money + takes time).
If someone asks the same question twice, why call Claude again?
Redis stores recent answers in memory:
- Cache hit: returns answer in <10ms (free)
- Cache miss: runs the full agent (~3-5 seconds, costs API credits)

This is exactly how production AI APIs work.
"""

import sys
import os
import hashlib
import json
import time

# Add parent directory to path so we can import our agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
from dotenv import load_dotenv

from agent.graph import build_agent

load_dotenv()


# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(
    title="DevOps Knowledge Assistant",
    description="""
    AI-powered assistant for DevOps teams.
    
    Ask questions about:
    - Kubernetes concepts and troubleshooting
    - FastAPI development
    - Support ticket status and management
    
    Built with LangGraph, Claude, ChromaDB, PostgreSQL, and Redis.
    """,
    version="1.0.0"
)

# CORS = Cross Origin Resource Sharing
# This allows a frontend (like a React app) to call our API
# Without this, browsers block requests from different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # In production, restrict this to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REDIS CACHE SETUP
# ============================================================

def get_redis_client():
    """
    Connects to our Redis instance running in Docker.
    Returns None if Redis isn't available (graceful degradation)
    so the API still works even without caching.
    """
    try:
        client = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True,
            socket_connect_timeout=2
        )
        client.ping()  # Test the connection
        return client
    except Exception:
        print("⚠️  Redis not available - running without cache")
        return None


# Initialize Redis and agent at startup
redis_client = get_redis_client()
agent = build_agent()

CACHE_TTL = 3600  # Cache answers for 1 hour (in seconds)


# ============================================================
# REQUEST/RESPONSE MODELS
# Pydantic models define exactly what our API accepts and returns
# FastAPI uses these for automatic validation and documentation
# ============================================================

class QuestionRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I fix a pod stuck in CrashLoopBackOff?"
            }
        }


class QuestionResponse(BaseModel):
    question: str
    answer: str
    query_type: str
    cached: bool          # Was this answer from cache or freshly generated?
    response_time_ms: int # How long did it take?


class HealthResponse(BaseModel):
    status: str
    redis: str
    agent: str


# ============================================================
# CACHE HELPER FUNCTIONS
# ============================================================

def get_cache_key(question: str) -> str:
    """
    Creates a unique cache key for each question.
    We use MD5 hash of the question so the key is always
    the same length regardless of question length.
    """
    return f"devops_assistant:{hashlib.md5(question.lower().strip().encode()).hexdigest()}"


def get_cached_answer(question: str):
    """Try to get a cached answer from Redis"""
    if not redis_client:
        return None
    try:
        key = get_cache_key(question)
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


def cache_answer(question: str, answer_data: dict):
    """Save an answer to Redis cache"""
    if not redis_client:
        return
    try:
        key = get_cache_key(question)
        redis_client.setex(
            key,
            CACHE_TTL,
            json.dumps(answer_data)
        )
    except Exception:
        pass


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", response_model=dict)
def root():
    """Root endpoint - confirms the API is running"""
    return {
        "message": "DevOps Knowledge Assistant API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Used by monitoring systems and load balancers to verify
    the service is running correctly.
    """
    redis_status = "connected" if redis_client else "unavailable"

    return HealthResponse(
        status="healthy",
        redis=redis_status,
        agent="ready"
    )


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """
    Main endpoint - accepts a question and returns an answer.

    Flow:
    1. Check Redis cache for existing answer
    2. If cached: return immediately (fast + free)
    3. If not cached: run the LangGraph agent
    4. Save new answer to cache
    5. Return answer with metadata
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    start_time = time.time()

    # Step 1: Check cache
    cached = get_cached_answer(request.question)
    if cached:
        response_time = int((time.time() - start_time) * 1000)
        print(f"⚡ Cache hit for: '{request.question[:50]}...'")
        return QuestionResponse(
            question=request.question,
            answer=cached["answer"],
            query_type=cached["query_type"],
            cached=True,
            response_time_ms=response_time
        )

    # Step 2: Run the agent
    print(f"🤖 Running agent for: '{request.question[:50]}...'")
    try:
        result = agent.invoke({
            "question": request.question,
            "query_type": "",
            "doc_results": [],
            "ticket_results": [],
            "answer": ""
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )

    response_time = int((time.time() - start_time) * 1000)

    # Step 3: Cache the answer
    answer_data = {
        "answer": result["answer"],
        "query_type": result["query_type"]
    }
    cache_answer(request.question, answer_data)

    print(f"✅ Answered in {response_time}ms | Type: {result['query_type']}")

    return QuestionResponse(
        question=request.question,
        answer=result["answer"],
        query_type=result["query_type"],
        cached=False,
        response_time_ms=response_time
    )


@app.get("/tickets/open", response_model=dict)
def get_open_tickets():
    """
    Convenience endpoint to get all open tickets directly.
    No AI needed for this - pure database query.
    """
    import sqlite3
    conn = sqlite3.connect("data/tickets.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticket_id, title, category, priority, status, assignee, created_at
        FROM tickets
        WHERE status = 'open'
        ORDER BY priority ASC, created_at ASC
    """)
    rows = cursor.fetchall()
    conn.close()

    columns = ["ticket_id", "title", "category",
               "priority", "status", "assignee", "created_at"]
    tickets = [dict(zip(columns, row)) for row in rows]

    return {
        "total": len(tickets),
        "tickets": tickets
    }


@app.delete("/cache", response_model=dict)
def clear_cache():
    """Clears all cached answers - useful during development"""
    if not redis_client:
        return {"message": "Redis not available"}
    try:
        keys = redis_client.keys("devops_assistant:*")
        if keys:
            redis_client.delete(*keys)
        return {"message": f"Cleared {len(keys)} cached entries"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))