import sys
import os
import hashlib
import json
import time
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "devops-knowledge-assistant")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis

from agent.graph import build_agent

app = FastAPI(
    title="DevOps Knowledge Assistant",
    description="""
    AI-powered assistant for DevOps teams.

    Ask questions about:
    - Kubernetes concepts and troubleshooting
    - FastAPI development
    - Support ticket status and management

    Built with LangGraph, Claude, ChromaDB, and Redis.
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_redis_client():
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True,
            socket_connect_timeout=2
        )
        client.ping()
        return client
    except Exception:
        print("⚠️  Redis not available - running without cache")
        return None


redis_client = get_redis_client()
agent = build_agent()

CACHE_TTL = 3600


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
    cached: bool
    response_time_ms: int


class HealthResponse(BaseModel):
    status: str
    redis: str
    agent: str


def get_cache_key(question: str) -> str:
    return f"devops_assistant:{hashlib.md5(question.lower().strip().encode()).hexdigest()}"


def get_cached_answer(question: str):
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
    if not redis_client:
        return
    try:
        key = get_cache_key(question)
        redis_client.setex(key, CACHE_TTL, json.dumps(answer_data))
    except Exception:
        pass


@app.get("/")
def root():
    return {
        "message": "DevOps Knowledge Assistant API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    redis_status = "connected" if redis_client else "unavailable"
    return HealthResponse(
        status="healthy",
        redis=redis_status,
        agent="ready"
    )


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    cached = get_cached_answer(request.question)
    if cached:
        response_time = int((time.time() - start_time) * 1000)
        print(f"⚡ Cache hit for: '{request.question[:50]}'")
        return QuestionResponse(
            question=request.question,
            answer=cached["answer"],
            query_type=cached["query_type"],
            cached=True,
            response_time_ms=response_time
        )

    print(f"🤖 Running agent for: '{request.question[:50]}'")
    try:
        result = agent.invoke({
            "question": request.question,
            "query_type": "",
            "doc_results": [],
            "ticket_results": [],
            "answer": ""
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    response_time = int((time.time() - start_time) * 1000)

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


@app.get("/tickets/open")
def get_open_tickets():
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
    return {"total": len(tickets), "tickets": tickets}


@app.delete("/cache")
def clear_cache():
    if not redis_client:
        return {"message": "Redis not available"}
    try:
        keys = redis_client.keys("devops_assistant:*")
        if keys:
            redis_client.delete(*keys)
        return {"message": f"Cleared {len(keys)} cached entries"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))