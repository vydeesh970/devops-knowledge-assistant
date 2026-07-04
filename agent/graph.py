import json
import sqlite3
import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "devops-knowledge-assistant")

CHROMA_DB_PATH = "data/chroma_db"
SQLITE_DB_PATH = "data/tickets.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ANTHROPIC_MODEL = "claude-sonnet-4-6"


class AgentState(TypedDict):
    question: str
    query_type: str
    doc_results: list
    ticket_results: list
    answer: str


print("🔧 Initializing agent components...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)
doc_collection = chroma_client.get_collection("devops_docs")

llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024
)

print("✅ Agent components ready\n")


def classify_query(state: AgentState) -> AgentState:
    print(f"🔍 Classifying question: '{state['question']}'")

    response = llm.invoke([
        SystemMessage(content="""You are a query classifier for a DevOps knowledge system.

Classify the user's question into exactly one of these categories:
- "docs" - asking how something works, how to fix something, or for technical explanation
- "tickets" - asking about support tickets, their status, priority, or assignee
- "both" - needs both technical documentation AND ticket information

Respond with ONLY one word: docs, tickets, or both.
No explanation, no punctuation, just the single word."""),
        HumanMessage(content=state["question"])
    ])

    query_type = response.content.strip().lower()

    if query_type not in ["docs", "tickets", "both"]:
        query_type = "both"

    print(f"   ✅ Classified as: '{query_type}'")
    return {**state, "query_type": query_type}


def search_docs(state: AgentState) -> AgentState:
    if state["query_type"] == "tickets":
        print("⏭️  Skipping doc search (tickets-only question)")
        return {**state, "doc_results": []}

    print("📚 Searching documentation...")

    query_embedding = embedding_model.encode(state["question"]).tolist()

    results = doc_collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )

    doc_results = []
    for i in range(len(results["documents"][0])):
        doc_results.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "filename": results["metadatas"][0][i]["filename"],
            "relevance_score": 1 - results["distances"][0][i]
        })

    print(f"   ✅ Found {len(doc_results)} relevant documentation chunks")
    for r in doc_results:
        print(f"   📄 {r['source']}/{r['filename']} (relevance: {r['relevance_score']:.2f})")

    return {**state, "doc_results": doc_results}


def search_tickets(state: AgentState) -> AgentState:
    if state["query_type"] == "docs":
        print("⏭️  Skipping ticket search (docs-only question)")
        return {**state, "ticket_results": []}

    print("🎫 Searching tickets...")

    question_lower = state["question"].lower()
    filters = []

    if "open" in question_lower:
        filters.append("status = 'open'")
    elif "in progress" in question_lower or "in_progress" in question_lower:
        filters.append("status = 'in_progress'")
    elif "resolved" in question_lower:
        filters.append("status = 'resolved'")
    elif "closed" in question_lower:
        filters.append("status = 'closed'")

    for priority in ["p1", "p2", "p3", "p4"]:
        if priority in question_lower:
            filters.append(f"priority = '{priority.upper()}'")

    if "kubernetes" in question_lower or "k8s" in question_lower:
        filters.append("category = 'kubernetes'")
    elif "fastapi" in question_lower:
        filters.append("category = 'fastapi'")
    elif "infrastructure" in question_lower or "terraform" in question_lower:
        filters.append("category = 'infrastructure'")

    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)
        print(f"   🔎 Applying filters: {' AND '.join(filters)}")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT ticket_id, title, category, priority, status, assignee,
               created_at, resolved_at, description
        FROM tickets
        {where_clause}
        LIMIT 10
    """)
    rows = cursor.fetchall()
    conn.close()

    columns = ["ticket_id", "title", "category", "priority",
               "status", "assignee", "created_at", "resolved_at", "description"]
    ticket_results = [dict(zip(columns, row)) for row in rows]

    print(f"   ✅ Found {len(ticket_results)} matching tickets")
    for t in ticket_results[:3]:
        print(f"   🎫 {t['ticket_id']} | {t['priority']} | {t['status']} | {t['title'][:50]}")

    return {**state, "ticket_results": ticket_results}


def generate_answer(state: AgentState) -> AgentState:
    print("✍️  Generating answer with Claude...")

    doc_context = ""
    if state.get("doc_results"):
        doc_context = "\n\n=== RELEVANT DOCUMENTATION ===\n"
        for i, doc in enumerate(state["doc_results"], 1):
            doc_context += f"\n[Doc {i}] Source: {doc['source']} | File: {doc['filename']}\n"
            doc_context += f"{doc['content']}\n"
            doc_context += "-" * 40

    ticket_context = ""
    if state.get("ticket_results"):
        ticket_context = "\n\n=== RELEVANT TICKETS ===\n"
        for ticket in state["ticket_results"]:
            ticket_context += f"\nTicket: {ticket['ticket_id']}\n"
            ticket_context += f"Title: {ticket['title']}\n"
            ticket_context += f"Priority: {ticket['priority']} | Status: {ticket['status']}\n"
            ticket_context += f"Category: {ticket['category']}\n"
            ticket_context += f"Assignee: {ticket['assignee']}\n"
            ticket_context += f"Created: {ticket['created_at']}\n"
            ticket_context += f"Description: {ticket['description']}\n"
            ticket_context += "-" * 40

    full_context = doc_context + ticket_context

    response = llm.invoke([
        SystemMessage(content="""You are a helpful DevOps Knowledge Assistant.

Answer questions about DevOps, Kubernetes, FastAPI, and infrastructure using
the provided context from our documentation and support ticket system.

Guidelines:
- Answer clearly and concisely
- Cite your sources (mention which doc or ticket you got info from)
- If you found relevant tickets, mention their ID and status
- If you found relevant docs, mention which documentation you used
- If the context doesn't contain the answer, say so honestly
- Use markdown formatting for code snippets and lists"""),
        HumanMessage(content=f"""Question: {state['question']}

Context from our knowledge base:
{full_context}

Please answer the question based on this context.""")
    ])

    print("   ✅ Answer generated")
    return {**state, "answer": response.content}


def route_after_classify(state: AgentState) -> Literal["search_docs", "search_tickets", "search_both"]:
    if state["query_type"] == "docs":
        return "search_docs"
    elif state["query_type"] == "tickets":
        return "search_tickets"
    else:
        return "search_both"


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_query)
    graph.add_node("search_docs", search_docs)
    graph.add_node("search_tickets", search_tickets)
    graph.add_node("generate", generate_answer)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "search_docs": "search_docs",
            "search_tickets": "search_tickets",
            "search_both": "search_docs",
        }
    )

    graph.add_conditional_edges(
        "search_docs",
        lambda state: "search_tickets" if state["query_type"] == "both" else "generate",
        {
            "search_tickets": "search_tickets",
            "generate": "generate"
        }
    )

    graph.add_edge("search_tickets", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


if __name__ == "__main__":
    agent = build_agent()

    test_questions = [
        "How do I fix a pod stuck in CrashLoopBackOff?",
        "Show me all open P1 tickets",
        "What are the open Kubernetes tickets and how do I debug them?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ QUESTION: {question}")
        print("=" * 60)

        result = agent.invoke({
            "question": question,
            "query_type": "",
            "doc_results": [],
            "ticket_results": [],
            "answer": ""
        })

        print(f"\n💬 ANSWER:\n{result['answer']}")