"""
graph.py

What this file does:
Builds the LangGraph agent that powers our DevOps Knowledge Assistant.

This is the brain of the entire project. It:
1. Receives a user question
2. Classifies what type of question it is
3. Routes to the right data source(s)
4. Generates a proper cited answer using Claude

WHY LANGGRAPH?
LangGraph lets us build AI workflows as a graph of connected steps.
Each step (called a "node") does one thing. Edges between nodes
define what happens next. This makes complex agent logic:
- Easy to understand (you can see the flow visually)
- Easy to debug (you can trace exactly which nodes ran)
- Easy to extend (add new nodes without breaking existing ones)

This is production-grade agent architecture - not a simple chain.
"""

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

# Load environment variables from .env file
# This is how we safely access our API key
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DB_PATH = "data/chroma_db"
SQLITE_DB_PATH = "data/tickets.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ANTHROPIC_MODEL = "claude-sonnet-4-6"


# ============================================================
# STATE
# This is the most important concept in LangGraph.
# "State" is a dictionary that gets passed between every node.
# Each node can READ from it and ADD to it.
# Think of it as a shared notepad that travels through the graph.
# ============================================================

class AgentState(TypedDict):
    # Input
    question: str                    # The user's original question

    # Set by classifier node
    query_type: str                  # "docs", "tickets", or "both"

    # Set by search nodes
    doc_results: list                # Relevant document chunks found
    ticket_results: list             # Relevant tickets found

    # Set by generator node
    answer: str                      # Final answer to return to user


# ============================================================
# INITIALIZE CLIENTS
# We load these once at startup, not on every request
# ============================================================

print("🔧 Initializing agent components...")

# Embedding model - converts questions to vectors for search
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# ChromaDB - our document vector database
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)
doc_collection = chroma_client.get_collection("devops_docs")

# Claude - our LLM for classification and answer generation
llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024
)

print("✅ Agent components ready\n")


# ============================================================
# NODE 1: CLASSIFIER
# Reads the question and decides which database(s) to search
# ============================================================

def classify_query(state: AgentState) -> AgentState:
    """
    Uses Claude to classify the user's question into one of:
    - "docs"    → question about how something works (search documentation)
    - "tickets" → question about support tickets (search ticket database)
    - "both"    → needs information from both sources

    Why use Claude for classification instead of keywords?
    Because "Redis connection issue" could be asking about:
    - How Redis works (docs question)
    - Status of a Redis ticket (ticket question)
    Claude understands context, keywords don't.
    """
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

    # Safety check - if Claude returns something unexpected, default to both
    if query_type not in ["docs", "tickets", "both"]:
        query_type = "both"

    print(f"   ✅ Classified as: '{query_type}'")

    return {**state, "query_type": query_type}


# ============================================================
# NODE 2: DOCUMENT SEARCH
# Searches ChromaDB for relevant documentation chunks
# ============================================================

def search_docs(state: AgentState) -> AgentState:
    """
    Searches our ChromaDB vector database for relevant documentation.

    Only runs if query_type is "docs" or "both".
    Returns the top 4 most semantically similar chunks.
    """

    # Skip if this question doesn't need docs
    if state["query_type"] == "tickets":
        print("⏭️  Skipping doc search (tickets-only question)")
        return {**state, "doc_results": []}

    print("📚 Searching documentation...")

    # Convert the question to a vector
    query_embedding = embedding_model.encode(state["question"]).tolist()

    # Search ChromaDB for similar chunks
    results = doc_collection.query(
        query_embeddings=[query_embedding],
        n_results=4  # return top 4 most relevant chunks
    )

    # Format results for easy use in the generator
    doc_results = []
    for i in range(len(results["documents"][0])):
        doc_results.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "filename": results["metadatas"][0][i]["filename"],
            "relevance_score": 1 - results["distances"][0][i]
            # ChromaDB returns distances, we convert to similarity scores
            # distance 0 = identical, distance 1 = completely different
            # so similarity = 1 - distance
        })

    print(f"   ✅ Found {len(doc_results)} relevant documentation chunks")
    for r in doc_results:
        print(f"   📄 {r['source']}/{r['filename']} "
              f"(relevance: {r['relevance_score']:.2f})")

    return {**state, "doc_results": doc_results}


# ============================================================
# NODE 3: TICKET SEARCH
# Searches SQLite for relevant support tickets
# ============================================================

def search_tickets(state: AgentState) -> AgentState:
    """
    Searches our SQLite database for relevant tickets.

    Only runs if query_type is "tickets" or "both".

    This does TWO types of search:
    1. Semantic: finds tickets by meaning using embeddings
    2. Structural: filters by status/priority if mentioned in question
    """

    # Skip if this question doesn't need tickets
    if state["query_type"] == "docs":
        print("⏭️  Skipping ticket search (docs-only question)")
        return {**state, "ticket_results": []}

    print("🎫 Searching tickets...")

    question_lower = state["question"].lower()

    # Build smart SQL filter based on what the user mentioned
    # This is the "structured" part of hybrid retrieval
    filters = []

    # Check if user mentioned a specific status
    if "open" in question_lower:
        filters.append("status = 'open'")
    elif "in progress" in question_lower or "in_progress" in question_lower:
        filters.append("status = 'in_progress'")
    elif "resolved" in question_lower:
        filters.append("status = 'resolved'")
    elif "closed" in question_lower:
        filters.append("status = 'closed'")

    # Check if user mentioned a specific priority
    for priority in ["p1", "p2", "p3", "p4"]:
        if priority in question_lower:
            filters.append(f"priority = '{priority.upper()}'")

    # Check if user mentioned a specific category
    if "kubernetes" in question_lower or "k8s" in question_lower:
        filters.append("category = 'kubernetes'")
    elif "fastapi" in question_lower:
        filters.append("category = 'fastapi'")
    elif "infrastructure" in question_lower or "terraform" in question_lower:
        filters.append("category = 'infrastructure'")

    # Build the WHERE clause
    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)
        print(f"   🔎 Applying filters: {' AND '.join(filters)}")

    # Query SQLite
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    query = f"""
        SELECT ticket_id, title, category, priority, status, assignee,
               created_at, resolved_at, description
        FROM tickets
        {where_clause}
        LIMIT 10
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    # Format results
    columns = ["ticket_id", "title", "category", "priority",
               "status", "assignee", "created_at", "resolved_at", "description"]

    ticket_results = [dict(zip(columns, row)) for row in rows]

    print(f"   ✅ Found {len(ticket_results)} matching tickets")
    for t in ticket_results[:3]:  # show first 3 in logs
        print(f"   🎫 {t['ticket_id']} | {t['priority']} | "
              f"{t['status']} | {t['title'][:50]}")

    return {**state, "ticket_results": ticket_results}


# ============================================================
# NODE 4: ANSWER GENERATOR
# Takes all search results and generates a final answer
# ============================================================

def generate_answer(state: AgentState) -> AgentState:
    """
    Uses Claude to generate a final answer based on:
    - The original question
    - Relevant documentation chunks found
    - Relevant tickets found

    This is where the magic comes together. Claude reads all
    the context we gathered and writes a helpful, cited answer.
    """
    print("✍️  Generating answer with Claude...")

    # Build context from doc results
    doc_context = ""
    if state.get("doc_results"):
        doc_context = "\n\n=== RELEVANT DOCUMENTATION ===\n"
        for i, doc in enumerate(state["doc_results"], 1):
            doc_context += f"\n[Doc {i}] Source: {doc['source']} | "
            doc_context += f"File: {doc['filename']}\n"
            doc_context += f"{doc['content']}\n"
            doc_context += "-" * 40

    # Build context from ticket results
    ticket_context = ""
    if state.get("ticket_results"):
        ticket_context = "\n\n=== RELEVANT TICKETS ===\n"
        for ticket in state["ticket_results"]:
            ticket_context += f"\nTicket: {ticket['ticket_id']}\n"
            ticket_context += f"Title: {ticket['title']}\n"
            ticket_context += f"Priority: {ticket['priority']} | "
            ticket_context += f"Status: {ticket['status']}\n"
            ticket_context += f"Category: {ticket['category']}\n"
            ticket_context += f"Assignee: {ticket['assignee']}\n"
            ticket_context += f"Created: {ticket['created_at']}\n"
            ticket_context += f"Description: {ticket['description']}\n"
            ticket_context += "-" * 40

    # Combine all context
    full_context = doc_context + ticket_context

    # Ask Claude to generate the answer
    response = llm.invoke([
        SystemMessage(content="""You are a helpful DevOps Knowledge Assistant.

Your job is to answer questions about DevOps, Kubernetes, FastAPI, and 
infrastructure using the provided context from our documentation and 
support ticket system.

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

    answer = response.content
    print("   ✅ Answer generated")

    return {**state, "answer": answer}


# ============================================================
# ROUTING LOGIC
# This tells LangGraph which node to go to next
# ============================================================

def route_after_classify(state: AgentState) -> Literal["search_docs", "search_tickets", "search_both"]:
    """
    After classification, decides which search path to take.
    This is called an "edge" in LangGraph - it connects nodes.
    """
    if state["query_type"] == "docs":
        return "search_docs"
    elif state["query_type"] == "tickets":
        return "search_tickets"
    else:
        return "search_both"


# ============================================================
# BUILD THE GRAPH
# This is where we wire all the nodes together
# ============================================================

def build_agent():
    """
    Assembles all the nodes into a LangGraph workflow.

    The graph looks like this:
    
    START → classify → (route) → search_docs ──────────→ generate → END
                              ↘ search_tickets ─────────↗
                              ↘ search_docs + search_tickets → generate → END
    """

    # Create the graph with our state definition
    graph = StateGraph(AgentState)

    # Add nodes (each node is a function that transforms state)
    graph.add_node("classify", classify_query)
    graph.add_node("search_docs", search_docs)
    graph.add_node("search_tickets", search_tickets)
    graph.add_node("generate", generate_answer)

    # Set the entry point - where the graph starts
    graph.set_entry_point("classify")

    # Add conditional routing after classification
    graph.add_conditional_edges(
        "classify",           # from this node
        route_after_classify, # use this function to decide
        {
            "search_docs": "search_docs",
            "search_tickets": "search_tickets",
            "search_both": "search_docs",  # for "both", start with docs
        }
    )

    # After doc search, either go to ticket search or generate
    graph.add_conditional_edges(
        "search_docs",
        lambda state: "search_tickets" if state["query_type"] == "both" else "generate",
        {
            "search_tickets": "search_tickets",
            "generate": "generate"
        }
    )

    # After ticket search, always generate the answer
    graph.add_edge("search_tickets", "generate")

    # After generating, we're done
    graph.add_edge("generate", END)

    # Compile the graph into a runnable agent
    return graph.compile()


# ============================================================
# MAIN - Test the agent
# ============================================================

def main():
    print("🚀 Building DevOps Knowledge Assistant agent...\n")

    agent = build_agent()

    print("=" * 60)
    print("🤖 AGENT READY - Running test questions")
    print("=" * 60)

    # Test questions that exercise different paths through the graph
    test_questions = [
        "How do I fix a pod stuck in CrashLoopBackOff?",
        "Show me all open P1 tickets",
        "What are the open Kubernetes tickets and how do I debug them?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ QUESTION: {question}")
        print("=" * 60)

        # Run the agent
        result = agent.invoke({
            "question": question,
            "query_type": "",
            "doc_results": [],
            "ticket_results": [],
            "answer": ""
        })

        print(f"\n💬 ANSWER:\n{result['answer']}")
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()