"""
load_to_db.py

What this file does:
Takes our embedded chunks and tickets from JSON files and loads them
into proper databases that can be searched instantly.

DATABASE 1 - ChromaDB (for documents):
- Stores all 1,630 document chunks with their embeddings
- Allows semantic search: "find chunks about pod crashes"
- Runs locally on your computer (free, no API needed)
- This is our local replacement for Pinecone

DATABASE 2 - SQLite (for tickets):
- Stores all 300 tickets in a proper structured database
- Allows both semantic search AND filtering
- Example: "find open P1 tickets about Kubernetes"
  → filters by status=open, priority=P1
  → THEN does semantic search on the results

Why two databases?
Documents and tickets are fundamentally different types of data:
- Documents = unstructured text, best searched by meaning
- Tickets = structured data with fields (status, priority, category)
  that need both filtering AND semantic search
This is what makes our project "hybrid retrieval" - a key
differentiator from basic RAG tutorials.
"""

import json
import sqlite3
import os
import chromadb
from chromadb.config import Settings


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_FILE = "data/processed/doc_chunks.json"
TICKETS_FILE = "data/processed/ticket_embeddings.json"
CHROMA_DB_PATH = "data/chroma_db"
SQLITE_DB_PATH = "data/tickets.db"


# ============================================================
# LOAD DOCUMENT CHUNKS INTO CHROMADB
# ============================================================

def load_docs_to_chromadb():
    """
    Loads all document chunks into ChromaDB.

    ChromaDB organizes data into "collections" - think of them
    like tables in a regular database. We create one collection
    called "devops_docs" that holds all our chunks.

    Each chunk stored in ChromaDB has 3 parts:
    1. id - unique identifier for this chunk
    2. embedding - the 384 numbers representing its meaning
    3. metadata - source, filename, etc. for filtering
    4. document - the actual text content
    """
    print("📚 Loading document chunks into ChromaDB...")

    # Load our embedded chunks from the JSON file
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"   ✅ Loaded {len(chunks)} chunks from JSON")

    # Create ChromaDB client
    # persist_directory tells ChromaDB to save to disk
    # so data survives when we restart our app
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete collection if it already exists (clean slate)
    try:
        client.delete_collection("devops_docs")
        print("   🗑️  Cleared existing collection")
    except Exception:
        pass  # Collection didn't exist yet, that's fine

    # Create fresh collection
    collection = client.create_collection(
        name="devops_docs",
        metadata={"hnsw:space": "cosine"}
        # cosine similarity = best for comparing text embeddings
        # it measures the ANGLE between vectors, not the distance
        # this works better for semantic similarity than euclidean distance
    )

    # ChromaDB likes to receive data in batches
    # We'll add 100 chunks at a time
    batch_size = 100
    total_added = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Prepare the batch data in the format ChromaDB expects
        ids = [chunk["chunk_id"] for chunk in batch]

        embeddings = [chunk["embedding"] for chunk in batch]

        documents = [chunk["content"] for chunk in batch]

        # Metadata = extra info we can filter by later
        metadatas = [{
            "source": chunk["source"],           # "fastapi" or "kubernetes"
            "filename": chunk["filename"],         # which file it came from
            "chunk_index": chunk["chunk_index"],   # position within document
        } for chunk in batch]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        total_added += len(batch)
        print(f"   📥 Added {total_added}/{len(chunks)} chunks...")

    print(f"   ✅ Successfully loaded {total_added} chunks into ChromaDB")
    print(f"   💾 Database saved to {CHROMA_DB_PATH}")
    return collection


# ============================================================
# LOAD TICKETS INTO SQLITE
# ============================================================

def load_tickets_to_sqlite():
    """
    Loads all tickets into SQLite database.

    SQLite is a lightweight database that runs as a single file
    on your computer. Perfect for development.

    We store tickets with ALL their fields so we can:
    - Filter by status: WHERE status = 'open'
    - Filter by priority: WHERE priority = 'P1'
    - Filter by category: WHERE category = 'kubernetes'
    - Search by meaning using the stored embeddings
    - Combine all of the above!
    """
    print("\n🎫 Loading tickets into SQLite...")

    # Load our embedded tickets from JSON
    with open(TICKETS_FILE, "r", encoding="utf-8") as f:
        tickets = json.load(f)

    print(f"   ✅ Loaded {len(tickets)} tickets from JSON")

    # Connect to SQLite (creates the file if it doesn't exist)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Drop table if it exists (clean slate)
    cursor.execute("DROP TABLE IF EXISTS tickets")

    # Create the tickets table
    # Each row = one ticket
    # Each column = one field
    cursor.execute("""
        CREATE TABLE tickets (
            ticket_id TEXT PRIMARY KEY,
            title TEXT,
            category TEXT,
            priority TEXT,
            status TEXT,
            assignee TEXT,
            created_at TEXT,
            resolved_at TEXT,
            description TEXT,
            embedding TEXT
        )
    """)

    # Insert all tickets
    for ticket in tickets:
        cursor.execute("""
            INSERT INTO tickets VALUES (
                :ticket_id, :title, :category, :priority,
                :status, :assignee, :created_at, :resolved_at,
                :description, :embedding
            )
        """, {
            "ticket_id": ticket["ticket_id"],
            "title": ticket["title"],
            "category": ticket["category"],
            "priority": ticket["priority"],
            "status": ticket["status"],
            "assignee": ticket["assignee"],
            "created_at": ticket["created_at"],
            "resolved_at": ticket["resolved_at"],
            "description": ticket["description"],
            "embedding": json.dumps(ticket["embedding"])
            # SQLite doesn't store lists natively
            # so we convert the embedding to a JSON string
        })

    # Save all changes to disk
    conn.commit()

    # Verify it worked by counting rows
    cursor.execute("SELECT COUNT(*) FROM tickets")
    count = cursor.fetchone()[0]

    # Show breakdown by status
    cursor.execute("""
        SELECT status, COUNT(*)
        FROM tickets
        GROUP BY status
        ORDER BY COUNT(*) DESC
    """)
    status_breakdown = cursor.fetchall()

    # Show breakdown by priority
    cursor.execute("""
        SELECT priority, COUNT(*)
        FROM tickets
        GROUP BY priority
        ORDER BY priority
    """)
    priority_breakdown = cursor.fetchall()

    conn.close()

    print(f"   ✅ Loaded {count} tickets into SQLite")
    print(f"   📊 By status: {dict(status_breakdown)}")
    print(f"   📊 By priority: {dict(priority_breakdown)}")
    print(f"   💾 Database saved to {SQLITE_DB_PATH}")


# ============================================================
# VERIFY EVERYTHING WORKS
# ============================================================

def verify_databases():
    """
    Runs a quick test search on both databases to make sure
    everything is working correctly before we move on.
    """
    print("\n🔍 Verifying databases with test searches...")

    # TEST 1: Search ChromaDB for a Kubernetes question
    print("\n   Test 1: Searching docs for 'CrashLoopBackOff'")

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection("devops_docs")

    # We need to embed our test query first
    # We'll use the same model we used for the chunks
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "How do I fix a pod stuck in CrashLoopBackOff?"
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # return top 3 most relevant chunks
    )

    print(f"   ✅ Found {len(results['documents'][0])} relevant chunks")
    for i, (doc, meta) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0]
    )):
        print(f"\n   Result {i+1}:")
        print(f"   Source: {meta['source']} | File: {meta['filename']}")
        print(f"   Preview: {doc[:150]}...")

    # TEST 2: Query SQLite for open P1 tickets
    print("\n   Test 2: Querying tickets for open P1 tickets")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ticket_id, title, priority, status
        FROM tickets
        WHERE status = 'open' AND priority = 'P1'
        LIMIT 5
    """)

    rows = cursor.fetchall()
    conn.close()

    print(f"   ✅ Found {len(rows)} open P1 tickets (showing up to 5):")
    for row in rows:
        print(f"   {row[0]} | {row[2]} | {row[3]} | {row[1]}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Loading data into databases...\n")

    # Make sure our output folder exists
    os.makedirs("data", exist_ok=True)

    # Load documents into ChromaDB
    load_docs_to_chromadb()

    # Load tickets into SQLite
    load_tickets_to_sqlite()

    # Verify both databases work
    verify_databases()

    print("\n" + "=" * 50)
    print("✅ PHASE 2 COMPLETE!")
    print("=" * 50)
    print("📚 ChromaDB: 1,630 document chunks loaded")
    print("🎫 SQLite:   300 tickets loaded")
    print("\n🚀 Next step: Build the LangGraph agent!")


if __name__ == "__main__":
    main()