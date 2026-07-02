"""
chunk_and_embed.py

What this file does:
Takes our raw documents and tickets and prepares them for AI search in 2 steps:

STEP 1 - CHUNKING:
Splits large documents into small overlapping pieces called "chunks"
Why? Because AI models can only process small amounts of text at once,
and we want to find the most relevant paragraph, not the whole document.

STEP 2 - EMBEDDING:
Converts each chunk into a list of numbers called a "vector"
Why? Because computers can't compare meaning directly, but they CAN
compare numbers. Similar meaning = similar numbers = found in search.

We use a FREE Hugging Face model for embeddings (no API cost).
Model: all-MiniLM-L6-v2
- Fast, lightweight, runs on your laptop
- Produces 384 numbers per chunk
- Good enough for production use cases
"""

import os
import json
import csv
from sentence_transformers import SentenceTransformer
import numpy as np

# ============================================================
# CONFIGURATION
# These numbers are tunable - worth understanding what they mean
# ============================================================

CHUNK_SIZE = 500        # How many words per chunk
CHUNK_OVERLAP = 50      # How many words overlap between consecutive chunks
                        # Overlap prevents important sentences getting cut in half

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free, fast, good quality

# Output paths
CHUNKS_OUTPUT = "data/processed/doc_chunks.json"
TICKET_EMBEDDINGS_OUTPUT = "data/processed/ticket_embeddings.json"


# ============================================================
# STEP 1A: LOAD DOCUMENTS
# ============================================================

def load_documents():
    """
    Reads all markdown files from our docs folder.
    Returns a list of documents, each with content and metadata.
    Metadata = information ABOUT the document (source, filename, category)
    """
    documents = []
    docs_base = "data/raw/docs"

    for category in ["fastapi", "kubernetes"]:
        category_path = os.path.join(docs_base, category)

        if not os.path.exists(category_path):
            print(f"   ⚠️  Folder not found: {category_path}")
            continue

        for filename in os.listdir(category_path):
            if not filename.endswith(".md"):
                continue

            filepath = os.path.join(category_path, filename)

            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()

            # Skip empty files or very short files (not useful)
            if len(content) < 100:
                continue

            documents.append({
                "content": content,
                "source": category,           # "fastapi" or "kubernetes"
                "filename": filename,          # e.g. "tutorial.md"
                "filepath": filepath
            })

    print(f"   ✅ Loaded {len(documents)} documents")
    return documents


# ============================================================
# STEP 1B: CHUNK DOCUMENTS
# ============================================================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits a long text into overlapping chunks.

    Example with chunk_size=5, overlap=2:
    Text: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Chunk 1: [1, 2, 3, 4, 5]
    Chunk 2: [4, 5, 6, 7, 8]   ← starts 2 words back (overlap)
    Chunk 3: [7, 8, 9, 10]
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move forward by chunk_size minus overlap
        # This creates the overlapping effect
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents):
    """
    Chunks all documents and adds metadata to each chunk.
    This metadata is crucial - it tells us WHERE each chunk came from
    so we can cite the source in our AI's answers.
    """
    all_chunks = []
    chunk_id = 0

    for doc in documents:
        chunks = chunk_text(doc["content"])

        for i, chunk_text_content in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "content": chunk_text_content,
                "source": doc["source"],
                "filename": doc["filename"],
                "chunk_index": i,           # Which chunk this is within the doc
                "total_chunks": len(chunks) # How many chunks this doc was split into
            })
            chunk_id += 1

    print(f"   ✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
    print(f"   📊 Average chunks per document: {len(all_chunks) // len(documents)}")
    return all_chunks


# ============================================================
# STEP 2A: EMBED DOCUMENT CHUNKS
# ============================================================

def embed_chunks(chunks, model):
    """
    Converts each chunk into a vector (list of numbers).
    The model reads the text and produces 384 numbers that
    capture the MEANING of that text.
    """
    print(f"   🔄 Embedding {len(chunks)} chunks...")
    print(f"   ⏱️  This may take 5-10 minutes on first run (model downloads once)")

    # Extract just the text content for embedding
    texts = [chunk["content"] for chunk in chunks]

    # This is where the magic happens - convert text to vectors
    # batch_size=32 means process 32 chunks at a time (memory efficient)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,    # Shows a progress bar
        convert_to_numpy=True
    )

    # Add the embedding to each chunk
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()  # Convert numpy to list for JSON

    print(f"   ✅ Embedded {len(chunks)} chunks")
    print(f"   📊 Each embedding has {len(chunks[0]['embedding'])} dimensions")
    return chunks


# ============================================================
# STEP 2B: EMBED TICKETS
# ============================================================

def load_and_embed_tickets(model):
    """
    Loads our support tickets and embeds their descriptions.
    This allows semantic search over tickets too -
    "find tickets about network issues" will find tickets even if
    they don't use the exact word "network"
    """
    tickets = []
    tickets_path = "data/raw/tickets.csv"

    with open(tickets_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(row)

    print(f"   ✅ Loaded {len(tickets)} tickets")

    # Embed the description field of each ticket
    descriptions = [t["description"] for t in tickets]
    embeddings = model.encode(
        descriptions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    for i, ticket in enumerate(tickets):
        ticket["embedding"] = embeddings[i].tolist()

    print(f"   ✅ Embedded {len(tickets)} tickets")
    return tickets


# ============================================================
# SAVE TO DISK
# ============================================================

def save_to_json(data, output_path):
    """Save our processed data to JSON files"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Show file size so we know it saved properly
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   💾 Saved to {output_path} ({size_mb:.1f} MB)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Starting chunking and embedding pipeline...\n")

    # Load the embedding model
    # First run: downloads ~90MB model file (one time only)
    # After that: loads instantly from cache
    print("📦 Loading embedding model...")
    print("   (First run downloads ~90MB - this is a one-time download)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"   ✅ Model loaded: {EMBEDDING_MODEL}\n")

    # STEP 1: Process documents
    print("📄 Step 1: Loading and chunking documents...")
    documents = load_documents()
    chunks = chunk_documents(documents)
    print()

    # STEP 2: Embed document chunks
    print("🔢 Step 2: Embedding document chunks...")
    embedded_chunks = embed_chunks(chunks, model)
    save_to_json(embedded_chunks, CHUNKS_OUTPUT)
    print()

    # STEP 3: Process tickets
    print("🎫 Step 3: Loading and embedding tickets...")
    embedded_tickets = load_and_embed_tickets(model)
    save_to_json(embedded_tickets, TICKET_EMBEDDINGS_OUTPUT)
    print()

    # Final summary
    print("=" * 50)
    print("✅ CHUNKING AND EMBEDDING COMPLETE!")
    print("=" * 50)
    print(f"📄 Document chunks: {len(embedded_chunks)}")
    print(f"🎫 Tickets:         {len(embedded_tickets)}")
    print(f"🔢 Vector size:     {len(embedded_chunks[0]['embedding'])} dimensions")
    print()
    print("📁 Output files:")
    print(f"   {CHUNKS_OUTPUT}")
    print(f"   {TICKET_EMBEDDINGS_OUTPUT}")
    print()
    print("🚀 Next step: Load these into ChromaDB and PostgreSQL!")


if __name__ == "__main__":
    main()