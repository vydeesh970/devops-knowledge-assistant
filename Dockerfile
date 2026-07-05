FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages in small groups so each step is cached
# If one group fails, Docker retries from that group only
RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir fastapi uvicorn python-dotenv redis

RUN pip install --no-cache-dir anthropic langchain langchain-anthropic

RUN pip install --no-cache-dir langgraph langsmith

RUN pip install --no-cache-dir chromadb

RUN pip install --no-cache-dir sentence-transformers

RUN pip install --no-cache-dir pandas numpy

COPY . .

RUN mkdir -p data/raw/docs data/processed data/chroma_db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]