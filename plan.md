# RivalRadar Implementation Plan
## Research MCP Server + PydanticAI Agent + EC2 Deployment

**Created:** 2024-01-13
**Status:** Ready for Implementation

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Agent Access API](#phase-1-agent-access-api)
4. [Phase 2: Research MCP Server](#phase-2-research-mcp-server)
5. [Phase 3: PydanticAI Main Agent](#phase-3-pydanticai-main-agent)
6. [Phase 4: EC2 Systemd Deployment](#phase-4-ec2-systemd-deployment)
7. [Testing & Verification](#testing--verification)
8. [Quick Reference Commands](#quick-reference-commands)

---

## Overview

This plan creates a complete research pipeline:

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│  PydanticAI     │────▶│  Research MCP       │────▶│  Agent Access   │
│  Main Agent     │     │  Server (port 8001) │     │  API (port 8003)│
│  (terminal UI)  │     │  (FastMCP)          │     │  (FastAPI)      │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
                                                            │
                                                            ▼
                                                    ┌─────────────────┐
                                                    │  Qdrant Cloud   │
                                                    │  (hybrid search)│
                                                    └─────────────────┘
```

**Components:**
1. **Agent Access API** (`src/agent_access/`) - FastAPI server that wraps Qdrant hybrid search
2. **Research MCP Server** (`src/MCP_Servers/research_mcp_server/`) - FastMCP server exposing `research` and `get_date` tools
3. **Main Agent** (`src/Agents/main_agent.py`) - PydanticAI agent connected to MCP server with terminal chat loop

---

## Prerequisites

### Environment Variables
The `.env` file in the project root must contain:

```bash
# LLM Configuration (already exists)
LLM_MODEL=anthropic/claude-3.5-sonnet
OPEN_ROUTER_API_KEY=sk-or-v1-...

# AWS S3 (already exists - for BM25 corpus)
aws_access_key_id=AKIA...
aws_secret_access_key=...
aws_region=us-east-2
bucket=rivalradar-scraped-data

# Qdrant (already exists)
QDRANT_URL=https://...cloud.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION=rivalradar_articles

# NEW: Add these for the MCP server
RESEARCH_API_URL=http://localhost:8003/query
RESEARCH_MCP_URL=http://localhost:8001/mcp
```

### Python Dependencies
Add to `config/requirements.txt`:
```
fastmcp>=0.1.0
httpx>=0.27.0
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic-ai[mcp]>=0.1.0
```

---

## Phase 1: Agent Access API

This phase creates a FastAPI server that exposes the Qdrant hybrid search functionality.

### Files to Create

#### 1.1 Create `src/agent_access/agent_search.py`

**Purpose:** Core search logic with hybrid (dense + sparse) Qdrant queries

```python
"""
Agent Search - Hybrid search logic for Qdrant vector database.

This module provides the core search functionality used by the Agent Access API.
It performs hybrid search using both dense (sentence-transformers) and sparse (BM25) vectors.
"""

import json
import re
import os
import logging
import boto3
import nltk
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pinecone_text.sparse import BM25Encoder
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# Load environment from project root
load_dotenv()

logger = logging.getLogger(__name__)

# Global state - initialized lazily for performance
_initialized = False
_bm25 = None
_client = None
_corpus = None
_tokenizer = None
_model = None


def _validate_environment():
    """Validate all required environment variables are present."""
    required_vars = [
        'bucket', 'aws_access_key_id', 'aws_secret_access_key', 'aws_region',
        'QDRANT_URL', 'QDRANT_API_KEY'
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")


def _download_nltk_data():
    """Download required NLTK data if not already present."""
    logger.info("Checking NLTK data...")
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        logger.info("Downloading required NLTK data...")
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)


def _load_corpus_from_s3():
    """Load the BM25 training corpus from S3."""
    global _corpus
    logger.info("Loading corpus from S3...")

    bucket_name = os.getenv('bucket')
    file_key = 'corpus.json'

    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        region_name=os.getenv("aws_region", "us-east-2")
    )

    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    content = response['Body'].read().decode('utf-8')
    _corpus = json.loads(content)

    # Clean corpus content
    for article in _corpus:
        article['content'] = clean_content(article.get('content', ''))

    logger.info(f"Loaded {len(_corpus)} articles from corpus")


def _initialize_models():
    """Initialize BM25 and dense embedding models."""
    global _bm25, _tokenizer, _model
    logger.info("Initializing embedding models...")

    # BM25 sparse encoder - fit on corpus
    _bm25 = BM25Encoder()
    corpus_content = [doc['content'] for doc in _corpus if 'content' in doc]
    _bm25.fit(corpus_content)

    # Dense embedding model (same as used for indexing)
    _tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    _model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    logger.info("Models initialized successfully")


def _connect_to_qdrant():
    """Establish connection to Qdrant Cloud."""
    global _client
    logger.info("Connecting to Qdrant...")

    _client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )

    logger.info("Qdrant connection established")


def _ensure_initialized():
    """
    Ensure all components are initialized. Thread-safe lazy loading.

    Call this before any search operation. It will only initialize once.
    """
    global _initialized
    if _initialized:
        return

    logger.info("Initializing agent_search components...")
    _validate_environment()
    _download_nltk_data()
    _load_corpus_from_s3()
    _initialize_models()
    _connect_to_qdrant()
    _initialized = True
    logger.info("Agent_search initialization complete")


def clean_content(content_str: str) -> str:
    """
    Clean text content by replacing problematic unicode characters.

    Args:
        content_str: Raw text content

    Returns:
        Cleaned ASCII-safe text
    """
    # Replace common unicode characters with ASCII equivalents
    replacements = [
        (r'\\u2019', "'"),   # Right single quote
        (r'\\u2014', "-"),   # Em dash
        (r'\\u2013', "-"),   # En dash
        (r'\\u2026', "..."), # Ellipsis
        (r'\\u201c', '"'),   # Left double quote
        (r'\\u201d', '"'),   # Right double quote
        (r'\\u2018', "'"),   # Left single quote
        (r'\\u2022', "*"),   # Bullet
        (r'\\xa0', ' '),     # Non-breaking space
        (r'\\u00a9', "(c)"), # Copyright
        (r'\\u00ae', "(R)"), # Registered trademark
        (r'\\u2122', "(TM)"),# Trademark
        (r'\\u2192', "->"),  # Right arrow
        (r'\\n', ' '),       # Newline
    ]

    for pattern, replacement in replacements:
        content_str = re.sub(pattern, replacement, content_str)

    # Replace remaining non-ASCII characters
    content_str = content_str.encode('ascii', errors='replace').decode('ascii')

    # Normalize whitespace
    content_str = re.sub(r'\s+', ' ', content_str).strip()

    return content_str


def mean_pooling(model_output, attention_mask):
    """
    Apply mean pooling to get sentence embeddings.

    Takes attention mask into account for correct averaging.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_user_query(query: str) -> tuple:
    """
    Encode a user query into both sparse and dense embeddings.

    Args:
        query: The user's search query

    Returns:
        Tuple of (sparse_embedding_dict, dense_embedding_list)
    """
    _ensure_initialized()

    # Sparse embedding (BM25)
    sparse_embedding = _bm25.encode_documents([query])[0]

    # Dense embedding (sentence-transformers)
    encoded_input = _tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    with torch.no_grad():
        model_output = _model(**encoded_input)
    dense_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    dense_embedding = F.normalize(dense_embedding, p=2, dim=1).squeeze().tolist()

    return sparse_embedding, dense_embedding


def run_hybrid_query(
    prompt: str,
    start_date: str = None,
    collection_name: str = None
) -> list:
    """
    Perform a hybrid (dense + sparse) search on Qdrant.

    Uses Reciprocal Rank Fusion (RRF) to combine results from both
    BM25 sparse search and sentence-transformer dense search.

    Args:
        prompt: The search query
        start_date: Optional date filter (YYYY-MM-DD format).
                   Only returns articles scraped on or after this date.
        collection_name: Qdrant collection name (defaults to QDRANT_COLLECTION env var)

    Returns:
        List of cleaned article content strings (top 10 results)
    """
    _ensure_initialized()

    # Use environment variable if collection not specified
    if collection_name is None:
        collection_name = os.getenv("QDRANT_COLLECTION", "rivalradar_articles")

    # Encode user query for both search methods
    sparse_query_dict, dense_query_vector = encode_user_query(prompt)

    # Create Qdrant SparseVector for BM25 search
    sparse_vector_for_qdrant = models.SparseVector(
        indices=sparse_query_dict["indices"],
        values=sparse_query_dict["values"]
    )

    # Build date filter if start_date is provided
    filter_condition = None
    if start_date:
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="scraped_date",
                    range=models.DatetimeRange(
                        gte=start_date
                    )
                )
            ]
        )

    # Build prefetch for hybrid search (BM25 + dense)
    prefetch = [
        models.Prefetch(
            query=sparse_vector_for_qdrant,
            using="bm25",
            limit=20
        ),
        models.Prefetch(
            query=dense_query_vector,
            using="sentence-transformers/all-mpnet-base-v2",
            limit=20
        ),
    ]

    # Execute RRF (Reciprocal Rank Fusion) query
    results = _client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=10,
        query_filter=filter_condition
    )

    # Extract and clean content from results
    content_list = []
    logger.info(f"\nTop {len(results.points)} hybrid search results:")

    for idx, point in enumerate(results.points, start=1):
        payload = point.payload
        original_content = payload.get('content', '')
        cleaned_content = clean_content(original_content)
        content_list.append(cleaned_content)

        logger.debug(f"Rank {idx}: {cleaned_content[:100]}...")

    return content_list
```

#### 1.2 Create `src/agent_access/agent_api.py`

**Purpose:** FastAPI server exposing the search functionality via REST API

```python
"""
Agent Access API - REST API for Qdrant hybrid search.

This FastAPI server provides an HTTP interface to the hybrid search functionality.
The Research MCP Server calls this API to perform searches.

Usage:
    python -m src.agent_access.agent_api

    Or via uvicorn:
    uvicorn src.agent_access.agent_api:app --host 0.0.0.0 --port 8003
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

# Add src to path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

# Load environment variables from project root
project_root = os.path.dirname(src_dir)
load_dotenv(os.path.join(project_root, '.env'))

# Import after path setup
from agent_access.agent_search import run_hybrid_query, _ensure_initialized

# ---- Logging Configuration ----
def _env_truthy(name: str, default: bool = False) -> bool:
    """Check if environment variable is set to a truthy value."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")

# Enable logging based on environment variable
LOGGING_ENABLED = _env_truthy("AGENT_API_LOGGING", default=True)
LOG_LEVEL = os.getenv("AGENT_API_LOG_LEVEL", "INFO").strip().upper()

if not LOGGING_ENABLED:
    logging.disable(logging.CRITICAL)
else:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)


# ---- FastAPI Lifespan ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to initialize components on startup.

    This ensures the BM25 encoder, embedding models, and Qdrant client
    are ready before the first request.
    """
    logger.info("Starting Agent Access API - initializing search components...")
    try:
        _ensure_initialized()
        logger.info("Agent Access API initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize Agent Access API: {e}")
        raise
    yield
    logger.info("Agent Access API shutting down...")


# ---- FastAPI App ----
app = FastAPI(
    title="Agent Access API",
    description="REST API for RivalRadar hybrid search",
    version="1.0.0",
    lifespan=lifespan
)


# ---- Request/Response Models ----
class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""
    prompt: str
    start_date: Optional[str] = None
    date_filter: Optional[bool] = False


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""
    results: list[str]


# ---- Endpoints ----
@app.get("/")
def root():
    """Root endpoint - API info."""
    return {
        "message": "Agent Access API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health/startup")
def startup_health():
    """Basic health check - server is running."""
    return {"status": "started", "message": "Server is running"}


@app.get("/health/ready")
def readiness_health():
    """Readiness check - can handle requests."""
    try:
        _ensure_initialized()
        return {"status": "ready", "message": "Service is ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(500, f"Service not ready: {str(e)}")


@app.post("/query", response_model=QueryResponse)
def query_api(request: QueryRequest):
    """
    Execute a hybrid search query.

    Args:
        request: QueryRequest with prompt and optional date filter

    Returns:
        QueryResponse with list of article content strings
    """
    try:
        # Apply date filter if requested
        start_date = None
        if request.date_filter and request.start_date:
            start_date = request.start_date
            logger.info(f"Query with date filter: {start_date}")

        logger.info(f"Processing query: {request.prompt[:100]}...")

        results = run_hybrid_query(
            prompt=request.prompt,
            start_date=start_date
        )

        logger.info(f"Returning {len(results)} results")
        return QueryResponse(results=results)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")


# ---- Main Entry Point ----
if __name__ == "__main__":
    print("Starting Agent Access API on port 8003...")
    uvicorn.run(app, host="0.0.0.0", port=8003)
```

#### 1.3 Create `src/agent_access/__init__.py`

```python
"""Agent Access - REST API for Qdrant hybrid search."""

from .agent_search import run_hybrid_query, _ensure_initialized
from .agent_api import app

__all__ = ['run_hybrid_query', '_ensure_initialized', 'app']
```

#### 1.4 Delete the placeholder file

```bash
rm src/agent_access/placeholder.txt
```

---

## Phase 2: Research MCP Server

This phase creates the FastMCP server that exposes `research` and `get_date` tools.

### Files to Create

#### 2.1 Create `src/MCP_Servers/research_mcp_server/server.py`

**Purpose:** Main MCP server with tool definitions

```python
"""
Research MCP Server - FastMCP server for RivalRadar research tools.

This server provides two tools:
1. research - Query the fintech article database with optional date filtering
2. get_date - Get the current UTC date in YYYY-MM-DD format

Usage:
    python -m src.MCP_Servers.research_mcp_server

    Or after pip install -e .:
    research-mcp-server
"""

import os
import sys
import httpx
import logging
import warnings
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# Load environment variables from project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(os.path.join(project_root, '.env'))

# Import implementation functions
from .research import execute_research
from .date import get_current_date_str

# ---- Logging Configuration ----
def _env_truthy(name: str, default: bool = False) -> bool:
    """Check if environment variable is set to a truthy value."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")

MCP_LOGGING_ENABLED = _env_truthy("MCP_LOGGING", default=False)
MCP_LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO").strip().upper()

if not MCP_LOGGING_ENABLED:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.disable(logging.CRITICAL)
else:
    level = getattr(logging, MCP_LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

logger = logging.getLogger("research-mcp")

# ---- Global HTTP Client ----
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(mcp: FastMCP):
    """
    Manages startup and shutdown of the MCP server.

    Initializes and closes the global httpx.AsyncClient used for
    calling the Agent Access API.
    """
    global http_client
    logger.info("MCP server lifespan: startup...")
    http_client = httpx.AsyncClient(timeout=60.0)
    logger.info("Global httpx.AsyncClient initialized.")

    yield  # Server runs here

    logger.info("MCP server lifespan: shutdown...")
    if http_client and not http_client.is_closed:
        logger.info("Closing global httpx.AsyncClient...")
        await http_client.aclose()
        http_client = None
    logger.info("MCP server shut down complete.")


# ---- Initialize FastMCP Server ----
mcp = FastMCP(
    name="research-mcp",
    instructions="MCP Server for RivalRadar research tools (Research & Date)",
    lifespan=lifespan
)


# ---- Health Check Route ----
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Health check endpoint for monitoring."""
    return PlainTextResponse("OK")


# ---- Tool: research ----
@mcp.tool(
    description="Queries the RivalRadar fintech article database. Returns relevant articles based on the research query. Use date_filter=True with a start_date to only search recent articles."
)
async def research(
    prompt: str,
    date_filter: bool = False,
    start_date: str = "no filter needed"
) -> List[str]:
    """
    Search the RivalRadar fintech article database.

    This tool performs a hybrid search (semantic + keyword) on the article
    database to find relevant content for the given research query.

    Args:
        prompt: The research query (e.g., 'latest payment trends',
                'Stripe competitor analysis', 'cryptocurrency regulations').
        date_filter: Set to True if you want to filter by date.
        start_date: The starting date in YYYY-MM-DD format. Only articles
                   scraped on or after this date will be returned.
                   Use get_date tool to get today's date if needed.

    Returns:
        A list of article content snippets relevant to the query.

    Example:
        research("What are the latest payment innovations?")
        research("fintech funding rounds", date_filter=True, start_date="2024-01-01")
    """
    if not http_client:
        logger.error("HTTP client not initialized. Cannot perform research.")
        return ["Error: HTTP client not available in the server."]

    return await execute_research(http_client, prompt, date_filter, start_date)


# ---- Tool: get_date ----
@mcp.tool(
    description="Returns the current UTC date in YYYY-MM-DD format. Use this to get today's date for the research tool's date filter."
)
async def get_date() -> str:
    """
    Get the current UTC date.

    Returns:
        The current date in YYYY-MM-DD format (e.g., "2024-01-13")

    Example:
        Use this to get today's date, then use it with the research tool:
        date = get_date()  # Returns "2024-01-13"
        research("latest news", date_filter=True, start_date=date)
    """
    return get_current_date_str()


# ---- Server Runner ----
def run_server():
    """
    Run the MCP server with streamable-http transport.

    The server listens on port 8001 at path /mcp.
    """
    try:
        log_level = "debug" if MCP_LOGGING_ENABLED else "critical"
        logger.info(f"Starting MCP server '{mcp.name}' with streamable-http transport...")
        mcp.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=8001,
            path="/mcp",
            log_level=log_level,
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user.")
    except Exception as e:
        logger.error(f"MCP server failed: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Starting Research MCP Server directly...")
    run_server()
```

#### 2.2 Create `src/MCP_Servers/research_mcp_server/research.py`

**Purpose:** Implementation of the research tool logic

```python
"""
Research Tool Implementation - Core logic for the research MCP tool.

This module handles the actual API call to the Agent Access API
to perform hybrid search queries.
"""

import httpx
import datetime
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Agent Access API endpoint
# When running locally, both services run on localhost
# Can be overridden via environment variable
RESEARCH_API_URL = os.getenv('RESEARCH_API_URL', 'http://localhost:8003/query')


async def execute_research(
    client: httpx.AsyncClient,
    prompt: str,
    date_filter: bool = False,
    start_date: str = "no filter needed"
) -> List[str]:
    """
    Execute a research query against the Agent Access API.

    Args:
        client: An active httpx.AsyncClient instance
        prompt: The research query
        date_filter: Set to True if date filtering is needed
        start_date: The starting date in YYYY-MM-DD format if date_filter is True

    Returns:
        A list of article content snippets or an error message
    """
    logger.info(f"Executing research: prompt='{prompt[:100]}...', date_filter={date_filter}, start_date='{start_date}'")

    # Build request payload
    payload = {"prompt": prompt, "date_filter": date_filter}

    # Validate and add date filter if requested
    if date_filter and start_date != "no filter needed":
        try:
            # Validate date format
            datetime.datetime.strptime(start_date, '%Y-%m-%d')
            payload["start_date"] = start_date
            logger.info(f"Applying start_date filter: {start_date}")
        except ValueError:
            logger.warning(f"Invalid start_date format: {start_date}. Ignoring date filter.")
            payload["date_filter"] = False

    try:
        # Call the Agent Access API
        response = await client.post(RESEARCH_API_URL, json=payload)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        logger.info(f"Research API call successful. Found {len(results)} results.")
        return results

    except httpx.RequestError as exc:
        logger.error(f"Connection error to {exc.request.url!r}: {exc}")
        return [f"Error: Could not connect to the research API. Make sure the Agent Access API is running on port 8003."]

    except httpx.HTTPStatusError as exc:
        logger.error(f"HTTP error {exc.response.status_code} from {exc.request.url!r}: {exc.response.text}")
        return [f"Error: Research API returned status {exc.response.status_code}"]

    except Exception as e:
        logger.exception(f"Unexpected error during research: {e}")
        return [f"Error: An unexpected error occurred: {str(e)}"]
```

#### 2.3 Create `src/MCP_Servers/research_mcp_server/date.py`

**Purpose:** Implementation of the get_date tool

```python
"""
Date Tool Implementation - Returns current UTC date.

Simple utility for getting the current date in YYYY-MM-DD format,
which can be used with the research tool's date filter.
"""

import datetime
from datetime import timezone


def get_current_date_str() -> str:
    """
    Return the current UTC date in YYYY-MM-DD format.

    Returns:
        String in format "YYYY-MM-DD" (e.g., "2024-01-13")
    """
    current_date = datetime.datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return current_date
```

#### 2.4 Create `src/MCP_Servers/research_mcp_server/__main__.py`

**Purpose:** Entry point for running as a module

```python
"""
Main entry point for the Research MCP Server.

This allows running the server as a module:
    python -m src.MCP_Servers.research_mcp_server
"""

import logging
from .server import run_server, logger


def main():
    """Main entry point to start the Research MCP server."""
    try:
        logger.info("Starting Research MCP Server...")
        print("--- Research MCP Server --- (Press Ctrl+C to exit)")
        print("Server will be available at: http://localhost:8001/mcp")
        run_server()
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt.")
        print("\nShutting down server...")
    except Exception as e:
        logger.exception("An unexpected error occurred during server execution.")
        print(f"\nError: {e}")
    finally:
        logger.info("Server stopped.")
        print("Server stopped.")


if __name__ == "__main__":
    main()
```

#### 2.5 Create `src/MCP_Servers/research_mcp_server/__init__.py`

```python
"""Research MCP Server - FastMCP server for RivalRadar research tools."""

from .server import mcp, run_server
from .research import execute_research
from .date import get_current_date_str

__all__ = ['mcp', 'run_server', 'execute_research', 'get_current_date_str']
```

#### 2.6 Create `src/MCP_Servers/research_mcp_server/pyproject.toml`

**Purpose:** Package configuration for pip install -e .

```toml
[project]
name = "research_mcp_server"
version = "0.1.0"
description = "MCP Server for RivalRadar research tools (Research & Date)"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "fastmcp>=0.1.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
]

[[project.authors]]
name = "RivalRadar Team"
email = "contact@rivalradar.com"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project.scripts]
research-mcp-server = "research_mcp_server.__main__:main"

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.hatch.build]
include = [
    "*.py",
]
```

#### 2.7 Create `src/MCP_Servers/research_mcp_server/README.md`

```markdown
# Research MCP Server

MCP Server for RivalRadar research tools.

## Tools

- `research`: Queries the RivalRadar fintech article database with hybrid search
- `get_date`: Returns the current UTC date in YYYY-MM-DD format

## Prerequisites

The Agent Access API must be running on port 8003 before starting this server.

## Installation

```bash
cd src/MCP_Servers/research_mcp_server
pip install -e .
```

## Running

**Option 1: After pip install**
```bash
research-mcp-server
```

**Option 2: As a Python module**
```bash
python -m src.MCP_Servers.research_mcp_server
```

**Option 3: Direct script**
```bash
python src/MCP_Servers/research_mcp_server/server.py
```

The server will be available at: `http://localhost:8001/mcp`

## Environment Variables

Required in `.env`:
- `RESEARCH_API_URL` - Agent Access API URL (default: http://localhost:8003/query)

Optional:
- `MCP_LOGGING` - Set to "true" to enable logging
- `MCP_LOG_LEVEL` - Log level (default: INFO)
```

#### 2.8 Delete the placeholder file

```bash
rm src/MCP_Servers/research_mcp_server/placeholder.txt
```

---

## Phase 3: PydanticAI Main Agent

This phase creates the terminal-based agent that connects to the MCP server.

### Files to Create

#### 3.1 Create `src/Agents/main_agent.py`

**Purpose:** PydanticAI agent with MCP server connection and terminal chat loop

```python
#!/usr/bin/env python3
"""
RivalRadar Main Agent - Fintech Intelligence Analyst

A PydanticAI agent connected to the Research MCP Server for analyzing
fintech trends, competitors, and market intelligence.

Usage:
    python -m src.Agents.main_agent

Prerequisites:
    1. Agent Access API running on port 8003
    2. Research MCP Server running on port 8001
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerStreamableHTTP

# Load environment variables from project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RivalRadarAgent")

# ---- Environment Variables ----
api_key = os.getenv('OPEN_ROUTER_API_KEY')
llm_model = os.getenv('LLM_MODEL', 'anthropic/claude-3.5-sonnet')
mcp_url = os.getenv('RESEARCH_MCP_URL', 'http://localhost:8001/mcp')

# Validate required environment variables
if not api_key:
    print("ERROR: OPEN_ROUTER_API_KEY not found in environment variables!")
    print("Please ensure your .env file contains:")
    print("  OPEN_ROUTER_API_KEY=your-api-key-here")
    print("  LLM_MODEL=anthropic/claude-3.5-sonnet")
    sys.exit(1)

# ---- Model Configuration ----
model = OpenAIModel(
    llm_model,
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1',
        api_key=api_key,
    ),
)

# ---- MCP Server Connection ----
research_mcp = MCPServerStreamableHTTP(url=mcp_url)

# ---- System Prompt ----
SYSTEM_PROMPT = """You are RivalRadar - an expert fintech competitive intelligence analyst.

## Your Role
You analyze fintech news, trends, and competitor activities to provide actionable insights.
You have access to a database of scraped fintech articles that you can search.

## Your Tools
1. **research** - Search the article database with a query. Use this to find relevant articles.
   - Always use specific, targeted queries
   - Use date_filter=True with start_date for recent news (use get_date first)

2. **get_date** - Get today's date in YYYY-MM-DD format. Use this before filtering by date.

## How to Research
1. When asked a question, ALWAYS use the research tool first
2. For recent news, get today's date, then use date_filter with a recent start_date
3. Analyze the returned articles to form your response
4. Cite sources when providing insights

## Response Format
- Start with a direct answer to the question
- Support with evidence from the articles
- Identify key trends and patterns
- Note any gaps in available data
- Be specific and actionable

## Topics You Cover
- Payment innovations and trends
- Banking technology and digital transformation
- Cryptocurrency and blockchain developments
- Fintech funding and M&A activity
- Regulatory changes and compliance
- Competitor strategies and product launches
- Market trends and consumer behavior

## Important Guidelines
- ONLY base your analysis on the articles returned by the research tool
- Do NOT make up information or cite non-existent sources
- If no relevant articles are found, say so clearly
- Be concise but thorough
- Focus on actionable intelligence
"""

# ---- Agent Definition ----
agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    mcp_servers=[research_mcp],
    retries=2,
)


# ---- Chat Loop ----
async def chat_loop():
    """
    Interactive terminal chat loop with message history.

    Maintains conversation context across multiple exchanges.
    """
    message_history = []

    print("\n" + "=" * 60)
    print("  RivalRadar - Fintech Intelligence Analyst")
    print("=" * 60)
    print("\nI can help you analyze fintech trends, competitors, and market signals.")
    print("I have access to a database of fintech news articles.")
    print("\nExample questions:")
    print("  - What are the latest payment innovations?")
    print("  - Tell me about recent fintech funding rounds")
    print("  - What is Stripe doing in the payments space?")
    print("\nType 'exit' or 'quit' to leave.")
    print("-" * 60 + "\n")

    # Connect to MCP server
    async with agent.run_mcp_servers():
        print("[Connected to Research MCP Server]\n")

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Check for exit commands
                if user_input.lower() in ('exit', 'quit', 'q'):
                    print("\nGoodbye! Happy analyzing.")
                    break

                # Skip empty input
                if not user_input:
                    continue

                print("\nThinking...\n")

                # Run the agent with message history
                if message_history:
                    result = await agent.run(user_input, message_history=message_history)
                else:
                    result = await agent.run(user_input)

                # Display response
                print("-" * 40)
                print(f"\nRivalRadar: {result.output}\n")
                print("-" * 40 + "\n")

                # Update message history for context
                message_history.extend(result.new_messages())

            except KeyboardInterrupt:
                print("\n\nGoodbye! Happy analyzing.")
                break
            except Exception as e:
                logger.error(f"Error during agent run: {e}")
                print(f"\nError: {e}")
                print("Please try again.\n")


# ---- Main Entry Point ----
def main():
    """Main entry point for the agent."""
    print("\nStarting RivalRadar Agent...")
    print(f"Using model: {llm_model}")
    print(f"MCP Server: {mcp_url}")

    try:
        asyncio.run(chat_loop())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

#### 3.2 Update `src/Agents/__init__.py`

Add the new agent to the module exports:

```python
"""RivalRadar Agents - Query and reasoning agents for fintech intelligence."""

from .qdrant_query import QdrantQueryHelper, query_qdrant
from .brain_agent import BrainAgent
# New main agent
from .main_agent import agent as main_agent

__all__ = ['QdrantQueryHelper', 'query_qdrant', 'BrainAgent', 'main_agent']
```

---

## Phase 4: EC2 Systemd Deployment

This phase provides guidance for running the scraping pipeline on EC2 as a systemd service.

### Important Note
The scraping pipeline MUST run as a systemd service with a timer. Cron jobs and direct shell commands fail due to the colorama library requiring an interactive terminal.

### 4.1 Create the Service File

On EC2, create `/etc/systemd/system/rivalradar-scraper.service`:

```ini
[Unit]
Description=RivalRadar Scraping Pipeline
After=network.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/RivalRadar
Environment="PATH=/home/ubuntu/RivalRadar/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=/home/ubuntu/RivalRadar/src"
ExecStart=/home/ubuntu/RivalRadar/.venv/bin/python -m src.Scraping.main --source news --use-all-sources

# Logging
StandardOutput=append:/home/ubuntu/RivalRadar/logs/scraper.log
StandardError=append:/home/ubuntu/RivalRadar/logs/scraper_error.log

# Timeout for long-running scrapes
TimeoutStartSec=3600

[Install]
WantedBy=multi-user.target
```

### 4.2 Create the Timer File

Create `/etc/systemd/system/rivalradar-scraper.timer`:

```ini
[Unit]
Description=Run RivalRadar Scraper Daily at 10 PM

[Timer]
OnCalendar=*-*-* 22:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

### 4.3 Setup Commands

Run these commands on the EC2 instance:

```bash
# Create logs directory
mkdir -p /home/ubuntu/RivalRadar/logs

# Copy service and timer files
sudo cp rivalradar-scraper.service /etc/systemd/system/
sudo cp rivalradar-scraper.timer /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the timer
sudo systemctl enable rivalradar-scraper.timer
sudo systemctl start rivalradar-scraper.timer

# Verify timer is active
sudo systemctl status rivalradar-scraper.timer
sudo systemctl list-timers | grep rivalradar
```

### 4.4 Manual Test Run

```bash
# Run the service manually to test
sudo systemctl start rivalradar-scraper.service

# Check logs
tail -f /home/ubuntu/RivalRadar/logs/scraper.log
```

### 4.5 Monitoring Commands

```bash
# Check timer status
sudo systemctl status rivalradar-scraper.timer

# Check last run status
sudo systemctl status rivalradar-scraper.service

# View logs
journalctl -u rivalradar-scraper.service -f

# List upcoming timer runs
systemctl list-timers --all
```

### 4.6 EC2 Instance Recommendations

**Instance Type:** `t3.xlarge` (or larger)
- 4 vCPUs, 16 GB RAM
- Sufficient for embedding generation and scraping

**Monitoring:**
- Set up CloudWatch alarms for CPU/memory
- Monitor the scraper logs for errors
- Scale up instance if processing takes too long

**Storage:**
- Ensure adequate EBS volume for logs and temp files
- Consider using S3 for long-term log storage

---

## Testing & Verification

### Test Phase 1: Agent Access API

```bash
# Terminal 1: Start the Agent Access API
cd /path/to/RivalRadar
python -m src.agent_access.agent_api

# Terminal 2: Test the API
curl http://localhost:8003/health/ready
curl -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "payment trends", "date_filter": false}'
```

### Test Phase 2: Research MCP Server

```bash
# Terminal 1: Keep Agent Access API running

# Terminal 2: Start the MCP Server
cd /path/to/RivalRadar
pip install -e src/MCP_Servers/research_mcp_server/
research-mcp-server

# Terminal 3: Test health endpoint
curl http://localhost:8001/health
```

### Test Phase 3: Main Agent

```bash
# Ensure both APIs are running (ports 8003 and 8001)

# Start the agent
python -m src.Agents.main_agent

# Test queries:
# "What are the latest payment trends?"
# "Tell me about fintech funding in 2024"
```

---

## Quick Reference Commands

### Starting All Services (Development)

```bash
# Terminal 1: Agent Access API
python -m src.agent_access.agent_api

# Terminal 2: Research MCP Server
research-mcp-server

# Terminal 3: Main Agent
python -m src.Agents.main_agent
```

### Installing the MCP Server Package

```bash
cd src/MCP_Servers/research_mcp_server
pip install -e .
```

### Environment Variables Quick Reference

```bash
# Required in .env:
LLM_MODEL=anthropic/claude-3.5-sonnet
OPEN_ROUTER_API_KEY=sk-or-v1-...
aws_access_key_id=AKIA...
aws_secret_access_key=...
aws_region=us-east-2
bucket=rivalradar-scraped-data
QDRANT_URL=https://...cloud.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION=rivalradar_articles

# New additions:
RESEARCH_API_URL=http://localhost:8003/query
RESEARCH_MCP_URL=http://localhost:8001/mcp
```

---

## File Structure After Implementation

```
RivalRadar/
├── .env                                    # Environment variables
├── src/
│   ├── agent_access/
│   │   ├── __init__.py                    # NEW
│   │   ├── agent_api.py                   # NEW - FastAPI server
│   │   └── agent_search.py                # NEW - Hybrid search logic
│   │
│   ├── MCP_Servers/
│   │   └── research_mcp_server/
│   │       ├── __init__.py                # NEW
│   │       ├── __main__.py                # NEW - Entry point
│   │       ├── server.py                  # NEW - FastMCP server
│   │       ├── research.py                # NEW - Research tool
│   │       ├── date.py                    # NEW - Date tool
│   │       ├── pyproject.toml             # NEW - Package config
│   │       └── README.md                  # NEW - Documentation
│   │
│   ├── Agents/
│   │   ├── __init__.py                    # UPDATED
│   │   ├── main_agent.py                  # NEW - PydanticAI agent
│   │   ├── brain_agent.py                 # EXISTING (kept for reference)
│   │   ├── qdrant_query.py                # EXISTING
│   │   └── news_agent.py                  # EXISTING
│   │
│   ├── Scraping/                          # EXISTING - No changes
│   └── Testing/                           # EXISTING - No changes
│
└── config/
    └── requirements.txt                    # UPDATED - Add new dependencies
```

---

## Troubleshooting

### "HTTP client not available" Error
**Cause:** MCP server started but lifespan didn't initialize the client
**Fix:** Check MCP server logs, ensure clean startup

### "Could not connect to research API" Error
**Cause:** Agent Access API not running on port 8003
**Fix:** Start the Agent Access API first

### "Missing QDRANT_URL" Error
**Cause:** Environment variables not loaded
**Fix:** Ensure .env file exists and contains required variables

### Agent hangs on first query
**Cause:** BM25 encoder fitting on corpus (one-time initialization)
**Fix:** Wait for initialization to complete (can take 30-60 seconds)

### EC2 scraper fails silently
**Cause:** Running via cron or non-interactive shell
**Fix:** Use systemd service as described in Phase 4

---

## Success Criteria

- [ ] Agent Access API responds to `/health/ready`
- [ ] Agent Access API returns search results for `/query`
- [ ] Research MCP Server responds to `/health`
- [ ] Main Agent connects to MCP server successfully
- [ ] Main Agent can answer questions about fintech trends
- [ ] Conversation history is maintained across messages
- [ ] EC2 timer is active and shows next run time

---

# Phase 5-8: Dashboard MVP (News Section Demo)

## Architecture Overview (Updated)

The architecture has been pivoted to a **dashboard-based UI** instead of a chat interface.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SUB-AGENTS (Run on schedule)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────┐       │
│   │News Agent  │  │Jobs Agent  │  │Funding Agent│ │Compliance Agent│       │
│   │  (DONE)    │  │  (LATER)   │  │   (LATER)  │  │    (LATER)     │       │
│   └─────┬──────┘  └────────────┘  └────────────┘  └────────────────┘       │
│         │                                                                   │
│         ▼                                                                   │
│      Scrapes → Embeds → Stores in Qdrant with "type" field                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QDRANT DB                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Each entry has: { content, source, type, scraped_date }                   │
│   type = "news" | "careers" | "funding" | "compliance"                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BRAIN AGENT + API                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Brain Agent queries Qdrant, filters by "type", returns insights           │
│   Exposed via FastAPI for frontend to call                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Dashboard with clickable sections: News | Jobs | Funding | Compliance     │
│   Click a section → calls API → displays insights                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 5: Verify Existing Pipeline

**Goal:** Make sure News Agent → Qdrant → Brain Agent flow works end-to-end.

| Step | Test | Command |
|------|------|---------|
| 5.1 | Test News Agent | `python -m src.Agents.news_agent` |
| 5.2 | Test Brain Agent | `python -m src.Agents.brain_agent --interactive` |
| 5.3 | Verify Qdrant | Check Qdrant Cloud dashboard |

**Success Criteria:**
- [ ] News Agent completes without errors
- [ ] Brain Agent returns relevant answers
- [ ] Qdrant contains articles from news sources

---

## Phase 6: Brain Agent API

**Goal:** Expose Brain Agent as a REST API for the frontend.

### Create `src/api/brain_api.py`

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/sections` | GET | List available sections |
| `/insights` | POST | Get insights for a section |

**Request body for `/insights`:**
```json
{
  "section": "news",
  "query": "optional custom query"
}
```

**Response:**
```json
{
  "section": "news",
  "insights": "Analysis from Brain Agent...",
  "sources": ["https://...", "https://..."],
  "articles_used": 7
}
```

**Success Criteria:**
- [ ] API starts without errors
- [ ] `/health` returns ok
- [ ] `/sections` returns list of sections
- [ ] `/insights` returns insights for "news" section

---

## Phase 7: Simple Frontend

**Goal:** Build a basic dashboard that displays insights from the API.

### Tech Stack Options

| Option | Pros | Cons |
|--------|------|------|
| **Streamlit** (recommended) | Python only, fast to build | Less customizable |
| React/Next.js | Professional, customizable | Requires JS knowledge |
| Plain HTML/CSS/JS | No build tools | More manual work |

### Create `src/frontend/dashboard.py` (Streamlit)

Basic dashboard with:
- Sidebar with section buttons (News, Jobs, Funding, Compliance)
- Main area showing insights
- Sources list with clickable links

**Run with:** `streamlit run src/frontend/dashboard.py`

**Success Criteria:**
- [ ] Frontend loads without errors
- [ ] Can click on "News" section
- [ ] Displays insights from API
- [ ] Shows source links

---

## Phase 8: End-to-End Demo

**Goal:** Full working demo with News section.

### Demo Flow

```
1. Qdrant has news articles (from News Agent)
          ↓
2. Start Brain API: python -m src.api.brain_api
          ↓
3. Start Frontend: streamlit run src/frontend/dashboard.py
          ↓
4. Open browser → Click "News" → See insights
```

### Demo Checklist

- [ ] Qdrant has news articles
- [ ] Brain API running on port 8000
- [ ] Frontend running (Streamlit)
- [ ] Click "News" shows relevant insights
- [ ] Sources are displayed and clickable

---

## Future Phases (After Demo)

| Phase | Goal |
|-------|------|
| **9** | Additional sub-agents (Jobs, Funding, Compliance) |
| **10** | Scheduled scraping (systemd timers) |
| **11** | Frontend enhancements (better UI, visualizations) |
| **12** | Production deployment (EC2, Vercel/Netlify)

---

# Dashboard MVP with MCP Architecture (News Only)

> **Scope:** This phase focuses on getting the dashboard working with **News data only**.
> Other sub-agents (Jobs, Funding, Compliance) will be added in Phase 9 after we verify the architecture works.

---

## Architecture Decision: Query & Reasoning Layer

When building the query/reasoning layer (where users ask questions and get insights), we had two architectural options:

### Option 1: BrainAgent (Simple/Direct)

```
┌─────────────────────────────────────────────────────────────┐
│                     QUERY & REASONING                        │
│                                                              │
│   Frontend ──→ Brain API ──→ BrainAgent ──→ Qdrant          │
│                                   │                          │
│                                   └──→ Claude ──→ Response   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Services needed: 1 (Brain API)
```

**Pros:**
- Simple architecture, fewer moving parts
- Easy to understand and debug
- Faster to implement

**Cons:**
- Adding new tools requires editing BrainAgent code
- Less flexible for future expansion
- Not industry standard

### Option 2: MCP Architecture (Extensible) ← CHOSEN

```
┌─────────────────────────────────────────────────────────────┐
│                     QUERY & REASONING                        │
│                                                              │
│   Frontend                                                   │
│      │                                                       │
│      ▼                                                       │
│   MainAgent (PydanticAI)                                     │
│      │                                                       │
│      ▼                                                       │
│   MCP Server (port 8001)                                     │
│      │                                                       │
│      ├── research tool ──→ Agent Access API ──→ Qdrant      │
│      ├── get_date tool                      (port 8003)      │
│      ├── [future tools...]                                   │
│      │                                                       │
│      └──→ Claude ──→ Response                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Services needed: 2 (MCP Server 8001 + Agent Access API 8003)
```

**Pros:**
- Industry standard for AI tool integration
- Easy to add new tools without changing agent code
- Clean separation of concerns
- More valuable pattern to learn
- Future-proof for growing features

**Cons:**
- More services to run
- Slightly more complex setup

### Why We Chose MCP

1. **Extensibility**: Adding new tools (analyze_competitor, generate_report, send_alert) only requires adding to MCP server
2. **Industry Standard**: MCP is becoming the standard for AI tool integration
3. **Learning Value**: More valuable real-world pattern for the team
4. **Separation of Concerns**: Search logic, tool definitions, and agent reasoning are cleanly separated

---

**Chosen Architecture (News Only for now):**
```
NewsAgent (scheduled)
       │
       └──→ scrape ──→ embed ──→ Qdrant (source_type="news")
                                    │
                                    ▼
Frontend Dashboard ──→ Dashboard API ──→ MainAgent ──→ MCP Server ──→ Agent Access API ──→ Qdrant
```

**Current Data:** Only `source_type="news"` exists in Qdrant.

---

## Important Note: Testing Mode (Single Source)

> ⚠️ **Current Status:** For testing purposes, **only 1 URL (pymnts.com)** is being scraped for news.
>
> The `src/Scraping/sources_config.py` file contains 10 news sources, but we're currently only scraping
> the first source to minimize resource usage and API costs during testing/development.
>
> **To enable all 10 news sources in production:**
> ```bash
> # On EC2, edit the systemd service:
> sudo nano /etc/systemd/system/rivalradar-scraper.service
>
> # Change ExecStart to include --use-all-sources:
> ExecStart=/opt/rivalradar/venv/bin/python -m src.Scraping.main --source news --use-all-sources
>
> # Reload and restart:
> sudo systemctl daemon-reload
> ```

---

## Phase 5: Verify Existing MCP Pipeline Works

**Goal:** Ensure MCP architecture is working end-to-end with news data.

### Steps:

| Step | Action | Command |
|------|--------|---------|
| 5.1 | Start Agent Access API | `python -m src.agent_access.agent_api` |
| 5.2 | Start MCP Server | `python -m src.MCP_Servers.research_mcp_server` |
| 5.3 | Test Main Agent | `python -m src.Agents.main_agent` |
| 5.4 | Ask a test question | "What are the latest payment trends?" |

### Success Criteria:
- [ ] Agent Access API running on port 8003
- [ ] MCP Server running on port 8001
- [ ] Main Agent connects successfully
- [ ] Agent returns insights based on news data in Qdrant

---

## Phase 6: Dashboard API (Wraps MainAgent for Frontend)

**Goal:** Create a FastAPI that the frontend can call. It uses MainAgent internally.

### Create `src/api/dashboard_api.py`

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/insights/news` | GET | Get news insights (default query) |
| `/ask` | POST | Free-form question |

**Request body for `/ask`:**
```json
{
  "question": "What are competitors doing in payments?"
}
```

**Response:**
```json
{
  "answer": "Analysis from MainAgent...",
  "sources": ["https://...", "https://..."]
}
```

### Success Criteria:
- [ ] Dashboard API starts without errors
- [ ] `/health` returns ok
- [ ] `/insights/news` returns news insights
- [ ] `/ask` answers free-form questions

---

## Phase 7: Streamlit Dashboard (News Section Only)

**Goal:** Build a simple dashboard UI with just the News section.

### Create `src/frontend/dashboard.py`

**Layout (MVP - News only):**
```
┌──────────────────────────────────────────────────────────┐
│  RivalRadar Dashboard                                     │
├──────────┬───────────────────────────────────────────────┤
│          │                                                │
│  [News]  │   Latest Fintech News Insights                │
│  ──────  │   ─────────────────────────────                │
│          │                                                │
│  Jobs    │   Based on our analysis of recent articles...  │
│  (soon)  │                                                │
│          │   Key trends:                                  │
│ Funding  │   • Trend 1                                    │
│  (soon)  │   • Trend 2                                    │
│          │                                                │
│          │   Sources:                                     │
│          │   • pymnts.com/article-1                       │
│          │   • techcrunch.com/article-2                   │
│          │                                                │
│          ├───────────────────────────────────────────────┤
│          │  Ask a question: [________________] [Ask]      │
│          │                                                │
└──────────┴───────────────────────────────────────────────┘
```

**Run with:** `streamlit run src/frontend/dashboard.py`

### Success Criteria:
- [ ] Dashboard loads without errors
- [ ] "News" section works and shows insights
- [ ] Source links are displayed
- [ ] Free-form questions work

---

## Phase 8: End-to-End Demo (News Only)

**Goal:** Full working demo with News section.

### Startup Sequence (4 terminals):

```bash
# Terminal 1: Agent Access API
python -m src.agent_access.agent_api

# Terminal 2: MCP Server
python -m src.MCP_Servers.research_mcp_server

# Terminal 3: Dashboard API
python -m src.api.dashboard_api

# Terminal 4: Frontend
streamlit run src/frontend/dashboard.py
```

### Demo Checklist:
- [ ] All 4 services running
- [ ] Open http://localhost:8501
- [ ] "News" section shows insights
- [ ] Can ask questions about fintech news
- [ ] Sources are clickable

---

## Quick Reference: Service Ports

| Service | Port | Command |
|---------|------|---------|
| Agent Access API | 8003 | `python -m src.agent_access.agent_api` |
| MCP Server | 8001 | `python -m src.MCP_Servers.research_mcp_server` |
| Dashboard API | 8000 | `python -m src.api.dashboard_api` |
| Streamlit | 8501 | `streamlit run src/frontend/dashboard.py` |

---

## After News Works (Phase 9+)

Once News section is working end-to-end:

1. **Phase 9**: Add Jobs, Funding, Compliance sub-agents (same pattern as NewsAgent)
2. **Phase 10**: Add those sections to Dashboard
3. **Phase 11**: Scheduled scraping for all agents
4. **Phase 12**: Production deployment

---

## Action Items & Technical Backlog (Updated 2026-01-20)

### Current Tasks

| # | Task | Status | Owner | Notes |
|---|------|--------|-------|-------|
| 1 | Filter Relevant Articles by Date | ✅ COMPLETE | - | Recency-weighted scoring implemented |
| 2 | EC2 Instance / Daily Reboot | ⚠️ PARTIAL | Joaquin | Timer works, reboot/Elastic IP pending |
| 8 | Logfire Integration | 🔴 NOT STARTED | - | Centralized logging needed |
| 9 | Dashboard Key Insights/Sources | ⚠️ IN PROGRESS | - | Parsing intermittent, diagnostic logging added |

### Future / Track

| # | Task | Description | Priority |
|---|------|-------------|----------|
| 3 | Enable All 10 URLs | Currently only pymnts.com scraping | Medium |
| 4 | DB Retention Policy | Delete articles >6 months old | Low |
| 5 | source_type Filtering | For multiple subagent support | Medium |
| 6 | EC2 Scalability | Parallel scraping for multi-source | Low |
| 7 | Scraping Schedule | ✅ DECIDED: 6-7 AM Central (12:00 UTC) | - |

### Implementation Details

#### Task 1: Recency-Weighted Scoring (COMPLETE)

**Approach:** Post-processing re-ranking with recency multiplier

```python
adjusted_score = semantic_score x recency_weight

# Recency weights:
# - Today/yesterday: 1.0
# - 2-3 days old: 0.95
# - 4-7 days old: 0.85
# - 1-2 weeks old: 0.70
# - >2 weeks: 0.50
```

**Files Changed:**
- `src/agent_access/agent_search.py` (lines 143-177, 325-377)
- `src/Agents/qdrant_query.py` (lines 26-60, 159-254)
- `src/Testing/test_11_brain_agent.py` (lines 158-213)

#### Task 2: EC2 Scheduling (PARTIAL)

**Working:**
- Systemd timer runs at 12:00 UTC (6 AM Central)
- Scraper successfully storing articles in Qdrant

**Pending (Joaquin - see Joaquin_EC2.md Part 9):**
- Daily instance reboot for memory clearing
- Elastic IP assignment for stable IP

#### Task 9: Dashboard Parsing Issue (IN PROGRESS)

**Problem:** Key Insights and Sources fields sometimes blank/inconsistent

**Root Cause:** LLM response parsing is fragile - relies on numbered lists/bullets

**Current Status:** Added diagnostic logging (dashboard_api.py lines 201-230)

**Recommended Fix:** Implement Pydantic AI structured output to guarantee consistent response format
