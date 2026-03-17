"""
Retrieval Service
=================
Pure retrieval microservice:
1. Receives question + user_profile.
2. Embeds the question.
3. Performs filtered vector search across library.chunk_embeddings.
4. Returns top-k chunks with metadata.
"""
import logging
import os
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Local import
from retrieval import vector_search

load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RetrievalService")

# Configuration
DATABASE_URL = os.environ["DATABASE_URL"]
EMBEDDER_NAME = os.environ.get("EMBEDDER_MODEL_NAME", "mixedbread-ai/mxbai-embed-large-v1")

# DB Initialization
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Embedder
try:
    from sentence_transformers import SentenceTransformer
    logger.info(f"⏳ Loading embedder: {EMBEDDER_NAME}")
    # Note: ensure dimension matches library.chunk_embeddings (1536 vs 1024)
    embedder = SentenceTransformer(EMBEDDER_NAME)
    logger.info("✅ Embedder loaded")
except Exception as e:
    logger.error(f"❌ Embedder not loaded: {e}")
    embedder = None

app = FastAPI(title="HSE Retrieval Service", version="1.1.0")


class UserProfile(BaseModel):
    university_id: Optional[str] = None
    campus_id: Optional[str] = None
    faculty_id: Optional[str] = None
    program_id: Optional[str] = None
    year: Optional[int] = None
    group_name: Optional[str] = None
    role: Optional[str] = None


class RetrievalRequest(BaseModel):
    question: str
    user_profile: UserProfile
    top_k: int = 5


class RetrievalResponse(BaseModel):
    question: str
    chunks: List[dict]


@app.get("/ping")
async def ping():
    return {
        "status": "ok", 
        "service": "Retrieval Service", 
        "embedder": EMBEDDER_NAME,
        "indexed_dim_expected": 1536  # Hint for infrastructure
    }


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(req: RetrievalRequest):
    """
    Main retrieval endpoint: embeds question and returns filtered chunks.
    No LLM calls, no chat history management.
    """
    if embedder is None:
        raise HTTPException(500, "Embedder model not loaded.")

    try:
        # 1. Vectorize query
        query_vec = embedder.encode(req.question, normalize_embeddings=True).tolist()
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(500, f"Error generating embedding: {e}")

    # 2. Search in DB
    async with async_session() as session:
        try:
            chunks = await vector_search(
                session=session, 
                query_embedding=query_vec, 
                user_profile=req.user_profile.model_dump(),
                top_k=req.top_k
            )
            logger.info(f"🔍 Question: {req.question[:50]}... | Found {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            raise HTTPException(500, f"Database search error: {e}")

    # 3. Return results
    return RetrievalResponse(
        question=req.question,
        chunks=chunks
    )
