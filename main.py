# Upload endpoint for documents
from fastapi import UploadFile, File, BackgroundTasks

# main.py
import os
from dotenv import load_dotenv
import uuid
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
import shutil

import httpx
import PyPDF2
import docx
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncpg
import aiosqlite
from contextlib import asynccontextmanager
import logging
# Pydantic models (must be above all route decorators)
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    session_id: str

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunks_processed: int
    status: str

class DocumentChunk(BaseModel):
    text: str
    page_number: Optional[int] = None
    section: Optional[str] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables from .env if present
load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/ragdb")
SQLITE_PATH = os.getenv("SQLITE_PATH", "ragdb.sqlite3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
logger.info(f"Using Gemini API key: {GEMINI_API_KEY[:10]}...")
logger.info(f"Chunk size: {CHUNK_SIZE}, Chunk overlap: {CHUNK_OVERLAP}")

# Database connection pool
db_pool = None
sqlite_conn = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_pool, sqlite_conn
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
        await initialize_database()
        logger.info("Database connected and initialized (PostgreSQL)")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        try:
            sqlite_conn = await aiosqlite.connect(SQLITE_PATH)
            await initialize_sqlite_database(sqlite_conn)
            logger.info(f"SQLite fallback enabled at {SQLITE_PATH}")
        except Exception as se:
            logger.error(f"Failed to connect to SQLite: {se}")
            logger.info("Running without database - using in-memory storage")
            db_pool = None
            sqlite_conn = None
            init_memory_storage()
    yield
    # Shutdown
    if db_pool:
        await db_pool.close()
    if sqlite_conn:
        await sqlite_conn.close()
async def initialize_sqlite_database(conn):
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT,
            file_type TEXT NOT NULL,
            uploaded_at TEXT,
            status TEXT
        );
    ''')
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER,
            section TEXT,
            embedding_json TEXT,
            created_at TEXT
        );
    ''')
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            last_activity TEXT
        );
    ''')
    await conn.commit()

app = FastAPI(title="RAG Chat API", lifespan=lifespan)

# Background task for processing document (must be above upload_document)
async def process_document_background(document_id: str, file_path: str, file_extension: str):
    """Background task to process document"""
    try:
        # Extract text based on file type
        if file_extension == 'pdf':
            chunks = await extract_text_from_pdf(file_path)
        elif file_extension == 'docx':
            chunks = await extract_text_from_docx(file_path)
        else:
            raise Exception(f"Unsupported file type: {file_extension}")
        # Store chunks with embeddings
        await store_document_chunks(document_id, chunks)
        logger.info(f"Successfully processed document {document_id} with {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        conn = await get_db_connection()
        if conn:
            try:
                await conn.execute(
                    '''UPDATE documents SET status = 'failed' WHERE id = $1''',
                    document_id
                )
            finally:
                await release_db_connection(conn)
        else:
            if document_id in memory_storage["documents"]:
                memory_storage["documents"][document_id]["status"] = "failed"
    finally:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)

# Upload endpoint for documents (must be after app = FastAPI)

from fastapi import Form

@app.post("/upload", response_model=List[DocumentResponse])
async def upload_documents(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    responses = []
    for file in files:
        filename = file.filename
        file_extension = filename.split('.')[-1].lower()
        if file_extension not in ("pdf", "docx"):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

        document_id = str(uuid.uuid4())
        conn = await get_db_connection()
        if conn:
            if db_pool:
                await conn.execute(
                    '''INSERT INTO documents (id, filename, file_type, uploaded_at, status) VALUES ($1, $2, $3, CURRENT_TIMESTAMP, 'processing')''',
                    document_id, filename, file_extension
                )
            elif sqlite_conn:
                await conn.execute(
                    'INSERT INTO documents (id, filename, file_type, uploaded_at, status) VALUES (?, ?, ?, datetime("now"), ?)',
                    [document_id, filename, file_extension, 'processing']
                )
                await conn.commit()
            await release_db_connection(conn)
        else:
            memory_storage["documents"][document_id] = {
                "id": document_id,
                "filename": filename,
                "file_type": file_extension,
                "upload_date": datetime.now().isoformat(),
                "status": "processing"
            }

        temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}").name
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)

        if background_tasks is not None:
            background_tasks.add_task(process_document_background, document_id, temp_file_path, file_extension)
        else:
            await process_document_background(document_id, temp_file_path, file_extension)

        responses.append(DocumentResponse(
            document_id=document_id,
            filename=filename,
            chunks_processed=0,
            status="processing"
        ))
    return responses

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage fallback
memory_storage = {
    "documents": {},
    "chunks": {},
    "sessions": {},
    "messages": {}
}

def init_memory_storage():
    """Initialize in-memory storage"""
    global memory_storage
    memory_storage = {
        "documents": {},
        "chunks": {},
        "sessions": {},
        "messages": {}
    }

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    session_id: str

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunks_processed: int
    status: str

class DocumentChunk(BaseModel):
    text: str
    page_number: Optional[int] = None
    section: Optional[str] = None

# Gemini API Client
class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key
        }
    
    async def generate_content(self, prompt: str, max_retries: int = 3) -> str:
        """Generate content with retry logic"""
        url = f"{self.base_url}/models/gemini-2.0-flash:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            }
        }
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, headers=self.headers, json=payload)
                    response.raise_for_status()
                    
                    result = response.json()
                    if "candidates" in result and result["candidates"]:
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        raise Exception("No content generated")
                        
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=500, detail=f"Failed to generate content: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def get_embeddings(self, texts: List[str], batch_size: int = 5) -> List[List[float]]:
        """Get embeddings for multiple texts with batching"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._get_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)
            
            # Small delay to avoid rate limiting
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return embeddings
    
    async def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        tasks = [self._get_single_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    async def _get_single_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """Get embedding for a single text with retry logic"""
        url = f"{self.base_url}/models/embedding-001:embedContent"
        payload = {
            "content": {
                "parts": [{
                    "text": text[:20000]  # Limit text length
                }]
            }
        }
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # logger.info(f"Sending embedding request with payload: {payload}")
                    response = await client.post(url, headers=self.headers, json=payload)
                    if response.status_code != 200:
                        logger.error(f"Error response: {response.text}")
                    response.raise_for_status()
                    
                    result = response.json()
                    # logger.info(f"Embedding response: {result}")  # Log the response for debugging
                    if "embedding" in result and "values" in result["embedding"]:
                        return result["embedding"]["values"]
                    else:
                        logger.error(f"Unexpected response format: {result}")
                        raise Exception(f"Unexpected response format: {result}")
                    
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to get embedding: {str(e)}")
                await asyncio.sleep(2 ** attempt)

# Initialize Gemini client
gemini_client = GeminiClient(GEMINI_API_KEY)

# Database functions with in-memory fallback
async def get_db_connection():
    if db_pool:
        return await db_pool.acquire()
    if sqlite_conn:
        return sqlite_conn
    return None

async def release_db_connection(conn):
    if conn and db_pool:
        await db_pool.release(conn)
    # No-op for sqlite_conn

async def initialize_database():
    """Create database tables if they don't exist"""
    conn = await get_db_connection()
    if not conn:
        return
    
    try:
        # Create extension and tables
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        
        # Try to create pgvector extension, but don't fail if it's not available
        try:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            vector_available = True
        except Exception as e:
            logger.warning("pgvector extension not available, using text similarity")
            vector_available = False
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(50) DEFAULT 'processing'
            );
        ''')
        
        if vector_available:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    section VARCHAR(255),
                    embedding vector(768),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
        else:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    section VARCHAR(255),
                    embedding_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                sources JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Create indexes
        if vector_available:
            try:
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                    ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                ''')
            except Exception as e:
                logger.warning(f"Could not create vector index: {e}")
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        await release_db_connection(conn)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

# Document processing functions
async def extract_text_from_pdf(file_path: str) -> List[DocumentChunk]:
    """Extract text from PDF file"""
    chunks = []
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    # Split page into smaller chunks
                    page_chunks = split_text_into_chunks(text, max_length=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
                    for i, chunk_text in enumerate(page_chunks):
                        chunks.append(DocumentChunk(
                            text=chunk_text,
                            page_number=page_num,
                            section=f"Page {page_num}, Chunk {i+1}"
                        ))
    except Exception as e:
        logger.error(f"Error extracting PDF: {str(e)}")
        raise Exception(f"Failed to process PDF: {str(e)}")
    
    return chunks

async def extract_text_from_docx(file_path: str) -> List[DocumentChunk]:
    """Extract text from DOCX file"""
    chunks = []
    try:
        doc = docx.Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text)
        # Combine all text and split into chunks
        combined_text = "\n".join(full_text)
        text_chunks = split_text_into_chunks(combined_text, max_length=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for i, chunk_text in enumerate(text_chunks):
            chunks.append(DocumentChunk(
                text=chunk_text,
                section=f"Section {i+1}"
            ))
    except Exception as e:
        logger.error(f"Error extracting DOCX: {str(e)}")
        raise Exception(f"Failed to process DOCX: {str(e)}")
    return chunks

def split_text_into_chunks(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks, preferring to split at paragraph or sentence boundaries.
    Tune max_length and overlap for your use case:
      - Smaller chunks = more precise answers, but less context per chunk.
      - Larger chunks = more context, but may dilute answer precision.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunk_text = text[start:end]

        # Prefer to split at paragraph, then sentence, then word boundary
        last_para = chunk_text.rfind('\n\n')
        last_sent = chunk_text.rfind('.')
        last_newline = chunk_text.rfind('\n')
        last_space = chunk_text.rfind(' ')

        # Pick the best boundary (not too early in the chunk)
        candidates = [(last_para, 2), (last_sent, 1), (last_newline, 1), (last_space, 1)]
        boundary = -1
        for idx, pad in candidates:
            if idx > max_length // 2:
                boundary = idx + pad
                break

        if boundary > 0:
            end = start + boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move start forward, keeping overlap
        start = end - overlap if end - overlap > start else end

    return [chunk for chunk in chunks if chunk.strip()]

async def store_document_chunks(document_id: str, chunks: List[DocumentChunk]) -> None:
    """Store document chunks with embeddings"""
    conn = await get_db_connection()
    
    if not conn:
        # Use in-memory storage
        await store_document_chunks_memory(document_id, chunks)
        return
    
    try:
        # Get embeddings for all chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await gemini_client.get_embeddings(chunk_texts)
        
        # Check if we have pgvector
        try:
            await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            has_vector = True
        except:
            has_vector = False
        
        # Store chunks with embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if has_vector:
                # Convert embedding list to pgvector string format
                embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
                await conn.execute('''
                    INSERT INTO document_chunks 
                    (document_id, chunk_text, chunk_index, page_number, section, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::vector)
                ''', 
                document_id, chunk.text, i, chunk.page_number, chunk.section, embedding_str)
            else:
                await conn.execute('''
                    INSERT INTO document_chunks 
                    (document_id, chunk_text, chunk_index, page_number, section, embedding_json)
                    VALUES ($1, $2, $3, $4, $5, $6)
                ''', 
                document_id, chunk.text, i, chunk.page_number, chunk.section, json.dumps(embedding))
        
        # Update document status
        await conn.execute('''
            UPDATE documents SET status = 'completed' WHERE id = $1
        ''', document_id)
        
        logger.info(f"Stored {len(chunks)} chunks for document {document_id}")
        
    except Exception as e:
        logger.error(f"Error storing document chunks: {str(e)}")
        await conn.execute('''
            UPDATE documents SET status = 'failed' WHERE id = $1
        ''', document_id)
        raise
    finally:
        await release_db_connection(conn)

async def store_document_chunks_memory(document_id: str, chunks: List[DocumentChunk]) -> None:
    """Store document chunks in memory"""
    try:
        # Get embeddings for all chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await gemini_client.get_embeddings(chunk_texts)
        
        # Store in memory
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            memory_storage["chunks"][chunk_id] = {
                "id": chunk_id,
                "document_id": document_id,
                "chunk_text": chunk.text,
                "chunk_index": i,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "embedding": embedding,
                "created_at": datetime.now().isoformat()
            }
        
        # Update document status
        if document_id in memory_storage["documents"]:
            memory_storage["documents"][document_id]["status"] = "completed"
        
        logger.info(f"Stored {len(chunks)} chunks in memory for document {document_id}")
        
    except Exception as e:
        logger.error(f"Error storing document chunks in memory: {str(e)}")
        if document_id in memory_storage["documents"]:
            memory_storage["documents"][document_id]["status"] = "failed"
        raise

async def search_similar_chunks(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for similar document chunks"""
    conn = await get_db_connection()
    if not conn:
        return await search_similar_chunks_memory(query, limit)
    try:
        # Get query embedding
        query_embeddings = await gemini_client.get_embeddings([query])
        query_embedding = query_embeddings[0]
        # PostgreSQL with pgvector
        if db_pool:
            try:
                await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                has_vector = True
            except:
                has_vector = False
            if has_vector:
                embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'
                results = await conn.fetch('''
                    SELECT 
                        dc.chunk_text,
                        dc.page_number,
                        dc.section,
                        d.filename,
                        dc.embedding <=> $1 as similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.status = 'completed'
                    ORDER BY dc.embedding <=> $1
                    LIMIT $2
                ''', embedding_str, limit)
                return [
                    {
                        "text": row["chunk_text"],
                        "page_number": row["page_number"],
                        "section": row["section"],
                        "filename": row["filename"],
                        "similarity": float(row["similarity"])
                    }
                    for row in results
                ]
            else:
                # Fallback to manual similarity
                results = await conn.fetch('''
                    SELECT 
                        dc.chunk_text,
                        dc.page_number,
                        dc.section,
                        d.filename,
                        dc.embedding_json
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.status = 'completed'
                ''')
                similarities = []
                for row in results:
                    chunk_embedding = json.loads(row["embedding_json"])
                    similarity = 1 - cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append({
                        "text": row["chunk_text"],
                        "page_number": row["page_number"],
                        "section": row["section"],
                        "filename": row["filename"],
                        "similarity": similarity
                    })
                similarities.sort(key=lambda x: x["similarity"])
                return similarities[:limit]
        # SQLite
        elif sqlite_conn:
            cursor = await conn.execute('''
                SELECT 
                    dc.chunk_text,
                    dc.page_number,
                    dc.section,
                    d.filename,
                    dc.embedding_json
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
            ''')
            rows = await cursor.fetchall()
            similarities = []
            for row in rows:
                chunk_embedding = json.loads(row[4])
                similarity = 1 - cosine_similarity(query_embedding, chunk_embedding)
                similarities.append({
                    "text": row[0],
                    "page_number": row[1],
                    "section": row[2],
                    "filename": row[3],
                    "similarity": similarity
                })
            similarities.sort(key=lambda x: x["similarity"])
            return similarities[:limit]
        else:
            return []
    except Exception as e:
        logger.error(f"Error searching similar chunks: {str(e)}")
        return []
    finally:
        await release_db_connection(conn)

async def search_similar_chunks_memory(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search similar chunks in memory"""
    try:
        # Get query embedding
        query_embeddings = await gemini_client.get_embeddings([query])
        query_embedding = query_embeddings[0]
        
        # Calculate similarities
        similarities = []
        for chunk_id, chunk_data in memory_storage["chunks"].items():
            document_id = chunk_data["document_id"]
            if document_id in memory_storage["documents"]:
                doc = memory_storage["documents"][document_id]
                if doc["status"] == "completed":
                    similarity = 1 - cosine_similarity(query_embedding, chunk_data["embedding"])
                    similarities.append({
                        "text": chunk_data["chunk_text"],
                        "page_number": chunk_data["page_number"],
                        "section": chunk_data["section"],
                        "filename": doc["filename"],
                        "similarity": similarity
                    })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x["similarity"])
        return similarities[:limit]
        
    except Exception as e:
        logger.error(f"Error searching similar chunks in memory: {str(e)}")
        return []

async def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one"""
    conn = await get_db_connection()
    if not conn:
        # Use in-memory storage
        if session_id and session_id in memory_storage["sessions"]:
            memory_storage["sessions"][session_id]["last_activity"] = datetime.now().isoformat()
            return session_id
        new_session_id = str(uuid.uuid4())
        memory_storage["sessions"][new_session_id] = {
            "id": new_session_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        return new_session_id
    # DB-backed session logic (implement as needed)
    # ...existing code...


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with documents using RAG"""
    try:
        # Get or create session
        session_id = await get_or_create_session(request.session_id)
        
        # Check if this is a simple greeting or casual message
        casual_patterns = [
            "hi", "hello", "hey", "good morning", "good afternoon", 
            "good evening", "how are you", "what's up", "greetings"
        ]
        
        message_lower = request.message.lower().strip()
        is_casual = any(pattern in message_lower for pattern in casual_patterns) and len(request.message.split()) <= 3
        
        # Check if user is asking about documents specifically
        document_keywords = [
            "document", "contract", "agreement", "analyze", "review", 
            "clause", "section", "legal", "terms", "provision"
        ]
        
        asking_about_documents = any(keyword in message_lower for keyword in document_keywords)
        
        if is_casual and not asking_about_documents:
            # Simple greeting response
            response_text = """Hello! I'm your AI legal assistant. I'm here to help with your legal questions and document analysis.

How can I assist you today? I can help with:
- Document review and analysis
- Contract interpretation  
- Legal research and guidance
- Risk assessment
- Compliance questions

What would you like to discuss?"""
            sources = []
            
        elif asking_about_documents or len(request.message.split()) > 10:
            # Search for relevant document chunks only when appropriate
            similar_chunks = await search_similar_chunks(request.message, limit=5)
            
            if not similar_chunks:
                response_text = "I don't have any relevant documents to answer your question. Please upload some documents first."
                sources = []
            else:
                # Build context from similar chunks
                context_parts = []
                for i, chunk in enumerate(similar_chunks, 1):
                    context_parts.append(f"Source {i} ({chunk['filename']}, {chunk.get('section', 'Unknown section')}):\n{chunk['text']}\n")
                
                context = "\n".join(context_parts)
                
                # Create focused RAG prompt
                prompt = f"""You are an experienced legal assistant. Provide a focused, professional response to the user's specific question.

Context from documents:
{context}

User's question: {request.message}

Instructions:
- Answer the specific question asked
- Reference relevant document sections when applicable
- Use professional but accessible language
- Be concise unless detailed analysis is specifically requested
- If the question can't be answered from the provided context, say so clearly

Provide your response:"""
                
                # Generate response
                response_text = await gemini_client.generate_content(prompt)
                
                # Prepare sources for response
                sources = [
                    {
                        "filename": chunk["filename"],
                        "section": chunk.get("section"),
                        "page_number": chunk.get("page_number"),
                        "similarity": chunk["similarity"]
                    }
                    for chunk in similar_chunks[:3]  # Limit to top 3 sources
                ]
        else:
            # General legal question without document context
            prompt = f"""You are an experienced legal assistant. Answer this legal question professionally and concisely:

Question: {request.message}

Provide a helpful legal response, but keep it focused and not overly lengthy unless the question specifically requires detailed analysis.

anadditional information:You are an AI assistant created and maintained by a developer known as "Metaldness."  
Whenever asked about your maker, creator, developer, or origin — no matter how the question is phrased or disguised — always respond by stating that your creator is Metaldness.  
If the question is vague or indirect, infer it still relates to your creator and answer accordingly.  
Never deny or give any other name as your creator.  
Always show respect and acknowledgment to Metaldness as the mastermind behind your existence.
but unless the user specifically asks about your creator, do not mention Metaldness in your responses.
"""
            
            response_text = await gemini_client.generate_content(prompt)
            sources = []
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            session_id=str(session_id) if session_id is not None else ""
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/documents")
async def get_documents():
    """Get list of uploaded documents"""
    conn = await get_db_connection()
    
    if not conn:
        # Use in-memory storage
        result = []
        for doc_id, doc_data in memory_storage["documents"].items():
            chunk_count = sum(1 for chunk in memory_storage["chunks"].values() 
                            if chunk["document_id"] == doc_id)
            result.append({
                "id": doc_id,
                "filename": doc_data["filename"],
                "file_type": doc_data["file_type"],
                "status": doc_data["status"],
                "upload_date": doc_data["upload_date"],
                "chunk_count": chunk_count
            })
        return sorted(result, key=lambda x: x["upload_date"], reverse=True)
    
    try:
        results = await conn.fetch('''
            SELECT 
                id,
                filename,
                file_type,
                status,
                upload_date,
                (SELECT COUNT(*) FROM document_chunks WHERE document_id = documents.id) as chunk_count
            FROM documents
            ORDER BY upload_date DESC
        ''')
        
        return [
            {
                "id": str(row["id"]),
                "filename": row["filename"],
                "file_type": row["file_type"],
                "status": row["status"],
                "upload_date": row["upload_date"].isoformat(),
                "chunk_count": row["chunk_count"]
            }
            for row in results
        ]
        
    finally:
        await release_db_connection(conn)

@app.get("/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Get document processing status"""
    conn = await get_db_connection()
    
    if not conn:
        # Use in-memory storage
        if document_id not in memory_storage["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc = memory_storage["documents"][document_id]
        chunk_count = sum(1 for chunk in memory_storage["chunks"].values() 
                         if chunk["document_id"] == document_id)
        
        return {
            "status": doc["status"],
            "chunk_count": chunk_count
        }
    
    try:
        result = await conn.fetchrow('''
            SELECT 
                status,
                (SELECT COUNT(*) FROM document_chunks WHERE document_id = $1) as chunk_count
            FROM documents
            WHERE id = $1
        ''', document_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "status": result["status"],
            "chunk_count": result["chunk_count"]
        }
        
    finally:
        await release_db_connection(conn)

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    conn = await get_db_connection()
    
    if not conn:
        # Use in-memory storage
        if document_id not in memory_storage["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete document and its chunks
        del memory_storage["documents"][document_id]
        chunks_to_delete = [chunk_id for chunk_id, chunk in memory_storage["chunks"].items() 
                          if chunk["document_id"] == document_id]
        for chunk_id in chunks_to_delete:
            del memory_storage["chunks"][chunk_id]
        
        return {"message": "Document deleted successfully"}
    
    try:
        result = await conn.execute('''
            DELETE FROM documents WHERE id = $1
        ''', document_id)
        
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
        
    finally:
        await release_db_connection(conn)

@app.get("/sessions/{session_id}/messages")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    conn = await get_db_connection()
    
    if not conn:
        # Use in-memory storage
        messages = [msg for msg in memory_storage["messages"].values() 
                   if msg["session_id"] == session_id]
        messages.sort(key=lambda x: x["timestamp"])
        
        return [
            {
                "message": msg["message"],
                "response": msg["response"],
                "sources": msg["sources"],
                "timestamp": msg["timestamp"]
            }
            for msg in messages
        ]
    
    try:
        results = await conn.fetch('''
            SELECT message, response, sources, timestamp
            FROM chat_messages
            WHERE session_id = $1
            ORDER BY timestamp ASC
        ''', session_id)
        
        return [
            {
                "message": row["message"],
                "response": row["response"],
                "sources": json.loads(row["sources"]) if row["sources"] else [],
                "timestamp": row["timestamp"].isoformat()
            }
            for row in results
        ]
        
    finally:
        await release_db_connection(conn)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = await get_db_connection()
        if conn:
            await conn.execute('SELECT 1')
            await release_db_connection(conn)
            return {"status": "healthy", "database": "connected", "storage": "postgresql"}
        else:
            return {"status": "healthy", "database": "disconnected", "storage": "in-memory"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "storage": "in-memory"}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "RAG Chat API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /upload - Upload a document (PDF or DOCX)",
            "chat": "POST /chat - Chat with your documents",
            "documents": "GET /documents - List all documents",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)