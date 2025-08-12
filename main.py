# main.py
import os
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
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/ragdb")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
logger.info(f"Using Gemini API key: {GEMINI_API_KEY[:10]}...")

# Database connection pool
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
        await initialize_database()
        logger.info("Database connected and initialized")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.info("Running without database - using in-memory storage")
        db_pool = None
        init_memory_storage()
    yield
    # Shutdown
    if db_pool:
        await db_pool.close()

app = FastAPI(title="RAG Chat API", lifespan=lifespan)

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
                    logger.info(f"Sending embedding request with payload: {payload}")
                    response = await client.post(url, headers=self.headers, json=payload)
                    if response.status_code != 200:
                        logger.error(f"Error response: {response.text}")
                    response.raise_for_status()
                    
                    result = response.json()
                    logger.info(f"Embedding response: {result}")  # Log the response for debugging
                    if "embedding" in result:
                        return result["embedding"]
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
    return None

async def release_db_connection(conn):
    if conn and db_pool:
        await db_pool.release(conn)

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
                    page_chunks = split_text_into_chunks(text, max_length=1000, overlap=100)
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
        text_chunks = split_text_into_chunks(combined_text, max_length=1000, overlap=100)
        
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
    """Split text into overlapping chunks"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at sentence or word boundary
        chunk_text = text[start:end]
        
        # Find last sentence boundary
        last_period = chunk_text.rfind('.')
        last_newline = chunk_text.rfind('\n')
        last_space = chunk_text.rfind(' ')
        
        boundary = max(last_period, last_newline, last_space)
        
        if boundary > start + max_length // 2:  # Only use boundary if it's not too early
            end = start + boundary + 1
        
        chunks.append(text[start:end].strip())
        start = max(start + max_length - overlap, end - overlap)
    
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
                await conn.execute('''
                    INSERT INTO document_chunks 
                    (document_id, chunk_text, chunk_index, page_number, section, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6)
                ''', 
                document_id, chunk.text, i, chunk.page_number, chunk.section, embedding)
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
        
        # Check if we have pgvector
        try:
            await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            has_vector = True
        except:
            has_vector = False
        
        if has_vector:
            # Use pgvector for similarity search
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
            ''', query_embedding, limit)
        else:
            # Manual similarity calculation
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
            
            # Calculate similarities manually
            similarities = []
            for row in results:
                chunk_embedding = json.loads(row["embedding_json"])
                similarity = 1 - cosine_similarity(query_embedding, chunk_embedding)
                similarities.append({
                    "chunk_text": row["chunk_text"],
                    "page_number": row["page_number"],
                    "section": row["section"],
                    "filename": row["filename"],
                    "similarity": similarity
                })
            
            # Sort by similarity and take top results
            similarities.sort(key=lambda x: x["similarity"])
            results = similarities[:limit]
        
        return [
            {
                "text": result["chunk_text"] if has_vector else result["chunk_text"],
                "page_number": result["page_number"],
                "section": result["section"],
                "filename": result["filename"],
                "similarity": float(result["similarity"])
            }
            for result in results
        ]
        
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
    
    try:
        if session_id:
            # Check if session exists
            result = await conn.fetchrow('''
                SELECT id FROM chat_sessions WHERE id = $1
            ''', session_id)
            
            if result:
                # Update last activity
                await conn.execute('''
                    UPDATE chat_sessions SET last_activity = CURRENT_TIMESTAMP 
                    WHERE id = $1
                ''', session_id)
                return session_id
        
        # Create new session
        new_session_id = str(uuid.uuid4())
        await conn.execute('''
            INSERT INTO chat_sessions (id) VALUES ($1)
        ''', new_session_id)
        
        return new_session_id
        
    finally:
        await release_db_connection(conn)

async def store_chat_message(session_id: str, message: str, response: str, sources: List[Dict[str, Any]]) -> None:
    """Store chat message and response"""
    conn = await get_db_connection()
    
    if not conn:
        # Use in-memory storage
        message_id = str(uuid.uuid4())
        memory_storage["messages"][message_id] = {
            "id": message_id,
            "session_id": session_id,
            "message": message,
            "response": response,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        return
    
    try:
        await conn.execute('''
            INSERT INTO chat_messages (session_id, message, response, sources)
            VALUES ($1, $2, $3, $4)
        ''', session_id, message, response, json.dumps(sources))
        
    finally:
        await release_db_connection(conn)

# API Routes

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a document"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    file_extension = file.filename.lower().split('.')[-1]
    if file_extension not in ['pdf', 'docx']:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    # Create document record
    document_id = str(uuid.uuid4())

    conn = await get_db_connection()
    if conn:
        try:
            await conn.execute(
                '''
                INSERT INTO documents (id, filename, file_type, status)
                VALUES ($1, $2, $3, 'processing')
                ''',
                document_id, file.filename, file_extension
            )
        finally:
            await release_db_connection(conn)
    else:
        memory_storage["documents"][document_id] = {
            "id": document_id,
            "filename": file.filename,
            "file_type": file_extension,
            "upload_date": datetime.now().isoformat(),
            "status": "processing"
        }
    
    # Save file to a temp path BEFORE starting background task
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}").name
    content = await file.read()
    with open(temp_file_path, "wb") as f:
        f.write(content)

    # Pass path instead of UploadFile
    background_tasks.add_task(process_document_background, document_id, temp_file_path, file_extension)
    
    return DocumentResponse(
        document_id=document_id,
        filename=file.filename,
        chunks_processed=0,
        status="processing"
    )


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
        if os.path.exists(file_path):
            os.unlink(file_path)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with documents using RAG"""
    try:
        # Get or create session
        session_id = await get_or_create_session(request.session_id)
        
        # Search for relevant document chunks
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
            
            # Create RAG prompt
            prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. If the context doesn't contain 
enough information to answer the question, say so clearly.

Context:
{context}

Question: {request.message}

Please provide a clear and helpful answer based on the context above. If you reference specific 
information, mention which source it came from."""
            
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
                for chunk in similar_chunks
            ]
        
        # Store chat message
        await store_chat_message(session_id, request.message, response_text, sources)
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            session_id=session_id
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