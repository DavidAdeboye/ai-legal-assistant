# AI Legal Assistant ⚖️

A sophisticated AI-powered legal document analysis platform that combines modern web technologies with advanced natural language processing to provide intelligent legal insights and document review capabilities.

## 🌟 Features

### Core Functionality
- **Document Upload & Processing**: Support for PDF and DOCX legal documents
- **AI-Powered Chat Interface**: Interactive Q&A with your legal documents using Google's Gemini 2.0 Flash
- **Advanced RAG (Retrieval-Augmented Generation)**: Intelligent document chunking and semantic search
- **Real-time Document Analysis**: Background processing with status tracking
- **Multi-format Support**: Handles various legal document types including contracts, briefs, and agreements

### User Experience
- **Modern Glassmorphism UI**: Beautiful, responsive interface with iOS 18-inspired design
- **Real-time Chat**: Instant responses with source citations
- **Document Management**: Upload, view, and delete documents with processing status
- **Analytics Dashboard**: Insights into document library and chat history
- **Session Management**: Persistent chat sessions across browser refreshes

### Technical Features
- **Flexible Database Support**: PostgreSQL with pgvector, SQLite fallback, or in-memory storage
- **Vector Embeddings**: Semantic search using Google's embedding-001 model
- **Scalable Architecture**: Async FastAPI backend with connection pooling
- **Error Handling**: Robust error handling with retry mechanisms
- **CORS Support**: Cross-origin resource sharing for web deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   Database      │
│   (Streamlit)   │◄──►│   (FastAPI)      │◄──►│ PostgreSQL/     │
│                 │    │                  │    │ SQLite/Memory   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Google Gemini   │
                    │  (AI Processing) │
                    └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key
- PostgreSQL (optional, falls back to SQLite/memory)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai-legal-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/ragdb
BACKEND_URL=http://localhost:8000
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

4. **Run the backend**
```bash
python main.py
```

5. **Run the frontend** (in a new terminal)
```bash
streamlit run frontend.py
```

6. **Access the application**
Open your browser to `http://localhost:8501`

## 📊 Database Setup

### PostgreSQL with pgvector (Recommended)
```sql
CREATE DATABASE ragdb;
\c ragdb;
CREATE EXTENSION vector;
CREATE EXTENSION "uuid-ossp";
```

### SQLite (Automatic Fallback)
The application automatically creates SQLite database if PostgreSQL is unavailable.

### In-Memory (Development)
For testing, the app can run entirely in memory without any database.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | **Required** |
| `DATABASE_URL` | PostgreSQL connection string | PostgreSQL default |
| `SQLITE_PATH` | SQLite database path | `ragdb.sqlite3` |
| `BACKEND_URL` | Backend API URL for frontend | `http://localhost:8000` |
| `CHUNK_SIZE` | Text chunk size for processing | `1000` |
| `CHUNK_OVERLAP` | Overlap between text chunks | `100` |

### Chunking Strategy
The application uses intelligent text chunking that:
- Prefers paragraph boundaries (`\n\n`)
- Falls back to sentence boundaries (`.`)
- Uses word boundaries as last resort
- Maintains configurable overlap for context preservation

## 🌐 Deployment

### Production Deployment

1. **Backend (FastAPI)**
```bash
# Using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000

# Using gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

2. **Frontend (Streamlit)**
```bash
streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
# Backend Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment
- **Backend**: Deploy on Render, Railway, or AWS Lambda
- **Frontend**: Deploy on Streamlit Cloud or Heroku
- **Database**: Use managed PostgreSQL (AWS RDS, Google Cloud SQL)

## 📡 API Reference

### Upload Documents
```http
POST /upload
Content-Type: multipart/form-data

files: List[UploadFile]
```

### Chat with Documents
```http
POST /chat
Content-Type: application/json

{
  "message": "What are the key terms in this contract?",
  "session_id": "optional-session-id"
}
```

### Get Documents
```http
GET /documents
```

### Delete Document
```http
DELETE /documents/{document_id}
```

### Health Check
```http
GET /health
```

## 🧠 AI Integration

### Google Gemini Integration
- **Model**: `gemini-2.0-flash` for text generation
- **Embeddings**: `embedding-001` for semantic search
- **Features**: Retry logic, rate limiting, batch processing

### RAG Implementation
1. **Document Processing**: Text extraction and intelligent chunking
2. **Vector Embeddings**: Generate embeddings for all chunks
3. **Semantic Search**: Find relevant chunks using cosine similarity
4. **Context Building**: Construct prompts with relevant context
5. **Response Generation**: Generate responses with source citations

## 🎨 Frontend Features

### Design System
- **Glassmorphism**: Modern translucent design with backdrop blur
- **Color Palette**: Green and coral theme with gradients
- **Typography**: SF Pro Display and Inter fonts
- **Responsive**: Mobile-friendly design

### Components
- **Chat Interface**: Real-time messaging with typing indicators
- **Document Cards**: Elegant document display with status indicators
- **Analytics Dashboard**: Statistics and insights
- **File Uploader**: Drag-and-drop with progress tracking

## 🔒 Security Considerations

- **API Key Protection**: Store Gemini API key securely
- **Input Validation**: Sanitize all user inputs
- **File Type Restriction**: Only allow PDF and DOCX files
- **Error Handling**: Don't expose internal errors to users
- **CORS Configuration**: Restrict origins in production

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Test Coverage
```bash
pytest --cov=main tests/
```

### Load Testing
```bash
# Test document upload
curl -X POST "http://localhost:8000/upload" -F "files=@test.pdf"

# Test chat endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is this document about?"}'
```

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check Gemini API key is set correctly
   - Verify internet connection
   - Check API quotas and billing

2. **Database Connection Error**
   - Verify PostgreSQL is running
   - Check connection string in DATABASE_URL
   - App will fallback to SQLite/memory automatically

3. **Document Processing Failed**
   - Ensure PDF/DOCX files are not corrupted
   - Check file size limits
   - Verify sufficient disk space

4. **Frontend Not Loading**
   - Check backend is running on correct port
   - Verify BACKEND_URL in environment
   - Check CORS settings

### Performance Optimization

1. **Large Documents**
   - Increase CHUNK_SIZE for better context
   - Reduce CHUNK_OVERLAP to save storage
   - Use PostgreSQL with pgvector for better performance

2. **Memory Usage**
   - Monitor memory usage with large document sets
   - Consider using Redis for session storage
   - Implement document cleanup procedures

## 📈 Monitoring

### Health Checks
- `/health` endpoint provides system status
- Database connectivity monitoring
- API status indicator in UI

### Logging
```python
# Configure logging level
logging.basicConfig(level=logging.INFO)

# Monitor key metrics
- Document processing times
- API response times  
- Error rates
- Database query performance
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use type hints for better code clarity
- Add docstrings for functions and classes
- Keep functions focused and small

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini**: AI model and embeddings
- **Streamlit**: Frontend framework
- **FastAPI**: Backend API framework
- **pgvector**: Vector similarity search
- **Creator**: Developed by Metaldness

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Include logs and error messages

## 🗺️ Roadmap

- [ ] Multi-language document support
- [ ] Advanced legal analytics
- [ ] Document comparison features
- [ ] Team collaboration tools
- [ ] Mobile application
- [ ] Integration with legal databases
- [ ] Custom AI model fine-tuning
- [ ] Advanced security features
