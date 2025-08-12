#!/bin/bash
# setup.sh - Quick setup script for RAG application

echo "🚀 Setting up RAG Application with Gemini API..."

# Create project directory
echo "📁 Creating project structure..."

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
echo "⚙️ Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file"
    echo "⚠️  IMPORTANT: Please edit .env file and add your Gemini API key!"
    echo "    Get your API key from: https://makersuite.google.com/app/apikey"
else
    echo "⚠️ .env file already exists"
fi

echo ""
echo "🎯 Setup Options:"
echo ""
echo "1. 💾 In-Memory Storage (Quick Start - No Database Required)"
echo "   - Data stored in memory only"
echo "   - Perfect for testing and development"
echo "   - Data lost when server restarts"
echo ""
echo "2. 🐘 PostgreSQL Database (Production Ready)"
echo "   - Persistent storage"
echo "   - Better performance with large documents"
echo "   - Supports vector similarity search"
echo ""

read -p "Choose option (1 for in-memory, 2 for PostgreSQL): " choice

if [ "$choice" = "2" ]; then
    echo ""
    echo "🐘 PostgreSQL Setup Instructions:"
    echo ""
    echo "1. Install PostgreSQL:"
    echo "   - Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "   - macOS: brew install postgresql"
    echo "   - Windows: Download from https://www.postgresql.org/download/"
    echo ""
    echo "2. Create database and user:"
    echo "   sudo -u postgres psql"
    echo "   CREATE DATABASE ragdb;"
    echo "   CREATE USER raguser WITH PASSWORD 'ragpassword';"
    echo "   GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;"
    echo "   \\q"
    echo ""
    echo "3. Optional: Install pgvector extension for better performance:"
    echo "   - Follow instructions at: https://github.com/pgvector/pgvector"
    echo ""
    echo "4. Update your .env file with the database URL:"
    echo "   DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/ragdb"
    echo ""
fi

echo ""
echo "🚀 Starting the application..."
echo "📖 API Documentation will be available at: http://localhost:8000/docs"
echo "🌐 API will be running at: http://localhost:8000"
echo ""

# Check if .env file has API key set
if grep -q "your_gemini_api_key_here" .env 2>/dev/null; then
    echo "❌ Please set your GEMINI_API_KEY in the .env file before starting!"
    echo "   Edit .env and replace 'your_gemini_api_key_here' with your actual API key"
    echo ""
else
    echo "▶️  To start the application, run:"
    echo "   source venv/bin/activate  # (or venv\\Scripts\\activate on Windows)"
    echo "   uvicorn main:app --reload"
    echo ""
    echo "🔧 Your application will:"
    if [ "$choice" = "2" ]; then
        echo "   - Use PostgreSQL for storage"
        echo "   - Support vector similarity search (if pgvector is installed)"
    else
        echo "   - Use in-memory storage (data lost on restart)"
        echo "   - Fall back gracefully if database is not available"
    fi
    echo "   - Process PDF and DOCX files"
    echo "   - Generate embeddings with Gemini API"
    echo "   - Provide RAG-based chat responses"
fi