# quick_setup.py - Python script to create all necessary files

import os

def create_env_example():
    """Create .env.example file"""
    content = """# .env.example
# Copy this to .env and fill in your actual values

# Gemini API Key (get from https://makersuite.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# Database URL (for local development)
DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/ragdb

# For production, use your actual database URL
# DATABASE_URL=postgresql://user:password@host:port/database

# Optional: Set log level
LOG_LEVEL=INFO
"""
    with open('.env.example', 'w') as f:
        f.write(content)
    print("✅ Created .env.example")

def create_env():
    """Create .env file from example"""
    if not os.path.exists('.env'):
        create_env_example()
        with open('.env.example', 'r') as f:
            content = f.read()
        with open('.env', 'w') as f:
            f.write(content)
        print("✅ Created .env file")
    else:
        print("⚠️ .env file already exists")

def create_requirements():
    """Create requirements.txt"""
    content = """fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.2
asyncpg==0.29.0
PyPDF2==3.0.1
python-docx==1.1.0
python-multipart==0.0.6
numpy==1.24.3
pydantic==2.5.0

# Optional: For PostgreSQL with vector support
# pgvector==0.2.4

# Development dependencies
pytest==7.4.3
pytest-asyncio==0.21.1
"""
    with open('requirements.txt', 'w') as f:
        f.write(content)
    print("✅ Created requirements.txt")

def main():
    print("🚀 Creating setup files...")
    
    # Create all necessary files
    create_env_example()
    create_env()
    create_requirements()
    
    print("\n📦 Now installing dependencies...")
    print("Run these commands:")
    print("1. python -m pip install --upgrade pip setuptools wheel")
    print("2. pip install -r requirements.txt")
    print("3. Edit .env file with your Gemini API key")
    print("4. uvicorn main:app --reload")

if __name__ == "__main__":
    main()