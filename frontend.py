import streamlit as st
import requests
import json
from typing import List
import time
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = "http://localhost:8000"

def upload_files(files):
    """Upload files to the API"""
    if not files:
        return None
    
    files_for_upload = [("files", file) for file in files]
    try:
        response = requests.post(f"{API_URL}/upload", files=files_for_upload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading files: {str(e)}")
        return None

def get_documents():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_URL}/documents")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []

def chat_with_documents(message: str, session_id: str = None):
    """Send chat message to API"""
    try:
        payload = {"message": message, "session_id": session_id}
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending message: {str(e)}")
        return None

def delete_document(document_id: str):
    """Delete a document"""
    try:
        response = requests.delete(f"{API_URL}/documents/{document_id}")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

def clear_chat_history():
    """Clear the chat history"""
    st.session_state.chat_history = []
    st.session_state.session_id = None

def format_timestamp():
    """Get formatted timestamp"""
    return datetime.now().strftime("%I:%M %p")

def main():
    # Custom CSS with modern design
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styles with iOS 18 glassmorphism */
        .stApp {
            font-family: 'SF Pro Display', 'Inter', sans-serif;
            background: linear-gradient(135deg, #105042 0%, #0d4436 25%, #0a3d30 50%, #08362a 75%, #063024 100%);
            min-height: 100vh;
            position: relative;
        }
        
        /* Add beautiful gradient overlay */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 20%, rgba(248, 112, 96, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(248, 112, 96, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 40% 60%, rgba(248, 112, 96, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }
        
        /* Main container with advanced glassmorphism */
        .main > div {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            border: 1px solid rgba(248, 112, 96, 0.15);
            border-radius: 24px;
            padding: 2.5rem;
            margin: 1rem;
            box-shadow: 
                0 8px 32px rgba(16, 80, 66, 0.3),
                0 2px 16px rgba(248, 112, 96, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
        }
        
        /* Enhanced header with glassmorphism */
        .main-header {
            background: rgba(248, 112, 96, 0.1);
            backdrop-filter: blur(40px);
            -webkit-backdrop-filter: blur(40px);
            border: 1px solid rgba(248, 112, 96, 0.2);
            color: #F87060;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 
                0 20px 40px rgba(16, 80, 66, 0.4),
                0 8px 32px rgba(248, 112, 96, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.15);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, rgba(248, 112, 96, 0.05) 0%, rgba(16, 80, 66, 0.05) 100%);
            pointer-events: none;
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 3rem;
            font-weight: 700;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #F87060 0%, #ff9a85 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 2px 4px rgba(248, 112, 96, 0.3);
        }
        
        .main-header p {
            margin: 1rem 0 0 0;
            font-size: 1.2rem;
            color: rgba(248, 112, 96, 0.9);
            font-weight: 400;
        }
        
        /* Chat message styling with glassmorphism */
        .chat-message {
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            position: relative;
            animation: slideInGlass 0.4s ease-out;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            overflow: hidden;
        }
        
        @keyframes slideInGlass {
            from { 
                opacity: 0; 
                transform: translateY(30px) scale(0.95);
                filter: blur(10px);
            }
            to { 
                opacity: 1; 
                transform: translateY(0) scale(1);
                filter: blur(0);
            }
        }
        
        .user-message {
            background: rgba(248, 112, 96, 0.15);
            border: 1px solid rgba(248, 112, 96, 0.3);
            color: #F87060;
            margin-left: 3rem;
            box-shadow: 
                0 8px 32px rgba(248, 112, 96, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .user-message::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(248, 112, 96, 0.1) 0%, rgba(255, 154, 133, 0.05) 100%);
            pointer-events: none;
        }
        
        .assistant-message {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            margin-right: 3rem;
            color: rgba(255, 255, 255, 0.95);
            box-shadow: 
                0 8px 32px rgba(16, 80, 66, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .message-header {
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.1rem;
        }
        
        .timestamp {
            font-size: 0.8rem;
            opacity: 0.7;
            margin-left: auto;
            font-weight: 400;
        }
        
        /* Document card with advanced glassmorphism */
        .document-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border: 1px solid rgba(248, 112, 96, 0.15);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 
                0 4px 20px rgba(16, 80, 66, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .document-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(248, 112, 96, 0.03) 0%, rgba(255, 255, 255, 0.02) 100%);
            pointer-events: none;
            transition: opacity 0.3s ease;
        }
        
        .document-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 
                0 20px 40px rgba(16, 80, 66, 0.4),
                0 8px 32px rgba(248, 112, 96, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.15);
            border-color: rgba(248, 112, 96, 0.4);
        }
        
        .document-card:hover::before {
            opacity: 1.5;
        }
        
        .document-card h4 {
            color: rgba(255, 255, 255, 0.95);
            font-weight: 600;
            font-size: 1.2rem;
        }
        
        .document-status {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1.25rem;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 1rem;
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
        }
        
        .status-complete {
            background: rgba(52, 211, 153, 0.15);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }
        
        .status-processing {
            background: rgba(251, 191, 36, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.3);
        }
        
        .status-error {
            background: rgba(248, 113, 113, 0.15);
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.3);
        }
        
        /* Enhanced button styling */
        .stButton > button {
            background: rgba(248, 112, 96, 0.2);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(248, 112, 96, 0.3);
            color: #F87060;
            padding: 1rem 2rem;
            border-radius: 15px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 
                0 4px 20px rgba(248, 112, 96, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(248, 112, 96, 0.1) 0%, rgba(255, 154, 133, 0.05) 100%);
            transition: opacity 0.3s ease;
            pointer-events: none;
        }
        
        .stButton > button:hover {
            transform: translateY(-4px) scale(1.05);
            background: rgba(248, 112, 96, 0.3);
            box-shadow: 
                0 12px 40px rgba(248, 112, 96, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            color: #fff;
        }
        
        .stButton > button:hover::before {
            opacity: 1.5;
        }
        
        /* File uploader with glassmorphism */
        .stFileUploader > div {
            border: 2px dashed rgba(248, 112, 96, 0.3);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            background: rgba(248, 112, 96, 0.05);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            transition: all 0.3s ease;
            color: rgba(248, 112, 96, 0.8);
        }
        
        .stFileUploader > div:hover {
            border-color: rgba(248, 112, 96, 0.6);
            background: rgba(248, 112, 96, 0.1);
            transform: translateY(-4px);
        }
        
        /* Sidebar with glassmorphism */
        .css-1d391kg {
            background: rgba(16, 80, 66, 0.3);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            border-right: 1px solid rgba(248, 112, 96, 0.15);
        }
        
        /* Enhanced stats cards */
        .stat-card {
            background: rgba(248, 112, 96, 0.1);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border: 1px solid rgba(248, 112, 96, 0.2);
            color: #F87060;
            padding: 2rem;
            border-radius: 18px;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 
                0 8px 32px rgba(248, 112, 96, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(248, 112, 96, 0.05) 0%, rgba(255, 154, 133, 0.02) 100%);
            pointer-events: none;
        }
        
        .stat-card:hover {
            transform: translateY(-4px);
            box-shadow: 
                0 16px 40px rgba(248, 112, 96, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.15);
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            display: block;
            background: linear-gradient(135deg, #F87060 0%, #ff9a85 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-label {
            font-size: 1rem;
            opacity: 0.9;
            margin-top: 0.75rem;
            font-weight: 500;
        }
        
        /* Source citations with glassmorphism */
        .source-citation {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-left: 4px solid #F87060;
            border: 1px solid rgba(248, 112, 96, 0.15);
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 12px;
            font-size: 0.95rem;
            color: rgba(255, 255, 255, 0.9);
        }
        
        /* Welcome message with advanced glassmorphism */
        .welcome-card {
            background: rgba(248, 112, 96, 0.1);
            backdrop-filter: blur(40px);
            -webkit-backdrop-filter: blur(40px);
            border: 1px solid rgba(248, 112, 96, 0.2);
            color: #F87060;
            padding: 4rem 3rem;
            border-radius: 25px;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 
                0 24px 60px rgba(16, 80, 66, 0.4),
                0 12px 40px rgba(248, 112, 96, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .welcome-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(248, 112, 96, 0.05) 0%, rgba(16, 80, 66, 0.03) 100%);
            pointer-events: none;
        }
        
        .welcome-card h2 {
            margin-bottom: 1.5rem;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #F87060 0%, #ff9a85 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .welcome-card p {
            font-size: 1.2rem;
            opacity: 0.9;
            line-height: 1.7;
            color: rgba(248, 112, 96, 0.9);
        }
        
        /* Chat input with glassmorphism */
        .stChatInput > div {
            border-radius: 30px;
            border: 2px solid rgba(248, 112, 96, 0.2);
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            transition: all 0.3s ease;
        }
        
        .stChatInput > div:focus-within {
            border-color: rgba(248, 112, 96, 0.5);
            box-shadow: 0 0 0 4px rgba(248, 112, 96, 0.15);
            background: rgba(255, 255, 255, 0.08);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(16, 80, 66, 0.2);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #F87060 0%, #ff9a85 100%);
            border-radius: 5px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #ff9a85 0%, #F87060 100%);
        }
        
        /* Text color fixes */
        .stMarkdown, .stText {
            color: rgba(255, 255, 255, 0.9);
        }
        
        /* Metric styling */
        .metric-container [data-testid="metric-container"] {
            background: rgba(248, 112, 96, 0.1);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(248, 112, 96, 0.15);
            border-radius: 15px;
            padding: 1.5rem;
        }
        
        /* Enhanced loading animation */
        .loading-dots {
            display: inline-block;
            color: #F87060;
        }
        
        .loading-dots::after {
            content: '';
            animation: dotsGlow 1.5s steps(5, end) infinite;
        }
        
        @keyframes dotsGlow {
            0%, 20% { content: ''; }
            40% { content: '.'; }
            60% { content: '..'; }
            80%, 100% { content: '...'; }
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "current_view" not in st.session_state:
        st.session_state.current_view = "chat"

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div class="main-header">
                <h1>⚖️ Legal AI</h1>
                <p>Your intelligent legal assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Get documents for stats
        documents = get_documents()
        
        # Statistics
        total_docs = len(documents)
        completed_docs = len([d for d in documents if d.get("status") == "completed"])
        chat_messages = len(st.session_state.chat_history)
        
        st.markdown(f"""
            <div class="stat-card">
                <span class="stat-number">{total_docs}</span>
                <div class="stat-label">Documents</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="stat-card">
                <span class="stat-number">{completed_docs}</span>
                <div class="stat-label">Processed</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="stat-card">
                <span class="stat-number">{chat_messages}</span>
                <div class="stat-label">Messages</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### Navigation")
        
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.current_view = "chat"
            st.rerun()
        
        if st.button("📄 Documents", use_container_width=True):
            st.session_state.current_view = "documents"
            st.rerun()
        
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.current_view = "analytics"
            st.rerun()
        
        # Quick actions
        st.markdown("### Quick Actions")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat_history()
            st.success("Chat history cleared!")
            time.sleep(1)
            st.rerun()
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Main content area
    if st.session_state.current_view == "chat":
        show_chat_interface(documents)
    elif st.session_state.current_view == "documents":
        show_document_management(documents)
    elif st.session_state.current_view == "analytics":
        show_analytics(documents)

def show_chat_interface(documents):
    """Display the chat interface"""
    if not documents or not any(d.get("status") == "completed" for d in documents):
        # Welcome screen
        st.markdown("""
            <div class="welcome-card">
                <h2>👋 Welcome to AI Legal Assistant</h2>
                <p>Upload your legal documents to start getting intelligent insights and answers. 
                I can help you analyze contracts, review legal documents, and answer questions about your files.</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Upload Documents", use_container_width=True):
                st.session_state.current_view = "documents"
                st.rerun()
        
        with col2:
            if st.button("📚 View Examples", use_container_width=True):
                st.info("Examples: 'What are the key terms in this contract?', 'Summarize the main obligations', 'What are the risks?'")
        
        with col3:
            if st.button("❓ Get Help", use_container_width=True):
                st.info("Upload PDF or DOCX files, then ask questions about their content. I'll provide detailed legal analysis.")
                
    else:
        # Chat interface
        st.markdown("""
            <div class="main-header">
                <h1>💬 Legal Chat</h1>
                <p>Ask questions about your uploaded documents</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Chat container
        chat_container = st.container()
        
        # Chat input
        user_message = st.chat_input("Ask me anything about your legal documents...")
        
        if user_message:
            timestamp = format_timestamp()
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": timestamp
            })
            
            with st.spinner("Analyzing your query..."):
                response = chat_with_documents(user_message, st.session_state.session_id)
                if response:
                    st.session_state.session_id = response["session_id"]
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["response"],
                        "sources": response.get("sources", []),
                        "timestamp": format_timestamp()
                    })
            st.rerun()

        # Display chat history
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <div class="message-header">
                                👤 You
                                <span class="timestamp">{msg.get('timestamp', '')}</span>
                            </div>
                            {msg['content']}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <div class="message-header">
                                ⚖️ Legal Assistant
                                <span class="timestamp">{msg.get('timestamp', '')}</span>
                            </div>
                            {msg['content']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if msg.get("sources"):
                        with st.expander("📚 Legal References & Citations"):
                            for idx, source in enumerate(msg["sources"], 1):
                                st.markdown(f"""
                                    <div class="source-citation">
                                        <strong>📄 Reference {idx}:</strong> {source.get('filename', 'Unknown')}
                                        <br/>
                                        <strong>Section:</strong> {source.get('section', 'General')}
                                        {f"<br/><strong>Page:</strong> {source.get('page', 'N/A')}" if source.get('page') else ""}
                                    </div>
                                """, unsafe_allow_html=True)

def show_document_management(documents):
    """Display document management interface"""
    st.markdown("""
        <div class="main-header">
            <h1>📄 Document Management</h1>
            <p>Upload and manage your legal documents</p>
        </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### Upload New Documents")
    uploaded_files = st.file_uploader(
        "Choose legal documents (PDF, DOCX)",
        accept_multiple_files=True,
        type=["pdf", "docx"],
        help="Upload contracts, legal briefs, court documents, or any legal text files"
    )
    
    if uploaded_files:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📤 Upload", type="primary"):
                with st.spinner("Processing documents..."):
                    result = upload_files(uploaded_files)
                    if result:
                        st.success(f"Successfully uploaded {len(uploaded_files)} documents!")
                        time.sleep(2)
                        st.rerun()
    
    # Document list
    if documents:
        st.markdown("### Your Document Library")
        
        for doc in documents:
            status_config = {
                "completed": ("✅", "Complete", "status-complete"),
                "processing": ("⏳", "Processing", "status-processing"),
                "error": ("❌", "Error", "status-error")
            }
            
            emoji, status_text, status_class = status_config.get(
                doc.get("status", "error"), 
                ("❓", "Unknown", "status-error")
            )
            
            with st.container():
                col1, col2, col3 = st.columns([6, 1, 1])
                
                with col1:
                    st.markdown(f"""
                        <div class="document-card">
                            <h4 style="margin: 0 0 0.5rem 0; color: #2d3748;">
                                📄 {doc.get('filename', 'Unknown Document')}
                            </h4>
                            <div class="document-status {status_class}">
                                {emoji} {status_text}
                            </div>
                            {f"<p style='margin-top: 1rem; color: #718096; font-size: 0.9rem;'>Uploaded: {doc.get('created_at', 'Unknown')}</p>" if doc.get('created_at') else ""}
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("🗑️", key=f"delete_{doc.get('id')}", help="Delete this document"):
                        if delete_document(doc.get('id')):
                            st.success("Document deleted!")
                            time.sleep(1)
                            st.rerun()
                
                with col3:
                    if doc.get("status") == "completed":
                        if st.button("💬", key=f"chat_{doc.get('id')}", help="Start chatting about this document"):
                            st.session_state.current_view = "chat"
                            st.rerun()

    else:
        st.markdown("""
            <div class="welcome-card">
                <h2>📚 No Documents Yet</h2>
                <p>Upload your first legal document to get started with AI-powered legal analysis.</p>
            </div>
        """, unsafe_allow_html=True)

def show_analytics(documents):
    """Display analytics interface"""
    st.markdown("""
        <div class="main-header">
            <h1>📊 Analytics Dashboard</h1>
            <p>Insights about your document library and chat history</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not documents:
        st.info("Upload some documents to see analytics!")
        return
    
    # Document statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_docs = len(documents)
    completed = len([d for d in documents if d.get("status") == "completed"])
    processing = len([d for d in documents if d.get("status") == "processing"])
    errors = len([d for d in documents if d.get("status") == "error"])
    
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Processed", completed)
    with col3:
        st.metric("Processing", processing)
    with col4:
        st.metric("Errors", errors)
    
    # Document types
    st.markdown("### Document Types")
    doc_types = {}
    for doc in documents:
        filename = doc.get('filename', '')
        ext = filename.split('.')[-1].upper() if '.' in filename else 'Unknown'
        doc_types[ext] = doc_types.get(ext, 0) + 1
    
    if doc_types:
        for doc_type, count in doc_types.items():
            st.write(f"**{doc_type}**: {count} documents")
    
    # Chat statistics
    st.markdown("### Chat Statistics")
    total_messages = len(st.session_state.chat_history)
    user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
    assistant_messages = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", total_messages)
    with col2:
        st.metric("Your Questions", user_messages)
    with col3:
        st.metric("AI Responses", assistant_messages)
    
    # Recent activity
    st.markdown("### Recent Activity")
    if st.session_state.chat_history:
        recent_messages = st.session_state.chat_history[-5:]  # Last 5 messages
        for msg in recent_messages:
            role_icon = "👤" if msg["role"] == "user" else "⚖️"
            role_name = "You" if msg["role"] == "user" else "Assistant"
            content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            
            st.markdown(f"""
                <div style="padding: 1rem; margin: 0.5rem 0; background: white; border-radius: 10px; border-left: 4px solid #667eea;">
                    <strong>{role_icon} {role_name}</strong>
                    <p style="margin: 0.5rem 0 0 0; color: #718096;">{content_preview}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No chat history yet. Start a conversation to see recent activity!")

if __name__ == "__main__":
    main()