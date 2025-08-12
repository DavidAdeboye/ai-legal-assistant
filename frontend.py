import streamlit as st
import requests
import json
from typing import List
import time

# Set page configuration
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide"
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

def main():
    st.title("📚 AI Legal Assistant")
    st.markdown("Upload legal documents and chat with them using AI")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📄 Document Management")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF/DOCX)",
            accept_multiple_files=True,
            type=["pdf", "docx"]
        )

        if st.button("Upload Selected Files"):
            if uploaded_files:
                with st.spinner("Uploading documents..."):
                    result = upload_files(uploaded_files)
                    if result:
                        st.success(f"Successfully uploaded {len(uploaded_files)} documents!")
                        time.sleep(1)  # Give the API time to process
                        st.rerun()  # Refresh the page to show new documents

        # Document list
        st.subheader("📋 Uploaded Documents")
        documents = get_documents()
        
        if documents:
            for doc in documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    status_color = "🟢" if doc["status"] == "completed" else "🟡" if doc["status"] == "processing" else "🔴"
                    st.write(f"{status_color} {doc['filename']}")
                    st.caption(f"Status: {doc['status']} | Chunks: {doc['chunk_count']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{doc['id']}", help="Delete document"):
                        if delete_document(doc['id']):
                            st.success("Document deleted!")
                            time.sleep(1)
                            st.rerun()
        else:
            st.info("No documents uploaded yet.")

    with col2:
        st.header("💬 Chat Interface")

        # Chat interface
        if documents:
            # Display chat history
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.write(f"You: {msg['content']}")
                else:
                    st.write(f"Assistant: {msg['content']}")
                    if msg.get("sources"):
                        with st.expander("View Sources"):
                            for idx, source in enumerate(msg["sources"], 1):
                                st.write(f"Source {idx}: {source['filename']}, "
                                       f"{source.get('section', 'N/A')}")

            # Chat input
            user_message = st.chat_input("Type your message here...")
            if user_message:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_message
                })

                # Get AI response
                with st.spinner("Getting response..."):
                    response = chat_with_documents(user_message, st.session_state.session_id)
                    if response:
                        st.session_state.session_id = response["session_id"]
                        # Add AI response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response["response"],
                            "sources": response["sources"]
                        })
                st.rerun()  # Refresh to show new messages
        else:
            st.info("Please upload some documents to start chatting!")

if __name__ == "__main__":
    main()
