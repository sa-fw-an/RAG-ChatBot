"""
UI components for the Streamlit application.
"""
import streamlit as st
from datetime import datetime

from modules.database import clear_chat_history, clear_vector_db

def render_sidebar():
    """Render the sidebar with document upload, text input, and URL input options."""
    st.header("Knowledge Base Configuration")
    
    tab1, tab2, tab3 = st.tabs(["File Upload", "Text Input", "URL Scraper"])
    
    # File Upload tab
    with tab1:
        uploaded_files = st.file_uploader(
            "Upload files", 
            type=["pdf", "txt", "docx", "csv", "json"], 
            accept_multiple_files=True
        )
    
    # Text Input tab
    with tab2:
        text_content = st.text_area("Enter text to add to knowledge base", height=200)
        text_title = st.text_input("Title for this text (optional)")
        add_text_button = st.button("Add Text to Knowledge Base")
    
    # URL Scraper tab  
    with tab3:
        url = st.text_input("Enter URL to scrape")
        url_button = st.button("Scrape URL")
    
    st.divider()
    
    # ADD THE WEB SEARCH TOGGLE HERE
    st.subheader("Search Settings")
    
    if "enable_web_search" not in st.session_state:
        st.session_state.enable_web_search = True
        
    enable_search = st.checkbox(
        "Enable web search",
        value=st.session_state.enable_web_search,
        help="When enabled, the chatbot can search the web for information not in the knowledge base"
    )
    
    if enable_search != st.session_state.enable_web_search:
        st.session_state.enable_web_search = enable_search
    
    st.divider()
    
    # About section
    st.subheader("About")
    st.write("This app creates a knowledge base chatbot that can answer questions based on your provided documents and information.")
    
    return (
        uploaded_files,
        {"content": text_content, "title": text_title, "submitted": add_text_button},
        {"url": url, "submitted": url_button}
    )

def render_chat_area(chat_history, is_kb_initialized=True):
    """Render the chat interface with message history."""
    st.subheader("Chat")
    
    # Display chat history
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "score" in message:
                st.caption(f"Relevance score: {message['score']:.2f}")

    # Chat input
    user_query = st.chat_input(
        "Ask a question:", 
        disabled=not is_kb_initialized,
        key="chat_input"
    )
    
    if not is_kb_initialized:
        st.info("Please add some content to your knowledge base first (upload files, add text, or scrape a URL).")

    return user_query

def render_stats_panel(vector_count, files, entries, models, current_model):
    """Render the stats and controls panel."""
    st.subheader("Knowledge Base Stats")
    
    # Display vector count
    st.metric("Vector Chunks", vector_count)
    
    # Display files
    with st.expander(f"Files ({len(files)})", expanded=False):
        if files:
            for file in files:
                st.caption(f"• {file}")
        else:
            st.caption("No files uploaded yet")
    
    # Display manual entries
    with st.expander(f"Text Entries ({len(entries)})", expanded=False):
        if entries:
            for entry in entries:
                st.caption(f"• {entry['title']} ({entry['timestamp'].split('T')[0]})")
        else:
            st.caption("No text entries added yet")
    
    # Model selection
    st.divider()
    st.subheader("Model Settings")
    
    selected_model = st.selectbox(
        "Language Model",
        options=models,
        index=models.index(current_model) if current_model in models else 0
    )
    
    if selected_model != current_model:
        st.session_state.current_model = selected_model
        st.rerun()
    
    # Add clear buttons
    st.divider()
    st.subheader("Reset Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Knowledge Base", type="secondary"):
            # Use the imported clear_vector_db function
            clear_vector_db()
            st.session_state.processed_files = set()
            st.session_state.manual_entries = []
            st.session_state.kb_initialized = False
            st.rerun()
    
    with col2:
        if st.button("Clear Chat History", type="secondary"):
            # Use the imported clear_chat_history function
            clear_chat_history()
            st.session_state.chat_history = []
            st.rerun()
            

def show_sources(context_items):
    """Display the sources of the information used to answer the query."""
    if not context_items:
        return
    
    # Separate web sources from knowledge base sources
    web_sources = [item for item in context_items if item.get("metadata", {}).get("type") == "web_search"]
    kb_sources = [item for item in context_items if item.get("metadata", {}).get("type") != "web_search"]
    
    with st.expander("View Sources", expanded=False):
        # If we used web search, show those sources first with special formatting
        if web_sources:
            st.markdown("### 🌐 Web Search Results")
            for i, item in enumerate(web_sources):
                source = item.get("source", "Unknown source")
                content = item.get("content", "").strip()
                
                st.markdown(f"**Source {i+1}: [{source}]({source})**")
                st.text_area(
                    f"Web content {i+1}:",
                    value=content,
                    height=100,
                    disabled=True,
                    key=f"web_source_{i}"
                )
                st.divider()
        
        # Then show knowledge base sources
        if kb_sources:
            if web_sources:
                st.markdown("### 📚 Knowledge Base Sources")
                
            for i, item in enumerate(kb_sources):
                source = item.get("source", "Unknown source")
                content = item.get("content", "").strip()
                
                st.markdown(f"**Source {i+1}: {source}**")
                st.text_area(
                    f"Content excerpt {i+1}:",
                    value=content,
                    height=100,
                    disabled=True,
                    key=f"kb_source_{i}"
                )
                st.divider()