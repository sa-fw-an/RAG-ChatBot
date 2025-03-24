import streamlit as st
import time
from datetime import datetime
import hashlib

from modules.document_processor import process_file, scrape_url
from modules.knowledge_extractor import extract_knowledge
from modules.web_search import enhance_with_web_search
from modules.database import (
    initialize_vector_db, 
    initialize_chat_db,
    store_embeddings, 
    retrieve_context,
    save_chat_message, 
    get_chat_history,
    store_learned_knowledge,
    remove_semantic_duplicates
)
from modules.models import (
    initialize_embedding_model,
    initialize_semantic_model,
    initialize_llm,
    get_available_models,
    query_llm
)
from modules.ui_components import (
    render_sidebar, 
    render_chat_area, 
    render_stats_panel,
    show_sources
)

# App configuration
st.set_page_config(page_title="Home Chatbot", page_icon="🤖", layout="wide")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = "deepseek-r1-distill-qwen-32b"
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'manual_entries' not in st.session_state:
    st.session_state.manual_entries = []
if 'kb_initialized' not in st.session_state:
    st.session_state.kb_initialized = False
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = f"conv_{int(time.time())}"
if 'conversations' not in st.session_state:
    st.session_state.conversations = [
        {'id': st.session_state.conversation_id, 'name': f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}", 'created_at': datetime.now().isoformat()}
    ]
if 'active_conversation_index' not in st.session_state:
    st.session_state.active_conversation_index = 0
# Add content hash registry for deduplication
if 'content_hashes' not in st.session_state:
    st.session_state.content_hashes = set()

# Main application logic
def main():
    st.title("😎Knowledge Base Home Chatbot😎")
    st.caption("Upload documents, add text, or provide URLs to build your knowledge base")
    
    # Initialize databases and models
    initialize_databases_and_models()
    
    # Layout: Sidebar and main content
    with st.sidebar:
        file_upload, text_input, url_input = render_sidebar()
        
        # Process uploaded files
        if file_upload:
            process_uploaded_files(file_upload)
            
        # Process text input
        if text_input['submitted'] and text_input['content']:
            process_text_input(text_input['content'], text_input['title'])
            
        # Process URL
        if url_input['submitted'] and url_input['url']:
            process_url_input(url_input['url'])
    
    # Main content with columns
    col1, col2 = st.columns([3, 1])
    
    # Chat interface
    with col1:
        user_query = render_chat_area(
            st.session_state.chat_history,
            is_kb_initialized=st.session_state.kb_initialized
        )
        
        if user_query:
            handle_user_query(user_query)
    
    # Stats and controls panel
    with col2:
        render_stats_panel(
            vector_count=st.session_state.vector_db.count() if hasattr(st.session_state, 'vector_db') else 0,
            files=st.session_state.processed_files,
            entries=st.session_state.manual_entries,
            models=get_available_models(),
            current_model=st.session_state.current_model
        )

# Initialize necessary databases and models
def initialize_databases_and_models():
    if 'vector_db' not in st.session_state:
        with st.spinner("Initializing vector database..."):
            st.session_state.vector_db = initialize_vector_db()
            
    if 'chat_db' not in st.session_state:
        with st.spinner("Initializing chat database..."):
            st.session_state.chat_db = initialize_chat_db()
            
    if 'embedding_model' not in st.session_state:
        with st.spinner("Loading embedding model..."):
            st.session_state.embedding_model = initialize_embedding_model()
            
    if 'semantic_model' not in st.session_state:
        with st.spinner("Loading semantic model..."):
            st.session_state.semantic_model = initialize_semantic_model()
            
    if 'llm' not in st.session_state or st.session_state.current_model_name != st.session_state.current_model:
        with st.spinner(f"Connecting to LLM ({st.session_state.current_model})..."):
            st.session_state.llm = initialize_llm(st.session_state.current_model)
            st.session_state.current_model_name = st.session_state.current_model
            
    # Check if we have any data
    if not st.session_state.kb_initialized:
        try:
            count_items = st.session_state.vector_db.count()
            if count_items > 0:
                st.session_state.kb_initialized = True
        except Exception as e:
            # If collection doesn't exist, reinitialize the database
            st.warning("Vector database collection not found. Reinitializing...")
            st.session_state.vector_db = initialize_vector_db()
            st.session_state.kb_initialized = False

# Process uploaded files
def process_uploaded_files(files):
    total_chunks_added = 0
    processed_files = []
    
    for file in files:
        if file.name in st.session_state.processed_files:
            continue
        
        with st.spinner(f"Processing {file.name}..."):
            text, metadata = process_file(file)
            
            if text.strip():
                chunks_added = add_to_knowledge_base(text, metadata)
                total_chunks_added += chunks_added
                processed_files.append(file.name)
                st.session_state.processed_files.add(file.name)
    
    if total_chunks_added > 0:
        st.sidebar.success(f"✅ Added {total_chunks_added} new chunks from {len(processed_files)} files")
        st.session_state.kb_initialized = True
    elif processed_files:
        st.sidebar.info("⚠️ All content already exists in the knowledge base")

# Process text input
def process_text_input(text, title):
    source = title if title else f"Manual entry {len(st.session_state.manual_entries) + 1}"
    
    # Save for reference
    st.session_state.manual_entries.append({
        "title": source,
        "content": text,
        "timestamp": datetime.now().isoformat()
    })
    
    # Process text
    chunks_added = add_to_knowledge_base(text, {"source": source, "type": "manual_entry"})
    
    if chunks_added > 0:
        st.sidebar.success(f"✅ Added {chunks_added} new chunks from text input")
        st.session_state.kb_initialized = True
    else:
        st.sidebar.info("⚠️ Content already exists in the knowledge base")

# Process URL
def process_url_input(url):
    with st.spinner(f"Scraping {url}..."):
        text, metadata = scrape_url(url)
        
        if text:
            chunks_added = add_to_knowledge_base(text, metadata)
            
            if chunks_added > 0:
                st.sidebar.success(f"✅ Added {chunks_added} new chunks from URL ({metadata.get('lines', 0)} lines)")
                st.session_state.kb_initialized = True
            else:
                st.sidebar.info("⚠️ Content already exists in the knowledge base")

# Add text to knowledge base with deduplication
def add_to_knowledge_base(text, metadata):
    from modules.text_processor import chunk_text
    
    # Generate content hash for deduplication
    content_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Skip if we've already processed this exact content
    if content_hash in st.session_state.content_hashes:
        return 0
    
    chunks, chunk_size = chunk_text(text)
    
    # Remove semantic duplicates before storing
    if hasattr(st.session_state, 'vector_db') and hasattr(st.session_state, 'embedding_model'):
        # Only perform semantic deduplication if we've initialized the vector DB
        chunks = remove_semantic_duplicates(
            chunks, 
            st.session_state.vector_db, 
            st.session_state.embedding_model,
            similarity_threshold=0.92
        )
    
    # If no chunks remain after deduplication, return 0
    if not chunks:
        return 0
    
    # Add content hash to metadata to aid future deduplication
    metadata = {
        **metadata,
        "content_hash": content_hash,
        "deduplication_timestamp": datetime.now().isoformat()
    }
    
    chunks_added = store_embeddings(
        chunks,
        st.session_state.vector_db,
        st.session_state.embedding_model,
        metadata
    )
    
    # If chunks were added successfully, add the hash to our registry
    if chunks_added > 0:
        st.session_state.content_hashes.add(content_hash)
    
    return chunks_added

def handle_user_query(user_query):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Add to chat history in session state
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Save to database
    save_chat_message(
        st.session_state.chat_db,
        st.session_state.conversation_id,
        "user",
        user_query
    )
    
    # Get AI response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get recent relevant chat history
            db_chat_history = get_chat_history(
                st.session_state.chat_db,
                st.session_state.conversation_id,
                max_messages=10
            )
            
            # Retrieve context from vector database
            relevant_context = retrieve_context(
                user_query,
                st.session_state.embedding_model,
                st.session_state.vector_db
            )

            used_web_search = False
            if st.session_state.enable_web_search:
                with st.spinner("Searching the web..."):
                    enhanced_context = enhance_with_web_search(user_query, relevant_context)
                    # Check if web results were added
                    used_web_search = len(enhanced_context) > len(relevant_context)
                    relevant_context = enhanced_context
            
            # Generate response
            ai_response, score = query_llm(
                user_query,
                st.session_state.llm,
                db_chat_history,
                relevant_context,
                st.session_state.semantic_model,
                model_name=st.session_state.current_model
            )
            
            # KNOWLEDGE EXTRACTION CODE STARTS HERE
            new_facts = extract_knowledge(user_query, ai_response)
            
            # If facts were extracted, store them in the vector database
            if new_facts:
                store_result = store_learned_knowledge(
                    st.session_state.vector_db,
                    st.session_state.embedding_model,
                    st.session_state.semantic_model,
                    new_facts,
                    context=f"From conversation on {datetime.now().strftime('%Y-%m-%d')}"
                )
                
                if isinstance(store_result, dict):
                    # Using new version with improved tracking
                    added_count = store_result.get("added", 0)
                    updated_count = store_result.get("updated", 0)
                    skipped_count = store_result.get("skipped", 0)
                    
                    # Build a notification message
                    message_parts = []
                    if added_count > 0:
                        message_parts.append(f"learned {added_count} new facts")
                    if updated_count > 0:
                        message_parts.append(f"corrected {updated_count} existing facts")
                    if skipped_count > 0:
                        message_parts.append(f"skipped {skipped_count} duplicate facts")
                    
                    if message_parts:
                        st.toast(f"I {' and '.join(message_parts)}!")
                else:
                    # Fallback for older version of the function
                    facts_added = store_result
                    if facts_added > 0:
                        st.toast(f"I learned {facts_added} new facts from our conversation!")
            # KNOWLEDGE EXTRACTION CODE ENDS HERE
            
        # Display response
        st.markdown(ai_response)
        st.caption(f"Relevance score: {score:.2f}")
        if used_web_search:
            st.info("ℹ️ This response was enhanced with web search results from Google.")
        # Show sources expander
        show_sources(relevant_context)
    
    # Add to chat history in session state
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": ai_response,
        "score": score,
        "context": relevant_context
    })
    
    # Save to database
    save_chat_message(
        st.session_state.chat_db,
        st.session_state.conversation_id,
        "assistant",
        ai_response,
        metadata={
            "score": score,
            "model": st.session_state.current_model
        }
    )

if __name__ == "__main__":
    main()