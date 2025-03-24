"""
Database management module for both vector database and chat history.
"""
import os
import json
import sqlite3
import chromadb
from datetime import datetime
import streamlit as st
from sentence_transformers import util

# Vector Database Functions
def initialize_vector_db():
    """Initialize ChromaDB client and create/retrieve a collection."""
    if not os.path.exists("./chroma_db_knowledge"):
        os.makedirs("./chroma_db_knowledge", exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db_knowledge")
    collection = chroma_client.get_or_create_collection(name="knowledge_base")
    return collection

def store_embeddings(chunks, collection, embedding_model, metadata=None):
    """Embed and store new text chunks in ChromaDB with enhanced deduplication."""
    if not chunks:
        return 0
        
    # Get existing documents
    existing_docs_result = collection.get(include=["documents"])
    existing_docs = existing_docs_result.get("documents", [])
    existing_docs_set = set(existing_docs) if existing_docs else set()
    
    # Filter out exact duplicates
    new_chunks = [chunk for chunk in chunks if chunk not in existing_docs_set]
    
    if not new_chunks:
        return 0
    
    # Check for semantic duplicates
    deduped_chunks = remove_semantic_duplicates(new_chunks, collection, embedding_model)
    
    if not deduped_chunks:
        return 0
        
    # Generate embeddings for truly new content
    with st.sidebar.status("Generating embeddings..."):
        embeddings = [embedding_model.embed_query(chunk) for chunk in deduped_chunks]
    
    # Generate metadata for each chunk
    base_metadata = metadata or {"source": "unknown"}
    metadatas = [
        {
            **base_metadata,
            "chunk_id": f"chunk_{hash(chunk) % 10000000}",  # Unique ID based on content hash
            "timestamp": datetime.now().isoformat(),
            "content_hash": str(hash(chunk))  # Store hash for future comparisons
        } 
        for chunk, idx in zip(deduped_chunks, range(len(deduped_chunks)))
    ]
    
    # Generate IDs
    ids = [f"doc_{hash(chunk) % 10000000}" for chunk in deduped_chunks]
    
    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=deduped_chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    return len(deduped_chunks)

def retrieve_context(query, embedding_model, collection, top_k=5):
    """Retrieve relevant documents from ChromaDB using embeddings."""
    query_embedding = embedding_model.embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    documents = results.get("documents", [[]])[0] if results else []
    metadatas = results.get("metadatas", [[]])[0] if results else []
    distances = results.get("distances", [[]])[0] if results else []
    
    # Format results with source information and relevance scores
    formatted_results = []
    
    for doc, meta, dist in zip(documents, metadatas, distances):
        source = meta.get("source", "Unknown source")
        # Convert distance to similarity score (1 = identical, 0 = completely different)
        # ChromaDB uses Euclidean distance, so we need to convert it
        similarity = max(0, 1 - dist / 2)  # Simple normalization
        
        formatted_results.append({
            "content": doc,
            "metadata": meta,
            "source": source,
            "similarity": similarity
        })
    
    return formatted_results

def clear_vector_db():
    """Clear all data from the vector database."""
    if not os.path.exists("./chroma_db_knowledge"):
        return False
        
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db_knowledge")
        chroma_client.delete_collection("knowledge_base")
        chroma_client.get_or_create_collection(name="knowledge_base")
        return True
    except Exception as e:
        st.error(f"Error clearing vector database: {str(e)}")
        return False

# Chat Database Functions
def initialize_chat_db():
    """Initialize SQLite database for chat history."""
    db_path = "chat_history.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        role TEXT,
        content TEXT,
        metadata TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    return db_path

def save_chat_message(db_path, conversation_id, role, content, metadata=None):
    """Save a chat message to the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ensure conversation exists
    cursor.execute(
        "INSERT OR IGNORE INTO conversations (conversation_id) VALUES (?)", 
        (conversation_id,)
    )
    
    # Update conversation timestamp
    cursor.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
        (conversation_id,)
    )
    
    # Insert message
    metadata_json = json.dumps(metadata) if metadata else "{}"
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, metadata_json)
    )
    
    conn.commit()
    conn.close()

def get_chat_history(db_path, conversation_id, max_messages=10, max_age_hours=24):
    """Retrieve chat history from the database with relevance filtering."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get recent messages from current conversation
    cursor.execute(
        """
        SELECT role, content, metadata, timestamp 
        FROM messages 
        WHERE conversation_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        """, 
        (conversation_id, max_messages)
    )
    
    messages = []
    for row in cursor.fetchall():
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        messages.append({
            'role': row['role'],
            'content': row['content'],
            'metadata': metadata,
            'timestamp': row['timestamp']
        })
    
    # Reverse to get chronological order
    messages.reverse()
    
    conn.close()
    return messages

def get_all_conversations(db_path, limit=10):
    """Get a list of all conversations in the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT c.conversation_id, c.created_at, c.updated_at,
               COUNT(m.message_id) as message_count
        FROM conversations c
        LEFT JOIN messages m ON c.conversation_id = m.conversation_id
        GROUP BY c.conversation_id
        ORDER BY c.updated_at DESC
        LIMIT ?
        """,
        (limit,)
    )
    
    conversations = []
    for row in cursor.fetchall():
        conversations.append(dict(row))
    
    conn.close()
    return conversations

def clear_chat_history():
    """Clear all chat history from the database."""
    db_path = "chat_history.db"
    if not os.path.exists(db_path):
        return False
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM conversations")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error clearing chat history: {str(e)}")
        return False
    
def store_learned_knowledge(collection, embedding_model, semantic_model, facts, context=None):
    """Store knowledge learned from conversations with enhanced deduplication."""
    if not facts:
        return {"added": 0, "updated": 0}
        
    # Metadata for new facts
    metadata_base = {
        "source": "chat_learning",
        "learned_at": datetime.now().isoformat(),
        "context": context or "General conversation"
    }
    
    # Track new and updated facts
    added_count = 0
    updated_count = 0
    skipped_count = 0
    
    # Process each fact
    for fact in facts:
        # Check for existing similar facts
        existing_entries = find_related_knowledge(collection, embedding_model, fact)
        
        # If we found a very similar fact (near duplicate)
        if is_near_duplicate(fact, existing_entries, semantic_model, threshold=0.95):
            skipped_count += 1
            continue
            
        # If we found a contradicting fact (needs updating)
        elif might_contradict(fact, existing_entries[0]['content'], semantic_model) if existing_entries else False:
            # Update existing knowledge
            updated = update_existing_knowledge(collection, fact, existing_entries[0], embedding_model)
            if updated:
                updated_count += 1
            
        # This is genuinely new information
        else:
            # Store as new fact
            chunks_added = store_embeddings(
                [fact], 
                collection, 
                embedding_model, 
                metadata_base
            )
            added_count += chunks_added
    
    return {
        "added": added_count,
        "updated": updated_count,
        "skipped": skipped_count
    }

def is_near_duplicate(new_fact, existing_entries, semantic_model, threshold=0.95):
    """Check if a fact is nearly identical to an existing entry."""
    if not existing_entries:
        return False
        
    # Check each existing entry for near-duplicates
    for entry in existing_entries:
        embedding1 = semantic_model.encode(new_fact, convert_to_tensor=True)
        embedding2 = semantic_model.encode(entry["content"], convert_to_tensor=True)
        
        similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
        
        # If very similar (above threshold), it's a near-duplicate
        if similarity > threshold:
            return True
            
    return False

def find_related_knowledge(collection, embedding_model, query, top_k=3):
    """Find existing knowledge that's related to the new fact."""
    query_embedding = embedding_model.embed_query(query)
    
    # Remove 'ids' from the include list - not supported in this ChromaDB version
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    documents = results.get("documents", [[]])[0] if results else []
    metadatas = results.get("metadatas", [[]])[0] if results else []
    distances = results.get("distances", [[]])[0] if results else []
    
    # Generate temporary IDs based on content - not ideal but works as a workaround
    # In a production system, you would want to use a stable ID system
    related_entries = []
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        if doc:
            # Generate a temporary ID based on content hash
            temp_id = f"doc_{hash(doc) % 10000000}"
            
            related_entries.append({
                "content": doc,
                "metadata": meta,
                "id": temp_id,  # Using a temporary ID
                "distance": dist
            })
    
    return related_entries

def might_contradict(new_fact, existing_fact, semantic_model, threshold=0.75):
    """Determine if two facts might contradict each other."""
    # High similarity but not identical suggests potential contradiction
    embedding1 = semantic_model.encode(new_fact, convert_to_tensor=True)
    embedding2 = semantic_model.encode(existing_fact, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    
    # Check for high similarity but not identical
    # This suggests the facts are about the same topic but might have different info
    return 0.5 < similarity < 0.95
    
def update_existing_knowledge(collection, new_fact, existing_entry, embedding_model):
    """Update existing knowledge with new information."""
    # Since we don't have real IDs, we'll need to find the document by content
    existing_content = existing_entry["content"]
    new_embedding = embedding_model.embed_query(new_fact)
    
    # First, we search for the exact content
    results = collection.query(
        query_texts=[existing_content],
        n_results=1,
        include=["documents", "metadatas"]
    )
    
    documents = results.get("documents", [[]])[0]
    
    if documents and documents[0] == existing_content:
        # We found the exact document, now we can modify it
        # But we need to delete and re-add since we don't have an ID
        
        # Add the new version with updated metadata
        chunks_added = store_embeddings(
            [new_fact],
            collection,
            embedding_model,
            {
                **existing_entry["metadata"],
                "updated_at": datetime.now().isoformat(),
                "previous_content": existing_content,
                "is_correction": True
            }
        )
        
        # We successfully updated by adding the new version
        return True
    
    # We couldn't find the exact document to update
    return False

def remove_semantic_duplicates(new_chunks, collection, embedding_model, similarity_threshold=0.92):
    """Remove chunks that are semantically very similar to existing content."""
    if not new_chunks or collection.count() == 0:
        return new_chunks
        
    deduped_chunks = []
    
    for chunk in new_chunks:
        # Create embedding for this chunk
        chunk_embedding = embedding_model.embed_query(chunk)
        
        # Query for similar existing chunks
        results = collection.query(
            query_embeddings=[chunk_embedding],
            n_results=1,
            include=["documents", "distances"]
        )
        
        # Check if there's a very similar document already
        if results and results.get("documents") and results["documents"][0]:
            distances = results.get("distances", [[1.0]])[0]
            
            if distances and distances[0]:
                similarity = 1.0 - (distances[0] / 2.0)  # Convert distance to similarity
                
                # If very similar, skip this chunk
                if similarity > similarity_threshold:
                    continue
        
        # If we reach here, this chunk is unique enough
        deduped_chunks.append(chunk)
    
    return deduped_chunks