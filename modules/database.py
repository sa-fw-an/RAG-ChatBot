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
import uuid

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
        
    existing_docs_result = collection.get(include=["documents"])
    existing_docs = existing_docs_result.get("documents", [])
    existing_docs_set = set(existing_docs) if existing_docs else set()
    
    new_chunks = [chunk for chunk in chunks if chunk not in existing_docs_set]
    
    if not new_chunks:
        return 0
    
    deduped_chunks = remove_semantic_duplicates(new_chunks, collection, embedding_model)
    
    if not deduped_chunks:
        return 0
        
    with st.sidebar.status("Generating embeddings..."):
        embeddings = [embedding_model.embed_query(chunk) for chunk in deduped_chunks]
    
    base_metadata = metadata or {"source": "unknown"}
    final_ids = []
    final_metadatas = []

    for chunk in deduped_chunks:
        doc_id = f"doc_{uuid.uuid4()}" 
        final_ids.append(doc_id)
        
        chunk_metadata = {
            **base_metadata,
            "chunk_id": doc_id,
            "timestamp": datetime.now().isoformat(),
            "content_hash": str(hash(chunk))
        } 
        final_metadatas.append(chunk_metadata)
    
    collection.add(
        ids=final_ids,
        documents=deduped_chunks,
        embeddings=embeddings,
        metadatas=final_metadatas
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

    formatted_results = []
    
    for i in range(len(documents)):
        doc = documents[i]
        meta = metadatas[i] if i < len(metadatas) else {}
        dist = distances[i] if i < len(distances) else 2.0

        source = meta.get("source", "Unknown source")
        similarity = max(0, 1 - dist / 2)
        
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
    
    cursor.execute(
        "INSERT OR IGNORE INTO conversations (conversation_id) VALUES (?)", 
        (conversation_id,)
    )
    
    cursor.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
        (conversation_id,)
    )
    
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
        return {"added": 0, "updated": 0, "skipped": 0}
        
    metadata_base = {
        "source": "chat_learning",
        "learned_at": datetime.now().isoformat(),
        "context": context or "General conversation"
    }
    
    added_count = 0
    updated_count = 0
    skipped_count = 0
    
    for fact in facts:
        existing_entries = find_related_knowledge(collection, embedding_model, fact)
        
        is_dup = False
        if existing_entries:
            is_dup = is_near_duplicate(fact, existing_entries, semantic_model, threshold=0.95)

        if is_dup:
            skipped_count += 1
            continue
        
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
        
    new_fact_embedding = semantic_model.encode(new_fact, convert_to_tensor=True)
    
    for entry in existing_entries:
        if entry.get("content") is None:
            continue
        
        existing_content_embedding = semantic_model.encode(entry["content"], convert_to_tensor=True)
        
        similarity = util.pytorch_cos_sim(new_fact_embedding, existing_content_embedding)[0][0].item()
        
        if similarity > threshold:
            return True
            
    return False

def find_related_knowledge(collection, embedding_model, query, top_k=3):
    """Find existing knowledge that's related to the new fact."""
    query_embedding = embedding_model.embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    doc_ids = results.get("ids", [[]])[0] if results and results.get("ids") else []
    documents = results.get("documents", [[]])[0] if results and results.get("documents") else []
    metadatas = results.get("metadatas", [[]])[0] if results and results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results and results.get("distances") else []
    
    related_entries = []

    num_results = 0
    if doc_ids:
        num_results = len(doc_ids)
    elif documents:
        num_results = len(documents)

    for i in range(num_results):
        doc_id = doc_ids[i] if i < len(doc_ids) else f"generated_id_{uuid.uuid4()}"
        doc = documents[i] if i < len(documents) else None
        meta = metadatas[i] if i < len(metadatas) else {}
        dist = distances[i] if i < len(distances) else 2.0 

        if doc:
            related_entries.append({
                "content": doc,
                "metadata": meta,
                "id": doc_id,
                "distance": dist
            })
    
    return related_entries

def might_contradict(new_fact, existing_fact, semantic_model, threshold=0.75):
    """Determine if two facts might contradict each other."""
    if not new_fact or not existing_fact:
        return False
    embedding1 = semantic_model.encode(new_fact, convert_to_tensor=True)
    embedding2 = semantic_model.encode(existing_fact, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    
    return 0.5 < similarity < 0.95
    
def update_existing_knowledge(collection, new_fact, existing_entry, embedding_model):
    """
    Update existing knowledge with new information.
    NOTE: This function currently adds a new version and does not delete the old one.
    """
    
    chunks_added = store_embeddings(
        [new_fact],
        collection,
        embedding_model,
        {
            **(existing_entry.get("metadata", {})),
            "updated_at": datetime.now().isoformat(),
            "previous_content": existing_entry.get("content"),
            "is_correction": True,
            "replaces_id_hint": existing_entry.get("id")
        }
    )
    
    return chunks_added > 0


def remove_semantic_duplicates(new_chunks, collection, embedding_model, similarity_threshold=0.92):
    """Remove chunks that are semantically very similar to existing content."""
    if not new_chunks or collection.count() == 0:
        return new_chunks
        
    deduped_chunks = []
    
    for chunk_idx, chunk in enumerate(new_chunks):
        if chunk is None:
            continue

        chunk_embedding = embedding_model.embed_query(chunk)
        
        results = collection.query(
            query_embeddings=[chunk_embedding],
            n_results=1,
            include=["documents", "distances"] 
        )
        
        is_duplicate = False
        if (results and 
            results.get("documents") and 
            results["documents"] and
            results["documents"][0] and
            results["documents"][0][0] is not None):
            
            distances = results.get("distances", [[2.0]])[0] 
            
            if distances and distances[0] is not None:
                similarity = max(0, 1 - (distances[0] / 2.0))
                
                if similarity > similarity_threshold:
                    is_duplicate = True
        
        if not is_duplicate:
            deduped_chunks.append(chunk)
    
    return deduped_chunks