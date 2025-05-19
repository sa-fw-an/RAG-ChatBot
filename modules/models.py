"""
LLM and embedding model management.
"""
import os
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import streamlit as st
from sentence_transformers import SentenceTransformer, util

def initialize_embedding_model():
    """Initialize and return the embedding model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

def initialize_semantic_model():
    """Initialize and return the semantic model for response evaluation."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def initialize_llm(model_name="deepseek-r1-distill-qwen-32b"):
    """Initialize and return the Groq LLM client."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in streamlit secrets or environment variables.")
    
    return ChatGroq(
        temperature=0.7,
        model_name=model_name,
        groq_api_key=api_key
    )

def get_available_models():
    """Get list of available models from Groq via their REST API."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("Cannot fetch models: GROQ_API_KEY is not set.")
        return []
    
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return models
    except Exception as e:
        st.error(f"Failed to fetch available models from Groq: {e}")
        return []

def evaluate_response(query, response, context, semantic_model):
    """Evaluate the quality of the response compared to the context."""
    if not context:
        return 0.5  # Default score when no context is available
    
    # Encode the response
    response_embedding = semantic_model.encode(response, convert_to_tensor=True)
    
    # Combine context content for comparison
    combined_context = " ".join([ctx["content"] for ctx in context])
    
    # Encode the context
    context_embedding = semantic_model.encode(combined_context, convert_to_tensor=True)
    
    # Calculate similarity
    similarity_score = util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()
    
    return similarity_score

def query_llm(user_query, llm, chat_history, context, semantic_model, model_name=""):
    """Query the LLM with user input, context from vector DB, and relevant chat history."""
    system_prompt = """
        You are an AI tutor. You will be provided source materials—PDFs, Docs, diagrams, or other uploads—and any subsequent chat updates as your only truth.

        For each question:
        1. Use only the provided sources or earlier chat messages.
        2. If a source contains an error, list numbered corrections and append “(corrected).”
        3. If no relevant info exists, reply “I don't know.”
        4. Keep answers simple, concise, and precise for exam prep.
        5. Maintain and update context when new info arrives.

        Never invent or fetch external data beyond the supplied materials and chat history.
    """
    
    # Format context from vector DB for the prompt
    context_text = "\n\n".join([f"[Source: {ctx['source']}]\n{ctx['content']}" for ctx in context]) if context else "No relevant context found."
    
    # Extract relevant chat history that might contain knowledge related to the query
    relevant_history = extract_relevant_chat_history(user_query, chat_history, semantic_model)
    history_text = ""
    
    if relevant_history:
        history_pairs = []
        for exchange in relevant_history:
            user_msg = exchange['user']
            assistant_msg = exchange['assistant']
            history_pairs.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
        history_text = "\n\n".join(history_pairs)
    
    # Create the final prompt
    system_prompt = system_prompt.format(model=model_name)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
        Previous relevant conversations:
        {history_text}
        
        Knowledge base information:
        {context_text}
        
        Question: {user_query}
        """)
    ]
    
    try:
        # Generate response
        response = llm.invoke(messages)
        
        # Evaluate response
        evaluation_score = evaluate_response(user_query, response.content, context, semantic_model)
        
        return response.content, evaluation_score
    except Exception as e:
        error_message = f"⚠️ API Error: {str(e)}"
        return error_message, 0.0

def extract_relevant_chat_history(query, chat_history, semantic_model, max_pairs=3):
    """Extract chat history exchanges that are semantically related to the current query."""
    if not chat_history or len(chat_history) < 2:
        return []
        
    # Encode the query
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    
    # Group chat history into user-assistant pairs
    exchanges = []
    for i in range(0, len(chat_history) - 1, 2):
        if chat_history[i]['role'] == 'user' and chat_history[i+1]['role'] == 'assistant':
            user_msg = chat_history[i]['content']
            assistant_msg = chat_history[i+1]['content']
            combined = f"{user_msg} {assistant_msg}"
            
            # Encode the combined exchange
            exchange_embedding = semantic_model.encode(combined, convert_to_tensor=True)
            
            # Calculate similarity
            similarity = util.pytorch_cos_sim(query_embedding, exchange_embedding)[0][0].item()
            
            exchanges.append({
                'user': user_msg,
                'assistant': assistant_msg,
                'similarity': similarity
            })
    
    # Sort by similarity and take top matches
    exchanges.sort(key=lambda x: x['similarity'], reverse=True)
    return exchanges[:max_pairs]