"""
Module for extracting and validating knowledge from chat conversations.
"""
import re
from datetime import datetime
from sentence_transformers import util

def extract_knowledge(user_query, assistant_response):
    """Extract factual knowledge from a chat exchange."""
    # Create a combined text of the exchange
    exchange = f"Q: {user_query}\nA: {assistant_response}"
    
    # Use heuristics to identify if the exchange contains factual knowledge worth storing
    knowledge_worthy = contains_factual_knowledge(exchange)
    
    if knowledge_worthy:
        # Extract the key facts from the exchange
        facts = extract_facts(exchange)
        return facts
    
    return []

def contains_factual_knowledge(exchange):
    """Determine if an exchange contains factual knowledge worth storing."""
    # Knowledge indicators - phrases that suggest factual content
    knowledge_indicators = [
        "is defined as", "means", "refers to", "is a", "are", "consists of",
        "comprises", "contains", "includes", "was invented", "was discovered",
        "was founded", "occurred in", "happened", "works by", "functions as",
        "actually", "in fact", "to clarify", "correction", "the truth is"
    ]
    
    # Correction indicators - phrases that suggest correcting information
    correction_indicators = [
        "correction", "actually", "in fact", "to clarify", "that's not correct",
        "that is incorrect", "this is wrong", "to correct", "I should correct",
        "let me fix", "that's a misconception", "contrary to", "more accurately"
    ]
    
    has_knowledge = any(indicator in exchange.lower() for indicator in knowledge_indicators)
    has_correction = any(indicator in exchange.lower() for indicator in correction_indicators)
    
    return has_knowledge or has_correction

def extract_facts(exchange):
    """Extract factual statements from the exchange."""
    # Better sentence splitting with regex
    sentences = re.split(r'(?<=[.!?])\s+', exchange)
    facts = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        # Skip questions and short statements
        if sentence.endswith("?") or len(sentence.split()) < 5:
            continue
            
        # Skip first-person statements and non-factual statements
        if re.match(r'^(I|We)\s', sentence) or "I think" in sentence or "I believe" in sentence:
            continue
            
        # Skip sentences that don't seem factual
        if not seems_factual(sentence):
            continue
            
        facts.append(sentence)
    
    return facts

def seems_factual(sentence):
    """Determine if a sentence appears to contain factual information."""
    # Keywords that suggest factual content
    factual_keywords = [
        "is", "are", "was", "were", "has", "have", "had", "consists", "contains",
        "comprises", "means", "refers to", "defined as", "founded", "created",
        "discovered", "invented", "developed", "established", "functions", "works"
    ]
    
    # Check if any factual keywords are present
    words = sentence.lower().split()
    return any(keyword in words for keyword in factual_keywords)

def is_contradiction(new_fact, existing_facts, similarity_threshold=0.85):
    """Determine if a new fact contradicts existing knowledge."""
    # This would use the semantic model to check for contradictions
    # For now, we'll return True if there's high similarity but potentially conflicting content
    
    # This implementation would need to be provided the semantic model and more sophisticated logic
    # We'll build the actual function in the database module where we have access to the model
    pass