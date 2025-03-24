"""
Text processing functionality for chunking and preparing text for embeddings.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=600, chunk_overlap=100):
    """Split the extracted text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks, chunk_size

def clean_text(text):
    """Clean and normalize text for better processing."""
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Basic text normalization
    text = text.strip()
    return text

def extract_keywords(text):
    """Extract potential keywords from text for improved retrieval."""
    # This is a simplified implementation - in a production system,
    # you might use NLTK, spaCy, or other NLP tools
    
    # Remove common stopwords (simplified version)
    stopwords = {"a", "an", "the", "in", "on", "at", "of", "for", "with", "by"}
    words = text.lower().split()
    keywords = [word for word in words if word not in stopwords and len(word) > 3]
    
    # Return top keywords by frequency
    from collections import Counter
    freq = Counter(keywords)
    return [word for word, _ in freq.most_common(10)]