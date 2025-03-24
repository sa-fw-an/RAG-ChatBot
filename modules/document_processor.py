"""
Module for processing different document types and extracting text.
"""
import os
from PyPDF2 import PdfReader
import docx2txt
import csv
import json
import io
import requests
from bs4 import BeautifulSoup
import streamlit as st

def process_file(file):
    """Process an uploaded file and extract its text content."""
    file_type = file.name.split(".")[-1].lower()
    
    processors = {
        "pdf": load_pdf,
        "txt": load_text,
        "docx": load_docx,
        "csv": load_csv,
        "json": load_json
    }
    
    if file_type in processors:
        return processors[file_type](file)
    else:
        st.sidebar.warning(f"Unsupported file type: {file_type}")
        return "", {"source": file.name, "error": "Unsupported file type"}

def load_pdf(file):
    """Load and extract text from a PDF file."""
    try:
        reader = PdfReader(file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        metadata = {
            "source": file.name,
            "type": "pdf",
            "pages": len(reader.pages)
        }
        return text, metadata
    except Exception as e:
        st.sidebar.error(f"⚠️ Error reading PDF: {str(e)}")
        return "", {"source": file.name, "error": str(e)}

def load_text(file):
    """Load and extract text from a text file."""
    try:
        text = file.getvalue().decode("utf-8")
        lines = len(text.split("\n"))
        metadata = {
            "source": file.name,
            "type": "text",
            "lines": lines
        }
        return text, metadata
    except Exception as e:
        st.sidebar.error(f"⚠️ Error reading text file: {str(e)}")
        return "", {"source": file.name, "error": str(e)}

def load_docx(file):
    """Load and extract text from a DOCX file."""
    try:
        text = docx2txt.process(file)
        lines = len(text.split("\n"))
        metadata = {
            "source": file.name,
            "type": "docx",
            "lines": lines
        }
        return text, metadata
    except Exception as e:
        st.sidebar.error(f"⚠️ Error reading DOCX file: {str(e)}")
        return "", {"source": file.name, "error": str(e)}

def load_csv(file):
    """Load and extract text from a CSV file."""
    try:
        csv_reader = csv.reader(io.StringIO(file.getvalue().decode('utf-8')))
        rows = list(csv_reader)
        text = "\n".join([", ".join(row) for row in rows])
        metadata = {
            "source": file.name,
            "type": "csv",
            "rows": len(rows)
        }
        return text, metadata
    except Exception as e:
        st.sidebar.error(f"⚠️ Error reading CSV file: {str(e)}")
        return "", {"source": file.name, "error": str(e)}

def load_json(file):
    """Load and extract text from a JSON file."""
    try:
        data = json.loads(file.getvalue().decode('utf-8'))
        text = json.dumps(data, indent=2)
        lines = len(json.dumps(data).split("\n"))
        metadata = {
            "source": file.name,
            "type": "json",
            "lines": lines
        }
        return text, metadata
    except Exception as e:
        st.sidebar.error(f"⚠️ Error reading JSON file: {str(e)}")
        return "", {"source": file.name, "error": str(e)}

def process_text(text, title=""):
    """Process manually entered text."""
    source = title if title else "Manual entry"
    metadata = {
        "source": source,
        "type": "manual_entry",
        "lines": len(text.split("\n"))
    }
    return text, metadata

def scrape_url(url):
    """Scrape text content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Extract title
        title = soup.title.string if soup.title else url
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        line_count = len(text.splitlines())
        
        metadata = {
            "source": url,
            "type": "url",
            "title": title,
            "lines": line_count
        }
        
        return text, metadata
    except Exception as e:
        st.sidebar.error(f"⚠️ Error scraping URL: {str(e)}")
        return "", {"source": url, "error": str(e)}