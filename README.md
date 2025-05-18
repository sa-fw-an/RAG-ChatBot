# RAG-ChatBot

<!--markdownlint-disable-->

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, ChromaDB, and a variety of LLMs.  
Upload documents, enter text, or scrape web pages to build a custom knowledge base. Ask questions and get concise, context-aware answers—enhanced with web search when needed—and have new facts learned and stored automatically.

---

## Features

- **Document Ingestion**  
  - PDF, TXT, DOCX, CSV, JSON uploads  
  - Manual text entry  
  - URL scraping via BeautifulSoup  
- **Vector Database**  
  - ChromaDB persistent store  
  - Exact- and semantic-duplicate filtering  
  - Add/update learned facts from conversations  
- **Chat History**  
  - SQLite storage of user/assistant messages  
  - Retrieve recent history for context  
- **LLM Integration**  
  - Configurable models via Groq API (e.g. `deepseek-r1-distill-qwen-32b`, `llama3-70b-8192`)  
  - Embedding model powered by Sentence-Transformers  
  - Response evaluation with semantic similarity  
- **Web Search Augmentation**  
  - Google Custom Search integration  
  - Fetch page content snippets to handle time-sensitive queries  
- **Streamlit UI**  
  - Sidebar for uploads, text & URL inputs  
  - Chat interface with relevance scores  
  - Stats panel: vector chunk count, uploaded files, manual entries, model selector  
  - “View Sources” expander showing KB & web search excerpts  
- **Knowledge Extraction**  
  - Heuristic-based identification of factual statements  
  - Deduplication, contradiction detection, and learning of new facts  

---

## Repository Structure

```
RAG-ChatBot/
├── .streamlit/
│   └── secrets.toml         # API keys (GROQ, Google)
├── modules/                 # Core functionality
│   ├── database.py          # Vector & chat DB management
│   ├── document_processor.py# File & URL text extraction
│   ├── knowledge_extractor.py # Heuristics for extracting facts
│   ├── models.py            # LLM & embedding initialization
│   ├── text_processor.py    # Chunking, cleaning, keyword extraction
│   ├── ui_components.py     # Streamlit sidebar & chat UI
│   └── web_search.py        # Google Custom Search integration
├── .gitignore
├── app.py                   # Streamlit application entrypoint
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

---

## Installation

1. Clone the repo:  
   ```bash
   git clone https://github.com/your-org/RAG-ChatBot.git
   cd RAG-ChatBot
   ```

2. Create & activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure secrets in `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   GOOGLE_API_KEY = "your_google_api_key"
   GOOGLE_CSE_ID   = "your_custom_search_id"
   ```

---

## Usage

```bash
streamlit run app.py
```

1. Use the **Sidebar** to upload files, enter text, or provide a URL.  
2. Watch the status messages as chunks are embedded and stored.  
3. Ask questions in the **Chat** panel—answers draw from uploaded content and optional Google search.  
4. View relevance scores and source excerpts via “View Sources.”  
5. The bot will automatically extract and store new factual knowledge from the conversation.

---

## Module Overview

- **modules/database.py**  
  Initialize/clear ChromaDB and SQLite chat DB; store & retrieve embeddings and messages; deduplicate & learn new facts.

- **modules/document_processor.py**  
  Load and parse PDF, text, DOCX, CSV, JSON files; scrape URLs; return raw text + metadata.

- **modules/knowledge_extractor.py**  
  Heuristics to detect & extract factual sentences from Q&A exchanges.

- **modules/models.py**  
  Initialize embedding (HuggingFaceEmbeddings) and semantic (SentenceTransformer) models; manage LLM clients and response scoring.

- **modules/text_processor.py**  
  Chunk text for embedding; clean whitespace; extract keyword candidates.

- **modules/ui_components.py**  
  Streamlit sidebar controls, chat history rendering, stats panel, and source display.

- **modules/web_search.py**  
  Perform Google Custom Search; fetch page excerpts; merge web results with KB context based on query sensitivity.

---

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add awesome feature"`
4. Push and open a Pull Request.

Please follow PEP8 and include docstrings for new modules/functions.

---

## Requirements

See `requirements.txt`. Key libraries include:

- streamlit  
- chromadb  
- sentence-transformers  
- langchain-huggingface, langchain-groq  
- PyPDF2, docx2txt, bs4, requests  
- sqlite3 (standard library)

---

## License

This project is released under the MIT License.
