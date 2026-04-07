# PageIndex Chat Demo

A drag-and-drop PDF chatbot built on top of [PageIndex](https://github.com/VectifyAI/PageIndex) — a vectorless, reasoning-based RAG system that uses hierarchical tree indexing instead of vector similarity search.

## What This Is

This is a personal experiment to explore **reasoning-based RAG** as an alternative to traditional vector database retrieval. It wraps the open-source PageIndex framework in a Streamlit chat interface where you can upload PDFs, index them, and ask questions — with the LLM navigating the document like a human expert would.

**Built on top of:** [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) (MIT License)

## How It Works

1. **Upload a PDF** — drop any document into the sidebar
2. **Automatic indexing** — PageIndex reads the document and builds a hierarchical tree structure (like a smart table of contents), with section titles, page ranges, and summaries
3. **Chat with your document** — ask questions and an agentic LLM navigates the tree, fetches relevant pages, and answers with page-level citations

### Why Not Vector Search?

| Traditional Vector RAG | PageIndex (This Project) |
|----------------------|--------------------------|
| Chunks documents into ~500 token pieces | Preserves natural document structure |
| Retrieves by embedding similarity | Retrieves by LLM reasoning over a tree index |
| Opaque "top-k nearest neighbors" | Fully traceable — shows which pages were consulted and why |
| Requires a vector database | Index is a single JSON file |
| Similarity ≠ relevance | Reasoning-based retrieval finds what's actually relevant |

### Agentic Multi-Tool Retrieval

The chat uses an agentic approach — the LLM gets three tools and can call them iteratively:

- `get_document()` — check document metadata and page count
- `get_document_structure()` — read the tree index to find relevant sections
- `get_page_content(pages="5-7")` — fetch specific pages

The agent makes multiple small fetches (like a human flipping through a book) rather than one massive retrieval, so there's no hard page limit and no context window overflow.

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key (or any LLM provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers))

### Installation

```bash
git clone https://github.com/pc9350/pageIndex-Demo.git
cd pageIndex-Demo
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
pip install streamlit openai-agents
```

### Configuration

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_key_here
```

The default model is `gpt-4o-2024-11-20` (configurable in `pageindex/config.yaml`).

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
├── app.py                  # Streamlit chat interface
├── pageindex/
│   ├── page_index.py       # Core tree index generation
│   ├── page_index_md.py    # Markdown support
│   ├── client.py           # PageIndexClient for indexing & retrieval
│   ├── retrieve.py         # Document/page retrieval tools
│   ├── utils.py            # LLM calls, JSON parsing, tree utilities
│   └── config.yaml         # Model and indexing configuration
├── examples/
│   ├── agentic_vectorless_rag_demo.py  # CLI-based agentic RAG example
│   └── documents/          # Sample PDFs with pre-generated indexes
├── cookbook/                # Jupyter notebook tutorials
├── requirements.txt
└── .env                    # Your API key (not committed)
```

## What I Changed From the Original

- **`app.py`** — New Streamlit chat interface with drag-and-drop upload, agentic multi-tool retrieval, streaming answers, and an explainer landing page
- **`pageindex/utils.py`** — Fixed `extract_json` to handle invalid backslash escapes from LaTeX-heavy PDFs
- **`pageindex/page_index.py`** — Added defensive type checks in `process_no_toc` to handle edge cases where JSON parsing returns a dict instead of a list

## Credits

This project is built on [PageIndex](https://github.com/VectifyAI/PageIndex) by [VectifyAI](https://github.com/VectifyAI), licensed under the MIT License.

> Mingtian Zhang, Yu Tang and PageIndex Team, "PageIndex: Next-Generation Vectorless, Reasoning-based RAG", PageIndex Blog, Sep 2025.

## License

MIT — same as the original PageIndex project.
