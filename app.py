import streamlit as st
import json
import os
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from pageindex import PageIndexClient
from pageindex.utils import extract_json, remove_fields

st.set_page_config(
    page_title="PageIndex Chat",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

WORKSPACE = Path("./demo_workspace")
UPLOAD_DIR = WORKSPACE / "uploads"


@st.cache_resource
def get_client():
    return PageIndexClient(workspace=WORKSPACE)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Get document metadata: name, description, page count, and status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_structure",
            "description": (
                "Get the document's tree structure index — section titles and page ranges — "
                "to identify which sections are relevant to the question."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": (
                "Retrieve the text content of specific pages. Use tight ranges and make "
                "multiple calls for different sections rather than one huge fetch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pages": {
                        "type": "string",
                        "description": (
                            "Page numbers or ranges, e.g. '5-7' for pages 5–7, "
                            "'3,8' for pages 3 and 8, '12' for page 12. "
                            "Keep each call to ~10 pages max."
                        ),
                    }
                },
                "required": ["pages"],
            },
        },
    },
]

AGENT_SYSTEM_PROMPT = """\
You are PageIndex, a document QA assistant that navigates documents like a human expert.

TOOL USE:
- Call get_document() first to check the document status and page count.
- Call get_document_structure() to see the section hierarchy and find relevant page ranges.
- Call get_page_content(pages="5-7") to read specific pages. Keep ranges tight (~10 pages) \
but make multiple calls if you need content from different sections.
- Before each tool call, write one sentence explaining what you're looking for.

Be thorough — if the answer spans multiple sections, fetch them all. \
You may call tools as many times as needed.
Answer based only on retrieved content. Give detailed answers and cite page numbers."""

MAX_TOOL_ROUNDS = 10


def slim_structure(structure):
    """Strip summaries and text to keep only titles + page ranges for navigation."""
    return remove_fields(structure, fields=["summary", "text", "node_id"])


def execute_tool(name, args, client, doc_id):
    if name == "get_document":
        return client.get_document(doc_id)
    if name == "get_document_structure":
        structure = json.loads(client.get_document_structure(doc_id))
        return json.dumps(slim_structure(structure), ensure_ascii=False)
    if name == "get_page_content":
        return client.get_page_content(doc_id, args.get("pages", "1"))
    return json.dumps({"error": f"Unknown tool: {name}"})


def run_agent_tools(model, doc_id, question, history, client, status_container):
    """Run the agentic tool-calling loop. Returns (messages, pages_consulted)."""
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for m in history[-6:]:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})

    pages_consulted = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = litellm.completion(
            model=model, messages=messages, tools=TOOLS, temperature=0
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            result = execute_tool(name, args, client, doc_id)

            if name == "get_page_content":
                pages_consulted.append(args.get("pages", ""))

            label = f"🔧 **{name}**"
            if args:
                label += f"({', '.join(f'{k}=\"{v}\"' for k, v in args.items())})"
            status_container.write(label)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return messages, pages_consulted


def stream_final_answer(model, messages):
    """Stream the agent's final text answer after all tool calls are complete."""
    stream = litellm.completion(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="none",
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def render_tree(nodes, level=0):
    for node in nodes:
        indent = "\u2003" * level
        marker = "├─ " if level > 0 else ""
        pages = f"pp. {node.get('start_index', '?')}–{node.get('end_index', '?')}"
        summary = node.get("summary", "")
        tip = summary[:120] + "…" if len(summary) > 120 else summary
        st.markdown(
            f"{indent}{marker}**{node.get('title', 'Untitled')}** &nbsp; `{pages}`"
        )
        if tip:
            st.caption(f"{indent}{'  ' * bool(level)}{tip}")
        if node.get("nodes"):
            render_tree(node["nodes"], level + 1)


# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

client = get_client()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📑 PageIndex Chat")
    st.caption("Vectorless, reasoning-based document QA")
    st.divider()

    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Drop a PDF to index and chat with it",
    )

    if uploaded is not None:
        already = next(
            (
                did
                for did, d in client.documents.items()
                if d.get("doc_name") == uploaded.name
            ),
            None,
        )
        if already:
            st.session_state.active_doc = already
            st.success(f"✓ {uploaded.name} already indexed")
        else:
            with st.status("Indexing document…", expanded=True) as status:
                st.write("Saving file…")
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                file_path = UPLOAD_DIR / uploaded.name
                file_path.write_bytes(uploaded.getvalue())

                st.write(f"Building tree index for **{uploaded.name}**…")
                st.write("This may take a few minutes for large documents.")
                try:
                    doc_id = client.index(str(file_path))
                    st.session_state.active_doc = doc_id
                    status.update(label="Indexing complete!", state="complete")
                except Exception as e:
                    status.update(label="Indexing failed", state="error")
                    st.error(str(e))

    st.divider()
    st.subheader("Documents")

    if not client.documents:
        st.info("No documents yet — upload a PDF above.")
    else:
        for doc_id, doc in client.documents.items():
            name = doc.get("doc_name", doc_id[:8])
            is_active = st.session_state.active_doc == doc_id
            if st.button(
                f"{'📖 ' if is_active else '📄 '}{name}",
                key=f"btn_{doc_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_doc = doc_id
                st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────
if st.session_state.active_doc is None:
    st.markdown(
        """
    # 📑 PageIndex Chat
    **Reasoning-based RAG** — no vector database, no chunking, human-like retrieval.

    Upload a PDF in the sidebar to get started.
    """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. Upload")
        st.markdown("Drop any PDF document into the sidebar.")
    with col2:
        st.markdown("### 2. Index")
        st.markdown(
            "PageIndex builds a hierarchical tree structure — like a smart table of contents."
        )
    with col3:
        st.markdown("### 3. Chat")
        st.markdown(
            "Ask questions and get answers with page-level citations. No vectors involved."
        )

    st.divider()

    st.markdown("## How It Works")
    st.markdown(
        "Traditional RAG systems chop your documents into small chunks, embed them "
        "as vectors, and retrieve by **similarity search**. This sounds reasonable, "
        "but similarity is not the same as relevance — especially for professional "
        "documents that require domain expertise and multi-step reasoning."
    )
    st.markdown(
        "**PageIndex** takes a completely different approach, inspired by how human "
        "experts actually read documents:"
    )

    step1, step2 = st.columns(2)
    with step1:
        st.markdown("#### 🌲 Step 1: Build a Tree Index")
        st.markdown(
            "When you upload a PDF, an LLM reads through it and generates a "
            "**hierarchical tree structure** — like an intelligent table of contents. "
            "Each node has a title, page range, and a summary of what that section covers. "
            "This is a one-time cost per document."
        )
    with step2:
        st.markdown("#### 🔍 Step 2: Reason Over the Tree")
        st.markdown(
            "When you ask a question, the LLM **reads the tree structure** and reasons "
            "about which sections are relevant — just like a human skimming a table of "
            "contents. It then fetches only the relevant pages and answers from those."
        )

    st.divider()

    st.markdown("## Why This Beats Vector-Based RAG")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**No Vector Database**
No embeddings, no Pinecone/Weaviate/Qdrant, no infrastructure to manage.
Your entire index is a single JSON file.

**No Chunking**
Documents keep their natural structure. No more tables split across chunks
or context lost at arbitrary 500-token boundaries.

**Better on Long Documents**
Vector RAG gets noisier with more pages. PageIndex gets *better* — more
hierarchy means more precise navigation.
"""
        )
    with c2:
        st.markdown(
            """
**Reasoning > Similarity**
Vector search finds text that *looks like* your query. PageIndex finds text
that *answers* your query — even when the wording is completely different.

**Fully Traceable**
Every answer shows exactly which pages were consulted and why. No more
opaque "top-5 nearest neighbors" — you can audit every retrieval decision.

**Future-Proof**
As LLMs get smarter, retrieval improves automatically. No need to
re-embed your entire corpus when a better model comes out.
"""
        )

    st.divider()

    st.markdown("## Performance")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("FinanceBench Accuracy", "98.7%", "State of the art")
    with m2:
        st.metric("Infrastructure Needed", "0", "No vector DB")
    with m3:
        st.metric("Index Format", "JSON", "Portable & inspectable")

else:
    doc_id = st.session_state.active_doc
    doc_info_raw = client.get_document(doc_id)
    doc_info = json.loads(doc_info_raw)

    header_col, btn_col = st.columns([5, 1])
    with header_col:
        st.title(f"📄 {doc_info.get('doc_name', 'Document')}")
    with btn_col:
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages[doc_id] = []
            st.rerun()

    desc = doc_info.get("doc_description", "")
    if desc:
        st.caption(desc)

    page_count = doc_info.get("page_count", doc_info.get("line_count", "?"))
    st.caption(f"{page_count} pages")

    with st.expander("View document tree structure"):
        structure = json.loads(client.get_document_structure(doc_id))
        if structure:
            render_tree(structure)
        else:
            st.write("No structure available.")

    st.divider()

    # Chat history
    if doc_id not in st.session_state.messages:
        st.session_state.messages[doc_id] = []

    messages = st.session_state.messages[doc_id]

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("pages_consulted"):
                st.caption(f"📖 Pages consulted: {msg['pages_consulted']}")

    if question := st.chat_input("Ask about this document…"):
        with st.chat_message("user"):
            st.markdown(question)
        messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.status("Reasoning over document…", expanded=True) as status:
                agent_messages, pages_consulted = run_agent_tools(
                    client.model, doc_id, question, messages, client, status
                )
                pages_str = ", ".join(pages_consulted) if pages_consulted else "none"
                status.update(
                    label=f"Retrieved pages: {pages_str}", state="complete",
                    expanded=False,
                )

            full_response = st.write_stream(
                stream_final_answer(client.model, agent_messages)
            )

            if pages_consulted:
                st.caption(f"📖 Pages consulted: {pages_str}")

        messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "pages_consulted": pages_str if pages_consulted else None,
            }
        )
