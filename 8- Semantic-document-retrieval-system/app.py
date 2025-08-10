import os
import shutil
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate


import sys
import streamlit as st

try:
    import chromadb
    st.sidebar.info(f"Python: {sys.executable}\nChromaDB: {chromadb.__version__}")
except Exception as e:
    st.sidebar.error(f"ChromaDB import failed: {e!r}")




# --------- Config ---------
load_dotenv()  # expects OPENAI_API_KEY in .env or st.secrets["OPENAI_API_KEY"]

DATA_PATH = Path("data/uploads")         # where uploaded files go
CHROMA_PATH = Path("chroma")             # where vector store persists
EMBED_MODEL = "text-embedding-3-small"   # keep same everywhere
CHAT_MODEL = "gpt-4o-mini"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
""".strip()

# --------- Helpers ---------
def ensure_dirs():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

def normalize_score(s: float) -> float:
    """
    Try to normalize various similarity/distance score styles into [0,1].
    This is defensive because different lib versions return slightly different scales.
    """
    if -1.0 <= s <= 1.0:
        return (s + 1.0) / 2.0                    # cosine similarity -> [0,1]
    if 0.0 <= s <= 2.0:
        return 1.0 - (s / 2.0)                    # cosine distance -> [0,1]
    if 0.0 <= s <= 1.0:
        return s                                   # already normalized
    # fallback (don’t crash)
    try:
        return max(0.0, min(1.0, float(s)))
    except Exception:
        return 0.0
    

import gc, time, shutil

def hard_reset_vector_store():
    st.cache_resource.clear()
    gc.collect()
    time.sleep(0.2)
    if CHROMA_PATH.exists():
        for _ in range(6):
            try:
                shutil.rmtree(CHROMA_PATH)
                break
            except PermissionError:
                time.sleep(0.5)
                gc.collect()

    # 2) try deleting the folder with a few retries (Windows locks can linger briefly)
    if CHROMA_PATH.exists():
        for _ in range(5):
            try:
                shutil.rmtree(CHROMA_PATH)
                break
            except PermissionError:
                time.sleep(0.5)
                gc.collect()


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL)

from langchain_community.document_loaders import TextLoader

def load_documents_from_dir(dirpath: Path) -> list[Document]:
    docs: list[Document] = []
    if not dirpath.exists():
        return docs

    # Markdown (recursive)
    md_loader = DirectoryLoader(
        str(dirpath),
        glob="**/*.md",
        show_progress=True,
        use_multithreading=True,
    )
    try:
        docs.extend(md_loader.load())
    except Exception as e:
        st.warning(f"Markdown loader issue: {e}")

    # TXT (recursive) — enable encoding autodetect
    txt_loader = DirectoryLoader(
        str(dirpath),
        glob="**/*.txt",
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True, "encoding": "utf-8"},
    )
    try:
        docs.extend(txt_loader.load())
    except Exception as e:
        st.warning(f"Text loader issue: {e}")

    return docs


def split_docs(documents: List[Document], chunk_size=300, chunk_overlap=100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)

def rebuild_index(documents: List[Document]) -> Tuple[int, int]:
    """
    Clears CHROMA_PATH and builds a fresh index. Returns (#docs, #chunks).
    """
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    chunks = split_docs(documents)
    db = Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),
        persist_directory=str(CHROMA_PATH),
    )
    db.persist()
    return len(documents), len(chunks)


@st.cache_resource(show_spinner=False)
def load_db():
    try:
        return Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=get_embeddings()
        )
    except Exception as e:
        # Likely old 1.x index or corrupted/locked DB -> reset folder
        st.warning(
            "Detected an incompatible or corrupted Chroma index. "
            "Resetting the vector store. Go to the Ingest tab and rebuild.\n\n"
            f"Details: {e}"
        )
        hard_reset_vector_store()
        return Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=get_embeddings()
        )



def retrieve(query: str, k: int = 3):
    db = load_db()
    return db.similarity_search_with_relevance_scores(query, k=k)

def answer_from_context(query: str, docs: List[Document]) -> str:
    context_text = "\n\n---\n\n".join([d.page_content for d in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query
    )
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return llm.predict(prompt)

# --------- UI ---------
st.set_page_config(page_title="Semantic Doc Search", page_icon="🔎", layout="wide")
ensure_dirs()

st.title("🔎 Semantic Document Retrieval — Streamlit UI")

with st.sidebar:
    st.header("Settings")
    st.caption("Uses OpenAI embeddings + Chroma. Keep the same embedding model when indexing & querying.")
    embed_model = st.text_input("Embedding model", EMBED_MODEL)
    chat_model = st.text_input("Chat model", CHAT_MODEL)
    if embed_model:
        st.session_state["embed_model"] = embed_model
    if chat_model:
        st.session_state["chat_model"] = chat_model

    st.divider()
    st.write("**Index controls**")
    k = st.slider("Top-k results", min_value=1, max_value=10, value=3, step=1)
    thresh = st.slider("Relevance threshold (normalized 0–1)", 0.0, 1.0, 0.2, 0.01)

    st.divider()
    st.caption("API key source:")
    st.code("OPENAI_API_KEY in .env or Streamlit secrets", language="bash")

tabs = st.tabs(["📄 Ingest", "❓ Query", "🧪 Inspect Index"])

# ---- Tab: Ingest ----
with tabs[0]:
    st.subheader("Upload or point to your docs")
    st.write("Upload `.md` or `.txt` files, then (re)build the index.")

    uploaded = st.file_uploader("Upload files", type=["md", "txt"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            dest = DATA_PATH / f.name
            with open(dest, "wb") as out:
                out.write(f.read())
        st.success(f"Saved {len(uploaded)} file(s) to `{DATA_PATH}`.")

    st.info(f"Current data dir: `{DATA_PATH}`")
    if st.button("🔁 Rebuild Index (from data/uploads)", type="primary"):
        docs = load_documents_from_dir(DATA_PATH)
        if not docs:
            st.error("No documents found. Add .md or .txt files first.")
        else:
            n_docs, n_chunks = rebuild_index(docs)
            st.success(f"Rebuilt index from {n_docs} docs into {n_chunks} chunks.")

    # Peek at a few docs
    if st.checkbox("Preview a few loaded documents"):
        docs = load_documents_from_dir(DATA_PATH)
        st.write(f"Found {len(docs)} doc(s). Showing first 3:")
        for d in docs[:3]:
            st.markdown(f"- **Source**: `{d.metadata.get('source')}`")
            st.code(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))

# ---- Tab: Query ----
with tabs[1]:
    st.subheader("Ask a question")
    query = st.text_input("Your question", placeholder="e.g., What is the capital of France?")
    run = st.button("Search & Answer", type="primary")

    if run and query.strip():
        if not CHROMA_PATH.exists() or not any(CHROMA_PATH.iterdir()):
            st.error("Vector store is empty. Please rebuild the index from the Ingest tab first.")
        else:
            with st.spinner("Retrieving…"):
                raw_results = retrieve(query, k=k)
            ...



        if not raw_results:
            st.warning("No results returned. Did you build the index?")
        else:
            # Normalize scores and filter by threshold
            normed = [(doc, normalize_score(score)) for (doc, score) in raw_results]
            kept = [(d, s) for (d, s) in normed if s >= thresh]

            st.write("### Retrieved Chunks")
            for i, (doc, score) in enumerate(normed, start=1):
                ok = "✅" if score >= thresh else "⚪"
                st.markdown(f"**[{i}] {ok} score={score:.3f}** | source = `{doc.metadata.get('source')}`")
                st.code(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

            if not kept:
                st.warning("All results were below the threshold. Adjust the slider or add better-matching docs.")
            else:
                with st.spinner("Calling LLM for final answer…"):
                    answer = answer_from_context(query, [d for d, _ in kept])
                st.success("Answer")
                st.write(answer)

# ---- Tab: Inspect Index ----
with tabs[2]:
    st.subheader("Index summary")
    if CHROMA_PATH.exists():
        files = list(DATA_PATH.rglob("*.*"))
        st.write(f"- Data dir: `{DATA_PATH}`")
        st.write(f"- Files: {len(files)}")
        st.write(f"- Vector store dir: `{CHROMA_PATH}` (delete to reset)")
        if st.button("🗑️ Delete vector store (clear index)"):
            hard_reset_vector_store()
            st.success("Vector store deleted. Rebuild from the Ingest tab.")


            # shutil.rmtree(CHROMA_PATH)
            # st.success("Deleted vector store. Rebuild when ready.")
    else:
        st.info("No vector store found yet. Build it from the **Ingest** tab.")
